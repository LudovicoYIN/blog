---
author: Ludovico
pubDatetime: 2026-05-28T14:00:00Z
title: TVM FFI（二）：类型系统与容器
featured: false
draft: false
tags: [论文精读, 量化]
  - AI编译器
description: Any/AnyView 所有权语义、Array/Map/Variant/Expected 容器、结构相等性与哈希、反射系统与 Dataclass。
---

[上一篇](/posts/tvm-ffi-deep-dive/) 讲了 TVM FFI 的骨架：对象系统、类型擦除、Packed Function 和 DLPack。这篇继续往下挖，覆盖数据层——值怎么在不同语言之间安全高效地传递。

## Any vs AnyView：值容器的所有权模型

`TVMFFIAny` 是 16 字节的带标签联合体，这是我们之前看到的 C 结构体。在 C++ API 层，它被包装成了两个不同的类：

```cpp
// AnyView — 借用语义，零开销
class AnyView {
    TVMFFIAny data_;  // 16 字节，直接嵌入
public:
    // 析构时什么都不做，不管理生命周期
    ~AnyView() = default;

    // 可以构造、拷贝、移动，不触发引用计数变化
    // 唯一需要做的就是在设为 None 时清空联合体
    void reset() {
        data_.type_index = kTVMFFINone;
        data_.zero_padding = 0;
        data_.v_int64 = 0;
    }
};

// Any — 拥有语义，析构时释放资源
class Any : public AnyView {
    // 析构时检查 type_index：
    //   如果是 Object → DecRef
    //   如果是 Tensor/String 等容器 → 析构容器
    //   如果是 int/float → 什么都不做
    ~Any() { Destroy(); }
};
```

两者的 C 内存布局完全一样，区别只在于**谁负责释放资源**：

| | `AnyView` | `Any` |
|---|---|---|
| 所有权 | 借用，不管生命周期 | 拥有，析构时释放 |
| 用途 | 函数参数（不收不释） | 返回值、存储（调用方接管） |
| 性能 | 零开销（无析构逻辑） | 有析构开销 |
| 与 C ABI 关系 | `const TVMFFIAny*` | `TVMFFIAny*`（调用方负责 `Destroy`） |

这就是为什么 packed function 签名长这样：

```c
typedef int (*TVMFFISafeCallType)(
    void* self,
    const TVMFFIAny* args,   // AnyView — 只是借用
    int32_t num_args,
    TVMFFIAny* result         // 调用方会接管这个值的所有权
);
```

参数全部是借用的（不需要释放），返回值是所有权转移（调用方负责释放）。这套设计避免了跨语言边界时的"谁负责释放"争议。

## 容器类型一览

TVM FFI 提供了一套通用的跨语言容器，全部在 [include/tvm/ffi/container/](https://github.com/apache/tvm/blob/main/3rdparty/tvm-ffi/include/tvm/ffi/container/) 下。它们的设计原则：**不可变容器用 COW（Copy-on-Write），可变容器直接暴露 API**。

### Array（不可变）

```cpp
// 类似 Python tuple 或 C++ std::vector（但不可变）
ffi::Array<PrimExpr> arr = {expr_a, expr_b, expr_c};

// 读操作
arr.size();     // 3
arr[0];         // expr_a
arr.empty();    // false

// "修改"操作创建新数组（COW）
auto arr2 = arr.push_back(expr_d);  // arr 不变，arr2 是新的

// Python 侧
// list / tuple 自动映射为 Array
mod.func([1, 2, 3])  // Python list → ffi::Array<IntImm>
```

底层是引用计数的堆对象，修改时先拷贝再写。适合 IR 节点的子节点列表这种读多写少的场景。

### List（可变）

```cpp
// 类似 Python list，可变
ffi::List<PrimExpr> list;
list.push_back(expr_a);
list.push_back(expr_b);
list[1] = expr_c;         // 原地修改
list.pop_back();
```

### Map（不可变）

```cpp
ffi::Map<ffi::String, PrimExpr> map;
auto map2 = map.insert("key", value);  // COW，返回新 Map
auto val = map2.get("key");            // 返回 Optional<PrimExpr>
```

### Dict（可变）

```cpp
ffi::Dict<ffi::String, PrimExpr> dict;
dict.set("key", value);   // 原地修改
dict.size();
dict.erase("key");
```

### String / Bytes

```cpp
// SmallStr — 短字符串内联存储（≤15 字节），避免堆分配
// String — 长字符串堆分配，引用计数
ffi::String s = "hello world";
s.size();     // 11
s.data();     // const char*
s.empty();    // false
```

Python 的 `str` / `bytes` 在传给 FFI 函数时会自动零拷贝映射为 `ffi::String` / `ffi::Bytes`。

### Tensor / Shape

```cpp
// Shape — 轻量形状描述
ffi::Shape shape = {1, 3, 224, 224};
shape.size();    // 4
shape[2];        // 224

// Tensor — 拥有数据内存的张量
ffi::Tensor tensor = ffi::Tensor::FromDLPack(dl_managed_tensor);
tensor.ndim();          // 几维
tensor.shape();         // 返回 Shape
tensor.dtype();         // 返回 DLDataType
tensor.data_ptr();      // void* — 指向实际数据
```

### Variant——类型安全的 union

```cpp
// 类似 C++17 的 std::variant，但支持跨语言
ffi::Variant<int64_t, double, String> v;

// 构造
v = 42;
v = 3.14;
v = String("hello");

// 类型检查与提取
if (auto* i = v.as<int64_t>()) { /* 是 int */ }
else if (auto* d = v.as<double>()) { /* 是 double */ }
```

底层用 `Any` 存储值，额外做了编译时类型检查。Python 侧同样支持：

```python
v = Variant(42)
assert isinstance(v, Variant)
```

### Expected——Rust 风格的 Result

对于可能失败的 API，tvm-ffi 提供了 `Expected<T, E>`：

```cpp
// 成功
Expected<int64_t> Divide(int64_t a, int64_t b) {
    if (b == 0)
        return Unexpected(Error("division by zero"));
    return a / b;
}

// 调用方
auto result = Divide(10, 2);
if (result) {
    int64_t value = *result;   // 5
} else {
    auto& err = result.error(); // Error 对象
}
```

比异常更显式，适合性能敏感的路径。

## 结构相等性与哈希

编译器里高频操作——比较两棵 IR 树是否"内容相同"。tvm-ffi 提供了一套遍历 IR 树的深度比较机制：

```cpp
// StructuralEqual — 深度比较两棵树
Any expr_a = ...;  // 一棵 (x + 1) * 2 的 IR 树
Any expr_b = ...;  // 另一棵同样的 IR 树，但指针不同

bool equal = StructuralEqual::Equal(expr_a, expr_b);
// equal == true — 因为 AST 结构相同

bool same_ptr = expr_a == expr_b;
// same_ptr == false — 因为是不同的堆对象
```

还支持找到第一个不匹配的位置：

```cpp
auto mismatch = StructuralEqual::GetFirstMismatch(expr_a, expr_b);
if (mismatch) {
    // mismatch->lhs 和 mismatch->rhs 是 AccessPath，指向不匹配的具体字段
    // 例如 "func.params[0].dtype"
}
```

StructuralHash 则对整棵树做哈希：

```cpp
int64_t hash = StructuralHash()(expr);
// 两棵结构相同的树会得到相同的哈希值
```

这套机制支撑了 TVM 编译器中所有的 Common Subexpression Elimination（CSE）、死代码消除（DCE）、以及各种子图匹配 pass。

两者都暴露到 Python：

```python
import tvm_ffi.structural as structural

equal = structural.equal(expr_a, expr_b)
h = structural.hash(expr)
```

## 反射系统

如何让 C++ 对象向 Python 暴露字段和方法？答案在 `reflection` 模块。

### ObjectDef——在 C++ 侧注册类型信息

```cpp
// 每个 Object 子类注册自己的字段
class IntImmNode : public PrimExprNode {
public:
    int64_t value;
    DataType dtype;

    static void RegisterReflection() {
        refl::ObjectDef<IntImmNode>()
            .def_ro("value", &IntImmNode::value)   // 只读字段
            .def_ro("dtype", &IntImmNode::dtype);
    }
};
```

注册后 Python 就能直接访问：

```python
imm = IntImm("int32", 42)
print(imm.value)   # 42 — 直接读 C++ 内存
print(imm.dtype)   # int32
```

### 全局函数注册

```cpp
// C++ 侧注册
TVM_FFI_REGISTER_GLOBAL_FUNC("math.add")
    .set_body([](int64_t a, int64_t b) {
        return a + b;
    });
```

```python
# Python 侧调用
result = tvm_ffi.get_global_func("math.add")(1, 2)
# result == 3
```

全局注册表是一个字符串到函数的映射，所有语言共享同一个命名空间。

### Dataclass 系统

[dataclass.h](https://github.com/apache/tvm/blob/main/3rdparty/tvm-ffi/include/tvm/ffi/extra/dataclass.h) 让 C++ 对象表现得像 Python `@dataclass`：

```python
from tvm_ffi import register_object, dataclasses

@register_object("ir.IntImm")
@dataclasses.c_class   # 自动生成 __init__, __repr__, __eq__, __hash__
class IntImm(Object):
    value: int
    dtype: DataType

# 用起来和普通 Python dataclass 一样
imm = IntImm(value=42, dtype="int32")
print(repr(imm))  # IntImm(value=42, dtype=int32)
```

`c_class` 装饰器根据 C++ 的反射信息自动生成这些方法，不需要手写任何胶水代码。

## 序列化

TVM FFI 支持两种序列化格式：

- **JSON**：人类可读，用于调试、日志、配置文件
- **Base64 编码的 FlatBuffer**：紧凑，用于缓存、网络传输

两者通过同一套 API：

```cpp
// 序列化
ffi::String json_str = SerializeToJSON(ir_module);
ffi::String flatbuffer_base64 = SerializeToBase64(ir_module);

// 反序列化
IRModule mod = DeserializeFromJSON(json_str);
IRModule mod = DeserializeFromBase64(flatbuffer_base64);
```

Python 侧同样支持：

```python
import tvm_ffi.serialization as ser

json_str = ser.dumps(ir_module)
mod = ser.loads(json_str)
```

## 本系列小结

第一篇讲了骨架（对象系统、类型擦除、packed function、DLPack），这一篇讲了数据层（容器、Any/AnyView、结构相等性、反射系统）。

还有最后一块拼图——生态与工程工具（Python/Rust 绑定、JIT 编译、stubgen、addons）——将在下一篇展开。
