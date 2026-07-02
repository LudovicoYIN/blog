---
author: Ludovico
pubDatetime: 2026-05-28T12:00:00Z
title: TVM FFI（一）：对象系统与调用约定
featured: true
draft: false
tags:
  - AI编译器
  - TVM
description: TVM FFI 的核心骨架：引用计数对象系统、类型擦除容器 TVMFFIAny、Packed Function 调用约定，以及 DLPack 零拷贝张量传递。
---

## 为什么需要 TVM FFI？

机器学习生态的碎片化：不同框架（PyTorch、JAX、PaddlePaddle）、不同语言（C++、Python、Rust）、不同硬件后端（CUDA、Metal、Vulkan），各有一套内部表示。一个 kernel 库作者如果想同时支持 PyTorch 和 JAX，通常需要为每个框架分别写绑定代码。

TVM FFI 解决的就是这个问题——它定义了一套**稳定的 C ABI 和一组跨语言的 FFI 工具**，让任何框架、任何语言、任何后端都能通过统一的接口交互。TVM 编译器本身的整个 IR 节点体系，也构建在这个 FFI 之上。

## 核心概念

### ABI vs API vs FFI

| 概念 | 含义 | 例子 |
|------|------|------|
| **API** | 源码级的调用约定 | `torch.add(x, y)` |
| **ABI** | 二进制级的交互规则 | 函数参数通过寄存器还是栈传递？struct 怎么对齐？ |
| **FFI** | 跨语言调用的胶水层 | Python 调用 `add_one_cpu.so` 里的 C++ 函数 |

ABI 比 API 更难：换一个编译器、换一个编译选项，就可能二进制不兼容。TVM FFI 选择用 C ABI 作为"通用语言"——所有主流语言都能调用 C 函数签名，这是唯一的共识。

### 两套接口

TVM FFI 提供了两套接口：

1. **底层 C ABI**：基于 `TVMFFIAny`（type-erased 16 字节容器）和 packed function（统一函数签名），稳定不变
2. **上层语言 API**：C++17、Python（Cython）、Rust 的原生 API，编译时类型安全，底层自动转换为 C ABI

## 对象系统

TVM 的对象系统是整个 FFI 的核心，它提供了引用计数 + 运行时类型信息（RTTI）+ 跨语言反射。

### 两层设计

每个 IR 节点拆成两部分，以 `IntImm` 为例：

```cpp
// Node 层：堆分配的实际数据，带引用计数和类型信息
class IntImmNode : public PrimExprNode {
public:
    int64_t value;       // 真正的数据
    static constexpr const char* _type_key = "ir.IntImm";
};

// Ref 层：栈上的智能指针句柄，值语义
class IntImm : public PrimExpr {
    // 内部持有 ObjectPtr<Object> data_;
    // operator-> 返回 const IntImmNode*
};
```

Node 层是继承树，Ref 层是配套的句柄。用户代码只操作 Ref，不需要手动管理内存。这和 Rust 的 `Arc<T>` / Swift 的引用计数类似，但额外带了一套运行时反射。

### 引用计数

每个堆对象头部嵌入 `TVMFFIObject`，其中 `combined_ref_count` 是一个 64 位原子变量：低 32 位是强引用计数，高 32 位是弱引用计数。

```cpp
// 增加强引用（原子操作）
void IncRef() {
    __atomic_fetch_add(&header_.combined_ref_count, 1, __ATOMIC_RELAXED);
}

// 减少强引用，归零时自动析构
void DecRef() {
    uint64_t old = __atomic_fetch_sub(&header_.combined_ref_count, 1, __ATOMIC_RELEASE);
    if (old == kCombinedRefCountBothOne) {
        // 强引用和弱引用都归零 → 调用 deleter 释放内存
        header_.deleter(&this->header_, kTVMFFIObjectDeleterFlagBitMaskBoth);
    }
}
```

### 运行时类型检查（替代 C++ RTTI）

TVM 不使用 C++ 的 `dynamic_cast`，而是自己实现了一套更快的 RTTI：

```cpp
// 每个子类注册一个 type_index
class IntImmNode : public PrimExprNode {
    TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.IntImm", IntImmNode, PrimExprNode);
};

// 使用时
PrimExpr expr = ...;
if (const IntImmNode* n = expr.as<IntImmNode>()) {
    // n->value 就是编译期常量
}
```

为什么不用 C++ RTTI？因为需要跨语言——Python 代码也要能判断 "这个 `ObjectRef` 到底是 `IntImmNode` 还是 `CallNode`"。这套类型信息通过反射暴露到 Python 侧。

### 类型层级

```
ffi::Object (带引用计数 + 类型索引的堆对象)
  ├── BaseExprNode
  │     ├── PrimExprNode (低层 POD 值：int, float...)
  │     └── RelaxExprNode (高层对象：tensor, function...)
  ├── TypeNode (统一类型系统)
  │     ├── PrimTypeNode  (int32, float32...)
  │     ├── FuncTypeNode  (arg_types → ret_type)
  │     └── TupleTypeNode
  └── StmtNode (所有语句)
```

## 类型擦除：`TVMFFIAny`

`TVMFFIAny` 是整个 ABI 的基石。它是一个 16 字节的带标签联合体：

```c
typedef struct {
    int32_t type_index;    // 标记：里面装了什么
    int32_t zero_padding;
    union {
        int64_t      v_int64;
        double       v_float64;
        void*        v_ptr;
        TVMFFIObject* v_obj;  // 堆对象指针
    };
} TVMFFIAny;
```

装箱/拆箱操作（完全在纯 C 中）：

```c
// 装箱：int → Any
TVMFFIAny Any_FromInt(int64_t value) {
    TVMFFIAny any;
    any.type_index = kTVMFFIInt;
    any.v_int64 = value;
    return any;
}

// 拆箱：Any → int
int64_t Any_GetInt(const TVMFFIAny* any) {
    if (any->type_index == kTVMFFIInt)
        return any->v_int64;
    if (any->type_index == kTVMFFIFloat)
        return (int64_t)(any->v_float64);  // 自动类型转换
    assert(0);
}
```

小值（int、float、指针等 ≤ 8 字节）直接嵌入 union 中，避免堆分配。大对象（`TVMFFIObject*`）存指针，由引用计数管理生命周期。

## Packed Function：统一的调用约定

所有导出的函数都遵循同一个二进制签名：

```c
typedef int (*TVMFFISafeCallType)(
    void* self,                    // 闭包上下文
    const TVMFFIAny* args,         // 参数数组
    int32_t num_args,
    TVMFFIAny* result              // 返回值
);
```

这意味着**所有函数在二进制层面都长一样**——参数和返回值全部类型擦除。调用一个函数：

```c
int64_t CallFunction(TVMFFIObject* func, int64_t x, int64_t y) {
    TVMFFIAny args[2];
    args[0] = Any_FromInt(x);
    args[1] = Any_FromInt(y);

    TVMFFIAny result;
    TVMFFIFunctionCall(func, args, 2, &result);

    return Any_GetInt(&result);
}
```

而 Kernel 开发者完全不需要关心这些底层细节——C++ API 会自动生成适配代码：

```cpp
// Kernel 开发者只需要写这个
void AddOne(TensorView x, TensorView y) {
    for (int64_t i = 0; i < x.size(0); ++i)
        ((float*)y.data_ptr())[i] = ((float*)x.data_ptr())[i] + 1;
}

// 一行宏：自动生成 packed function 包装 + 符号导出
TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, AddOne);
```

## 零拷贝张量传递：DLPack

TVM FFI 的张量传递基于 [DLPack](https://dmlc.github.io/dlpack/latest/) 协议——一个由 TVM 社区推动、现已被 PyTorch/JAX/CuPy 广泛采纳的开放标准。

```cpp
// 获取 Tensor 的底层 DLDevice 和 DLTensor
TensorView x;
DLDevice device = x.device();     // {kDLCPU, 0} 或 {kDLCUDA, 0}
void* raw_ptr = x.data_ptr();     // GPU 显存或 CPU 内存指针，零拷贝
DLDataType dtype = x.dtype();     // {kDLFloat, 32, 1}
```

当一个 `torch.Tensor` 传入 TVM FFI 函数时，不会发生任何数据拷贝——只是把内部的 `DLTensor*` 指针传递过去。这也是为什么 TVM FFI 能做到"极低开销"。

## 实战：三个层次看同一段代码

### 第一层：写 Kernel（C++）

```cpp
// compile/add_one_cpu.cc
#include <tvm/ffi/tvm_ffi.h>

void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
    int64_t n = x.size(0);
    float* x_data = static_cast<float*>(x.data_ptr());
    float* y_data = static_cast<float*>(y.data_ptr());
    for (int64_t i = 0; i < n; ++i)
        y_data[i] = x_data[i] + 1;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, AddOne);
```

编译：`cmake --build .` → 产出 `add_one_cpu.so`。

### 第二层：消费 Kernel（C++）

```cpp
// load/load_cpp.cc
Module mod = Module::LoadFromFile("add_one_cpu.so");
Function add_one = mod->GetFunction("add_one_cpu").value();
add_one(x, y);  // x=[1,2,3,4,5] → y=[2,3,4,5,6]
```

### 第三层：消费 Kernel（Python）

```python
import torch
from tvm_ffi import Module

mod = Module.load_from_file("add_one_cpu.so")
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.empty_like(x)

mod.add_one_cpu(x, y)   # PyTorch tensor 直接传入，零拷贝
# y == tensor([2., 3., 4., 5., 6.])
```

C++ 写的 `.so`，C++ 和 Python 加载的是**同一个二进制文件**，没有任何中间层翻译或序列化开销。

### 极致：Python JIT 编译 C++/CUDA

```python
import tvm_ffi.cpp

mod = tvm_ffi.cpp.load_inline(
    name="hello",
    cpp_sources=r"""
        void add_one_cpu(TensorView x, TensorView y) {
            for (int i = 0; i < x.size(0); ++i)
                ((float*)y.data_ptr())[i] = ((float*)x.data_ptr())[i] + 1;
        }
        void add_one_cuda(TensorView x, TensorView y);
    """,
    cuda_sources=r"""
        __global__ void AddOneKernel(float* x, float* y, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) y[idx] = x[idx] + 1;
        }
        void add_one_cuda(TensorView x, TensorView y) {
            int64_t n = x.size(0);
            cudaStream_t stream = TVMFFIEnvGetStream(x.device());
            AddOneKernel<<<(n+255)/256, 256, 0, stream>>>(
                (float*)x.data_ptr(), (float*)y.data_ptr(), n);
        }
    """,
    functions=["add_one_cpu", "add_one_cuda"],
)

# CPU
mod.add_one_cpu(x, y)
# CUDA — 同样的调用方式
mod.add_one_cuda(x.cuda(), y.cuda())
```

CPU 和 GPU kernel 用同样的 ABI 包装，调用方无感知。

## 架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│                     Kernel 开发者                            │
│                                                             │
│  void AddOne(TensorView x, TensorView y) { ... }            │
│  TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one, AddOne)             │
└────────────────────────┬────────────────────────────────────┘
                         │ 编译为 .so / .dll
┌────────────────────────▼────────────────────────────────────┐
│                  C ABI 层（稳定，不变）                        │
│                                                             │
│  TVMFFIAny args[] ──→ packed function ──→ TVMFFIAny result │
│  TVMFFIObject*   (引用计数堆对象)                              │
│  DLTensor*       (零拷贝张量)                                  │
│  TVMFFIFunctionCall / TVMFFIFunctionGetGlobal               │
└──────┬──────────────┬──────────────────┬────────────────────┘
       │              │                  │
┌──────▼──────┐ ┌─────▼──────┐ ┌───────▼──────┐
│  C++ 消费   │ │ Python 消费 │ │  Rust 消费    │
│             │ │             │ │              │
│  Module::   │ │  Module.    │ │  (开发中)     │
│  LoadFrom   │ │  load_from  │ │              │
│  File(so)   │ │  file(so)   │ │              │
└─────────────┘ └─────────────┘ └──────────────┘
```

## 在 TVM 编译器中的角色

TVM 编译器的整个 IR 节点体系都建立在 `ffi::Object` 之上：

```
TVM 编译器
├── IR Core (tvm::ir)       ← ffi::Object 继承体系
├── TIR (tvm::tirx)         ← 张量级 IR 节点
├── Relax (tvm::relax)      ← 图级 IR 节点
├── Target Codegen          ← 通过 packed function 调用后端
├── Meta Schedule           ← Python 调优通过 FFI 调用 C++ 调度原语
└── Runtime                 ← 通过 DLPack 零拷贝传递张量
```

这意味着 TVM 的所有组件——无论是 C++ 核心还是 Python 脚本——都通过同一套 FFI 机制通信。一个 Python 写的调度 pass 可以直接操作 C++ 构造的 TIR IR 树，因为每个 `ForNode`、`BufferLoadNode` 都通过 `ffi::Object` 暴露了 Python 可访问的字段。

## 总结

TVM FFI 解决的是一个看似简单但至关重要的问题：**让不同语言写的 ML 组件能高效协作**。它的设计哲学很清晰：

1. **C ABI 是唯一的共识**——用 C 函数签名作为跨语言边界
2. **类型擦除 + 运行时类型反射**——`TVMFFIAny` 负责传值，`ffi::Object` 负责管理复杂对象
3. **零拷贝**——基于 DLPack，张量在框架之间传递时不做数据拷贝
4. **Kernel 开发者无感知**——写 C++ 的用 `TVM_FFI_DLL_EXPORT_TYPED_FUNC`，不需要关心打包/拆包

这套基础设施不只在 TVM 内部使用——它正在成为 ML 生态的通用组件：FlashInfer、TileLang、cuteDSL 等项目都在基于 TVM FFI 发布 kernel 库。
