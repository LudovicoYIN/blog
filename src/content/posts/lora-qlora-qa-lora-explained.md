---
author: Ludovico
pubDatetime: 2026-08-10T08:00:00+08:00
title: "从 LoRA 到 QLoRA、QA-LoRA：低秩更新如何进入量化模型"
featured: false
draft: false
tags:
  - LoRA
  - QLoRA
  - QA-LoRA
  - 量化
  - 大语言模型
description: 用一个完整的数值例子解释 LoRA、QLoRA 和 QA-LoRA 的训练、推理与合并流程，并拆开 QA-LoRA 官方实现中的 group pooling 和 qzeros 写回。
---

LoRA、QLoRA、QA-LoRA 的名字很像，但它们解决的问题并不完全相同：

- **LoRA**：不改动大矩阵，只训练一个低秩更新。
- **QLoRA**：把基座模型压成低比特，再训练 LoRA，重点是省显存。
- **QA-LoRA**：让 LoRA 的更新结构和量化分组对齐，重点是训练后能更自然地合回量化模型。

本文最后一部分以 [QA-LoRA 官方实现](https://github.com/yuhuixu1993/qa-lora) 为准，解释它为什么可以直接修改 GPTQ 的 `qzeros`。

## 先看总图

![LoRA、QLoRA 和 QA-LoRA 的流程对比](@/assets/images/lora/lora-qlora-qalora-flow.svg)

QA-LoRA 论文仓库还提供了一张更完整的结构图，直接对比了三种方法的训练和推理路径：

![QA-LoRA 论文仓库中的 LoRA、QLoRA、QA-LoRA 总览图](@/assets/images/lora/qa-lora-overview.png)

## 一、LoRA：给大矩阵旁边加一条小路

假设线性层的权重矩阵是：

```text
W: [Din, Dout]
```

普通微调会直接更新 `W`。如果 `W` 很大，显存和优化器状态都会很重。

LoRA 冻结原来的 `W`，额外增加两个小矩阵：

```text
A: [Din, r]
B: [r, Dout]
```

其中 `r` 远小于 `Din` 和 `Dout`。前向计算变成：

$$
y=xW+xAB
$$

训练时：

```text
W：冻结
A：更新
B：更新
```

最终的权重更新是：

$$
\Delta W=AB
$$

因为中间维度只有 `r`，所以 `ΔW` 的秩最多是 `r`。这意味着 LoRA 不会允许权重矩阵任意变化，而是用少数几个方向来修改它。

### 一个最小例子

令：

```text
Din = 4, Dout = 2, r = 1
```

```text
A = [ 2.0,
      0.5,
     -1.0,
      1.5 ]^T
B = [0.1, -0.2]
```

则：

```text
ΔW = A × B
    = [ 2.0 ] × [0.1, -0.2]
      [ 0.5 ]
      [-1.0 ]
      [ 1.5 ]

    = [ 0.20, -0.40 ]
      [ 0.05, -0.10 ]
      [-0.10,  0.20 ]
      [ 0.15, -0.30 ]
```

如果 `r=2`，就会有两个基础方向，更新可以写成两个外积的相加：

$$
\Delta W=a_1b_1+a_2b_2
$$

## 二、QLoRA：把 W 换成 4 bit，但保留 LoRA 的训练方式

QLoRA 的基座权重不是 FP16，而是低比特权重：

```text
W_fp16 -> W4
```

反量化后再参与计算：

$$
y=x\,\operatorname{dequant}(W4)+xAB
$$

或者写成有效权重：

$$
W_{eff}=\operatorname{dequant}(W4)+\frac{\alpha}{r}AB
$$

训练过程是：

1. 把基座权重量化为 4 bit，常见实现会使用 NF4、双重量化等技巧。
2. 冻结 `W4`。
3. 只训练 FP16/BF16 的 `A` 和 `B`。
4. 每次前向时，对 `W4` 反量化或使用融合 kernel 参与矩阵乘。
5. 训练结束后保存 `W4 + LoRA adapter`。

QLoRA 的核心收益是：大部分模型权重用 4 bit 保存，训练时不需要为完整 FP16 权重维护梯度和优化器状态。

### QLoRA 推理时发生什么

通常保留两部分：

```text
量化基座 W4
LoRA A、B
```

推理时：

```text
x -> dequant(W4) -> xW4
x -> A -> B       -> xAB
两路相加          -> y
```

所以 QLoRA 的部署形态通常仍然是：

```text
量化基座 + 独立 LoRA adapter
```

它的重点是**低显存微调**，不是让 adapter 消失。

## 三、QA-LoRA：让更新的粒度和量化 group 对齐

QA-LoRA 使用 GPTQ 量化，并设置 `group_size`。假设：

```text
输入维度 Din = 4
group_size = 2
输出维度 Dout = 2
```

输入被分成两个 group：

```text
Group 1 = [x1, x2]
Group 2 = [x3, x4]
```

QA-LoRA 的关键不是把 `A` 和 `B` 都复制成很多份，而是先把每个 group 的输入聚合：

$$
\operatorname{pool}(x)=
\left[
\frac{x_1+x_2}{2},
\frac{x_3+x_4}{2}
\right]
$$

池化前后维度是：

```text
x       : [Din]              = [4]
pool(x) : [Din/group_size]   = [2]
```

接着再经过低秩矩阵：

```text
A: [Din/group_size, r] = [2, 1]
B: [r, Dout]            = [1, 2]
```

QA-LoRA 的前向可以写成：

$$
y=xW+\operatorname{pool}(x)AB
$$

## 四、把官方代码的计算完整展开

我们使用一个具体数字：

```text
x = [1, 2, 3, 4]

A = [2.0,
     0.5]

B = [0.1, -0.2]
```

### 第一步：池化

```text
pool(x)
= [(1+2)/2, (3+4)/2]
= [1.5, 3.5]
```

### 第二步：经过 A

```text
pool(x)A
= 1.5×2.0 + 3.5×0.5
= 3.0 + 1.75
= 4.75
```

### 第三步：经过 B

```text
pool(x)AB
= 4.75 × [0.1, -0.2]
= [0.475, -0.95]
```

这就是 LoRA 分支的输出。

## 五、为什么说 group 内的更新相同

把上面的式子重新展开：

$$
\begin{aligned}
\operatorname{pool}(x)AB
={}&x_1\frac{A_1B}{2}+x_2\frac{A_1B}{2}\\
&+x_3\frac{A_2B}{2}+x_4\frac{A_2B}{2}
\end{aligned}
$$

Group 1 的更新向量是：

```text
A1 × B / group_size
= 2.0 × [0.1, -0.2] / 2
= [0.1, -0.2]
```

因此 Group 1 中的两列权重都得到同一个更新：

```text
第 1 列：[0.1, -0.2]
第 2 列：[0.1, -0.2]
```

Group 2 的更新向量是：

```text
A2 × B / group_size
= 0.5 × [0.1, -0.2] / 2
= [0.025, -0.05]
```

```text
第 3 列：[0.025, -0.05]
第 4 列：[0.025, -0.05]
```

注意这里的“相同”是：

> 同一个输入 group 内，不同输入列共享同一个更新向量。

不是说 `[0.1, -0.2]` 这个向量里的每个输出元素都相同。

## 六、为什么 QA-LoRA 可以直接改 qzeros

GPTQ 的分组反量化形式可以简化写成：

$$
w=(q-z)\times s
$$

同一个 group 共享量化参数 `z` 和 `s`。

如果给这个 group 的每个输入列都增加同一个更新 `Δw`，那么：

$$
(q-z)\times s+\Delta w
$$

可以改写为：

$$
\left(q-z-\frac{\Delta w}{s}\right)\times s
$$

因此只要修改：

$$
z_{new}=z-\frac{\Delta w}{s}
$$

就可以把 LoRA 更新写进 zero-point。

这正是仓库 [merge.py](https://github.com/yuhuixu1993/qa-lora/blob/main/merge.py) 做的事情：

```python
qzeros -= (B @ A).T * scale / group_size / scales
```

流程是：

```text
B @ A
  -> 得到每个 group 的更新向量
  -> 除以 group_size，还原 avgpool 的平均因子
  -> 除以 GPTQ scale，换算成 zero-point 的变化
  -> 直接写回 qzeros
```

所以这个实现不是：

```text
反量化 W4
  -> 加上 LoRA
  -> 重新量化 q、scale、zero
```

而是：

```text
保持 q 和 scale
  -> 直接修改 qzeros
  -> 得到合并后的 GPTQ 模型
```

## 七、三者放在一起

| 方法    | 基座权重      | LoRA 输入 | 训练后部署                |
| ------- | ------------- | --------- | ------------------------- |
| LoRA    | FP16/BF16     | 原始 `x`  | FP 权重 + adapter，或合并 |
| QLoRA   | 4 bit，冻结   | 原始 `x`  | 量化基座 + 独立 adapter   |
| QA-LoRA | GPTQ 分组量化 | `pool(x)` | 通过分组结构写回量化参数  |

三者的公式可以这样记：

```text
LoRA:
y = xW + xAB

QLoRA:
y = x·dequant(W4) + xAB

QA-LoRA:
y = x·dequant(W4) + pool(x)AB
```

最关键的变化只发生在 LoRA 分支的输入：

```text
LoRA / QLoRA：使用完整 x
QA-LoRA：      先按量化 group 对 x 做 pool
```

正是这个 `pool(x)`，让同一 group 内的输入列共享相同更新，从而与 GPTQ 的 group-wise `qzero` 对齐。

## 最后记住

- LoRA 解决的是：怎样用很少的参数表达权重更新。
- QLoRA 解决的是：怎样在低比特基座上低显存训练 LoRA。
- QA-LoRA 解决的是：怎样让 LoRA 更新的结构和量化分组对齐，训练后直接合回量化参数。

QA-LoRA 并不是“每个 group 只有一个数”，而是：

```text
每个 group 有自己的 A 系数
所有 group 共享 B 的低秩方向
同一个 group 内的输入列共享一个更新向量
```

这就是它能够直接修改 `qzeros` 的原因。

## 参考

- [QA-LoRA 官方仓库](https://github.com/yuhuixu1993/qa-lora)
- [QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2309.14717)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
