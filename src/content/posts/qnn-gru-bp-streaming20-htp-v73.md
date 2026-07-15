---
author: Ludovico
pubDatetime: 2026-07-15T09:00:00+08:00
title: QAIRT 2.42 适配流式多通道语音增强 GRU：从错误 Lowering 到可验证的 HTP 图
featured: false
draft: false
tags:
  - 高通
  - 音频模型
  - 模型部署
description: 基于 bp_streaming20 的一次完整排查：GRU 展开、长序列 native cell、QAIRT 源码开关、Einsum layout bug 与流式 I/O 对齐。
---

本文记录 QAIRT 2.42 在 QCS8550 / HTP v73 上适配流式 GRU 网络的过程，覆盖 GRU 语义、图规模、后端执行稳定性和非 GRU 算子的 layout 错误。

模型文件名是 `bp_streaming20_sim.onnx`，但从 ONNX 图看，它更像一个**流式多通道复数频谱语音增强网络**：输入是复数频谱实部/虚部、DOA 与跨帧 cache；前端构造频谱和空间特征，decoder 使用 intra/inter 双路径 RNN，输出增强后的两路复数频谱和新的流式状态。

本文所有结论基于 QAIRT `2.42.0.251225`、FP16 图、QCS8550 HTP v73 的实际转换和板端执行。

## 先读图，不要先猜模型

原始图有 8 个输入：

| 输入 | shape | 含义推断 |
|---|---:|---|
| `x_real`, `x_imag` | `[1, 2, 257]` | 两路复数频谱的实部与虚部 |
| `doa` | `[1]` | 到达角或空间先验 |
| `gru_state` | `[2, 257, 12]` | 编码端循环状态 |
| `attn_k_cache`, `attn_v_cache` | `[257, 10, 12]` | 流式 attention cache |
| `tconv1_r_cache` | `[1, 257, 12]` | 实部时序卷积 cache |
| `tconv1_i_cache` | `[1, 257, 2]` | 虚部时序卷积 cache |

图中可见 `Cos/Sin`、成对实虚部计算、`Einsum` 外积、`feat_emb`、`scm_emb`、`intra_rnn`、`inter_rnn`。其中 `scm` 可以合理理解为空间协方差矩阵（spatial covariance matrix）特征分支；`intra/inter` 是典型 dual-path RNN 的命名。

模型按帧执行：每次调用消费上一帧 cache，输出下一帧 cache。接口 layout 参与状态传递语义。

## 六个 GRU 的形状

模型共有 6 个 ONNX GRU，且全部使用 `linear_before_reset=1`：

| 节点 | direction | 逻辑 shape | hidden |
|---|---|---:|---:|
| `GRU_emb/GRU` | forward | `[seq=1, batch=257, input=24]` | 12 |
| `GRU_emb/GRU_1` | forward | `[1, 257, 12]` | 12 |
| `intra_rnn/rnn1/GRU` | bidirectional | `[257, 1, 6]` | 3 |
| `intra_rnn/rnn2/GRU` | bidirectional | `[257, 1, 6]` | 3 |
| `inter_rnn/rnn1/GRU` | forward | `[1, 257, 6]` | 6 |
| `inter_rnn/rnn2/GRU` | forward | `[1, 257, 6]` | 6 |

部署策略由 sequence 长度和 batch 大小决定：

```text
4 个短 GRU: sequence = 1, batch = 257
2 个长双向 GRU: sequence = 257, batch = 1
```

前者展开一个 cell 的代价有限；后者若按 257 个时间步、双向、两层展开，图规模会急剧增长。

## 默认 GRU lowering 先在 002 模型上暴露问题

QAIRT 的 ONNX converter 默认先按时间维 unroll，再将 cell 展开为 `MatMul/Add/Sigmoid/Tanh/Mul/Sub` 等基础算子。

相关逻辑位于：

```text
converters/common/converter_ir/op_graph_optimizations.py
  unroll_gru_time_steps()
  expand_gru_op_structure()
```

较短的 `002.onnx` 在 HTP 上出现默认 GRU lowering 数值错误。处理方式是在 ONNX 层按 `linear_before_reset=1` 公式手工展开 GRU，并用 ONNX Runtime 验证等价性。

`bp_streaming20` 需要单独评估图规模。

## 为什么不能把 bp_streaming20 的所有 GRU 都手工展开

手工展开长双向 GRU 时，每个时间步都要构造 update/reset/candidate 三组门计算，还需要状态串联、slice、concat 和方向处理。两个 `seq=257` 的双向 GRU 展开后会形成数万级基础算子。

该图在数学上可行，部署成本较高：

- converter、model library 编译和 context prepare 明显变慢；
- context 体积、内存和调度开销上升；
- 图变大后更难定位别的 layout 或数值问题；
- 这并没有利用 HTP 本身的 GRU cell 能力。

目标是保留**单时间步** native QNN Gru cell，避免继续展开 cell 内部算子。

## 从 QAIRT 源码找到两个独立开关

源码里 GRU 的行为由三个开关控制：

```text
multi_time_steps_gru
unroll_gru_time_steps
expand_gru_op_structure
```

三个开关分别控制不同阶段：

```text
multi_time_steps_gru
  保留多时间步 GRU，由后端处理整个 sequence

unroll_gru_time_steps
  将 sequence 拆成 T 个单时间步 Gru cell

expand_gru_op_structure
  将每个 Gru cell 继续展开为 MatMul/Add/Sigmoid/Tanh/... 基础算子
```

官方 ONNX 路径最终会展开 GRU。实验使用 converter 入口副本，仅修改强制设置：

```python
args.unroll_gru_time_steps = True
args.expand_gru_op_structure = False
```

这样会保留 native 的单步 `QNN Gru`，但 sequence=257 的 GRU 仍会成为 257 个 cell；一个双向 GRU 则是 514 个 cell。

## Multi-time-step 与完整 native 图的执行边界

最初使用不 unroll 的多时间步 native GRU。转换、compose 和 context prepare 可以通过，板端在 `Executing Graphs` 阶段失败；相关实验出现 HTP skel DMA error `1100`。短序列 probe 同样出现明显误差。故障覆盖前向、双向和 `linear_before_reset` 配置。

随后改为：

```text
长双向 GRU: unroll = True, expand = False
```

即 257 个 native single-step cell。单独抽取的长双向 GRU 在 v73 上精度正常，说明单步 native cell 的语义没有问题。

6 个 GRU 全部改为 native cell 后，完整图包含 1,032 个 cell。context 可以生成，执行阶段仍可能失败。context 生成和 `qnn-net-run` 执行需要作为独立关卡验证。

## 最终 GRU 策略：短 GRU 展开，长双向 GRU 保留 native cell

最终采用混合策略：

```text
sequence=1 的 4 个 forward GRU
  -> 手工展开为基础算子

sequence=257 的 2 个 bidirectional GRU
  -> QAIRT unroll 为单时间步 native QNN Gru cell
  -> 不再展开 cell 内部结构
```

该策略对应两组 shape 的后端行为：短 GRU 使用基础算子展开；长 GRU 使用 native cell，避免数万级基础算子图。

## FP16 精度仍然差：不要继续怀疑 GRU，做逐层分析

混合图执行后仍有大误差，部分 cache cosine 接近 0，中间 tensor 出现 NaN。该误差量级超过 FP16 舍入误差范围。

使用 target-side context 的中间输出能力进行逐层比对：

```sh
qnn-context-binary-generator --set_output_tensors tensor_a,tensor_b
qnn-net-run --retrieve_context ...
```

每次选择一组 ONNX 中间 tensor，导出 QNN 结果，再与同一输入下的 ONNX Runtime 对比。比较目标是定位**第一个错误 tensor**。

## 首个错误：Einsum 的 rank-5 broadcast Mul

最早看到明显偏差的位置在 `scm_relu/PRelu`。该节点的 slope 为负数，一度怀疑 HTP 的 PRelu 或 opset 16 parser 有问题。

把 PRelu 等价改写为：

```text
y = Relu(x) + slope * (x - Relu(x))
```

后，误差仍然存在。继续向前导出中间 tensor，发现 SCM Conv 前的特征已经坏了，进一步定位到先前为兼容性手工展开的 4 个 Einsum：

```text
...ct,...et -> ...tce
```

最初写法是：

```text
Transpose + Unsqueeze + Mul
```

在 ONNX Runtime 完全正确，但 HTP 对两个 rank-5 tensor 的 broadcast `Mul` 做了错误的 layout lowering。四个外积结果的 cosine 都已明显失真，后续 `Add/Sub` 开始出现 NaN。

这些张量本质上是 outer product：

```text
[1, 257, 1, 2, 1] @ [1, 257, 1, 1, 2]
-> [1, 257, 1, 2, 2]
```

将 `Mul` 换成等价的 batched `MatMul`：

```text
Transpose + Unsqueeze + MatMul
```

ONNX Runtime 输出保持等价，HTP 特征分支恢复正常。经验：**ONNX broadcast 合法时，HTP layout lowering 仍需验证；高 rank outer product 使用 MatMul 表达。**

另四个 `bfc,bfc->bf` Einsum 则安全地改写为：

```text
Mul + ReduceSum(axis=-1)
```

## 最后一个陷阱：流式 passthrough cache 的 I/O layout

模型有三个输出只是输入 cache 的 passthrough：

```text
new_attn_k_cache = Identity(attn_k_cache)
new_attn_v_cache = Identity(attn_v_cache)
new_tconv1_i_cache = Identity(tconv1_i_cache)
```

即使传入：

```sh
--preserve_io layout --preserve_onnx_output_order
```

QAIRT 仍可能先删除 `Identity`，使输出暴露为内部 NFC layout：

| ONNX shape | 错误暴露的 QNN shape |
|---:|---:|
| `[257, 10, 12]` | `[257, 12, 10]` |
| `[1, 257, 2]` | `[1, 2, 257]` |

数值本身并未错误，但应用必须额外 transpose 才能按 ONNX 语义读取。这会把接口兼容问题伪装成精度问题。

修复是把三个 Identity 改为等价但不会被 Identity pass 删除的输出锚点：

```text
Identity(x)  ->  Add(x, 0)
```

之后 `--preserve_io layout` 能在图出口插入 `NFC -> NCF` 转置，QNN 输出 shape 与 ONNX 完全一致。

## 最终验证

最终图使用 FP16 native I/O：

```sh
--float_bitwidth 16
--preserve_io layout
--preserve_onnx_output_order
```

没有使用 `--preserve_io datatype`。这样外部 I/O 是 FP16，不会额外插入 `FLOAT32 -> FLOAT16` Convert 节点。context 在 QCS8550/v73 目标端生成，使用 `qnn-net-run --retrieve_context` 验证。

同一组 deterministic mock 输入下，QNN 输出直接按 ONNX shape 读取：

| 输出 | max abs | MAE | cosine |
|---|---:|---:|---:|
| `output` | 0.0062496662 | 0.00051888858 | 0.999999081 |
| `new_gru_state` | 0.0057431459 | 0.00043708686 | 0.999999691 |
| `new_attn_k_cache` | 0.00097632408 | 0.00014102271 | 0.999999428 |
| `new_attn_v_cache` | 0.0014824867 | 0.00014048010 | 0.999999560 |
| `new_tconv1_r_cache` | 0.010823771 | 0.00055843621 | 0.999999374 |
| `new_tconv1_i_cache` | 0.00096940994 | 0.00014568148 | 1.000000000 |

这些是合理的 FP16 relaxed-precision 误差，不再有 layout 造成的伪大误差。

## 可复用的排查顺序

排查顺序：

```text
先按 GRU 的 seq/batch 分组
  -> 评估完全展开后的图规模
  -> 查看 converter 是否有 unroll 和 expand 的独立开关
  -> 单算子验证 native cell，再验证完整图
  -> FP16 偏差异常时立即逐层比对
  -> 找第一个错误 tensor
  -> 高 rank broadcast 外积优先改为 MatMul
  -> 对 streaming cache 检查“值、shape、物理 layout”三件事
```

QAIRT/HTP 上的 GRU 可用性由 sequence 长度、单步 unroll、cell 展开方式、图内其他算子的 layout 和目标端 context prepare 共同决定。converter、context 和板端执行应分别验证。
