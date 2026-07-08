---
author: Ludovico
pubDatetime: 2026-07-08T00:00:00Z
title: RKNN OCR Rec INT8 量化：Weight Outlier、CLE 与 Hybrid 的取舍
featured: false
draft: false
tags:
  - RKNN
  - 端侧部署
description: 记录一次 OCR recognition presoftmax 模型在 RKNN W8A8 量化下的精度排查：对比 normal、KL、MMSE，尝试手动 Cross-Layer Equalization 处理 weight outlier，并分析为什么最终仍需要 hybrid。
---

这篇记录一次 OCR recognition 模型在 RKNN 上做 INT8 量化的精度排查。

模型是 presoftmax 输出，输入宽度有 `w320` 和 `w640` 两版。目标是把 RKNN W8A8 的输出 cosine 尽量拉高，同时判断 weight outlier 和 Cross-Layer Equalization (CLE) 到底能不能解决问题。

结论先放前面：

1. `mmse` 是本次纯 W8A8 下整体最好的量化算法。
2. CLE 可以明显改善部分 RKNN build 阶段的 weight outlier warning，但不能保证最终 logits cosine 提升。
3. `normal` 量化最受 outlier 影响，CLE 后有明显改善。
4. `kl_divergence` 在 CLE 后反而下降，说明分布 KL 最优不等于任务输出最优。
5. 如果目标是 `cosine >= 0.999`，纯 INT8 + CLE 不够，需要继续走 RKNN `auto_hybrid` 或手动 hybrid。

## 实验结果

![RKNN OCR Rec INT8 CLE 对比](/images/rknn-ocr-rec-int8-cle-compare.png)

### w320

| 配置 | cosine_min | cosine_mean | 变化 |
| - | - | - | - |
| `normal` | `0.524568052` | `0.917384625` | baseline |
| `normal + CLE` | `0.619720233` | `0.923002650` | min 提升，mean 小幅提升 |
| `kl_divergence` | `0.753880422` | `0.949090446` | baseline |
| `kl_divergence + CLE` | `0.699861627` | `0.946783263` | min 下降，mean 小幅下降 |
| `mmse` | `0.923029723` | `0.984708844` | baseline，w320 最优 |
| `mmse + CLE` | `0.876696159` | `0.984592642` | min 下降，mean 基本持平 |

### w640

| 配置 | cosine_min | cosine_mean | 变化 |
| - | - | - | - |
| `normal` | `0.057584906` | `0.805615253` | baseline |
| `normal + CLE` | `0.213995877` | `0.842987104` | min 提升，mean 提升 |
| `kl_divergence` | `0.467775398` | `0.880270489` | baseline |
| `kl_divergence + CLE` | `0.413393508` | `0.875421858` | min 下降，mean 小幅下降 |
| `mmse` | `0.815402347` | `0.951946904` | baseline |
| `mmse + CLE` | `0.840134289` | `0.955012016` | min 提升，mean 小幅提升，w640 最优 |

## 三种量化算法的表现

### normal

`normal` 更接近直接按统计范围做量化，容易被极端权重值拉大 scale。CLE 压缩了部分 outlier 权重范围，所以 `normal` 在两个宽度上都有提升：

```text
w320 normal: 0.524568052 -> 0.619720233
w640 normal: 0.057584906 -> 0.213995877
```

但提升后仍然远低于可交付精度要求，说明 weight outlier 不是唯一问题。

### kl_divergence

`kl_divergence` 的目标是让量化前后张量分布的 KL 散度较小。CLE 会改变中间激活分布和权重分布，校准阶段选出的 clip range 也会变化。

本次实验中，CLE 后 `kl_divergence` 在 w320、w640 上都下降：

```text
w320 kl_divergence: 0.753880422 -> 0.699861627
w640 kl_divergence: 0.467775398 -> 0.413393508
```

这说明 KL 分布最优不等于最终 OCR logits cosine 最优。对于 presoftmax OCR rec 模型，局部 time step 或字符类别 logit 的扰动会被最终向量 cosine 放大。

### mmse

`mmse` 会搜索更优的量化 scale / clip range，使量化前后的均方误差更小。它不会修改原始 ONNX，但会影响 `.rknn` 内部生成的 INT8 权重和量化参数。

本次结果：

```text
w320 mmse: 0.923029723 -> 0.876696159
w640 mmse: 0.815402347 -> 0.840134289
```

w320 上 CLE 破坏了原本更适合 `mmse` 的量化范围；w640 上 CLE 对 `mmse` 有小幅收益。整体看，`mmse` 仍是纯 W8A8 下的首选，但是否叠加 CLE 需要按模型尺寸分别验证。

## CLE 的基本原理

CLE 是 Cross-Layer Equalization，即跨层均衡。它通过对相邻线性层做等价缩放，降低层间或通道间的动态范围差异。

对两个连续 Conv：

```text
y = Conv2(Conv1(x))
```

可以做如下等价变换：

```text
Conv1.weight[channel] *= scale
Conv1.bias[channel]   *= scale
Conv2.weight[:, channel] /= scale
```

FP32 数学结果基本不变，但权重范围更均衡。对 INT8 来说，如果某些 channel 有极端 outlier，量化 scale 会被拉大，普通值的有效精度会下降。CLE 的目标就是缓解这个问题。

## RKNN build 里的 outlier

RKNN build 阶段提示如下权重存在异常值：

```text
conv2d_74.w_0       abs_mean 0.35  abs_std 0.58  outlier  11.571
conv2d_89.w_0       abs_mean 0.79  abs_std 1.24  outlier -46.043
conv2d_116.w_0      abs_mean 1.64  abs_std 3.56  outlier  40.535
conv2d_120.w_0      abs_mean 0.88  abs_std 2.93  outlier -75.035
conv2d_122.w_0      abs_mean 0.37  abs_std 0.45  outlier  12.960, 12.377
conv2d_66.w_0       abs_mean 0.81  abs_std 1.98  outlier -28.995
```

其中 `conv2d_66.w_0` 的原始 ONNX 权重本身不大，RKNN warning 更可能来自 Conv + BatchNorm folding 后的等效权重。

## 手动 CLE 怎么做

### 三组安全 Conv pair

对以下相邻 Conv 做标准 cross-layer scaling：

```text
conv2d_72.w_0  + p2o.pd_op.reshape.4.0  -> conv2d_74.w_0
conv2d_87.w_0  + p2o.pd_op.reshape.18.0 -> conv2d_89.w_0
conv2d_114.w_0 + p2o.pd_op.reshape.45.0 -> conv2d_116.w_0
```

处理后：

```text
conv2d_74.w_0   max 11.571 -> 3.786
conv2d_89.w_0   max 46.043 -> 2.878
conv2d_116.w_0  max 40.535 -> 2.533
```

### input-scale 吸收

对 `conv2d_120.w_0` 和 `conv2d_66.w_0` 使用输入缩放：

```text
input * scale -> Conv(weight / scale)
```

`conv2d_66.w_0` 按 BatchNorm folding 后的等效权重计算 scale。

### residual-aware scale

`conv2d_122.w_0` 后面进入 residual add。直接做：

```text
Conv(weight / scale) -> bias / scale -> Mul(scale)
```

很容易被 RKNN 重新 fold 回 Conv，导致 outlier warning 仍然出现。

因此改成 residual 两路一起缩放，再由后继 Conv 吸收：

```text
conv122 branch / scale
skip branch    / scale
next conv input weights * scale
```

后继吸收层：

```text
conv2d_67.w_0
conv2d_65.w_0
```

## 等价性验证

手动 CLE 后，使用 ONNX Runtime 对比原始 ONNX 与 CLE ONNX：

```text
w320 cos 0.999999999999, mae about 4.5e-6
w640 cos 0.999999999998, mae about 4.5e-6
```

这说明手动 CLE 在 FP32/ORT 下基本等价。后续 RKNN W8A8 精度变化主要来自量化行为，而不是 ONNX 重写本身把模型改坏。

## 最后怎么选

本次比较后的策略是：

1. 纯 W8A8 优先使用 `mmse`。
2. w320 不建议使用 `mmse + CLE`，原始 ONNX + `mmse` 更好。
3. w640 可以保留 `mmse + CLE` 作为候选，但收益有限。
4. 如果要求 `cosine >= 0.999`，继续走 `auto_hybrid` 或手动 hybrid。
5. layerwise analysis 需要优先关注 logits 前后、归一化、Add、ReduceMean、cast 等敏感节点，而不是只看 weight outlier。

这次比较有一个比较明确的教训：weight outlier 是真实问题，但不一定是最终精度瓶颈。CLE 能让权重看起来更健康，却不保证任务输出更好。对于 RKNN 这类部署链路，最终还是要把 layerwise analysis、量化算法、hybrid 回退一起看，不能只盯 build warning。
