---
author: Ludovico
pubDatetime: 2026-07-08T09:30:00+08:00
title: SeqMSE 原理：用输出重构误差选择量化 Encoding
featured: false
draft: false
tags:
  - 量化
  - PTQ
description: SeqMSE 的核心思想、候选 encoding 搜索、reconstruction loss 计算、block-wise 优化方式，以及它和 min/max、AdaRound、GPTQ 的区别。
---

SeqMSE, Sequential MSE, 是一种后训练量化中的 encoding 搜索方法。

它不是简单按 min/max 直接定量化范围，而是为每一层准备多组候选 encoding，用 calibration 数据实际跑该层，然后选择让层输出重构误差最小的那组 encoding。

以 AIMET 的实现为例，SeqMSE 主要用于优化 **weight / parameter encoding**。它用层输出 activation tensor 的 reconstruction error 作为评价指标，但被搜索和冻结的是 Linear/Conv 的 weight quantizer encoding。这里的 activation 指中间张量，不是 activation quantizer。

## 核心问题

普通 PTQ 常见做法是：

```text
统计 tensor min/max -> 计算 scale/zero -> quant/dequant
```

这种方法的问题是容易受 outlier 影响。如果少数极端值把 min/max 拉得很宽，大多数数值会落在很少的量化等级里，导致量化误差变大。

SeqMSE 换了一个目标：不再只问“这个 tensor 的范围是多少”，而是问：

```text
哪一个量化范围能让这一层的输出最接近浮点输出？
```

## 总体流程

以 AIMET 的 weight SeqMSE 为例，流程是：

1. 准备 FP32 model、QuantSim model 和 calibration data。
2. 临时关闭 activation quantizers，避免 activation QDQ 干扰当前层 weight encoding 的选择。
3. 只保留受支持模块的 weight quantizer，例如 Linear/Conv。
4. 按模型拓扑顺序遍历每个支持的模块。
5. 对当前模块的 weight encoding min/max 生成一组候选 encoding。
6. 对每个候选 encoding 做 weight quant/dequant。
7. 用 calibration activation tensor 跑当前模块。
8. 计算量化输出和浮点输出之间的 reconstruction loss。
9. 选择 loss 最小的 encoding，并 freeze 当前模块的 weight quantizer。
10. 继续处理下一层。

![AIMET SeqMSE 总体流程](/blog/images/seqmse/figure_01.png)

这个过程叫 sequential，是因为它按网络拓扑顺序逐层处理。前面层的 weight encoding 确定后，会影响后续层在 QuantSim 路径中看到的输入分布。注意，这里的影响主要来自前序层 weight quantization；SeqMSE 阶段 activation quantizers 通常是关闭的。

## Encoding 是什么

Encoding 指量化参数，通常包括：

- `min/max`: 浮点裁剪范围。
- `scale`: 每个整数刻度对应的浮点步长。
- `zero`: 零点。
- `bitwidth`: 量化位宽。
- `sym/asym`: 对称或非对称量化。

常见 affine quantization 可以写成：

```text
q = clamp(round(x / scale) + zero, qmin, qmax)
x_hat = (q - zero) * scale
```

SeqMSE 搜索的本质就是：尝试多组 `min/max`，由它们导出不同的 `scale/zero`，然后看哪组让输出误差最小。

## Activation tensor 和 activation quantizer

这两个词要分开：

- **activation tensor**：网络中间的浮点张量，例如某层输入 `x`、某层输出 `y`。
- **activation quantizer**：负责给 activation tensor 计算 scale/zero，并对 activation 做 QDQ 的量化器。

SeqMSE 里参与的是 activation tensor，不参与的是 activation quantizer。

对一层 Linear：

```text
y = x @ w
```

SeqMSE 会用 calibration 里采到的 `x` 去测试不同 weight encoding 的好坏：

```text
y_ref = x @ w
y_q   = x @ wq
loss  = MSE(y_q, y_ref)
```

这里 `x` 和 `y` 都是 activation tensor，它们参与前向和 loss；但 `x` 和 `y` 不会在 SeqMSE 阶段再经过 activation quantizer。

## 候选范围怎么生成

AIMET 的典型候选生成方式是从当前 weight quantizer 里取初始范围。这里的 `min/max` 是 **weight encoding 的 min/max**，不是输入 activation 的 min/max：

```python
max_tensor = quant_module.param_quantizers["weight"].get_max()
min_tensor = quant_module.param_quantizers["weight"].get_min()
```

然后按 `num_candidates` 做线性缩放：

```python
cand_max = max_tensor / num_candidates * (candidate_idx + 1)
cand_min = min_tensor / num_candidates * (candidate_idx + 1)
```

如果 `num_candidates = 20`，候选就是：

```text
5%, 10%, 15%, ..., 100% of original range
```

也就是说，SeqMSE 会尝试不同程度地裁剪 weight 的原始 encoding 范围。较小范围会牺牲 weight outlier，换取主体分布更细的量化刻度；较大范围保留 outlier，但主体分布的量化分辨率会下降。

![Candidate range 搜索](/blog/images/seqmse/figure_02.png)

## Loss 怎么算

对 Linear 层，核心比较是：

```text
float output:      y_ref = x_ref        @ w
quantized output:  y_q   = x_quant_path @ wq
loss:              L(y_q, y)
```

其中：

- `x_ref`: reference 分支使用的输入 activation tensor。
- `x_quant_path`: QuantSim 路径中采到的输入 activation tensor，具体取法由策略决定。
- `w`: 浮点 weight。
- `wq`: 当前候选 encoding 下 quant/dequant 后的 weight。
- `L`: reconstruction loss，常见是 MSE。

SeqMSE 阶段通常不会再把当前层输出 `y_q` 送进 activation quantizer。也就是说，它比较的是当前层 float output tensor 和 weight-QDQ 后的 output tensor，而不是“output activation 再量化后”的结果。

AIMET 默认支持的 loss 包括：

- `mse`: mean squared error。
- `l1`: absolute error。
- `sqnr`: signal-to-quantization-noise ratio 的负值，优化时等价于最大化 SQNR。

默认 MSE 可以理解为：

```text
loss = sum((y_q - y)^2)
```

在代码里，它会保留 channel 维度上的 loss，以支持 per-channel 或 block-wise 选择：

```python
loss = loss_fn(xw, xqwq, reduction="none").sum(0)
```

## Block-wise SeqMSE

对 Linear，AIMET 会按 input channel 方向切 block。每个 block 可以独立计算 reconstruction loss，因此可以为不同 block/channel 选择不同候选。

![Linear block-wise loss](/blog/images/seqmse/figure_03.png)

直观上：

1. 把输入 `x` reshape 成多个 input-channel block。
2. 把 weight `w` 也按 input-channel block 切开。
3. 对每个候选 weight encoding 计算 `x_quant_path @ wq`。
4. 与 `x @ w` 对比。
5. 得到每个 block 的 loss。
6. 为每个 block 选择 loss 最小的候选。

这比 per-tensor 搜索更细，但也更依赖 backend 是否支持对应粒度的 weight quantization。AIMET 的 Linear block-wise 逻辑通常沿 `in_channels` 方向分块，隐含要求 block size 能整除该维度；如果不能整除，通常需要换 block size、fallback 到 per-tensor/per-channel，或者在量化配置阶段拒绝。

## 输入 activation 的三种取法

AIMET 的 SeqMSE 有一个 `inp_symmetry` 参数，它不是量化 symmetric/asymmetric 的意思，而是决定重构时 `x_ref` 和 `x_quant_path` 怎么取。这里的 `x_quant_path` 不是 activation quantizer 量化后的 `x`，而是从 QuantSim model 路径采到的当前层输入；由于前序层 weight quantizer 可能已经开启并冻结，它会包含前序 weight quantization 造成的误差。

### `asym`

```text
x_ref        = FP32 model input activation tensor
x_quant_path = QuantSim model input activation tensor
```

这种方式更接近完整量化路径对比，因为当前层的 quant 分支输入已经包含前面层 weight quantization 造成的误差。但它也把前序层误差和当前层 weight encoding 误差混在了同一个 loss 里。

### `symfp`

```text
x_ref        = FP32 model input activation tensor
x_quant_path = FP32 model input activation tensor
```

这种方式只看当前层 weight quantization 本身造成的误差，不把前序层的量化误差引入当前层。

### `symqt`

```text
x_ref        = QuantSim model input activation tensor
x_quant_path = QuantSim model input activation tensor
```

这是 AIMET 默认值。它用 QuantSim 路径中当前层实际会看到的输入分布做优化，但 float reference 和 quantized output 使用同一份输入，从而把优化重点放在当前层 `w -> wq` 的误差上。

## 为什么要按顺序做

SeqMSE 是 sequential 的原因是：量化误差会沿网络传播。

如果一次性独立优化所有层，每层看到的输入可能都还是理想 FP32 activation。但 SeqMSE 逐层冻结前序层 weight encoding 后，第 N 层在 QuantSim 路径中采到的输入会带有前 N-1 层 weight quantization 的影响。前面层一旦 freeze encoding，后面层的输入分布就随之确定。

因此逐层做有两个好处：

- 后面的层能看到更接近真实量化推理的输入分布。
- 每一层的局部优化目标更稳定。

它不是全局最优算法，但在 PTQ 场景下成本低、实现清晰，通常比纯 min/max 或简单 clipping 更稳。

## SeqMSE 与普通 min/max 的区别

普通 min/max：

```text
只看 tensor 自身分布。
```

SeqMSE：

```text
看这个 tensor 被量化后，对当前层输出造成多大影响。
```

普通 min/max 的目标是覆盖范围；SeqMSE 的目标是输出重构质量。

这也是为什么它能主动裁掉 outlier：如果裁掉少量 outlier 能显著降低主体数据的量化误差，并让层输出更接近 FP32，那么 SeqMSE 会选择更窄的范围。

## 它优化的是什么，不优化什么

以 AIMET Torch 实现为准：

优化：

- Linear weight encoding。
- Conv2d weight encoding。
- 低 bit weight quantization，尤其是 4bit。
- 可支持 per-channel/block-wise weight encoding。

不直接优化：

- activation encoding 或 activation scale/offset。
- attention 内部 softmax/rotary/cache 等非 Linear 权重量化。
- 任意算子的全局误差。
- 端到端 PPL 或 accuracy 本身。

它使用 output activation tensor 的 MSE 做评价，但搜索对象仍然是 weight/parameter encoding。

## 简化伪代码

```python
def seq_mse(model_fp, model_q, calib_data, num_candidates=20):
    disable_activation_quantizers(model_q)
    disable_unsupported_param_quantizers(model_q)

    for name, module_fp in ordered_supported_modules(model_fp):
        module_q = get_quant_module(model_q, name)

        x_ref, x_quant_path = collect_inputs(module_fp, module_q, calib_data)
        min0, max0 = module_q.weight_quantizer.get_min_max()

        losses = []
        candidates = []

        for i in range(num_candidates):
            ratio = (i + 1) / num_candidates
            cand_min = min0 * ratio
            cand_max = max0 * ratio

            module_q.weight_quantizer.set_encoding(cand_min, cand_max)
            wq = quant_dequant(module_q.weight, module_q.weight_quantizer)

            y_ref = linear_or_conv(x_ref, module_fp.weight)
            y_q = linear_or_conv(x_quant_path, wq)

            loss = reconstruction_loss(y_q, y_ref)
            losses.append(loss)
            candidates.append((cand_min, cand_max))

        best = argmin(losses)
        module_q.weight_quantizer.set_encoding(*candidates[best])
        module_q.weight_quantizer.freeze()

    restore_quantizers(model_q)
```

## 优点

- 比纯 min/max 更关注实际输出误差。
- 对 weight-only 或低 bit weight quantization 很有效。
- 不需要反向传播，不需要训练标签。
- 可以逐层执行，计算量可控。
- 可以扩展到 per-channel/block-wise encoding。

## 局限

- 它是局部优化，不保证端到端全局最优。
- calibration 数据质量会明显影响结果。
- 候选集合太窄可能找不到好 encoding，太宽则成本上升。
- 默认 AIMET 实现主要优化 weight encoding，不是 activation encoding。
- 如果 backend 不支持对应粒度的 encoding，算法选出来也无法落地。

## 和 AdaRound / GPTQ 的区别

SeqMSE：

- 搜索 min/max encoding。
- 通常不改 weight 数值本身，只改量化范围。
- 目标是层输出重构误差。

AdaRound：

- 优化 rounding decision。
- 重点是决定每个 weight 往上还是往下取整。
- 通常需要一个小的优化过程。

GPTQ：

- 利用 Hessian 近似做逐列/逐块 weight quantization。
- 更偏 LLM weight-only quantization。
- 会补偿量化误差。

可以粗略理解为：

```text
SeqMSE: 选更好的尺子范围。
AdaRound: 选更好的取整方向。
GPTQ: 量化时顺手补偿误差。
```
