---
author: Ludovico
pubDatetime: 2026-08-26T10:00:00+08:00
title: QNN W8A16 的 INT32 Bias Overflow：从极小 Scale 到稳定部署
featured: false
draft: false
tags:
  - QNN
  - 量化部署
  - Qualcomm
description: 记录 QNN HTP W8A16 中 INT32 bias 溢出的根因、公式、QueOpt 修复方式，以及在真实板端的验证结果。
---

在 QNN HTP 的 W8A16 部署中，模型可能出现一种很难从最终输出直接判断的掉点：转换成功，部分输入也正常，但另一类输入的 logits 整体异常，甚至全部翻成同一个类别。

这类问题的根因可能不是输入 layout，而是 **bias 的 INT32 编码溢出**。

## 先看公式

对于使用 INT8 weight、INT16 activation 的 Conv 或 Gemm，bias 的组合 scale 为：

```text
bias_scale = input_scale * weight_scale
bias_int32 = round(bias_fp32 / bias_scale)
```

INT32 有限幅，因此必须满足：

```text
abs(bias_fp32) / (input_scale * weight_scale) <= 2^31 - 1
```

反过来，weight scale 至少需要达到：

```text
required_weight_scale = abs(bias_fp32)
                       / ((2^31 - 1) * input_scale)
```

如果实际 weight scale 小于这个值，bias 的整数表示就可能超过 INT32 范围，最终在 QNN 中发生饱和或溢出。

## TSC 模型里的具体根因

问题出现在 SE 分支：

```text
feature
  -> GlobalAveragePool
  -> fc1 Conv
  -> ReLU
  -> fc2 Conv
```

`fc1` 是一个很小的降维层，weight 大约只有 `0.002`。它把 pooled feature 压到了最大约 `0.0228`，ReLU 又让约 `73%` 的值变成 0。

因此，`fc2` 的输入范围非常小：

```text
input_scale ≈ 6.95e-7
```

但 `fc2` 的 bias 最大约为 `1.2458`，原始 weight scale 也很小。两者相乘后，bias scale 极小，使得：

```text
bias_int32 ≈ 7.36e10
```

这个数约为 INT32 上限的 34 倍。转换链路仍可能完成，但运行时输出已经不可靠。

## QueOpt 的修复

修复不是修改 bias 的浮点值，也不是在导出的 encoding 文件上做事后补丁，而是在量化过程中调整对应的 weight scale。

`QNNBiasScaleAdjustmentPass` 的执行顺序是：

```text
activation calibration
  -> QNNBiasScaleAdjustmentPass
  -> passive bias scale 推导
  -> QNN converter
```

对每个带 bias 的计算层，执行：

```text
weight_scale = max(
    weight_scale,
    abs(bias) / ((2^31 - 1) * input_scale)
)
```

具体处理方式：

| 算子                 | 权重粒度    | 处理                                  |
| -------------------- | ----------- | ------------------------------------- |
| Conv / ConvTranspose | per-channel | 按输出通道分别抬高必要的 weight scale |
| Gemm                 | per-tensor  | 取所有输出通道所需 scale 的最大值     |

这样做只会抬高确实不满足 INT32 范围约束的通道，不会修改权重浮点值；代价是这些通道的 INT8 权重分辨率略有降低，换来 bias 编码合法。

## Scale 下限的统一策略

此前尝试过对 QNN W8A16 单独设置一个较大的 weight scale 下限，但 `0.01 / 127` 会粗化大量小权重通道，反而损失精度。

当前采用统一的 observer 下限：

```python
OBSERVER_MIN_SCALE = 1e-4 / ((2 ** 7) - 1)
```

不再在 `QNNQuantizer` 中注入 QNN 专属 weight floor。这样所有 observer 共用同一底线，同时由 `QNNBiasScaleAdjustmentPass` 精确处理真正的 bias overflow 层。

## 真实板端验证

在 Qualcomm SM7325（v68）上，用相同的 QNN context 和完整测试集验证：

| 模型              | 样本 | 分类一致率 | 平均 cosine（0 类 / 1 类） |
| ----------------- | ---: | ---------: | -------------------------: |
| TSC               |   20 |      20/20 |        0.999904 / 0.999947 |
| hand_rotation_t12 |  128 |    128/128 |        0.999994 / 0.999998 |

另一个平台回传的 `model (6).zip` 也在同一块板上完成 128 条全量验证：

```text
0 类：64/64，cosine 0.99999979
1 类：64/64，cosine 0.99999892
总体：128/128
```

## 排查时应该看什么

遇到 W8A16 掉点时，建议按下面顺序确认：

1. 从 ONNX 或校准中间结果检查 activation 的 absmax 和 scale。
2. 找出对应计算层的 bias 最大值和 weight scale。
3. 计算 `abs(bias) / (input_scale * weight_scale)`，检查是否超过 `2^31 - 1`。
4. 查看调整后的 encoding 是否满足 `weight_scale >= required_weight_scale`。
5. 不只看 host 仿真，必须用同一输入在目标板端运行 context，再比较 logits、cosine 和分类结果。

这个问题的关键是：**极小 activation scale 本身不一定是错误，但它会放大后续 bias 编码的风险；真正需要约束的是 bias 的整数表示范围。**
