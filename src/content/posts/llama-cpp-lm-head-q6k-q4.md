---
author: Ludovico
pubDatetime: 2026-06-05T10:00:00Z
title: llama.cpp 里的 lm-head：Q6_K、Q4_0，以及为什么 4B 不一定提速
featured: false
draft: false
tags:
  - 量化
  - 论文精读
  - 端侧大模型
description: 从 Qwen3/Qwen3.5 的 tied embedding 看 lm-head 性能瓶颈：0.6B 改 output-q4_0 有收益，4B 上收益有限甚至可能变慢。
---

Qwen3-0.6B 能从 58 token/s 到 70 token/s，很大一部分来自 lm-head。

这件事容易说糊。这里把细节拆开。

## Qwen3 的 lm-head 在哪里

很多 Qwen GGUF 没有单独的 `output.weight`，而是 tied embedding：

```text
token_embd.weight 同时用于输入 embedding 和输出 lm-head
```

Qwen3-0.6B 原始 Q4_0 文件里，tensor 类型是：

```text
type f32:  113 tensors
type q4_0: 193 tensors
type q4_1:   3 tensors
type q6_K:   1 tensors
```

这唯一的 `q6_K` 就是：

```text
token_embd.weight
```

最后 logits 计算就是：

```text
token_embd.weight x result_norm -> result_output
```

profile 里能看到它落在 CPU：

```text
MUL_MAT result_output
token_embd.weight [CPU]
result_norm       [HTP0]
```

原因很直接：当前 HTP 后端不支持这个大 `q6_K` lm-head。

## lm-head 是不是 FP 计算

不是完整反量化成 FP 矩阵再算。

以 `q4_0` 为例，CPU kernel 是：

```text
q4_0 weight x q8_0 activation -> f32 output
```

源码里 `Q4_0` 的 vec dot 是：

```cpp
ggml_vec_dot_q4_0_q8_0
```

核心逻辑是：

```cpp
sumf += sumi * scale_x * scale_y;
*s = sumf;
```

也就是说：

1. 权重是 4-bit + block scale。
2. activation 会转成 dot kernel 需要的 `q8_0` 临时格式。
3. 每个 block 做整数 dot。
4. 乘 scale 后累加到 float。
5. 输出 logits 是 FP32。

所以更准确的说法是：

```text
量化权重参与 mixed quant dot，结果是 f32 logits
```

## 为什么把 0.6B 的 output 改 q4_0 有用

原始 0.6B：

```text
token_embd.weight: q6_K
模型大小: 358.78 MiB
```

重新量化：

```bash
./build-host/bin/llama-quantize \
  --allow-requantize \
  --output-tensor-type q4_0 \
  --token-embedding-type q4_0 \
  Qwen3-0.6B-Q4_0.gguf \
  Qwen3-0.6B-Q4_0-output-Q4_0.gguf \
  Q4_0 4
```

结果：

```text
token_embd.weight: q6_K -> q4_0
模型大小: 358.78 MiB -> 319.96 MiB
tg64: 58.59 -> 67-70 token/s
```

这个收益不是因为 lm-head 上了 HTP。

相反，lm-head 仍然主要在 CPU 上算，只是从 CPU 上的 `q6_K` matmul 变成了更快的 `q4_0` matmul。

## 强行把 lm-head 放 HTP，反而慢

我们加过实验开关，让大 lm-head 也能被 HTP claim。

结果：

```text
CPU q4_0 lm-head: 67-70 t/s
HTP q4_0 lm-head: 65.74 t/s
```

HTP compute buffer 从大约 6.5 MiB 涨到 74.7 MiB，CPU compute buffer 降了，但 token/s 变差。

这说明当前 v73 HTP 上，大 vocab lm-head 不是好任务。大矩阵不等于一定适合 HTP，还要看 VTCM、repack、数据搬运和 split 形态。

## Qwen3.5-4B 也是 Q6_K lm-head

检查 `Qwen3.5-4B-Q4_0.gguf`：

```text
output_norm.weight  F32   [2560]
token_embd.weight   Q6_K  [2560, 248320]
```

同样没有单独 `output.weight`，也是 tied embedding。

所以理论上也可以做：

```bash
./build-host/bin/llama-quantize \
  --allow-requantize \
  --output-tensor-type q4_0 \
  --token-embedding-type q4_0 \
  Qwen3.5-4B-Q4_0.gguf \
  Qwen3.5-4B-Q4_0-output-Q4_0.gguf \
  Q4_0 4
```

但 4B 不一定像 0.6B 那样涨。

## 为什么 4B 不一定提速

0.6B 的主体 matmul 很小，lm-head 和调度开销占比高。改 lm-head 类型，收益明显。

4B 不一样：

```text
模型主体更大
每层 matmul 更重
decode 时间主要花在 transformer block
lm-head 占比下降
```

实测 Qwen3.5-4B 原始 Q4_0 在 HTP 上大约：

```text
pp128 = 132.82 t/s
tg64  = 10.55 t/s
```

如果 output-q4_0 没有涨，甚至更慢，不奇怪。说明瓶颈不在尾部 head，或者 q4_0 带来的计算收益被其他开销吃掉了。

这不是实验失败。它告诉我们：

```text
0.6B 的优化不能直接外推到 4B
```

## 质量风险

把 lm-head 从 `q6_K` 改成 `q4_0` 是性能优化，不是无损优化。

可能影响：

```text
logits 排序
小概率 token 选择
多轮对话稳定性
中英文混合输出
```

0.6B 上它能正常对话，但如果要正式使用，需要拿真实 prompt 做质量对比。

我的建议：

| 场景 | 建议 |
|---|---|
| 追速度、做部署验证 | output-q4_0 值得用 |
| 追质量、长对话 | 保留 q6_K，或做 A/B |
| 4B/更大模型 | 先 profile，别默认改 |

## 结论

lm-head 优化要看占比。

对 Qwen3-0.6B：

```text
q6_K lm-head 是瓶颈
q4_0 output 能明显提速
```

对 Qwen3.5-4B：

```text
尾部也是 Q6_K
但主体计算更重
output-q4_0 不保证有收益
```

这类优化最怕凭感觉。先看 tensor 类型，再看 scheduler split，最后看真实 `eval time`。数字比直觉可靠。

