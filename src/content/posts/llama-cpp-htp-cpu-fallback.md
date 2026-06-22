---
author: Ludovico
pubDatetime: 2026-06-05T09:00:00Z
title: llama.cpp HTP 调优：少用 CPU 为什么反而慢
featured: false
draft: false
tags: [论文精读, 大模型]
  - 端侧大模型
description: 用 Qwen3-0.6B 的 profile 解释一个朴素误区：减少 CPU 参与不等于更快。HTP attention、SET_ROWS、图切分和真实 token/s 的关系。
---

这次调 Qwen3-0.6B 时，有一个很自然的目标：

> CPU 留给调度，计算尽量交给 HTP。

听起来合理。实际跑下来，结果反过来：CPU 用得更少，速度更慢。

这篇只记录这个现象。

## 两条路径

同一个模型：

```text
Qwen3-0.6B-Q4_0-output-Q4_0.gguf
```

路径 A：尽量让 HTP 承接算子，不设置 `OPFILTER`。

```bash
GGML_HEXAGON_OPPOLL=1
```

结果：

```text
tg64 = 30.32 ± 1.14 t/s
```

短 completion profile 里：

```text
eval time = 377.78 ms / 16 runs
23.61 ms/token
42.35 token/s
```

路径 B：过滤 HTP 上不划算的 attention/KV 算子。

```bash
GGML_HEXAGON_OPPOLL=1
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS'
```

结果：

```text
tg64 = 67-70 t/s
```

从名字看，路径 A 更像 NPU 加速；从结果看，路径 B 更快。

## no-filter profile 里看到什么

打开 HTP profile 后，no-filter 路径里 `SET_ROWS` 和 `FLASH_ATTN_EXT` 都确实在 HTP 上：

```text
SET_ROWS       6-11 us / op
FLASH_ATTN_EXT 245-296 us / layer
```

`SET_ROWS` 不是大问题。它是 KV cache 写入，单次十微秒以内。

真正贵的是 `FLASH_ATTN_EXT`。Qwen3-0.6B 有 28 层，decode 单 token 每层一次 attention。按 250us 算：

```text
250 us * 28 = 7 ms
```

单 token 要跑到 70 token/s，预算大约是：

```text
1000 ms / 70 = 14.3 ms/token
```

只 attention 就吃掉一半预算，其他 matmul、fallback、采样还没有算。

## 为什么 HTP attention 在这里不划算

这里不是说 HTP attention 写错了，也不是说 HTP 不适合 LLM。

问题是 Qwen3-0.6B 太小。

decode 是单 token 场景，算力利用很难拉满。HTP attention 要处理：

```text
Q/K/V view
KV cache
mask
VTCM scratch
DMA prefetch
worker pool
同步
```

这些固定成本对 4B、7B 可能还能摊掉。对 0.6B，matmul 变小后，固定成本就露出来了。

所以最优策略不是“算子能上 HTP 就上”，而是：

```text
大 matmul 上 HTP
小而碎、调度重的算子 fallback
```

## 单独 filter 更差

还试过只过滤一个：

| 策略 | tg64 |
|---|---:|
| no filter，`FLASH_ATTN_EXT` + `SET_ROWS` 都上 HTP | 30.32 ± 1.14 |
| 只 filter `FLASH_ATTN_EXT` | 21.54 ± 8.82 |
| 只 filter `SET_ROWS` | 17.45 ± 5.36 |
| filter `FLASH_ATTN_EXT|SET_ROWS` | 67.42-69.90 |

这说明问题不只是某个 kernel 慢。图切分也很关键。

只把其中一个算子留在 HTP，会产生更差的 HTP/CPU 交替形态。每一段 split 都要同步、搬数据、调度。小模型上，这些开销很容易压过计算收益。

## mask 裁剪实验失败了

我们试过改 HTP `FLASH_ATTN_EXT`，想在 decode 时根据 mask 尾部的无效位置缩短 KV 长度。

尝试了两种判断：

```text
fp16 -inf   = 0xFC00
fp16 -65504 = 0xFBFF
```

结果没有变化：

```text
FLASH_ATTN_EXT 仍然约 250-290 us / layer
端到端仍然约 41-42 token/s
```

补丁撤回了。

这个失败也有价值：HTP kernel 里看到的 mask，不是简单“尾部全负无穷”的形态。后续如果要优化，不能在 kernel 里靠猜 mask。更靠谱的方向是从 llama graph / KV cache view 构造处拿到真实有效 KV 长度。

## CPU fallback 不是坏事

这次最容易误解的是：

```text
OPFILTER = CPU fallback = 不好
```

实测不是。

对这个模型，这条路径最快：

```bash
GGML_HEXAGON_OPPOLL=1
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS'
```

它的含义是：

```text
HTP: Q4_0 matmul、主要 dense 计算
CPU: attention/KV 小算子、lm-head logits
```

混合执行不是妥协，而是当前硬件和 kernel 形态下的最优点。

## 什么时候应该少用 CPU

如果目标是系统调度余量、电源、热设计，少用 CPU 是合理目标。

但如果目标是 token/s，要先看 profile。

这次 no-filter 路径确实减少了 CPU 计算，但速度只有：

```text
41-42 token/s
```

最高速度路径是：

```text
67-70 token/s
```

所以两个目标要分开说：

| 目标 | 当前策略 |
|---|---|
| 最高 token/s | filter `FLASH_ATTN_EXT|SET_ROWS` |
| 少用 CPU | no-filter，但需要继续优化 HTP attention |

如果要同时做到少用 CPU 和高 token/s，下一步不是调参数，而是改 HTP `FLASH_ATTN_EXT` 的 decode 路径。

