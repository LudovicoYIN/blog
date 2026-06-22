---
author: Ludovico
pubDatetime: 2026-06-05T08:00:00Z
title: llama.cpp 跑 Qwen3-0.6B：HTP 上从 20 到 70 token/s
featured: true
draft: false
tags: [论文精读, 量化]
  - 端侧大模型
description: 一次 Snapdragon Hexagon HTP 上的真实调优记录：OPPOLL、算子过滤、lm-head 量化，以及为什么 70 token/s 不是“全上 NPU”跑出来的。
---

这次目标很简单：在 Snapdragon / Hexagon HTP 上，把 `Qwen3-0.6B-Q4_0.gguf` 的 decode 从 20 多 token/s 往上推，先过 50 token/s，再看能不能继续靠近 100 token/s。

最后稳定结果是：

```text
Qwen3-0.6B-Q4_0-output-Q4_0.gguf
pp128 = 789-850 t/s
tg64  = 67-70 t/s
```

真实 completion 也能对上：

```text
eval time = 1812.68 ms / 127 runs
14.27 ms/token
70.06 token/s
```

这篇只讲怎么得到这个结论。

## 基线

最早的裸 HTP 路径并不好看。Qwen3-0.6B Q4_0 在 HTP 上只有 20 多 token/s，和我们对“NPU 加速”的直觉不一致。

原因不是模型没上 HTP。日志里能看到：

```text
ggml-hex: Hexagon Arch version v73
load_tensors: layer 0 assigned to device HTP0
...
ggml-hex: HTP0 profile-op MUL_MAT
```

权重和主要 matmul 确实在 HTP 上。慢点在别处：小模型 decode 的固定调度成本、attention/KV 小算子成本，以及最后 lm-head CPU split。

先加第一步优化：

```bash
GGML_HEXAGON_OPPOLL=1
```

完整命令：

```bash
cd /data/local/tmp/llama.cpp

LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
./bin/llama-bench \
  -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0.gguf \
  -p 128 -n 64 -t 4 \
  -ngl 99 -dev HTP0 -fa on -r 3
```

`OPPOLL` 很关键。它让 DSP queue completion 从等待模式变成轮询，减少 FastRPC / HTP 调度延迟。0.6B 模型很小，单 token matmul 不大，所以这种固定调度开销占比很高。

这个阶段大概是：

```text
tg64 = 39-43 t/s
```

这一步已经从 20 多 token/s 推到了 40 左右，但还不够。

阶段结果可以先这样记：

| 阶段 | 关键变化 | tg64 |
|---|---|---:|
| 初始 HTP | `-ngl 99 -dev HTP0` | 20+ t/s |
| 降低调度等待 | `GGML_HEXAGON_OPPOLL=1` | 39-43 t/s |
| 过滤 HTP attention/KV | `GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS'` | 58.59 t/s |
| 输出层 q4_0 | `token_embd.weight: q6_K -> q4_0` | 67-70 t/s |

## 第一个反直觉点：不是所有算子都适合上 HTP

开 Hexagon profile 后能看到，decode 阶段 HTP 不只在跑 matmul，也在跑：

```text
FLASH_ATTN_EXT
SET_ROWS
```

原始 profile 里，`FLASH_ATTN_EXT` 单次大约 250us，Qwen3-0.6B 有 28 层，累计就是 7ms 级别。对大模型这可能不算什么；对 0.6B，它就是主要延迟之一。

于是试了：

```bash
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS'
```

这个名字容易误解。它不是“指定这些算子跑 HTP”，而是反过来：过滤 HTP claim，让这些算子 fallback。

完整命令：

```bash
LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS' \
./bin/llama-bench \
  -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0.gguf \
  -p 128 -n 64 -t 4 \
  -ngl 99 -dev HTP0 -fa on -r 3
```

结果：

```text
pp128 = 894.71 ± 1.02 t/s
tg64  =  58.59 ± 0.61 t/s
```

这一步把目标从 50 token/s 以下推到了 58 token/s。

## 为什么 `-fa on` 还要保留

容易犯的错是：既然 HTP 的 `FLASH_ATTN_EXT` 慢，那是不是关掉 flash attention？

不是。

实测四组：

| 配置 | `-fa` | tg64 |
|---|---:|---:|
| `OPPOLL=1` | off | 36.92 |
| `OPPOLL=1` | on | 43.45 |
| `OPPOLL=1` + filter `FLASH_ATTN_EXT|SET_ROWS` | off | 33.02 |
| `OPPOLL=1` + filter `FLASH_ATTN_EXT|SET_ROWS` | on | 58.59 |

结论很直接：`-fa on` 要保留。我们只是不要让 HTP 执行这个 `FLASH_ATTN_EXT` kernel。

## 第二个瓶颈：lm-head

58 token/s 后继续看 scheduler split，发现最后一个 CPU split 很重：

```text
MUL_MAT: result_output
src0 = token_embd.weight
src1 = result_norm
backend = CPU
```

原始模型 tensor 类型：

```text
type f32:  113 tensors
type q4_0: 193 tensors
type q4_1:   3 tensors
type q6_K:   1 tensors
```

唯一的 `q6_K` 就是 `token_embd.weight`。Qwen3 这里是 tied embedding，最后 lm-head 也用它。HTP 当前不支持这个大 `q6_K` lm-head，所以落到 CPU。

CPU 上算 lm-head 本身没错，但 `q6_K` 在这里很贵。

于是把 output / token embedding 重新量成 `q4_0`：

```bash
./build-host/bin/llama-quantize \
  --allow-requantize \
  --output-tensor-type q4_0 \
  --token-embedding-type q4_0 \
  /home/luke/code/quetecl_deploy/llm/Qwen3-0.6B-Q4_0.gguf \
  /home/luke/code/quetecl_deploy/llm/Qwen3-0.6B-Q4_0-output-Q4_0.gguf \
  Q4_0 4
```

变化：

```text
token_embd.weight: q6_K -> q4_0
model size: 358.78 MiB -> 319.96 MiB
BPW: 5.05 -> 4.50
```

再跑同一套 HTP 配置：

```text
tg64 = 69.90 ± 0.47 t/s
```

后续复测稳定在：

```text
67-70 t/s
```

## 这不是把 lm-head 推上 HTP

我们也试过强行让 q4_0 lm-head 上 HTP。

代码里原本有保护：

```cpp
if (ggml_nrows(src0) > 16 * 1024) {
    return false;
}
```

这是为了避免大 vocab lm-head 进 HTP 后 VTCM / buffer 代价过高。加实验开关后，lm-head 确实能被 HTP 承接，但结果是：

```text
CPU q4_0 lm-head: 67-70 t/s
HTP q4_0 lm-head: 65.74 t/s
```

所以当前 v73 HTP 上，大 vocab lm-head 不适合强推到 HTP。让 CPU 算 q4_0 lm-head 反而更快。

## 最终命令

```bash
cd /data/local/tmp/llama.cpp

LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS' \
./bin/llama-bench \
  -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0-output-Q4_0.gguf \
  -p 128 -n 64 -t 4 \
  -ngl 99 -dev HTP0 -fa on -r 3
```

真实生成用：

```bash
LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS' \
./bin/llama-completion \
  -m /data/local/tmp/gguf/Qwen3-0.6B-Q4_0-output-Q4_0.gguf \
  -p "hello /no_think" -n 128 \
  -t 4 -tb 4 \
  -c 1024 -b 128 -ub 128 \
  -ngl 99 -dev HTP0 -fa on \
  -no-cnv --no-warmup --no-display-prompt --no-mmap
```

## 这次学到的

1. 小模型上，HTP 的固定调度成本很敏感。
2. 裸 HTP 只有 20 多 token/s；`OPPOLL=1` 才把它拉到 40 左右。
3. 最高 token/s 不是“所有算子都上 NPU”，而是 matmul 留在 HTP，不划算的小算子 fallback。
4. Qwen3-0.6B 的 tied lm-head 如果是 `q6_K`，CPU logits 会成为瓶颈。
5. 把 lm-head 从 `q6_K` 改成 `q4_0` 能明显提速，但要接受一点输出质量风险。
6. bench 的 `tg64` 和真实 `llama-cli` 对话速度不是同一个口径；`llama-completion` 的 `eval time` 更接近 bench。
