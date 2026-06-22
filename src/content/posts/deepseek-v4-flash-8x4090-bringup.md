---
author: Ludovico
pubDatetime: 2026-06-15T08:00:00Z
title: 在 8x4090 上硬跑 DeepSeek-V4-Flash 的踩坑随记
featured: true
draft: false
tags:
  - 论文精读
  - 量化
  - 大模型推理
  - vLLM
description: 本来想看看 4090 能不能凑合跑 DeepSeek-V4-Flash，跑是跑通了，但离"能用"还很远。一篇随记。
---

本来想的是：8 张 4090 加起来 192G 显存，FP8 量化一下，跑个 DeepSeek-V4-Flash 应该勉强能玩吧。结果是跑通了，但也只是"跑通了"。

最小请求能返回 `OK`，官方 benchmark 脚本能过。但首 token 等 40 秒，输出每秒 2 token，就算并发拉满也只是在排队——**4090 跑 DeepSeek-V4 不具备实用性**。

## 能跑的配置

环境变量和启动参数：

```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NVIDIA_TF32_OVERRIDE=0

python -m vllm.entrypoints.openai.api_server \
  --model /home/q/models/deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --cpu-offload-gb 14.81 \
  --max-model-len 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  -O1
```

`cpu-offload-gb 14.81` 是个精确数字，少一点直接 OOM。基本是把 4090 显存吃满之后往外溢出的量。

## 为什么一开始连启动都费劲

DeepSeek-V4 的代码假设你有一台 Hopper 机器。4090 是 Ada/SM89，跟这个假设在三个地方撞墙：

1. **nvcc 不支持 `sm_89`**。本机的 `/usr/bin/nvcc` 编不了 `-arch=sm_89`，任何跑着跑着突然 JIT 一段 CUDA kernel 的路径全炸。
2. **高性能路径默认走 Hopper-only 实现**。FP8 block-scaled linear、FlashMLA、MHC、FlashInfer sampler……这些默认 kernel 跟你 4090 没关系。
3. **错误不发生在启动阶段**。模型加载完、服务 started 日志打出来都是正常的，真正炸的是第一个请求走到 decode/sampling/sparse attention 的时候。

所以好几次我以为是"快成功了"，结果换一个请求长度又崩了。

## 修了什么

策略就一条：**不硬开 4090 不支持的优化，把走不通的路径全部降级到 PyTorch reference 或 Triton fallback。**

改了大概 8 类补丁，涉及 vLLM 主线十几个文件。每个都是"不改就跑不过去"——

**编译配置**：`CompilationConfig` 里缺 `encoder_compilation_time` 字段导致启动期就报错；SM89 上 cudagraph 在 `nonzero()`、boolean mask、`copy_()` 这些位置需要 PIECEWISE 切图，不然 graph capture 直接挂。

**FP8 block-scaled linear**：这是第一个请求就炸的地方。4090 没有 Hopper 的 FP8 硬件路径，Triton 路径里 `float8_e8m0fnu` 类型也不存在。补了一个纯 PyTorch reference 的 emulation kernel，准确但不快。

**FlashMLA / 稀疏 attention**：原生 `_flashmla_C` 不可用，稀疏 decode 的索引计算 (`compute_swa_indices_and_lens` 等) 和 KV cache dequantize + gather 也要补 SM89 fallback。这部分补完，长上下文才不崩。

**Compressor / indexer / KV cache 写入流水线**：这链路依赖 cutedsl 做 fused indexer，4090 上只能用 Python reference 替掉。补了一整套 compressor → RMSNorm → RoPE → FP8 quant → KV cache write 的 fallback。

**MHC（Multi-Head Compression）**：整件事里最折腾的一个。MHC 路径在首请求时触发 TileLang/TVM 编译，编译命令直接调 `nvcc --cubin -arch=sm_89`，worker 当场去世，返回 HTTP 500。补了 `mhc_pre`、`mhc_post`、`mhc_fused_post_pre`、`hc_head` 全部 reference 实现，再把 fused-with-norm 的 TileLang 路径拆成 reference MHC + Python RMSNorm。

**FlashInfer sampler 禁用**：SM89 上 FlashInfer 做本地 JIT 还是会碰到 `sm_89` 工具链问题，直接标记 unsupported 回退到 native sampler。

修完这些之后，终于拿到了第一个 `{"choices":[{"message":{"content":"OK"}}]}`。

## 跑起来之后的性能

注意：`--max-num-seqs 1` 意味着一口气只能处理一个请求，benchmark 里的"并发"实际上是排队。

短请求（128 in / 32 out）：

| 并发 | 吞吐 | Mean TTFT | Mean TPOT |
| --- | --- | --- | --- |
| 1 | 2.04 tok/s | 0.96s | 476ms |
| 2 | 2.05 tok/s | 12.66s | 475ms |

并发 2 的时候 TTFT 直接从 1s 飙到 12.7s —— 因为根本不能并行，第二个请求纯在排队。

长上下文：

| 输入长度 | Mean TTFT | Mean TPOT |
| --- | --- | --- |
| 4096 | 39s | 492ms |
| 8192 | 41s | 488ms |
| 16384 | 114s | 477ms |

16384 长度的首 token 等将近两分钟。这还算能跑通的，32k 直接 `400 Bad Request`，虽然不 OOM 但就是拒绝服务。

多轮对话倒是全过了，4 轮 × 4 请求，平均 TTFT 1.16s。但受限 `max-num-seqs 1`，实际还是串行。

核心问题不是显存——engine 日志显示 KV cache 容量有 1,272,663 tokens，理论上能撑 38 路 32k。问题是 reference/fallback 路径太慢了，而且并发等于排队，**这机器连"凑合用"的标准都够不上**。

## 结论

如果你看到这篇文章在想"4090 是不是也能跑 DeepSeek-V4"——能跑，但只限于能跑。输出 2 token/s，首 token 等半分钟到两分钟，没什么实用价值。真要部署还是等官方出更轻量的量化版本，或者直接上 Hopper。
