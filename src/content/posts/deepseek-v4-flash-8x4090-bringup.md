---
author: Ludovico
pubDatetime: 2026-06-15T08:00:00Z
title: DeepSeek-V4-Flash 在 8x RTX 4090 上的 bring-up 记录
featured: true
draft: false
tags:
  - 大模型推理
  - vLLM
description: 把 DeepSeek-V4-Flash 在 8x4090/SM89 上从"服务能启动"推到"首请求返回 200"，记录了每一类补丁的含义、故障消除顺序和官方 benchmark 结果。
---

目标：在 8x NVIDIA RTX 4090（SM89 / Ada）上，通过 `vllm.entrypoints.openai.api_server` 跑通 `DeepSeek-V4-Flash`，不是只启动服务，而是 `/v1/chat/completions` 真正返回正确结果。

最终状态：服务启动成功，最小请求返回 `OK`，官方短请求、多轮、`4096/8192/16384` 长上下文均已验证通过。

## 最终可工作的配置

环境变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:512'
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=600
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1
export NVIDIA_TF32_OVERRIDE=0
```

启动命令：

```bash
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

验证请求：

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "DeepSeek-V4-Flash",
    "messages": [{"role": "user", "content": "你好，请只回复OK。"}],
    "max_tokens": 8,
    "temperature": 0
  }'
# → {"choices":[{"message":{"content":"OK"}}]}
```

## 为什么一开始跑不起来

问题不是单点故障，而是多层叠加。

### 启动参数层

两个非内核问题让排查方向跑偏：

1. `python -m vllm.entrypoints.openai.api_server` 不会把位置参数自动当成 `--model`，服务可能悄悄回落到默认模型。
2. `-O.mode=PIECEWISE` 不是合法写法，需要用 `-O1` 或显式传 `--compilation-config`。

### 4090 / SM89 与主线内核不匹配

`DeepSeek-V4-Flash` 不是单一路径的纯 FP8 模型。Dense/attention 一部分是 FP8 block-scaled，MoE/compressor/KV cache/稀疏解码又会牵出 Triton、TileLang、FlashMLA、FlashInfer、cutedsl、DeepGEMM 等多条路径。

在 4090 上，三个核心矛盾：

1. 设备是 SM89，不是 Hopper/SM90。
2. 本机 `/usr/bin/nvcc` 不支持 `-arch=sm_89`，任何运行期 JIT 到 nvcc 的路径直接失败。
3. 主线 DeepSeek-V4 高性能实现默认偏向 Hopper/Blackwell，或假设本机可以即时编译 CUDA 内核。

所以"服务启动成功"不等于"首请求成功"——大量错误在第一次真实 decode/sampling/sparse attention/MHC 路径上才暴露。

## 补丁策略

不是硬把 4090 伪装成 Hopper，也不是强行开启不支持的 JIT。策略是：

1. 修启动参数和兼容字段。
2. 对 Hopper-only 或会触发 `nvcc -arch=sm_89` 运行期编译的路径，显式绕开。
3. 尽量复用仓库里已有的 PyTorch/Triton/reference 实现。
4. 必要时补本地 reference/emulation/Triton fallback，只覆盖真正挡路的切片。

即"把不兼容路径有控制地降级到可运行路径"，不是"打开更多优化"。

## 补丁分组

### 1. 启动与编译配置兼容

`vllm/config/compilation.py`、`vllm/compilation/backends.py`：

- 恢复 `encoder_compilation_time` 字段，避免启动期统计访问缺字段报错。
- 把 SM89 reference 自定义算子加入 `CompilationConfig._attention_ops`，让 PIECEWISE cudagraph 在 `.nonzero()`、布尔 mask、`copy_()` 等不适合完整图捕获的位置切图。

### 2. 统一 reference 开关

`vllm/utils/deep_gemm.py`：

- 新增 `_use_sm86_reference()`，统一承载 `SM < 90` 的降级判断。
- 在 Ampere/Ada 上优先选择 reference/fallback 路径。
- 留环境变量 `VLLM_SM86_DEEPSEEK_V4_REF` 做强制开关。

### 3. FP8 block-scaled linear 降级

`vllm/model_executor/kernels/linear/scaled_mm/emulation.py`、`.../linear/__init__.py`、`.../quantization/utils/fp8_utils.py`、`.../scaled_mm/triton.py`：

- 新增 `EmulationFp8BlockScaledMMKernel`，PyTorch reference 纯软件 fallback。
- SM89 reference 模式下强制 `Fp8LinearMethod` 选 emulation kernel。
- 修正 Triton 环境中 `float8_e8m0fnu` 不存在的问题，以及 SM<89 上 FP8 Triton 路径不该下发的问题。

### 4. FlashMLA 与稀疏解码 fallback

`vllm/v1/attention/ops/flashmla.py`、`vllm/third_party/flashmla/`、`vllm/utils/fp8_paged_mqa_logits_sm86.py`：

- 原生 `_flashmla_C` 不可用时，SM8x 走 Python/Triton fallback。
- 提供 SM8x 稀疏 decode Triton 核和 paged MQA logits Triton 核。
- 保留 DeepSeek-V4 稀疏 decode 的算法形状，但用 4090 能承受的实现替掉不兼容主线 kernel。

### 5. 稀疏元数据、KV gather、partial states reference 化

`vllm/v1/attention/backends/mla/sparse_swa.py`、`vllm/models/deepseek_v4/common/ops/cache_utils.py`、`vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py`、`.../save_partial_states.py`：

- `compute_swa_indices_and_lens`、`compute_global_topk_indices_and_lens` 增加 Python reference 路径。
- `dequantize_and_gather_k_cache` 增加 SM89 fallback，SM89 上禁用 cutedsl。
- `save_partial_states` 增加 reference 实现。

### 6. Compressor / indexer / quant cache fallback

`vllm/models/deepseek_v4/compressor.py`、`vllm/model_executor/layers/deepseek_compressor.py`、`vllm/v1/attention/ops/deepseek_v4_ops/`：

- head_dim=128 和 head_dim=512 的 compressor 路径分别接入 SM89 pyref/fallback。
- SM89 上关闭依赖 cutedsl 的 fused indexer。
- 新增 `deepseek_v4_ops` 目录，放置压缩→RMSNorm→RoPE→FP8 quant→KV cache 写入整条流水线的 fallback。

### 7. MHC / TileLang 绕行（首请求 500 的最后一根刺）

`vllm/model_executor/kernels/mhc/tilelang.py`、`vllm/model_executor/layers/mhc.py`、`vllm/models/deepseek_v4/nvidia/model.py`：

- MHC 路径在首请求时触发 TileLang/TVM 编译，编译命令调用 `nvcc --cubin -arch=sm_89`，本机 nvcc 不支持，worker 直接崩。
- 补齐 `mhc_pre`、`mhc_post`、`mhc_fused_post_pre`、`hc_head` 的 reference 实现。
- `norm_weight` 存在时，新增 `_rms_norm_bf16()`，把 fused-with-norm 的 TileLang 路径拆成 reference MHC + Python 侧 RMSNorm。

### 8. FlashInfer sampler 禁用 JIT

`vllm/v1/sample/ops/topk_topp_sampler.py`：

- SM89 上 FlashInfer sampler 做本地 JIT 仍会碰到 `sm_89` nvcc 工具链问题。
- 标记为 unsupported，回退到 native sampler。

## 故障消除顺序

1. 修启动参数与 `CompilationConfig` → 服务能稳定用目标模型启动。
2. 修 FP8 block-scaled linear kernel 选择 → 不死在不支持的 FP8 backend。
3. 补 FlashMLA / 稀疏 helper / compressor / indexer SM89 fallback → attention 和 KV cache 流程走通。
4. 禁用 FlashInfer sampler JIT → 请求尾部不再炸。
5. 修 MHC TileLang 请求期 JIT → 首请求从 500 变 200。

## 官方 Benchmark 结果

测试全部使用仓库自带官方脚本（`vllm.entrypoints.cli.main bench serve` 和 `benchmarks/multi_turn/benchmark_serving_multi_turn.py`），非手写压测。

注意：当前 `--max-num-seqs 1`，并发 > 1 时体现的是排队和 TTFT 恶化，不是引擎真正并行执行。

### 显存与 KV cache

启动后 engine 日志：

- GPU KV cache size: 1,272,663 tokens
- Maximum concurrency for 32,768 tokens per request: 38.84x

理论容量大，但 `--max-num-seqs 1` + fallback 路径，不等价于可稳定支撑 38 路 32k。

### 短请求吞吐

128 in / 32 out：

| 并发 | 成功/失败 | 总时长 | 请求吞吐 | 输出吞吐 | Mean TTFT | Mean TPOT |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4/0 | 62.87s | 0.064 req/s | 2.04 tok/s | 963ms | 476ms |
| 2 | 4/0 | 62.59s | 0.064 req/s | 2.05 tok/s | 12656ms | 475ms |

并发 2 时 TTFT 从 0.96s 升到 12.66s，说明基本只是客户端排队。TPOT 稳定在 ~475ms/token。

### 长上下文

经过两轮修复（FlashMLA sparse prefill 分块化、MQA logits reference 分块化）后的结果：

| 输入长度 | 成功/失败 | 总时长 | Mean TTFT | Mean TPOT |
| --- | --- | --- | --- | --- |
| 4096 | 1/0 | 46.51s | 39126ms | 492ms |
| 8192 | 1/0 | 48.51s | 41191ms | 488ms |
| 16384 | 1/0 | 121.13s | 113977ms | 477ms |

`32768 in / 16 out` 不触发 OOM，但被 API 层以 `400 Bad Request` 拒绝，服务在请求后仍健康。

### 多轮对话

`num_clients=1, max_active_conversations=1, max_num_requests=4, max_turns=4`：

- 4/0 全部成功，benchmark runtime 58.96s
- mean ttft 1162ms, mean tpot 498ms, mean latency 14734ms
- mean input 315 tokens, mean output 29 tokens

短上下文多轮功能正确，但受 `--max-num-seqs 1` 限制，本质仍是单路串行。

## 当前边界

1. 大量路径是 reference/fallback/emulation，正确性优先于性能。
2. 短请求和保守多轮已验证可用，但并发提升不线性带来吞吐提升。
3. 长上下文已验证到 16384；32k 当前被 API 拒绝，不能直接宣称可用。
4. 本机 nvcc 仍不支持 `sm_89`，任何新的运行期 JIT 路径理论上仍可能踩雷。

