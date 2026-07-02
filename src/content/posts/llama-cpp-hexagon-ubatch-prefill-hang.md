---
author: Ludovico
pubDatetime: 2026-06-23T08:30:00Z
title: "llama.cpp Hexagon HTP 长 Prompt 默认 ubatch=512 卡住的定位与修复"
featured: false
draft: false
tags:
  - 端侧大模型
  - llama.cpp
description: Qwen3-4B Q4_0 在 Snapdragon HTP 后端上处理超过 512 token 的 prompt 时，默认 ubatch-size=512 会在 prefill 阶段卡住；最终定位到 HMX matmul 对小 remainder batch 启用异步 pipeline 后不返回。
---

## 背景

在板端用 `llama.cpp` 的 Hexagon HTP 后端测试小远 Qwen3-4B Q4_0 回复质量时，短问题可以正常完成，但一条稍长的对话总结题出现了“像卡死”的现象。

测试命令使用：

```bash
LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS' \
./bin/llama-completion \
  -m /data/local/tmp/gguf/Qwen3-4B-Q4_0.gguf \
  -sysf /data/local/tmp/llama_system_prompt.txt \
  -p "请根据给出的对话进行上下文理解，总结对话内容，对话：... /no_think" \
  --single-turn \
  --simple-io \
  --no-display-prompt \
  -t 4 \
  --ctx-size 2048 \
  -fa on \
  -ngl 99 \
  --device HTP0 \
  --temp 0 \
  -n 1 \
  --perf
```

把 `-n` 压到 1 后仍然 10 秒不出首 token，说明问题不在 decode 太长，而在 prompt prefill 阶段。

## 稳定复现

### 正常样例：短 Prompt

短问题 “切洋葱时为啥会流泪 /no_think” 对应 prompt 约 407 tokens，小于默认 `n_ubatch=512`。

结果正常：

```text
prompt eval time = 2136.59 ms / 407 tokens (190.49 tokens/s)
short_htp_exit=0
```

### 卡住样例：长 Prompt

对话总结题对应 prompt 约 554 tokens，超过默认 `n_ubatch=512`。使用默认 ubatch 时：

```bash
timeout 10s adb shell '... -n 1 --perf'
```

结果：

```text
generate: n_ctx = 2048, n_batch = 2048, n_predict = 1, n_keep = 0
host_exit=124
```

也就是 10 秒内没有完成 `prompt eval`，没有进入正常 perf footer。

### CPU 对照

同一条长 prompt，CPU/no offload 能完成：

```bash
-fa off -ngl 0 -n 1 --perf
```

结果：

```text
prompt eval time = 13787.42 ms / 554 tokens (40.18 tokens/s)
long_cpu_exit=0
```

说明 prompt 本身、chat template、tokenizer、`llama-completion` 主流程都不是根因。

### HTP 规避：放大 ubatch

同一条长 prompt，显式设置：

```bash
--batch-size 2048 --ubatch-size 2048
```

结果正常：

```text
prompt eval time = 2924.68 ms / 554 tokens (189.42 tokens/s)
long_htp_ub2048_exit=0
```

### 禁用 graph reuse 仍会卡

同一条长 prompt，默认 `ubatch=512`，额外设置：

```bash
LLAMA_GRAPH_REUSE_DISABLE=1
```

结果仍然停在：

```text
generate: n_ctx = 2048, n_batch = 2048, n_predict = 1, n_keep = 0
```

没有进入 `prompt eval` footer。说明这个现象不是 `llama_context::process_ubatch()` 复用旧 graph 导致。

### 增加 Hexagon watchdog 后的错误

在 Hexagon host 后端增加 `GGML_HEXAGON_WAIT_TIMEOUT_MS=10000` 后，同一条默认 `ubatch=512` 的长 prompt 不再无限等待，而是在 10 秒左右明确报错：

```text
ggml-hex: HTP0 timed out waiting for 1 pending HTP batch(es) after 10000 ms
```

栈顶位置：

```text
ggml_hexagon_session::flush_pending
ggml_backend_sched_graph_compute_async
llama_context::graph_compute
llama_context::process_ubatch
llama_context::decode
```

同样带 watchdog，但加 `--batch-size 2048 --ubatch-size 2048` 后正常完成：

```text
prompt eval time = 2755.63 ms / 554 tokens (201.04 tokens per second)
long_htp_ub2048_exit=0
```

因此稳定复现边界是：

```text
prompt tokens <= 512: HTP 正常
prompt tokens >  512: 默认 ubatch=512 可能卡住
prompt tokens >  512 + --ubatch-size 2048: HTP 正常
```

## 根因定位

`llama.cpp` 默认：

```text
n_batch  = 2048
n_ubatch = 512
```

长 prompt prefill 会被拆成多个 physical ubatch，例如：

```text
554 tokens => 512 tokens + 42 tokens
```

CPU 路径可以跨 ubatch 正常执行；Hexagon HTP 后端在这个跨 ubatch prefill 场景下卡住。当前命令里设置了 op filter：

```text
GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT|SET_ROWS'
```

注意这里的 `OPFILTER` 语义是“匹配到的 op 不由 Hexagon claim”，不是“只启用这些 op”。所以这条命令实际是在把 `FLASH_ATTN_EXT` / `SET_ROWS` 排除出 Hexagon 后端。结合 `LLAMA_GRAPH_REUSE_DISABLE=1` 后仍然卡住，可以先排除 graph reuse。

把 Hexagon op batching 和 queue depth 都降到 1 后，watchdog 打印出了卡住的单 op：

```text
MUL_MAT 9728:2560 x 9728:42 -> 2560:42
```

也就是第二个 physical ubatch 的 remainder `42 tokens` 触发了 HTP `MUL_MAT` 不返回。继续做两个对照：

```bash
GGML_HEXAGON_OPFILTER="MUL_MAT|FLASH_ATTN_EXT|SET_ROWS"
```

可以跑通，但 prefill 只有约 `47.79 tokens/s`，说明卡点确实在 Hexagon `MUL_MAT`。

```bash
GGML_HEXAGON_USE_HMX=0
```

也可以跑通，但 prefill 只有约 `22.47 tokens/s`，说明问题集中在 HMX matmul 路径，而不是普通 HVX matmul。

代码里 `hmx_matmul_2d_f32()` 原来使用：

```c
const bool use_pipeline = (m > 32);
```

这会让 `m=42` 这种只有两个 HMX row tile 的小 remainder 进入异步 pipeline。把阈值改成至少超过两个 tile 再启用 pipeline：

```c
const bool use_pipeline = (m > 2 * HMX_FP16_TILE_N_ROWS);
```

之后默认 `ubatch=512` 的长 prompt 可以正常完成。

## 修复验证

修复 HMX pipeline 阈值后，原始长 prompt 不再需要 `--ubatch-size 2048`，默认 `ubatch=512` 能完成：

```text
prompt eval time = 2974.84 ms / 554 tokens (186.23 tokens per second)
host_exit=0
```

显式 `--batch-size 2048 --ubatch-size 2048` 的正例仍然正常：

```text
prompt eval time = 2703.62 ms / 554 tokens (204.91 tokens per second)
host_exit=0
```

旧包上仍然可以用 `--batch-size 2048 --ubatch-size 2048` 规避；修复后的包不需要这个规避。

## 最小复现命令

```bash
# 预期卡住或 timeout
timeout 10s adb shell 'cd /data/local/tmp/llama.cpp && \
LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER="FLASH_ATTN_EXT|SET_ROWS" \
./bin/llama-completion \
  -m /data/local/tmp/gguf/Qwen3-4B-Q4_0.gguf \
  -sysf /data/local/tmp/llama_system_prompt.txt \
  -p "请根据给出的对话进行上下文理解，总结对话内容，对话：记者：近期有市民反映，部分小区的垃圾分类执行情况不佳，存在混投现象，您怎么看？社区工作人员：确实，我们在巡查中也发现了这个问题。主要是一些居民对垃圾分类的标准不太清楚，加上有时候垃圾投放点设置不合理，导致大家图方便就混投了。记者：那社区有什么改进措施吗？社区工作人员：我们打算加强宣传，通过举办讲座、发放宣传册等方式，提高居民的垃圾分类意识。同时，也会优化垃圾投放点的设置，让居民投放更方便。记者：这些措施大概什么时候能实施？社区工作人员：宣传活动本周就能启动，投放点的优化工作预计下个月完成。 /no_think" \
  --single-turn --simple-io --no-display-prompt \
  -t 4 --ctx-size 2048 \
  -fa on -ngl 99 --device HTP0 \
  --temp 0 -n 1 --perf'

# 预期正常
timeout 20s adb shell 'cd /data/local/tmp/llama.cpp && \
LD_LIBRARY_PATH=./lib \
ADSP_LIBRARY_PATH=./lib \
GGML_HEXAGON_OPPOLL=1 \
GGML_HEXAGON_OPFILTER="FLASH_ATTN_EXT|SET_ROWS" \
./bin/llama-completion \
  -m /data/local/tmp/gguf/Qwen3-4B-Q4_0.gguf \
  -sysf /data/local/tmp/llama_system_prompt.txt \
  -p "同上 /no_think" \
  --single-turn --simple-io --no-display-prompt \
  -t 4 --ctx-size 2048 --batch-size 2048 --ubatch-size 2048 \
  -fa on -ngl 99 --device HTP0 \
  --temp 0 -n 1 --perf'
```
