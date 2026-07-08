---
author: Ludovico
pubDatetime: 2026-06-23T08:30:00Z
title: "llama.cpp Hexagon HTP 长 Prompt 默认 ubatch=512 卡住的定位与修复"
featured: false
draft: false
tags:
  - 端侧大模型
  - llama.cpp
description: 本来只是想在 Hexagon HTP 上跑一条稍长一点的总结题，结果 `llama.cpp` 默认 `ubatch=512` 直接卡在 prefill。最后一路排到 HMX matmul，发现是小 remainder batch 走异步 pipeline 后没回来。
---

## 背景

这个问题不是一开始就很明显。

我是在板端用 `llama.cpp` 的 Hexagon HTP 后端测小远 Qwen3-4B Q4_0 的回复质量。短问题一直都正常，所以最开始我还以为只是某条输入比较倒霉，或者 decode 太慢。但换成一条稍长一点的“对话总结”题后，现象很稳定：程序像卡死了一样，首 token 很久都不出来。

测试命令大概是这样：

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

这里我把 `-n` 压到 1，就是想先把 decode 的影响尽量排掉。如果只生成 1 个 token 还要等 10 秒，那问题大概率不在“生成太慢”，而是在生成前面的那一段，也就是 prompt prefill。

## 稳定复现

### 正常样例：短 Prompt

先看一个正常样例。

短问题“切洋葱时为啥会流泪 /no_think”对应的 prompt 大约是 407 tokens，还没超过默认的 `n_ubatch=512`。

这时结果是正常的：

```text
prompt eval time = 2136.59 ms / 407 tokens (190.49 tokens/s)
short_htp_exit=0
```

### 卡住样例：长 Prompt

再看会卡住的样例。

那条对话总结题对应的 prompt 大约是 554 tokens，刚好超过默认 `n_ubatch=512`。这时候如果什么都不改，直接用默认 ubatch 跑：

```bash
timeout 10s adb shell '... -n 1 --perf'
```

结果：

```text
generate: n_ctx = 2048, n_batch = 2048, n_predict = 1, n_keep = 0
host_exit=124
```

结果就是 10 秒内连 `prompt eval` 都没走完，perf footer 也没出来。换句话说，它不是“慢”，而是根本没顺利跑过去。

### CPU 对照

为了确认不是 prompt 本身有问题，我拿同一条长 prompt 去跑 CPU/no offload：

```bash
-fa off -ngl 0 -n 1 --perf
```

结果：

```text
prompt eval time = 13787.42 ms / 554 tokens (40.18 tokens/s)
long_cpu_exit=0
```

CPU 能跑完，虽然慢很多，但至少能结束。这基本说明 prompt 本身、chat template、tokenizer，甚至 `llama-completion` 的主流程都没什么大问题。卡点还是在 Hexagon 这条链路上。

### HTP 规避：放大 ubatch

接着我试了一个很朴素的规避办法：直接把 ubatch 放大。

```bash
--batch-size 2048 --ubatch-size 2048
```

结果居然就好了：

```text
prompt eval time = 2924.68 ms / 554 tokens (189.42 tokens/s)
long_htp_ub2048_exit=0
```

### 禁用 graph reuse 仍会卡

接下来想确认是不是 graph reuse 这类状态复用问题，所以我在默认 `ubatch=512` 的基础上，又额外关掉了 graph reuse：

```bash
LLAMA_GRAPH_REUSE_DISABLE=1
```

结果仍然停在：

```text
generate: n_ctx = 2048, n_batch = 2048, n_predict = 1, n_keep = 0
```

还是一样，卡在进入 `prompt eval` footer 之前。至少到这一步，可以先把“旧 graph 复用坏了”这个方向放掉。

### 增加 Hexagon watchdog 后的错误

继续往下排时，我给 Hexagon host 后端加了一个 watchdog：`GGML_HEXAGON_WAIT_TIMEOUT_MS=10000`。

这样做的好处很直接：它不再无限等，而是 10 秒左右给我一个明确报错：

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

同样带 watchdog，但如果把 `--batch-size 2048 --ubatch-size 2048` 一起加上，就又能正常结束：

```text
prompt eval time = 2755.63 ms / 554 tokens (201.04 tokens per second)
long_htp_ub2048_exit=0
```

到这里，复现边界就比较清楚了：

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
