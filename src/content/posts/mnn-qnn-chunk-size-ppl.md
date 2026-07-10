---
author: Ludovico
pubDatetime: 2026-07-10T00:00:00+08:00
title: MNN QNN 里为什么 chunk size 会影响 PPL
featured: false
draft: false
tags:
  - 端侧大模型
  - MNN
description: 记录一次 Qwen3-0.6B 在 MNN QNN 上排查 PPL 异常的过程，解释为什么 chunk size 不只是输入切块参数，而是同时决定 prefill 图 shape、padding 路径和 ppl_eval 的上下文窗口，因此会真实影响困惑度。
---

这篇文章记录：为什么在 MNN QNN 里，改了 `chunk size` 之后，`ppl_eval` 的结果会明显变化。

一开始这个现象看起来不太合理。

直觉上，`chunk size` 像是一个“跑模型时怎么切 prompt”的工程参数，不应该改变模型本身。既然权重没变，PPL 为什么会变，甚至从不合理的高值回落到正常值？

这次排查下来，结论很明确：

```text
在 MNN QNN 这条链路里，chunk size 不是单纯的输入切块参数。
它同时决定了 prefill 图的离线 shape、padding 行为、最后一个 logits 的选择方式，
以及 ppl_eval 的上下文窗口长度。
所以 chunk size 会真实影响 PPL。
```

## 现象

这次实验对象是：

```text
Model: Qwen3-0.6B
Backend: MNN QNN offline
```

前面一度出现过一个很迷惑的现象：

- CPU baseline PPL 在合理范围
- 但最原始的 QNN baseline PPL 很高（chunk size == 128），看起来量化损失很重
- 但把 QNN 图重新按真实 `256/512` 的 chunk size 生成以后，PPL 明显下降

这次排查里比较关键的几组数字是：

```text
CPU baseline 512: 24.975338
QNN baseline 128: 49.674919
```

这里的含义要说清楚：

- `CPU baseline 24.975338` 对应的是 `512` 上下文窗口
- `QNN baseline 49.674919` 对应的是最原始的 `128` chunk 规格

也正因为这两个 baseline 规格并不一致，才会在最初把问题看得很混乱。后面把 chunk size 统一之后，再比较才有意义。

统一规格之后，实际测到的结果是：

```text
QNN 256:    36.293194
QNN 512:    24.503946
```

这说明前面那个异常高的 `49.674919`，是评测时 `chunk size` 和离线 QNN 图规格没对齐。

## chunk size 到底是什么

在 MNN LLM 里，`chunk_limits` 常常写成这样：

```json
"chunk_limits": [512, 1]
```

它的意思是：

- `512`：prefill 阶段按 512 token 规格处理
- `1`：decode 阶段按 1 token 规格处理

所以这里的 `chunk size`，本质上描述的是：

```text
长 prompt 在 prefill 阶段一次处理多长
```

也就是：

- prefill 的切块长度
- prefill graph 的目标 shape
- ppl_eval 的上下文窗口基准

decode 阶段通常是单 token，自然就是 `1`。

## 为什么它会影响 PPL

原因要拆成三层看。

### 1. `ppl_eval` 本身就依赖 `chunk_limits`

MNN 自带的 `ppl_eval` 不是一次把整篇文本整段送进去，而是按窗口滑动计算 loss。

代码在 [ppl_eval.cpp](MNN/transformers/llm/engine/tools/ppl_eval.cpp:68)：

```cpp
size_t stride = 512;
size_t contextLength = stride + stride / 2;
std::shared_ptr<MNN::Transformer::LlmConfig> lmConfig(new MNN::Transformer::LlmConfig(llmPath));
if (lmConfig->config_.contains("chunk_limits")) {
    contextLength = lmConfig->config_["chunk_limits"][0].get<int>();
    stride = (contextLength / 3) * 2;
}
```

也就是说：

- `contextLength` 直接取 `chunk_limits[0]`
- `stride` 也跟着变
- 后续每个 loss window 的边界都会跟着变

所以哪怕完全不考虑 QNN，单看 `ppl_eval` 的滑窗方式，`chunk size` 也会影响最终的平均 loss。

### 2. QNN 离线图是按 chunk size 编出来的

这次链路里更关键的是，QNN offline model 不是“一个动态图跑所有长度”，而是生成时就把 prefill shape 固化进图里。

这一步是：

```bash
python /home/luke/code/LLM/MNN/transformers/llm/export/npu/generate_llm_qnn.py \
  --model /home/luke/code/LLM/models/Qwen3-0.6B-MNN-QNN-smooth-seqmse-qnn512 \
  --soc_id 43 \
  --dsp_arch v73 \
  --mnn_path /home/luke/code/LLM/MNN/build \
  --chunk_size 512
```

也就是说：

```text
--chunk_size 512
```

不是单纯给运行时的提示，而是在生成 `graph0.bin` 这类 prefill graph 时就已经决定了图的输入规格。

前面我们还专门修过一个问题：`generate_llm_qnn.py` 之前有硬编码 `128`，导致即便你以为自己在做 `512` 版，实际离线图还是按 `128` 的路径生成。

这个修复之后，QNN PPL 才回到了合理区间。

### 3. MNN 在 prefill 时会按有效 block size 做切块和 padding

MNN 的 LLM 路径里，`chunk_limits` 最终会进入 `mValidBlockSize`，并影响 prefill 时怎么 split 输入，以及最后一块怎么 pad。

代码在 [llm.cpp](MNN/transformers/llm/engine/src/llm.cpp:132)：

```cpp
if (mConfig->config_.contains("chunk_limits")) {
    auto size_limit = mConfig->config_["chunk_limits"];
    for (size_t i = 0; i < size_limit.size(); i++) {
        mValidBlockSize.emplace_back(size_limit[i].get<int>());
    }
    std::sort(mValidBlockSize.begin(), mValidBlockSize.end());
    mBlockSize = mValidBlockSize[mValidBlockSize.size()-1];
}
```

真正 forward 时，如果最后一段不满一个整块，会走 padding 路径。代码在 [llm.cpp](MNN/transformers/llm/engine/src/llm.cpp:720) 附近：

```cpp
if (blockRemain < forwardSize) {
    // Pad
    hasPad = true;
    ...
}
...
if (hasPad) {
    auto logitSize = logits[0]->getInfo()->dim[2];
    mGenerateParam->validLogitStart = ((int)addSize - 1) * logitSize;
    mGenerateParam->validLogitSize = logitSize;
}
```

这一段的意思是：

- 末块不足时会 pad 到某个有效 block size
- 然后再通过 `validLogitStart/validLogitSize` 去取真正该用的最后一个 logits

所以当 `chunk size` 变化时，变的不只是“切块长度”，还包括：

- 是否进入 padding
- pad 到多长
- 最终从哪一段 logits 里取最后一个 token 的结果

量化模型对这些边界条件本来就更敏感，因此 PPL 变化是完全可能的。

## 为什么这次 128 和 512 差这么大

这次不是正常的小范围波动，而是前后差距比较明显。原因在于之前存在“配置和图不一致”的问题。

具体来说，曾经有过这种组合：

```text
1. config_qnn.json / ppl_eval 认为自己在用 256 或 512
2. 实际 QNN graph 还是按 128 规格生成的
```

在这种情况下，测到的 PPL 已经不是“同一条推理路径上的 teacher-forcing loss”了，而是：

- 图 shape
- prefill 切块
- padding 路径
- logits 选取位置

都混在一起的结果。

后面把这三处统一之后，PPL 就稳定了：

```text
QNN 生成图使用的 chunk_size
config_qnn.json 里的 chunk_limits[0]
ppl_eval 传入的 max_length / 运行时上下文
```