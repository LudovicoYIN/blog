---
author: Ludovico
pubDatetime: 2026-06-26T12:00:00Z
title: MNN QNN 离线模式跑通 Qwen3-4B 大模型
featured: false
draft: false
tags:
  - 端侧大模型
  - MNN
description: 完整记录从本地编译 MNN、生成 QNN 离线模型、准备高通 QNN 依赖，到在 QCS8550/8 Gen 2 板端按 MNN 文档跑通 Qwen3-4B 的全过程，以及中间遇到的权限、路径和库问题。
---

这篇记录一次 MNN + QNN 跑 Qwen3-4B 的完整 bring-up。

目标不是写一份抽象教程，而是把这次实际机器上的路径、命令、报错和最终能跑的方式都沉淀下来。以后再换模型或换板子，可以直接对照这里排查。

## 环境

Host：

```text
Project root: /home/luke/code/LLM
MNN root:     /home/luke/code/LLM/MNN
Model:        /home/luke/code/LLM/models/Qwen3-4B-MNN-QNN-Omni-hqq
QNN SDK:      /opt/qcom/aistack/qairt/2.42.0.251225
NDK:          /opt/ndk
```

Device：

```text
Board:      kalama
SOC:        QCS8550 / 8 Gen 2
SOC ID:     43
DSP arch:   v73
Work dir:   /data/local/tmp/MNN
```

板端检查：

```bash
adb shell 'getprop ro.soc.model; getprop ro.board.platform; getprop ro.vendor.qti.soc_id; getprop ro.product.board; getprop ro.hardware'
```

实际输出：

```text
QCS8550
kalama

kalama
qcom
```

SOC ID / DSP 架构来自 MNN 文档里的表：

| 芯片    | SOC ID | HEXAGON ARCH |
| ------- | -----: | -----------: |
| 8 Gen 1 |     36 |           69 |
| 8 Gen 2 |     43 |           73 |
| 8 Gen 3 |     57 |           75 |
| 8 Elite |     69 |           79 |

所以这块板子用：

```text
--soc_id 43
--dsp_arch v73
HEXAGON_ARCH=73
```

## 关键认知：QNN LLM 的 backend_type 应该是 CPU

一开始很容易误判：`config_qnn.json` 里写着 `"backend_type": "cpu"`，`llm_bench` 表格里也显示 backend 是 `CPU`，看起来像没跑 QNN。

但 MNN 文档里明确说，QNN 离线模式是把 QNN 离线产物封装在 Plugin 算子里，而 Plugin 算子注册在 CPU 后端。因此离线 QNN runtime 的外层 backend type 就应该是 CPU。

也就是说：

```json
{
  "llm_model": "qnn/llm.mnn",
  "backend_type": "cpu"
}
```

这是 MNN QNN LLM 离线模式的正常配置，不是错误。

判断 QNN 是否被调用，不能只看 `backend CPU`，而要看：

```text
1. 模型是否使用 config_qnn.json
2. llm_model 是否指向 qnn/llm.mnn
3. QNN so 是否成功加载
4. 性能是否和纯 CPU 基线有明显差异
```

本次模型的 `config_qnn.json`：

```json
{
  "llm_model": "qnn/llm.mnn",
  "llm_weight": "llm.mnn.weight",
  "backend_type": "cpu",
  "thread_num": 4,
  "precision": "low",
  "memory": "low",
  "sampler_type": "mixed",
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "min_p": 0.05,
  "tfs_z": 1.0,
  "typical": 0.95,
  "repetition_penalty": 1.0,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "penalty_window": 0,
  "n_gram": 8,
  "ngram_factor": 1.0,
  "tokenizer_file": "tokenizer.mtok",
  "chunk_limits": [128, 1]
}
```

## Host 侧编译 MNN

QNN LLM 分两套构建：

```text
Host build:
  用于生成 QNN 离线模型，MNN_QNN_CONVERT_MODE=ON

Android build:
  用于板端运行，MNN_QNN_CONVERT_MODE=OFF，MNN_WITH_PLUGIN=ON
```

### Host 侧编译

```bash
cd /home/luke/code/LLM/MNN
mkdir -p build
cd build

export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_BUILD_CONVERTER=ON \
  -DMNN_BUILD_TOOLS=ON \
  -DMNN_QNN=ON \
  -DMNN_QNN_CONVERT_MODE=ON \
  -DMNN_WITH_PLUGIN=OFF \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON

make -j"$(nproc)"
```

需要确认这些 host 工具存在：

```text
/home/luke/code/LLM/MNN/build/MNNConvert
/home/luke/code/LLM/MNN/build/generateLlmIO
/home/luke/code/LLM/MNN/build/compilefornpu
/home/luke/code/LLM/MNN/build/MNN2QNNModel
```

### Android 侧编译

```bash
cd /home/luke/code/LLM/MNN/project/android
mkdir -p build
cd build

export ANDROID_NDK=/opt/ndk
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225

../build_64.sh \
  -DMNN_BUILD_LLM=ON \
  -DMNN_BUILD_LLM_OMNI=ON \
  -DMNN_QNN=ON \
  -DMNN_QNN_CONVERT_MODE=OFF \
  -DMNN_WITH_PLUGIN=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON
```

这份仓库的 Android 产物实际在：

```text
/home/luke/code/LLM/MNN/project/android/build/llm_demo
/home/luke/code/LLM/MNN/project/android/build/llm_bench
/home/luke/code/LLM/MNN/project/android/build/libMNN.so
```

这次实际使用 `project/android/build`。

## 生成 QNN 离线模型

当前模型目录已经是 QNN 离线模型：

```text
/home/luke/code/LLM/models/Qwen3-4B-MNN-QNN-Omni-hqq
├── config.json
├── config_qnn.json
├── llm.mnn
├── llm.mnn.weight
├── tokenizer.mtok
└── qnn/
    ├── llm.mnn
    ├── graph0.bin
    ├── graph1.bin
    └── ...
```

如果需要重新生成，用：

```bash
export QNN_SDK_ROOT=/opt/qcom/aistack/qairt/2.42.0.251225
export PATH=/home/luke/miniforge3/envs/mnn/bin:$PATH

cd /home/luke/code/LLM/MNN/transformers/llm/export

/home/luke/miniforge3/envs/mnn/bin/python npu/generate_llm_qnn.py \
  --model /home/luke/code/LLM/models/Qwen3-4B-MNN-QNN-Omni-hqq \
  --soc_id 43 \
  --dsp_arch v73 \
  --mnn_path /home/luke/code/LLM/MNN/build \
  --cache_path /home/luke/code/LLM/models/Qwen3-4B-MNN-QNN-Omni-hqq-qnn-cache \
  --chunk_size 128
```

脚本完成后会在模型目录生成：

```text
qnn/
config_qnn.json
```

MNN 文档还特别说明：

```text
构建完成后，model 目录下的 llm.mnn 及 llm.mnn.weight 不再需要，可以删除以减少文件总大小。
```

不过本次为了排查和可回退，没有删。

## 推哪些东西到板子

QNN 跑法只需要推三类东西：

```text
1. MNN Android 可执行文件：
   - llm_demo
   - llm_bench

2. 运行时 so：
   - libMNN.so
   - libQnnHtp.so
   - libQnnHtpV73Stub.so
   - libQnnHtpV73Skel.so
   - libQnnSystem.so

3. QNN 离线模型目录：
   - Qwen3-4B-MNN-QNN-Omni-hqq/
   - 里面必须有 config_qnn.json 和 qnn/llm.mnn、qnn/graph*.bin
```

MNN 文档建议的执行方式是：exe、MNN so、QNN so 放在同一个工作目录，`cd` 到目录里运行。

所以最终采用：

```text
/data/local/tmp/MNN
├── llm_demo
├── llm_bench
├── libMNN.so
├── libQnnHtp.so
├── libQnnHtpV73Stub.so
├── libQnnHtpV73Skel.so
├── libQnnSystem.so
└── model/
    └── Qwen3-4B-MNN-QNN-Omni-hqq/
```

完整推送命令如下。

```bash
adb root
adb shell 'setenforce 0; rm -rf /data/local/tmp/MNN; mkdir -p /data/local/tmp/MNN/model'

# MNN Android runtime
adb push /home/luke/code/LLM/MNN/project/android/build/llm_demo /data/local/tmp/MNN/
adb push /home/luke/code/LLM/MNN/project/android/build/llm_bench /data/local/tmp/MNN/
adb push /home/luke/code/LLM/MNN/project/android/build/libMNN.so /data/local/tmp/MNN/

# QNN runtime
adb push /opt/qcom/aistack/qairt/2.42.0.251225/lib/aarch64-android/libQnnHtp.so /data/local/tmp/MNN/
adb push /opt/qcom/aistack/qairt/2.42.0.251225/lib/aarch64-android/libQnnHtpV73Stub.so /data/local/tmp/MNN/
adb push /opt/qcom/aistack/qairt/2.42.0.251225/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so /data/local/tmp/MNN/
adb push /opt/qcom/aistack/qairt/2.42.0.251225/lib/aarch64-android/libQnnSystem.so /data/local/tmp/MNN/

# QNN offline model
adb push /home/luke/code/LLM/models/Qwen3-4B-MNN-QNN-Omni-hqq /data/local/tmp/MNN/model/
```

推送后修执行权限：

```bash
adb shell 'chmod 755 /data/local/tmp/MNN/llm_demo /data/local/tmp/MNN/llm_bench /data/local/tmp/MNN/*.so'
```

如果不修，会遇到：

```text
/system/bin/sh: ./llm_bench: can't execute: Permission denied
```

## 怎么跑 QNN

板端跑 QNN 前先确认 root 和 permissive：

```bash
adb root
adb shell 'setenforce 0'
adb shell 'id; getenforce'
```

期望看到 `uid=0(root)` 和 `Permissive`。

对话：

```bash
adb shell 'cd /data/local/tmp/MNN && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=. && ./llm_demo model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json'
```

测速：

```bash
adb shell 'cd /data/local/tmp/MNN && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=. && ./llm_bench -m model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json -a cpu -t 4 -p 32 -n 32 -rep 1'
```

## 最终验证结果

`llm_demo` 能完成 tuning 并回复：

```text
config path is model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json
CPU Group: [ 0  1  2 ], 307200 - 2016000
CPU Group: [ 3  4  5  6 ], 499200 - 2803200
CPU Group: [ 7 ], 595200 - 3187200
(last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK))=0x 4100d4e0 in _getInfoArm, 1244
The device supports: i8sdot:1, fp16:1, i8mm: 1, sve2: 0, sme2: 0
main, 276, cost time: 54.313999 ms
Prepare for tuning opt Begin
stateNumber=0 in init, 1270
...
Prepare for tuning opt End
main, 284, cost time: 10061.630859 ms

User:
A:
Okay, the user sent "_hi" which is a common way to start a conversation in many languages. They might just be greeting me or starting to talk. Let me respond warmly to encourage them to share more details abouttheir query. Keepit simple and welcoming. Maybe say hello and ask how I can help.


Hi! How can I help youtoday？ 😊
```

bench 结果：

```text
| model                          |  modelSize | backend    | threads | precision  | test          |        t/s |
| ------------------------------ | ---------: | ---------- | ----: | ---------- | ------------- | ---------: |
| Qwen3-4B-MNN-QNN-Omni-hqq      |   1.99 GiB | CPU        |     4 | Low        | pp32          | 28.21 ± 0.00 |
| Qwen3-4B-MNN-QNN-Omni-hqq      |   1.99 GiB | CPU        |     4 | Low        | tg32          | 8.66 ± 0.00 |
```

这里的 `backend CPU` 是 MNN 离线 QNN Plugin 的外层 backend，不应该直接解读为没有调用 QNN。

## 中间遇到的完整报错

### 1. ADSP_LIBRARY_PATH 写法错误

错误写法是把多个路径直接用分号放在 shell 命令里，但没有整体加引号：

```bash
export ADSP_LIBRARY_PATH=/data/local/tmp/qwen35/lib-qnn;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp
```

shell 会把分号后面的路径当成命令执行，于是出现：

```text
/system/bin/sh: /vendor/dsp/cdsp: inaccessible or not found
/system/bin/sh: /vendor/lib/rfsa/adsp: can't execute: Is a directory
/system/bin/sh: /system/lib/rfsa/adsp: inaccessible or not found
```

正确写法：

```bash
export ADSP_LIBRARY_PATH="/data/local/tmp/qwen35/lib-qnn;/vendor/lib/rfsa/adsp;/vendor/dsp/cdsp;/system/lib/rfsa/adsp"
```

按 MNN 文档同目录运行时可以更简单：

```bash
cd /data/local/tmp/MNN
export ADSP_LIBRARY_PATH=.
```

### 2. 忘记 chmod 导致不能执行

```text
/system/bin/sh: /data/local/tmp/qwen35/bin-qnn/llm_demo: can't execute: Permission denied
```

或：

```text
/system/bin/sh: ./llm_bench: can't execute: Permission denied
```

修复：

```bash
adb shell 'chmod 755 /data/local/tmp/MNN/llm_demo /data/local/tmp/MNN/llm_bench /data/local/tmp/MNN/*.so'
```

### 3. 传错 config 路径或命令行被折行污染

有一次交互 shell 中命令被折行污染，路径尾部被拼坏：

```text
Unable to open llm_config file: /data/local/tmp/qwen35/model-qwen3-4b-qnn/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.jsollm_config.json
CPU Group: [ 0  1  2 ], 307200 - 2016000
CPU Group: [ 3  4  5  6 ], 499200 - 2803200
CPU Group: [ 7 ], 595200 - 3187200
(last_midr & (CPUINFO_ARM_MIDR_IMPLEMENTER_MASK | CPUINFO_ARM_MIDR_PART_MASK))=0x 4100d4e0 in _getInfoArm, 1244
The device supports: i8sdot:1, fp16:1, i8mm: 1, sve2: 0, sme2: 0
[Error]: tokenizer file not found: /data/local/tmp/qwen35/model-qwen3-4b-qnn/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.jsotokenizer.txt
LLM init error
main, 276, cost time: 0.085000 ms
```

这个不是模型问题，是命令字符串错了。正确路径必须是：

```text
model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json
```

### 4. 没有 root / permissive 时不能稳定跑 QNN

当前板子上需要先切到 root 并关闭 SELinux enforcing：

```bash
adb root
adb shell 'setenforce 0'
```

确认：

```text
uid=0(root) gid=0(root) ... context=u:r:su:s0
Permissive
```

## 清理板端旧文件

如果要从头来，保留已有 `gguf` 和 `llama.cpp`，清掉别的测试碎片：

```bash
adb shell 'cd /data/local/tmp && find . -maxdepth 1 ! -name . ! -name gguf ! -name llama.cpp -exec rm -rf {} +'
```

本次最终文档式目录是 `/data/local/tmp/MNN`，可以直接重建：

```bash
adb shell 'rm -rf /data/local/tmp/MNN && mkdir -p /data/local/tmp/MNN/model'
```

## 当前结论

这次已经确认：

```text
1. MNN Android QNN runtime 可以本地编译出来。
2. Qwen3-4B-MNN-QNN-Omni-hqq 的 qnn/llm.mnn 和 graph*.bin 可以推到 QCS8550 板端。
3. 按 MNN 文档布局，exe / libMNN.so / QNN so 同目录，模型放 model/ 下，可以跑通 llm_demo。
4. config_qnn.json 的 backend_type=cpu 是离线 QNN Plugin 模式的设计，不是错误。
5. 板端运行前需要 root + setenforce 0。
```

目前可复用的最短命令：

```bash
adb root
adb shell 'setenforce 0'
adb shell 'cd /data/local/tmp/MNN && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=. && ./llm_demo model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json'
```

性能复测：

```bash
adb shell 'cd /data/local/tmp/MNN && export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=. && ./llm_bench -m model/Qwen3-4B-MNN-QNN-Omni-hqq/config_qnn.json -a cpu -t 4 -p 32 -n 32 -rep 1'
```
