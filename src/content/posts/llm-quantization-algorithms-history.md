---
author: Ludovico
pubDatetime: 2026-06-21T12:00:00Z
title: 量化算法时间线
featured: true
draft: false
tags:
  - 论文精读
  - 量化
  - LLM
  - GPTQ
  - AWQ
  - SmoothQuant
  - OmniQuant
  - QuIP
  - QLoRA
  - 系统优化
description: 从 2017 年 CNN 量化到 2026 年 1-bit LLM，梳理量化算法发展脉络，覆盖 PTQ、QAT、GPTQ、AWQ、SmoothQuant、OmniQuant、QuIP、QLoRA 等核心算法的原理与实践对比。
---

## 量化是什么

用低精度数值（INT8/INT4/FP8）替代高精度数值（FP32/FP16），减少模型体积和计算量。

**核心公式**：

$$W_{quantized} = \text{round}\left(\frac{W_{fp32}}{s} + z\right)$$

其中 $s$ 是缩放因子（scale），$z$ 是零点（zero point）。

**收益直接看数字**：

| 精度 | 体积压缩 | 计算加速 | 典型精度损失 |
|------|---------|---------|-------------|
| FP32 → FP16 | 2x | 1.5-2x | <1% |
| FP16 → INT8 | 2x | 2-4x | 1-3% |
| FP16 → INT4 | 4x | 4-8x | 3-8% |
| FP16 → INT2 | 8x | 8-16x | 8-15% |
| FP16 → INT1 | 16x | 16-32x | 15-30% |

## 发展时间线

```mermaid
timeline
    title 量化算法发展时间线
    2017 : Learnable Quantization<br>(Courbariaux et al.)
          : 首次提出可学习量化
    2018 : QAT (Quantization-Aware Training)<br>(Jacob et al.)
          : 训练时模拟量化误差
          : HOOVER (Hinton et al.)
          : 1-bit 权重量化
    2019 : 8-bit Matrix Multiplication<br>(Micikevicius et al.)
          : 混合精度训练
    2020 : Mixed Precision Training<br>(Korthikanti et al.)
          : FP16/BF16 成为标准
    2022 : LLM.int8() (Dettmers et al.)
          : 大模型 INT8 推理
          : GPTQ (Frantar et al.)
          : 逐层 Hessian 优化量化
          : SmoothQuant (Liu et al.)
          : 激活值量化突破
    2023 : AWQ (Lin et al.)
          : 激活感知权重量化
          : OmniQuant (Shao et al.)
          : 全方位校准量化
          : QuIP (Lee et al.)
          : 最优传输理论量化
    2024 : QLoRA (Dettmers et al.)
          : 量化 LoRA 微调
          : QuIP# (Tseng et al.)
          : Hadamard 非相干处理
          : FP8 训练 (Microsoft)
          : 硬件原生 FP8 支持
    2025 : 1-bit LLM (Wang et al.)
          : 极端低位量化
          : 稀疏量化统一框架
```

## 核心算法

### 1. PTQ（Post-Training Quantization）

训练完成后直接量化，不需要重新训练。

**流程**：

```mermaid
graph LR
    A[FP32 模型] --> B[采集校准数据]
    B --> C[计算 scale/zero_point]
    C --> D[权重/激活值量化]
    D --> E[INT8/INT4 模型]
```

**关键参数**：

- **校准数据量**：通常 32-512 条样本足够
- **量化粒度**：per-tensor（整个张量一个 scale）vs per-channel（每通道独立 scale）
- **量化方法**：Round-to-Nearest (RTN)、MinMax、Percentile

**per-channel vs per-tensor**：

```
per-tensor: 整个权重矩阵共用一个 scale
[1.2, 0.8, 1.5, 0.3]  →  scale = 1.5

per-channel: 每行/每列独立 scale
[1.2, 0.8, 1.5, 0.3]  →  [s1, s2, s3, s4]
```

Per-channel 精度更高，但增加存储和计算复杂度。

### 2. QAT（Quantization-Aware Training）

训练过程中模拟量化误差，让模型学会适应低精度。

**流程**：

```mermaid
graph TD
    A[FP32 权重] --> B[前向传播]
    B --> C[插入 Fake Quantize 节点]
    C --> D[STE 直通估计器]
    D --> E[反向传播]
    E --> A
```

**STE（Straight-Through Estimator）**：

```python
# 前向：量化
def fake_quantize(x, scale, zero_point):
    return round(x / scale + zero_point) - zero_point

# 反向：直接传梯度（忽略量化操作）
def fake_quantize_backward(grad):
    return grad  # 直通
```

**QAT vs PTQ 对比**：

| 维度 | PTQ | QAT |
|------|-----|-----|
| 训练时间 | 0 | 需要重新训练 |
| 精度损失 | 较大 | 较小 |
| 校准数据 | 需要 | 不需要 |
| 适用场景 | 快速部署 | 精度敏感 |

### 3. LLM.int8()

2022 年提出，首个针对大模型的 INT8 量化方案。

**核心洞察**：

- 大模型中存在 **outlier**（异常值），少量权重/激活值远大于均值
- 直接量化会导致这些 outlier 被截断，精度大幅下降

**解决方案**：

```python
# Hessian-based outlier detection
hessian = compute_hessian(model, calibration_data)
outlier_mask = hessian > threshold

# 对 outlier 保持 FP16，其余量化为 INT8
W_quantized = torch.where(outlier_mask, W_fp16, quantize(W_fp16))
```

**效果**：

- LLaMA-7B INT8 推理，精度损失 <1%
- 显存占用从 14GB 降至 7GB

### 4. GPTQ（GPT Quantization）

2022 年提出，逐层优化的量化算法。

**核心思想**：

利用 Hessian 矩阵的二阶信息，指导权重量化时的误差最小化。

**算法流程**：

```mermaid
graph TD
    A[加载 FP16 模型] --> B[逐层处理]
    B --> C[计算当前层 Hessian]
    C --> D[按 Hessian 对角线排序权重]
    D --> E[逐列量化，利用二阶信息修正]
    E --> F[固定该层，处理下一层]
    F --> G[完成所有层量化]
```

**数学原理**：

$$\min_{W_q} \|W - W_q\|_H^2 = (W - W_q)^T H (W - W_q)$$

其中 $H$ 是 Hessian 矩阵，近似为对角矩阵。

**关键参数**：

- **per-group size**：通常 128，将权重分组量化
- **校准数据**：128 条文本样本
- **量化精度**：支持 INT4、INT3、INT2

**GPTQ vs PTQ 对比**：

| 维度 | PTQ | GPTQ |
|------|-----|------|
| 量化精度 | INT8 | INT4/INT3/INT2 |
| 校准数据 | 需要 | 需要 |
| 计算复杂度 | 低 | 高（需计算 Hessian） |
| 精度损失 | 3-5% | 1-2% |
| 适用场景 | 快速部署 | 极致压缩 |

### 5. AWQ（Activation-aware Weight Quantization）

2023 年提出，MLSys 2024 Best Paper。

**核心洞察**：

- 权重不是同等重要的
- 某些激活通道对模型输出影响更大
- 保护这些"重要权重"可以显著减少量化误差

**算法流程**：

```mermaid
graph TD
    A[加载 FP16 模型] --> B[采集激活值分布]
    B --> C[识别重要激活通道]
    C --> D[计算缩放因子 s]
    D --> E[缩放权重 W = W * s]
    E --> F[对缩放后权重量化]
    F --> G[反向缩放输出]
```

**数学原理**：

$$s_i = \sqrt{\frac{\text{mean}(|A_i|)}{\text{max}(|A_i|)}}$$

其中 $A_i$ 是第 $i$ 个激活通道的值。

**关键特点**：

- **无需重新训练**：纯 PTQ 方法
- **激活感知**：根据激活值分布调整量化策略
- **硬件友好**：支持 INT4、INT3

**AWQ vs GPTQ 对比**：

| 维度 | GPTQ | AWQ |
|------|------|-----|
| 量化方法 | Hessian 优化 | 激活感知缩放 |
| 计算复杂度 | 高 | 低 |
| 校准数据 | 需要 | 需要 |
| 精度损失 | 1-2% | 0.5-1.5% |
| 适用场景 | 极致压缩 | 平衡精度与速度 |

### 6. SmoothQuant

2022 年提出，解决激活值量化难题。

**核心问题**：

激活值中存在 outlier，直接量化会导致精度大幅下降。

**解决方案**：

```python
# 将激活值的 outlier 转移到权重中
def smooth_quant(W, A, alpha=0.5):
    # 计算激活值的 scale
    A_scale = torch.max(torch.abs(A), dim=-1, keepdim=True)[0]
    
    # 平滑因子
    smooth_factor = A_scale ** (-alpha)
    
    # 转移到权重
    W_smooth = W * smooth_factor
    
    # 激活值除以平滑因子
    A_smooth = A / smooth_factor
    
    return W_smooth, A_smooth
```

**效果**：

- 激活值量化误差降低 50%
- 支持 INT8 激活值 + INT8 权重
- 推理速度提升 2-4x

### 7. OmniQuant

2023 年提出，全方位校准量化算法。

**核心洞察**：

- 现有量化方法只关注权重或激活值之一
- OmniQuant 同时优化权重和激活值的量化参数
- 通过可微优化框架统一处理

**算法流程**：

```mermaid
graph TD
    A[加载 FP16 模型] --> B[块级校准]
    B --> C[学习量化参数]
    C --> D[优化 scale/zero_point]
    D --> E[量化权重和激活值]
    E --> F[INT4 模型]
```

**数学原理**：

$$\min_{s, z} \mathbb{E}_{x} \|\mathcal{Q}(W, s, z) x - W x\|_2^2$$

其中 $\mathcal{Q}$ 是量化函数，$s$ 和 $z$ 是可学习的量化参数。

**关键特点**：

- **全方位校准**：同时优化权重和激活值
- **可微优化**：通过梯度下降学习量化参数
- **支持 W4A16/W4A4**：权重 4-bit，激活值 16-bit 或 4-bit

**OmniQuant vs AWQ 对比**：

| 维度 | AWQ | OmniQuant |
|------|-----|-----------|
| 优化目标 | 权重 | 权重 + 激活值 |
| 优化方法 | 启发式缩放 | 可微优化 |
| 量化精度 | INT4 | INT4/INT2 |
| 计算复杂度 | 低 | 中 |
| 适用场景 | 快速部署 | 精度敏感 |

### 8. QuIP / QuIP#

**QuIP（2023）**：最优传输理论量化。

**核心思想**：

将量化问题建模为最优传输问题，理论上保证量化误差最小。

**数学原理**：

$$\min_{W_q} \mathbb{W}_2(W, W_q)$$

其中 $\mathbb{W}_2$ 是 Wasserstein-2 距离。

**关键特点**：

- **理论保证**：最优传输理论保证误差最小
- **支持 INT2**：可将权重压缩到 2-bit
- **Incoherence Processing**：非相干处理减少 outlier

**QuIP#（2024）**：Hadamard 非相干处理。

**核心改进**：

- 使用 Hadamard 矩阵进行非相干处理
- 进一步减少量化误差
- 支持 INT1（1-bit）量化

**QuIP vs QuIP# 对比**：

| 维度 | QuIP | QuIP# |
|------|------|-------|
| 非相干处理 | 随机旋转 | Hadamard 矩阵 |
| 量化精度 | INT2 | INT1/INT2 |
| 计算复杂度 | 中 | 低 |
| 精度损失 | 2-3% | 1-2% |

### 9. QLoRA

2023 年提出，量化 LoRA 微调。

**核心问题**：

- 全量微调需要大量显存
- LoRA 减少了参数量，但基座模型仍需 FP16

**解决方案**：

```mermaid
graph TD
    A[FP16 预训练模型] --> B[4-bit NormalFloat 量化]
    B --> C[双量化存储 scale]
    C --> D[冻结量化权重]
    D --> E[插入 LoRA 适配器]
    E --> F[仅训练 LoRA 参数]
    F --> G[反量化 + LoRA 输出]
```

**关键创新**：

1. **4-bit NormalFloat**：非均匀量化，适应权重分布
2. **双量化**：scale 也量化，进一步减少显存
3. **梯度反量化**：反向传播时反量化权重

**效果**：

- LLaMA-65B 微调显存从 130GB 降至 24GB
- 单张 48GB GPU 可微调 65B 模型
- 精度损失 <1%

**QLoRA vs LoRA 对比**：

| 维度 | LoRA | QLoRA |
|------|------|-------|
| 基座模型精度 | FP16 | 4-bit |
| 微调显存 | 高 | 低 |
| 精度损失 | 0 | <1% |
| 适用场景 | 资源充足 | 资源受限 |

### 10. FP8 量化

2024 年提出，硬件原生 FP8 支持。

**核心问题**：

- INT8 量化需要反量化，增加计算开销
- FP8 可以直接在硬件上计算

**FP8 格式**：

- **E4M3**：4-bit 指数 + 3-bit 尾数，范围更大
- **E5M2**：5-bit 指数 + 2-bit 尾数，精度更高

**效果**：

- H100 GPU 原生支持 FP8 matmul
- 推理速度提升 2x
- 精度损失 <1%

## 量化算法对比总览

| 算法 | 年份 | 量化精度 | 优化方法 | 计算复杂度 | 适用场景 |
|------|------|---------|---------|-----------|---------|
| PTQ | 2017 | INT8 | MinMax | 低 | 快速部署 |
| QAT | 2018 | INT8 | 训练模拟 | 高 | 精度敏感 |
| LLM.int8() | 2022 | INT8 | Outlier 保护 | 中 | 大模型 INT8 |
| GPTQ | 2022 | INT4 | Hessian 优化 | 高 | 极致压缩 |
| SmoothQuant | 2022 | INT8 | 激活值平滑 | 中 | 激活值量化 |
| AWQ | 2023 | INT4 | 激活感知缩放 | 低 | 平衡精度与速度 |
| OmniQuant | 2023 | INT4 | 可微优化 | 中 | 精度敏感 |
| QuIP | 2023 | INT2 | 最优传输 | 中 | 极端压缩 |
| QuIP# | 2024 | INT1 | Hadamard 非相干 | 低 | 极致压缩 |
| QLoRA | 2023 | 4-bit | LoRA 微调 | 中 | 量化微调 |
| FP8 | 2024 | FP8 | 硬件原生 | 低 | GPU 推理 |

## 实践建议

**选型决策树**：

```mermaid
graph TD
    A[需要量化吗？] -->|是| B[精度要求？]
    B -->|高| C[QAT 或 OmniQuant]
    B -->|中| D[GPTQ 或 AWQ]
    B -->|低| E[PTQ INT8]
    C --> F[显存足够？]
    F -->|是| G[INT4]
    F -->|否| H[INT8]
    D --> I[需要激活值量化？]
    I -->|是| J[SmoothQuant]
    I -->|否| K[GPTQ INT4]
    E --> L[INT8 PTQ]
```

**具体场景**：

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 服务器部署 | FP8/INT8 | 硬件支持好，精度损失小 |
| 边缘设备 | INT4 GPTQ/AWQ | 压缩比高，推理速度快 |
| 移动端 | INT4 AWQ | 激活感知，精度高 |
| 极致压缩 | INT2 QuIP/INT1 QuIP# | 压缩比 8-16x |
| 快速部署 | PTQ INT8 | 无需重新训练 |
| 量化微调 | QLoRA | 单卡微调大模型 |

## 总结

量化已经从"精度损失可接受的压缩技巧"发展为"大模型部署的标配"。

**关键趋势**：

1. **低位量化成为常态**：INT4 已成为边缘设备标准
2. **激活值量化突破**：SmoothQuant 解决了激活值 outlier 问题
3. **全方位校准**：OmniQuant 同时优化权重和激活值
4. **量化 + 微调**：QLoRA 让量化模型也能高效微调
5. **硬件原生支持**：FP8、INT4 成为 GPU/NPU 标准格式
6. **极端压缩**：QuIP/QuIP# 支持 INT2/INT1 量化

**实践原则**：

- 精度优先 → QAT/OmniQuant
- 速度优先 → PTQ INT8
- 压缩优先 → GPTQ/QuIP INT4
- 平衡 → AWQ INT4
- 微调 → QLoRA

量化不是银弹，但选对算法能让模型部署成本降低一个数量级。
