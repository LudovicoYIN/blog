---
author: Ludovico
pubDatetime: 2026-06-22T10:45:00Z
title: "[论文精读] MODE: 面向 MoE 多模态大模型的模态分解专家混合精度量化"
featured: false
draft: false
tags:
  - 论文精读
  - 量化
  - MoE
  - 多模态
  - PTQ
description: MODE 发现 MoE-MLLM 量化中存在的跨模态和模态内两个频率偏差，通过模态分解频率 + 量化敏感性 + ILP 分配策略，在 W3A16 下将性能损失控制在 2.9% 以内。
---

## 论文信息

| 字段 | 内容 |
|------|------|
| **标题** | MODE: Modality-Decomposed Expert-Level Mixed-Precision Quantization for MoE Multimodal LLMs |
| **作者** | Yuanteng Chen, Peisong Wang*, Zhilei Liu, Nanxin Zeng, Yuantian Shao, Shiqiang Lang, Tao Liu, Chuangyi Li, Qinghao Hu, Gang Li, Jing Liu, Jian Cheng* |
| **机构** | 中国科学院自动化研究所、中国科学院大学人工智能学院、中关村实验室 |
| **arXiv** | [arxiv.org/abs/2606.17118](https://arxiv.org/abs/2606.17118) |
| **PDF** | [arxiv.org/pdf/2606.17118](https://arxiv.org/pdf/2606.17118) |
| **类别** | cs.LG, cs.AI |
| **发表日期** | 2026-06-15 |

**一句话总结：** MODE 揭示了 MoE-MLLM 量化中因视觉 token 数量碾压文本 token 而导致的**跨模态频率偏差**，以及视觉 token 内部冗余导致的**模态内频率偏差**，通过模态分解频率估计 + 量化敏感性评估 + ILP 整数规划分配策略，在 W3A16 下将平均性能损失控制在 **2.9%** 以内。

## 研究背景

### 2.1 问题场景

MoE 多模态大模型（MoE-MLLM）如 Qwen3-VL-30B-A3B-Instruct 拥有数十亿参数，BF16 下权重内存高达 **62 GB**，远超消费级 GPU（RTX 4090 24GB）的承载能力。Post-Training Quantization（PTQ）无需重新训练即可大幅压缩模型，但现有方法要么面向 Dense MLLM，要么面向 MoE LLM，**缺乏专门针对 MoE-MLLM 的量化方案**。

### 2.2 现有方法的不足

| 方法类别 | 代表工作 | 问题 |
|---------|---------|------|
| Dense MLLM 量化 | MBQ, Speed-Q | 将模型视为整体，忽略 MoE 稀疏激活和专家贡献不均 |
| MoE LLM 量化 | MC-MoE, MoEQuant | 利用 MoE 结构特性，但忽略模态差异 |
| MoE-MLLM 量化 | VEQ-ME | 初步探索，性能提升有限 |

### 2.3 核心洞察：两层频率偏差

MODE 团队发现，当将 MoE LLM 的"专家激活频率即重要性"范式迁移到 MoE-MLLM 时，存在两个被忽视的偏差：

**偏差一：跨模态偏差（Cross-Modal Bias）**
- 单张图像编码为数百至数千个视觉 token，而文本 prompt 通常较短
- 视觉 token 数量 >> 文本 token 数量
- 全局频率统计完全被视觉侧路由模式主导
- **文本关键专家被系统性低估**

**偏差二：模态内偏差（Intra-Vision Bias）**
- 视觉 token 中仅少数携带核心语义，大部分冗余
- 冗余 token 同样参与路由投票，扭曲频率统计
- **冗余偏好专家被高估，语义关键专家被低估**

## 核心方法

### 3.1 整体 Pipeline

![MODE 整体流程图](/blog/papers/2606.17118/img_in_image_box_145_146_1043_490.jpg)

> Figure 5: Pipeline of MODE. 模态维度的频率和敏感性逐专家分析，通过 ILP 在预算下分配专家级比特宽度。

MODE 包含三个核心步骤：

```mermaid
flowchart TD
    A[校准集] --> B[Step 1: 模态分解频率]
    A --> C[Step 2: 模态量化敏感性]
    B --> D[文本频率 f_t(e)]
    B --> E[视觉频率 f_v(e)]
    C --> F[文本敏感性 δ_T(e,b)]
    C --> G[视觉敏感性 δ_V(e,b)]
    D & E & F & G --> H[Step 3: ILP 整数规划]
    H --> I[专家比特宽度分配]
    I --> J[GPTQ 校准]
    J --> K[量化模型]
```

### 3.2 Step 1: 模态分解专家选择频率

**关键设计：** 分别收集文本 token 和**关键视觉 token**（每层注意力最高的前 20%）的专家选择频率，层内归一化。

$$\tilde{f}_t(e), \quad \tilde{f}_v(e)$$

**关键视觉 token 筛选（借鉴 SparseVLM）：**

$$\tilde{\boldsymbol{p}} = \frac{1}{L_t}\sum_{i=1}^{L_t}\boldsymbol{P}_i$$

其中 $\boldsymbol{P}$ 是文本到视觉的注意力子块，$\tilde{p}_j$ 越大表示视觉 token $j$ 对当前文本查询越相关。

**创新点：** 不同于 SparseVLM 在浅层一次性裁剪，MODE **逐层自适应**筛选关键视觉 token，避免跨层偏差。

### 3.3 Step 2: 模态量化敏感性

频率反映专家被选中的频率，但不等同于量化损失。MODE 进一步评估每个专家在每种模态下的量化敏感性：

$$\delta_M(e,b) = \frac{1}{|\mathcal{D}|}\sum_{x\in\mathcal{D}} D_{KL}\left(p(x) \mathrel{\Big\|} p_M^{(e\to b)}(x)\right)$$

其中 $M \in \{T, V\}$ 表示模态，$p(x)$ 为全精度输出 logit，$p_M^{(e\to b)}(x)$ 为仅将专家 $e$ 量化到 $b$ bit 且仅应用于模态 $M$ token 时的输出 logit。

**隔离策略：** 量化权重仅应用于对应模态的 token，确保每个敏感性分数忠实反映量化该专家对特定模态的影响。

### 3.4 Step 3: ILP 整数规划比特宽度分配

引入二元指示变量 $x_{e,b} \in \{0,1\}$，表示专家 $e$ 是否被量化为 $b$ bit：

$$\min \sum_e \sum_{b\in\mathcal{B}} \left[\bar{f}_t(e)\cdot\delta_T(e,b) + \bar{f}_v(e)\cdot\delta_V(e,b)\right] x_{e,b}$$

约束条件：

$$\sum_e \sum_{b\in\mathcal{B}} b\cdot x_{e,b} = n\cdot k, \quad \sum_{b\in\mathcal{B}} x_{e,b} = 1$$

其中 $n$ 为专家数量，$k$ 为目标平均比特数。

**设计精妙之处：** 每一项 $\bar{f}_M(e) \cdot \delta_M(e,b)$ 同时耦合了专家 $e$ 对模态 $M$ 的**重要性**（频率）和**脆弱性**（敏感性），最小化总成本自然驱动求解器将高精度分配给"既频繁被选又高度敏感"的专家。

### 3.5 概念验证实验

论文通过控制实验验证了两个维度偏差的存在性。固定注意力层为 4-bit，所有 MoE 专家初始为 2-bit，逐步将专家提升至 3-bit，比较四种排序策略：

| 策略 | 频率信号 | 含义 |
|------|---------|------|
| (a) Global | $\bar{f}_{total}$ | 全局频率（基线） |
| (b) Text+Vision | $\frac{1}{2}\bar{f}_{text} + \frac{1}{2}\bar{f}_{vision}$ | 模态平衡 |
| (c) Text+Key | $\frac{1}{2}\bar{f}_{text} + \frac{1}{2}\bar{f}_{key}$ | 仅关键视觉 token |
| (d) Text+Redundant | $\frac{1}{2}\bar{f}_{text} + \frac{1}{2}\bar{f}_{red}$ | 仅冗余视觉 token |

**结果：** (b) >> (a)，确认模态平衡的重要性；(c) > (b) > (d)，确认关键视觉 token 比冗余 token 提供更忠实的专家重要性信号。

![概念验证结果](/blog/papers/2606.17118/img_in_image_box_146_152_1044_479.jpg)

> Figure 4: 概念验证。专家从 2-bit 逐步提升至 3-bit，四种排序策略在 ChartQA 和 MMBench 上的表现验证了两层频率偏差。

![性能对比](/blog/papers/2606.17118/img_in_image_box_620_451_1044_725.jpg)

> Figure 1: Qwen3-VL-30B-A3B-Instruct 在 3-bit 权重量化（W3A16）下的性能对比。MODE 平均损失仅 2.84%。

## 实验结果

### 4.1 主实验结果

在三个 MoE-MLLM 家族（Qwen3-VL-30B-A3B-Instruct, Kimi-VL-A3B-Instruct, InternVL3.5-30B-A3B）上，覆盖 10 个多模态基准，评估 3-bit 和 2-bit 设置。

**Qwen3-VL-30B-A3B-Instruct (W3A16) 关键数据：**

| 方法 | MMMU | MMBench | MMStar | ChartQA | TextVQA | POPE | **Avg.** |
|------|------|---------|--------|---------|---------|------|---------|
| BF16 基线 | 52.56 | 86.86 | 60.05 | 85.20 | 83.37 | 89.92 | **72.65** |
| GPTQ-W3 | 44.00 | 76.63 | 50.82 | 78.15 | 75.20 | 84.10 | 64.43 |
| MBQ-W3 | 46.20 | 78.90 | 52.10 | 79.80 | 76.50 | 85.30 | 65.68 |
| MC-MoE-W3 | 48.10 | 80.20 | 54.30 | 81.20 | 78.10 | 86.50 | 67.28 |
| VEQ-ME-W3 | 49.80 | 81.50 | 55.80 | 82.40 | 79.30 | 87.20 | 68.69 |
| **MODE-W3** | **50.33** | **82.80** | **55.68** | **83.00** | **80.65** | **87.80** | **69.89** |

**核心指标：**
- W3A16 下 Qwen3-VL-30B 平均损失仅 **2.84%**
- W3A16 下 Kimi-VL-A3B 平均损失仅 **2.08%**
- W2A16 极端设置下超越最强基线 **4%+**

### 4.2 与 QuaRot 旋转量化的兼容性

| 方法 | Bits | MMMU | VizWiz | TextVQA |
|------|------|------|--------|---------|
| Baseline (BF16) | 16 | 52.56 | 71.64 | 83.37 |
| MODE | 3 | 50.33 | 68.82 | 80.65 |
| **MODE + QuaRot** | **3** | **50.81** | **69.17** | **81.50** |
| MODE | 2 | 44.63 | 63.39 | 75.45 |
| **MODE + QuaRot** | **2** | **46.49** | **66.02** | **77.10** |

QuaRot 在 MODE 基础上进一步提升，2-bit 下平均提升约 **1.5%**，证实两者正交可组合。

### 4.3 校准集鲁棒性

| 校准集 | MMMU | VizWiz | InfoVQA | MMStar | ChartQA | **Avg.** |
|--------|------|--------|---------|--------|---------|---------|
| ShareGPT4V | 50.33 | 68.82 | 76.62 | 55.68 | 83.00 | **66.89** |
| Flickr30k | 49.61 | 68.27 | 77.04 | 54.75 | 82.54 | **66.44** |
| LLaVA-Next | 49.80 | 68.50 | 76.80 | 55.10 | 82.80 | **66.60** |

切换校准集引入的波动 < 0.5%，方法对校准数据选择不敏感。

### 4.4 部署效率

| 模型 | 总内存 (GB) | 激活内存 (GB) | 平均精度 (%) |
|------|------------|-------------|-------------|
| Qwen3-VL-30B (BF16) | 62.14 | 7.78 | 72.65 |
| LLaVA-OneVision-7B (BF16) | 14.14 | 3.52 | 70.01 |
| **MODE 量化 (W3A16)** | **4.40** | **2.20** | **68.01** |
| Qwen3-VL-2B (BF16) | 16.06 | 16.06 | 62.61 |

**关键结论：** 30B MoE-MLLM 经 MODE 量化后总内存从 62 GB 降至 **14 GB**，可轻松部署在单张 RTX 4090（24 GB）上，精度仅损失约 5%。

## 个人评价

### 5.1 创新点

1. **问题发现精准：** 首次明确指出 MoE-MLLM 量化中存在的两层频率偏差（跨模态 + 模态内），这是之前所有 MoE 量化方法共同忽视的盲区。

2. **方法设计简洁有效：** 模态分解频率 + 量化敏感性 + ILP 分配，三步走逻辑清晰，没有引入复杂的训练或微调过程，纯 PTQ 方案。

3. **关键视觉 token 筛选的逐层自适应：** 借鉴 SparseVLM 但做了关键改进——逐层筛选而非浅层一次性裁剪，避免了跨层注意力分布变化带来的偏差。

4. **ILP 公式的精妙耦合：** 频率 × 敏感性同时衡量重要性和脆弱性，在预算约束下自然优化分配。

### 5.2 局限性

1. **实验规模有限：** 仅在 30B 级别模型上验证，未覆盖 Qwen3-VL-235B-A22B 等更大规模模型。

2. **视觉模块未量化：** 视觉编码器仅占 <3% 内存，但方法本身未涵盖视觉模块量化设计。

3. **推理速度未优化：** 使用 BitBLAS 存储混合精度权重，但端到端推理速度仍落后于 vLLM 等优化框架，因为 vLLM 尚未原生支持混合精度 MoE。

4. **校准集依赖：** 虽然鲁棒性实验显示对校准集不敏感，但方法仍需要校准数据收集频率和敏感性信号。

### 5.3 对后续研究的影响

- 为 MoE-MLLM 量化确立了一个新的研究范式：**模态感知**是必须的
- ILP 分配框架可直接扩展到更多模态（音频、视频等）
- 与 QuaRot 的正交性表明混合精度 + 旋转量化是未来方向
- 关键视觉 token 的逐层筛选策略可迁移到 MoE-MLLM 的推理加速领域

## 相关论文

| 论文 | 方向 | 简要介绍 |
|------|------|---------|
| **MC-MoE** (Huang et al., 2025) | MoE 混合精度量化 | 利用专家激活频率作为重要性指标，为 MoE LLM 分配混合精度，但未考虑模态差异 |
| **VEQ-ME** (Qin et al., 2026) | MoE-MLLM 量化 | 首次针对 MoE-MLLM 的 PTQ 方法，基于 token-expert affinity 的模态感知 Hessian 目标 |
| **MoEQuant** (Hu et al., 2025) | MoE 量化 | 通过专家平衡校准策略解决 MoE 的激活不平衡问题 |
| **MBQ** (Li et al., 2024) | MLLM 量化 | 认识到视觉和语言 token 的量化敏感性差异，引入梯度敏感性度量平衡跨模态重建质量 |
| **QuaRot** (Ashkboos et al., 2024) | 旋转量化 | 通过 Hadamard 旋转抑制权重和激活中的离群值，实现 4-bit 推理，与 MODE 正交可组合 |

---

> 整理者：Nancy | 数据源：arXiv (2606.17118) + PaddleOCR-VL-1.6 | 更新时间：2026-06-22
