---
author: Ludovico
pubDatetime: 2026-06-22T06:00:00Z
title: "[论文精读] Rethinking Shrinkage Bias in LLM FP4 Pretraining: UFP4 统一量化配方"
featured: false
draft: false
tags: [论文精读, 量化]
  - 论文精读
  - 量化
  - FP4训练
  - E2M1
  - E1M2
  - UFP4
description: 蚂蚁集团提出 UFP4 统一 4-bit 训练配方，揭示 E2M1 格式固有 Shrinkage Bias，在 124B MoE 预训练中将 BF16 相对损失误差从 1.73% 降至 1.39%。
---

## 论文信息

| 字段 | 内容 |
|------|------|
| **标题** | Rethinking Shrinkage Bias in LLM FP4 Pretraining: Geometric Origin, Systemic Impact, and UFP4 Recipe |
| **作者** | Qian Zhao, Kunlong Chen, Changxin Tian, Zhonghui Jiang, Haitao Zhang, Chaofan Yu, Peijie Jiang, Mingliang Gong, Jia Liu, Ziqi Liu, Zhiqiang Zhang\*, Jun Zhou |
| **机构** | 蚂蚁集团 Ling Team |
| **日期** | 2026-06-19 |
| **arXiv** | [2606.20381](https://arxiv.org/abs/2606.20381) |
| **代码** | 未公开 |

## 研究背景

FP4 训练被视为 LLM 预训练的下一站。NVIDIA Blackwell/Rubin 和 AMD MI350 系列均已原生支持 FP4 计算路径，理论上可将显存和计算开销再减半。但当前所有 FP4 训练方案几乎全部围绕 **E2M1**（2 指数位 + 1 尾数位）展开——MXFP4、NVFP4、Quartet II、TetraJet-v2 无一例外。

本文核心质疑：**E2M1 真的是 FP4 训练的最佳格式吗？**

作者发现 E2M1 的 **非均匀量化网格** 存在结构性缺陷——其 RTNE（Round-to-Nearest-Even）舍入区间的几何不对称性导致系统性负舍入误差（Shrinkage Bias）。这个偏差在深层网络中呈指数级累积，且 RHT（Random Hadamard Transform）不仅不能缓解，反而会加剧问题。

## 核心方法

### 3.1 几何起源

考虑归一化幅度 $t = |x|/s$，对内部量化级别 $q_i$，其 RTNE 舍入区间为：

$$\mathcal{B}_i = \left(\frac{q_{i-1}+q_i}{2}, \frac{q_i+q_{i+1}}{2}\right)$$

区间左右宽度分别为 $\ell_i = \frac{q_i-q_{i-1}}{2}$ 和 $r_i = \frac{q_{i+1}-q_i}{2}$。

当 $r_i > \ell_i$ 时（即右侧区间更宽），条件期望舍入误差为：

$$\mathbb{E}[\rho_G(t)-t \mid t\in\mathcal{B}_i] = \frac{\ell_i - r_i}{2} < 0$$

这就是 **Shrinkage Bias**——量化值系统性地向零收缩。

**E2M1 的非均匀网格** 在间距转换点存在不对称区间。例如 E2M1 非负幅度 $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ 中，$q_i=2$ 的条件偏差为 **-0.125**，$q_i=4$ 同样存在不对称。

**E1M2 的均匀网格** 满足 $\ell_i = r_i$ 对所有区间成立，从根本上消除这一偏差源。

![Figure 1a: E2M1 vs E1M2/INT4 量化网格对比](/blog/papers/2606.20381/img_in_image_box_932_886_1053_940.jpg)

### 3.2 系统性影响：跨层累积与 RHT 恶化

Shrinkage Bias 不是零均值噪声。在 GEMM $Z = AB^\top$ 中，量化操作符可分解为：

$$\hat{A} = \alpha_A A + R_A, \quad \hat{B} = \alpha_B B + R_B$$

其中 $\alpha_A < 1$ 表示与原始信号对齐的分量衰减。代入得：

$$Z_q = \alpha_A\alpha_B AB^\top + \text{残差噪声}$$

对于 $K$ 个连续量化 GEMM，初始干净信号被累积缩放：

$$\prod_{k=1}^{K}\eta_k = \prod_{k=1}^{K}(1-\delta_k) \approx \exp\left(-\sum_{k=1}^{K}\delta_k\right)$$

**关键洞察：RHT 使问题恶化。** RHT 将异常值能量分散到所有坐标，将张量从"动态范围受限"转变为"局部分辨率受限"。E2M1 的宽动态范围优势不再重要，瓶颈转移到典型幅度的精确表示。RHT 将数据质量推入 E2M1 最不对称的舍入区间，导致 $\Delta$SQNR < 0。而 E1M2 均匀网格安全地将展平分布转化为更高保真度（$\Delta$SQNR > 0）。

![Figure 1b: 124B MoE BF16 相对损失退化](/blog/papers/2606.20381/img_in_image_box_932_886_1053_940.jpg)



### 4.1 设计原则

一旦 RHT 将张量从动态范围受限转为局部分辨率受限，4-bit 网格必须优先保局部幅度而非极端动态范围。

### 4.2 UFP4 配方

| 配置项 | E2M1 基线 | **UFP4 (E1M2)** |
|--------|-----------|------------------|
| 格式 | E2M1 | **E1M2/INT4 均匀网格** |
| 量化块大小 | $1\times16$ | $1\times16$ |
| Scale 层级 | FP32 单层 | FP32 单层 |
| RHT 范围 | 仅 bwd_dw | **fwd_y, bwd_dx, bwd_dw（全路径）** |
| RHT 块大小 | 16 | 16 |
| 随机舍入范围 | dY | dY |

核心差异仅两点：
1. **均匀网格** 消除几何 Shrinkage Bias
2. **全路径 RHT** 覆盖三个 GEMM（FPROP、DGRAD、WGRAD）

![Figure 3: UFP4 配方概览](/blog/papers/2606.20381/img_in_image_box_932_886_1053_940.jpg)

### 4.3 为什么全路径 RHT 可行？

现有 E2M1 方案（如 NVFP4）通常只将 RHT 限制在 bwd_dw 路径，因为 RHT 在非叶子路径（fwd_y、bwd_dx）上会加剧 E2M1 的几何误差。UFP4 的关键洞察是：**问题不在 RHT 本身，而在 E2M1 与 RHT 后张量分布的失配**。均匀网格下，RHT 扩展到全部三条路径反而保持稳定的长期收益。

## 实验结果

### 5.1 单张量量化诊断

对 outlier-heavy 的 `linear_fc2/fwd_x` 张量，RHT 反转了格式排名：

| 指标 | RHT 前 E2M1 | RHT 前 E1M2 | RHT 后 E2M1 | RHT 后 E1M2 |
|------|-------------|-------------|-------------|-------------|
| SQNR (dB) | 21.90 | 19.94 | 20.00 | **23.19** |
| 有效桶比 | — | — | — | **0.97** |

### 5.2 端到端训练损失

| 模型 | E2M1 相对误差 | UFP4 (E1M2) 相对误差 | 改善 |
|------|--------------|---------------------|------|
| Dense 1.5B | 1.2570% | **0.9673%** | ↓ 23.1% |
| MoE 7.9B | 2.3596% | **1.8469%** | ↓ 21.7% |
| MoE 124B | 1.7308% | **1.3863%** | ↓ 19.9% |

### 5.3 缩放定律验证

在 10M-324M MoE 模型族上遵循 Ling scaling-law 协议，E1M2 曲线始终低于 E2M1，且 FP4-to-BF16 差距随计算量增加而缩小。

### 5.4 消融实验（Dense 1.5B, E1M2）

| 配置 | 平均 LM Loss | $\Delta$ Loss |
|------|-------------|---------------|
| 无 RHT + SR | 1.89202 | 0.00000 |
| RHT on bwd_dw | 1.88721 | -0.00481 |
| RHT on fwd_y+bwd_dw | — | -0.00644 |
| 全路径 RHT | — | **-0.01123** |
| 全路径 RHT + SR on dY | — | **-0.01579** |

### 5.5 性能开销

融合 RHT+quantization 内核在 SM90 和 SM100 上分别为 standalone quantization 的 **1.06x** 和 **1.07x**，未融合方案则为 1.62x 和 1.41x。

## 个人评价

### 优点

1. **理论扎实**：从网格几何出发形式化定义 Shrinkage Bias，给出跨层累积的指数近似推导，不是纯经验性发现。
2. **反直觉洞察**：RHT 在 E2M1 下有害，在 E1M2 下有益——同一操作因量化格式不同而效果反转，这个发现对量化社区有广泛启示。
3. **大规模验证**：124B MoE 长程预训练，不是小模型玩具实验。
4. **工程务实**：融合内核开销仅 6-7%，证明方案在硬件层面可行。
5. **对硬件设计的直接影响**：明确建议未来加速器支持 E1M2/INT4 均匀网格作为一等 FP4 训练原语。

### 局限

1. **硬件依赖**：当前 NVIDIA/AMD 硬件原生支持 E2M1，E1M2 需要硬件更新或软件模拟。蚂蚁团队提到华为 Ascend 960（HiFloat4 S1P2）可能是首个原生支持平台。
2. **未测试 E1M2 在推理场景的表现**：E2M1 的宽动态范围在推理时可能有优势，本文未讨论。
3. **MixFP4 对比缺失**：Zou et al. 2026 提出的自适应 E2M1/E1M2 混合方案未纳入对比。
4. **代码未开源**：无法复现 UFP4 配方。

### 行业意义

这篇论文对 FP4 训练生态有潜在颠覆性影响：

- **对 NVIDIA**：Blackwell/Rubin 的 FP4 路径锁定 E2M1，如果 E1M2 确实更优，需要软件模拟层或下一代硬件调整。
- **对华为 Ascend**：HiFloat4 的 S1P2 统一格式与 UFP4 理念高度契合，可能成为首个原生支持平台。
- **对量化研究**：将讨论从"如何更好地用 E2M1"转向"E2M1 是否应该是默认格式"，这是一个范式级别的转变。

**总结**：UFP4 用理论+实验证明了 E2M1 不是 FP4 训练的最优解。在均匀网格上，RHT 可以安全扩展到全部三条 GEMM 路径，带来实质性的训练质量提升。这是一个值得量化社区严肃对待的结果。
