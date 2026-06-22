---
author: Ludovico
pubDatetime: 2026-06-22T12:00:00Z
title: "[论文精读] VIMPO: Value-Implicit Policy Optimization for LLMs"
featured: false
draft: false
tags: [论文精读, 大模型]
  - 论文精读
  - RLVR
  - 大模型
  - 强化学习
  - 推理优化
description: "无需 Critic 的 RLVR 新范式：从 KL 正则最优条件导出隐式价值函数，实现 Token 级信用分配，全面超越 GRPO。"
---

## 论文信息

| 字段 | 内容 |
|------|------|
| **标题** | VIMPO: Value-Implicit Policy Optimization for LLMs |
| **作者** | Zhewei Kang (UC Berkeley), Aosong Feng (Yale), Dawn Song (UC Berkeley), Sergey Levine (UC Berkeley), Xuandong Zhao (UC Berkeley) |
| **arXiv** | [2606.20008](https://arxiv.org/abs/2606.20008) |
| **PDF** | [PDF](https://arxiv.org/pdf/2606.20008) |
| **代码** | [GitHub](https://github.com/backprop07/VIMPO) |

**一句话总结：** VIMPO 从 KL 正则化 RL 的最优条件出发，推导出完全由策略-参考模型对数比值表示的隐式价值函数，在无需训练 Critic 的前提下实现 Token 级信用分配，在数学推理基准上全面超越 GRPO。

---

## 研究背景

### 核心矛盾：简单性 vs. 信用分配粒度

RLVR（Reinforcement Learning with Verifiable Rewards）已成为提升 LLM 推理能力的核心范式。现有方法分两大阵营：

| 阵营 | 代表方法 | 优势 | 劣势 |
|------|----------|------|------|
| **Actor-Critic** | PPO, VAPO | Token 级密集监督 | 需训练 Value 网络，存在 Critic-Policy 共适配不稳定 |
| **Critic-Free** | GRPO, DAPO | 训练简单，无需 Value 网络 | 轨迹级 Advantage 广播到所有 Token，信用分配粗糙 |

**GRPO 的局限：** 对同一 Prompt 采样 G 条回复，计算组内标准化奖励作为 Advantage，然后**广播到该回复的所有 Token**。这意味着关键推理步骤和连接性 Token 获得相同的梯度信号——无法区分"决定性推理"与"常规过渡"。

后续改进（FIPO 的 future-KL 重加权、基于 Attention 的 anchor token 识别）都是事后修补，而非从根本上解决目标函数设计问题。

### VIMPO 的核心洞察

作者认为这个矛盾**不是根本性的**。关键思路：

> 将自回归生成建模为**确定性转移 MDP**，分析 KL 正则化策略优化的最优条件，可得到一个**恒等式**：最优价值函数可以用策略相对于冻结参考模型的对数比值隐式表示。

不需要学习单独的 Critic——价值函数隐含在策略本身中。

---

## 核心方法

### 3.1 确定性 MDP 建模

自回归生成中，给定下一个 Token $a_t$，状态转移是确定性的：

$$P(s_{t+1} \mid s_t, a_t) = 1$$

其中 $s_t = (x, a_0, ..., a_{t-1})$ 是 Token 前缀状态。

### 3.2 核心价值恒等式

KL 正则化最优策略满足局部最优条件：

$$\pi^*(a \mid s_t) = \frac{\pi_{\text{ref}}(a \mid s_t)}{Z(s_t)} \exp\left(\frac{1}{\beta} Q^*(s_t, a)\right)$$

结合确定性 Bellman 方程 $Q^*(s_t, a_t) = r(s_t, a_t) + \gamma V^*(s_{t+1})$，消去 $\ln Z$ 后得到论文的核心恒等式：

$$\boxed{\beta \ln\frac{\pi^*(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} = r(s_t, a_t) + \gamma V^*(s_{t+1}) - V^*(s_t) + \beta \text{KL}^*(s_t)}$$

**这个公式是 VIMPO 的骨架。** 它表明：在最优条件下，每个 Token 的策略-参考对数比值等于 Bellman 残差加上 KL 修正项。

由此直接导出三个关键性质：

1. **闭式价值递推：** 重排为 $\gamma V^*(s_{t+1}) - V^*(s_t) = \beta \ln(\pi^*/\pi_{\text{ref}}) - \beta \text{KL}^* - r$，给定锚定值 $V^*(s_0)$ 即可前向递推，无需学习 Critic。

2. **零均值信号：** $\beta \ln(\pi^*/\pi_{\text{ref}}) - \beta \text{KL}^*$ 在 $a_t \sim \pi^*$ 下的期望为零（因为 $\text{KL}^*$ 恰好是对数比值的期望），这一性质是推导出来的而非强加的，赋予 VIMPO 训练稳定性。

3. **闭式单步 Advantage：** Bellman 残差形式直接给出 $\hat{A}_t^{\text{TD}} = \beta \ln(\pi^*/\pi_{\text{ref}}) - \beta \text{KL}^*$，可直接接入 PPO。

### 3.3 隐式价值训练

**终端锚定条件：** 轨迹末端无未来奖励，$V_\pi(s_T) = 0$。这是一个无参数的监督目标——最优时策略隐含价值必须在每条轨迹末端归零。

训练目标：

$$\mathcal{L}_V(\pi) = \frac{1}{2} V_\pi(s_T)^2$$

对于 RLVR 中常见的**仅末端奖励**设置（$r(s_k, a_k) = 0$ for $k < T-1$），展开后得到操作形式：

$$\mathcal{L}_V(\pi) = \frac{1}{2} \left[ \sum_{k=0}^{T-1} \left( \beta \ln\frac{\pi(a_k \mid s_k)}{\pi_{\text{ref}}(a_k \mid s_k)} - \beta \text{sg}[\text{KL}_\pi(s_k)] \right) - (R_{\text{final}} - \overline{R}_{\text{final}}) \right]^2$$

**直觉：** 损失鼓励轨迹上累积的策略-参考对数比值跟踪居中后的最终奖励，将策略隐含价值函数与 rollout 结果对齐，不引入学习型 Critic。

### 3.4 PPO 集成

闭式 Advantage 估计：

$$\hat{A}_t^{\text{TD}} = \beta \log\frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{ref}}(a_t \mid s_t)} - \beta \text{KL}_{\pi_\theta}(s_t)$$

可进一步按 GAE 形式累积多步：

$$\hat{A}_t^\lambda = \sum_{\ell=0}^{T-t-1} (\gamma\lambda)^\ell \hat{A}_{t+\ell}^{\text{TD}}$$

归一化后接入 PPO Actor 损失。

**最终训练目标：**

$$\mathcal{L}_{\text{VIMPO}}(\pi) = \mathcal{L}_V(\pi) + c_A \mathcal{L}_A(\pi)$$

### 3.5 算法流程

```
Algorithm 1: VIMPO Training Step
─────────────────────────────────────
for each training step do
  1. 采样 prompt x, 生成 G 条 rollout
  2. 计算最终奖励 R⁽ⁱ⁾ 和组均值 R̄
  3. 对每个 token t 计算:
     ρ_t = β · log(π_θ / π_ref)    # 对数比值
     κ_t = β · KL(π_θ || π_ref)    # KL 散度
     A_t = ρ_t - κ_t               # 隐式 Advantage
  4. 价值损失 L_V = ½ Σ [Σ(ρ_t - sg[κ_t]) - (R⁽ⁱ⁾ - R̄)]²
  5. Actor 损失 L_A = PPO-Clip(A_t, π_θ, π_old)
  6. 更新 θ ← θ - η∇(L_V + c_A · L_A)
end for
```

### 3.6 方法架构

![VIMPO 方法总览](/blog/papers/2606.20008/img_in_image_box_188_92_1016_303.jpg)

*Figure 1: VIMPO 架构。给定 prompt $q$，策略生成回复 $o$，评分获得结果奖励 $r$。VIMPO 用此奖励训练策略隐含价值损失，同时策略与冻结参考模型定义 Token 级 TD 信号用于形成 Actor Advantage。无需训练显式 Critic。*

![VIMPO 信用分配特性](/blog/papers/2606.20008/img_in_image_box_569_1205_1062_1345.jpg)

*Figure 2: VIMPO 是无 Critic 方法，但具备 Token 级信用分配能力。*

### 3.7 与 GRPO 的对比

| 维度 | GRPO | VIMPO |
|------|------|-------|
| Critic | 无 | 无（隐式） |
| Advantage 粒度 | 轨迹级（广播到所有 Token） | Token 级（策略隐含 TD 信号） |
| 奖励融入 | 直接通过 Advantage | 通过价值损失目标 |
| Actor 更新 | PPO-Clip | PPO-Clip（使用策略隐含 Advantage） |
| 训练目标 | 单一组相对目标 | 价值损失 + Actor 损失 |
| 噪声奖励鲁棒性 | 较弱 | 较强（奖励与策略更新解耦） |

### 3.8 Mermaid 流程图

```mermaid
graph TD
    A[采样 Prompt] --> B[策略生成 G 条 Rollout]
    B --> C[计算最终奖励 Rⁱ]
    C --> D[组均值标准化 R̄]
    D --> E[对每条轨迹每个 Token]
    E --> F[计算 ρ_t = β·log π/π_ref]
    E --> G[计算 κ_t = β·KL π||π_ref]
    F --> H[隐式 Advantage A_t = ρ_t - κ_t]
    G --> H
    H --> I[归一化 A_t]
    I --> J[PPO Actor 损失 L_A]
    F --> K[累积 ρ_t - sg κ_t]
    D --> L[计算 Rⁱ - R̄]
    K --> M[价值损失 L_V = ½ Σ 累积 - 奖励差²]
    L --> M
    J --> N[联合更新 θ = θ - η∇ L_V + c_A·L_A]
    M --> N
```

---

## 实验结果

### 4.1 实验设置

- **基座模型：** Qwen3-4B-Base
- **训练数据：** Guru Math 子集，54.4K 样本
- **评测基准：** MATH-500, AIME 2024, AIME 2025, OlympiadBench
- **对比方法：** Naive GRPO, Token-level GRPO, VIMPO
- **VIMPO 超参：** $\beta = 5 \times 10^{-4}$, $c_A = 5 \times 10^{-3}$

### 4.2 主实验结果

| 方法 | MATH-500 | AIME24 | AIME25 | OlympiadBench | 平均 |
|------|----------|--------|--------|---------------|------|
| Qwen3-4B-Base | 54.0 | 8.6 | 3.6 | 18.0 | 21.1 |
| Naive GRPO | 80.4 | 20.0 | 14.6 | 31.3 | 36.6 |
| GRPO | 79.6 | 19.3 | 17.6 | 33.2 | 37.4 |
| **VIMPO** | **81.6** | **21.7** | **20.8** | **34.1** | **39.6** |

**关键发现：**
- VIMPO 在所有 4 个基准上均超越 GRPO
- AIME 2025 提升最大（17.6 → 20.8），竞赛级推理受益最明显
- 回复长度曲线显示 VIMPO 的提升并非单纯来自更长的回复——后期回复长度下降而验证精度仍高于 GRPO

### 4.3 训练动态

| 训练步数 | 训练准确率 (GRPO) | 训练准确率 (VIMPO) | AIME24 (GRPO) | AIME24 (VIMPO) |
|----------|-------------------|--------------------|---------------|----------------|
| 0 | 0.08 | 0.08 | 0.09 | 0.09 |
| 100 | 0.17 | 0.18 | 0.12 | 0.13 |
| 200 | 0.22 | 0.24 | 0.16 | 0.20 |
| 400 | 0.25 | 0.28 | 0.19 | 0.22 |
| 500 | 0.26 | 0.29 | 0.20 | 0.21 |

VIMPO 在训练后期展现出更强的学习动态，训练准确率和验证准确率均领先。

### 4.4 噪声奖励鲁棒性测试

25% 奖励翻转条件下，训练 200 步后的表现：

| 指标 | GRPO (噪声) | VIMPO (噪声) |
|------|-------------|-------------|
| 训练准确率 | 0.17 | 0.24 |
| MATH-500 | 0.77 | 0.78 |

**VIMPO 在噪声奖励下保持显著优势。** 原因分析：VIMPO 的外部奖励通过价值损失目标进入，而非直接作为 Actor Advantage。Actor 更新不直接正比于单个奖励标签，降低了对腐败奖励的敏感度。

### 4.5 消融实验

| 设置 | 200 步训练准确率 | VO KL |
|------|-------------------|-------|
| 仅价值损失 ($c_A=0$) | 0.165 | 低 |
| **VIMPO 完整** ($c_A=0.005$) | **最高** | 较高 |
| 高 $\beta$, 高 $c_A$ | 低 | 极低 |

- 仅价值损失可用但学习较慢
- Actor 更新显著加速学习，但以更大的策略漂移为代价
- 过强的 $\beta$ 和 $c_A$ 会过度约束更新，降低 Actor 组件的收益
- **建议方向：** 自适应/退火 $\beta$ 和 $c_A$，早期大值稳定，后期小值允许进一步改进

### 4.6 训练熵分析

| 训练步数 | Naive GRPO | GRPO | VIMPO |
|----------|-----------|------|-------|
| 0 | 0.85 | 1.05 | 0.80 |
| 200 | 0.15 | 0.08 | 0.05 |
| 550 | 0.07 | 0.025 | 0.025 |

两个 GRPO 变体呈现单调熵下降（与 Cui et al. 2025 讨论的熵崩溃一致）。VIMPO 进入低熵区域后出现小幅波动而非单调下降，表明 Token 级隐式 Advantage 允许更局部的策略分布调整。

---

## 个人评价

### 创新点

1. **理论优雅：** 从 KL 正则最优条件直接导出闭式价值表示，不需要启发式构造。核心恒等式（公式 4）是一个真正的理论贡献。
2. **架构简洁：** 无需额外网络，价值函数完全由策略参数化。训练目标只有两个标量超参（$\beta$, $c_A$）。
3. **奖励-策略解耦：** 外部奖励通过价值损失融入，Actor 使用策略内部信号。这一设计天然增强了对噪声奖励的鲁棒性。
4. **Token 级信用分配：** 不靠事后修补（FIPO 的 future-KL、Attention anchor），而是从目标函数本身导出。

### 局限性

1. **固定参考策略和 $\beta$：** 训练后期策略远离初始化时，固定参考可能过度约束。作者建议自适应 $\beta$ 和定期更新参考策略。
2. **精确 KL 计算开销：** 需要策略和参考模型的完整 next-token 分布，对大模型/长回复有非平凡开销。
3. **实验范围有限：** 仅数学推理、4B 模型、单种子。未对比调优的 Actor-Critic 基线（PPO/VAPO）。
4. **未验证到代码/工具使用/机器人等领域。**

### 对后续研究的影响

- **RLVR 方向：** 为 Critic-Free 方法提供了更精细的信用分配理论框架，可能替代 GRPO 成为新的基线。
- **Value 函数建模：** 展示了价值函数可以"隐含"在策略中，而非必须单独学习——这一思路可能推广到其他 RL 设置。
- **噪声奖励场景：** 奖励-策略解耦设计对实际 RLVR（验证器/奖励规则不完美）有直接实用价值。

---

## 相关论文

1. **GRPO** (Shao et al., 2024) — DeepSeekMath 的核心方法，通过组相对优势消除 Critic，是 VIMPO 的主要对比基线。
2. **DAPO** (Yu et al., 2025) — 在 GRPO 基础上加入动态采样、Token 级损失聚合和解耦裁剪，VIMPO 的实验设置参考了 DAPO 的训练配方。
3. **VAPO** (Yue et al., 2025) — Actor-Critic 路线，通过价值预训练、解耦 GAE 实现 Token 级信用分配，VIMPO 试图在无需 Critic 的情况下达到相同效果。
4. **FIPO** (Ma et al., 2026) — 用 future-KL 因子重加权 GRPO 优势，属于事后修补；VIMPO 从目标函数层面解决了同样的问题。
5. **DPO** (Rafailov et al., 2023) — 展示了 KL 正则最优策略可通过对数比值表示，VIMPO 将这一思路推广到在线 RL 场景。

---

> 整理者：Nancy | 数据源：arXiv (2606.20008) + PaddleOCR-VL-1.6 解析 | 更新时间：2026-06-22
