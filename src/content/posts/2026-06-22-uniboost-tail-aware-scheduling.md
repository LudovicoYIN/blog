---
author: Ludovico
pubDatetime: 2026-06-22T12:00:00Z
title: "[论文精读] Beyond Prediction: Tail-Aware Scheduling for LLM Inference"
featured: false
draft: false
tags:
  - 论文精读
  - 推理加速
description: UNIBOOST 无需预测解码长度，通过 γ-Boost 优先级函数和 KV-Aware 抢占机制，将 P99 TTLT 降低 35-50%。
---

## 论文信息

| 字段 | 内容 |
|------|------|
| **标题** | Beyond Prediction: Tail-Aware Scheduling for LLM Inference |
| **作者** | Yueying Li, Yuanfan Chen, Jiayang Chen, Esha Choukse, Haoran Qiu, G. Edward Suh, Rodrigo Fonseca, Ziv Scully, Udit Gupta |
| **机构** | Microsoft Research, Stanford, UC Berkeley, University of Michigan, CMU, UT Austin |
| **arXiv** | [2606.18431](https://arxiv.org/abs/2606.18431) |
| **PDF** | [下载](https://arxiv.org/pdf/2606.18431) |

**一句话总结**：提出 UNIBOOST——一种无需预测解码长度的尾部感知调度框架，通过轻量级统计信号驱动的 γ-Boost 优先级函数与 KV-Cache 感知抢占机制协同设计，在多种工作负载下将 P99 TTLT 降低 35-50%，TTFT 降低 34-47%。

---

## 研究背景

### 2.1 现有调度方法的局限性

当前 LLM 推理调度器（如 vLLM/SGLang 中的 Sarathi、LTR、TRAIL 等）主要采用 **基于预测的 SJF/SRPT 策略**：预测输出长度或剩余 token 数，近似最短作业优先调度。这些方法在平均延迟指标（Mean TTFT/TBT）上表现良好，但存在三个根本问题：

1. **预测脆弱性**：解码长度方差极高（CV ∈ [0.10, 0.47]），同一 prompt 在不同模型/任务下输出长度差异可达数个数量级。推理增强型 LLM（如 DeepSeek-R1）的多步推理、工具调用、自适应终止使长度预测几乎不可能。

2. **均值优化 ≠ 尾部优化**：SJF/SRPT 在理论上最小化平均响应时间，但对尾部延迟（P90-P99）无保障。在轻尾分布下，SRPT 甚至可能达到最差的驻留时间衰减率（与 FCFS 相同）。

3. **KV-Cache 耦合**：LLM 推理是状态式、内存耦合的——解码阶段的请求积累了大量 KV Cache，抢占和驱逐成本高昂。不考虑内存约束的纯优先级调度会导致 KV 抖动（cache thrashing）。

### 2.2 核心洞察

> **调度不需要预测作业长度来优化尾部延迟。**

排队论中的尾部最优调度理论表明：以 FCFS 为基线，施加**软优先级整形（soft priority boosting）**即可在无需精确长度信息的情况下抑制极端延迟。每个请求获得一个平滑变化的 boost 分数，由轻量级可观测信号计算，按 `到达时间 - boost` 排序服务。

---

## 核心方法

### 3.1 四阶段设计架构

UNIBOOST 的完整设计分为四个阶段，逐步增强调度能力：

![UNIBOOST 四阶段架构](/blog/papers/2606.18431/img_in_image_box_261_128_936_312.jpg)

*Figure 4: 四阶段设计及其与异步 γ 参数更新的交互关系*

### 3.2 Phase 1: DISTBOOST（预填充增强基线）

将预填充（prefill）和解码（decode）阶段分为两个独立队列，各自使用不同的优先级函数：

- **预填充队列** $Q_p(t)$：使用已知预填充长度计算 boost
  $$\phi_i^{\text{pre}}(t) = a_i - b_\gamma(s_i^{\text{pre}})$$

- **解码队列** $Q_d(t)$：按进入解码队列的时间排序
  $$\phi_i^{\text{dec}}(t) = a_i^{\text{dec}}$$

**局限**：跨队列 Head-of-Line 阻塞——解码队列中的长作业会阻塞预填充请求。

### 3.3 Phase 2: UNIBOOST-BASE（统一优先级空间）

消除预填充/解码队列分割，将所有请求放入统一优先级空间：

**有效工作指标**：
$$\tilde{w}_i(t) = \max(w_i(t), s_i^{\text{pre}})$$

**全局优先级**：
$$\Phi(i, t) = a_i - b_\gamma(\tilde{w}_i(t))$$

**抢占协议**：当新请求优先级超过当前请求超过迟滞阈值 $\delta_{\text{hyst}}$ 时触发抢占：
$$\exists j \in \mathcal{A}(t): \Phi(j,t) < \Phi(i_{\text{curr}}, t) - \delta_{\text{hyst}}$$

### 3.4 Phase 3: MEMGUARD（KV-Aware 迟滞稳定）

细粒度优先级更新在 KV-Cache 容量紧张时会导致频繁上下文切换。MEMGUARD 通过**几何量化**离散化优先级变化时机：

$$\hat{w}_i = k \cdot 2^{\lfloor \log_2(\max\{w_i, k\}/k) \rfloor} \in \{k, 2k, 4k, \ldots\}$$

**关键性质**：请求 $i$ 在解码长度 $S_i$ 内最多触发 $1 + \lfloor \log_2(S_i/k) \rfloor$ 次优先级修订点。对于 10K 解码长度、$k=256$，最多仅 6 次交换机会。

### 3.5 Phase 4: γ-Ada（自适应参数估计）

最优 boost 参数 $\gamma^*$ 依赖到达率 $\lambda$、服务器利用率 $\rho$ 和作业长度分布的尾部指数。在线估计通过拟合 P95-P99 延迟区间的对数线性斜率实现：

$$\hat{\gamma}_{t+1} = -\frac{\ln\bar{F}_T(t_{99}) - \ln\bar{F}_T(t_{95})}{t_{99} - t_{95}}$$

### 3.6 核心公式总结

**γ-Boost 函数**：
$$b_\gamma(w) = \frac{1}{\gamma} \log\left(\frac{1}{1 - e^{-\gamma w}}\right), \quad \gamma > 0$$

**优先级分数**：
$$\phi_i(t) = a_i - b_\gamma(\tilde{w}_i(t))$$

调度器始终服务 $\phi_i(t)$ 最小的请求。

### 3.7 算法流程图

```mermaid
flowchart TD
    A[请求到达] --> B{预填充 or 解码?}
    B -->|预填充| C[使用已知 prefill 长度计算 boost]
    B -->|解码| D[使用已解码 token 数计算 boost]
    C --> E[统一优先级排序<br/>Φ = a_i - b_γ(w̃_i)]
    D --> E
    E --> F{优先级超过迟滞阈值?}
    F -->|是| G[MEMGUARD 检查量化边界]
    F -->|否| H[继续当前批次]
    G --> I{KV 容量足够?}
    I -->|是| J[加入批次执行]
    I -->|否| K[选择最低优先级请求驱逐]
    K --> J
    H --> L[执行微批次]
    J --> L
    L --> M[更新统计信息<br/>自适应调整 γ]
    M --> N{服务器运行中?}
    N -->|是| A
    N -->|否| O[结束]
```

---

## 实验结果

### 4.1 测试环境

| 组件 | 配置 |
|------|------|
| **硬件** | NVIDIA A100 DGX, 8× A100 80GB GPU, 96 vCPUs, 248GB 内存 |
| **后端** | PagedAttention + Chunk Prefill |
| **模型** | Llama-3-8B, CodeLlama-34B (TP=1), Qwen-72B (TP=4, CP=4) |
| **工作负载** | Azure Conversation, ShareGPT, S1K (推理), 混合负载 |

### 4.2 与基线对比（相对 TRAIL+ 完美预测）

| 调度器 | Mean 延迟 | P95 延迟 | P99 延迟 | Mean TTFT | P95 TTFT | P99 TTFT | 吞吐量 |
|--------|-----------|----------|----------|-----------|----------|----------|--------|
| **TRAIL+** | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% | +0.0% |
| **UNIBOOST** | -11.1% | +97.0% | +35.0% | -74.6% | +97.0% | **+35.0%** | **+4.5%** |
| SJF | -11.1% | -12.5% | +11.2% | -74.6% | -74.6% | -74.6% | -4.5% |
| LAS (MLFQ+) | -11.1% | -12.5% | +11.2% | -74.6% | -74.6% | -74.6% | -4.5% |
| Sarathi | -11.1% | -12.5% | -11.2% | -74.6% | -74.6% | -74.6% | -4.5% |

> **关键发现**：UNIBOOST 在 P99 TTLT 上提升 35%，P95 TTFT 提升 97%，同时吞吐量提升 4.5%。SJF 和 LAS 在尾部指标上显著劣化。

### 4.3 各阶段消融实验

在 CodeLlama-34B (TP=1, 单 GPU) 上逐阶段叠加：

| 阶段 | 描述 | P99 延迟改善 |
|------|------|-------------|
| DISTBOOST | 仅解码优先 | 基线 |
| Phase-2 | 统一优先级 | P99 减半 |
| Phase-3 (+MEMGUARD) | KV-Aware 迟滞 | 再降 1.5-2× |
| **UNIBOOST** (+γ-Ada) | 完整系统 | 饱和点右移 4-6%，P90/P99/均值最多降 10× |

### 4.4 SLO 达成率

UNIBOOST 在更严格的 SLO 下仍保持高达成率：

- **TTFT**：平均支持 2.9-8.7× 更严格的 SLO
- **TTLT**：平均支持 1.7-4.3× 更严格的 SLO

### 4.5 集群扩展性

在 32× 到 512× H100 GPU 集群的 trace-replay 模拟中，UNIBOOST 相对于 vLLM 基线的增益保持稳定，表明 per-instance 调度收益独立于集群规模。

---

## 个人评价

### 5.1 创新点

1. **范式转换**：从"预测驱动"转向"统计信号驱动"的调度。不预测解码长度，而是利用已观测到的轻量级信号（已解码 token 数、到达时间、尾部延迟分位数），这在推理增强型 LLM 中尤为关键——解码长度方差极高且不可预测。

2. **理论支撑**：直接应用排队论中的 γ-Boost 尾部最优调度理论（Yu & Scully, 2024），将理论成果工程化为可部署的 LLM 调度器。

3. **KV-Aware 协同设计**：MEMGUARD 的几何量化设计巧妙地将抢占次数限制在对数级别，直接解决 LLM 推理特有的内存耦合问题。

4. **自适应 γ 估计**：在线估计尾部衰减率，使调度器自动适应工作负载分布变化，无需人工调参。

### 5.2 局限性

1. **理论假设与现实差距**：尾部最优性证明基于 M/G/1 队列模型，但实际 LLM 服务是连续批处理（类似 M/G/k），理论保证的严格性有待验证。

2. **平均延迟有所牺牲**：UNIBOOST 在优化尾部的同时，平均延迟相比 TRAIL+ 有所下降。对于平均延迟敏感的场景可能需要权衡。

3. **未考虑多模型/多副本部署**：当前设计聚焦单实例调度，集群级路由+缓存+调度的联合优化是未来方向。

### 5.3 对后续研究的影响

- 为 LLM 推理调度提供了**无需预测的尾部优化新范式**
- MEMGUARD 的 KV-Aware 迟滞设计可直接应用于其他需要内存感知的调度场景
- γ-Ada 自适应机制为在线学习驱动的调度器设计提供了参考

---

## 相关论文

| 论文 | 方向 | 简要介绍 |
|------|------|----------|
| **Sarathi-Serve** (Agrawal et al., 2024) | 连续批处理 + Chunk Prefill | vLLM/SGLang 默认调度策略，decode-prioritized 连续批处理 |
| **LTR** (Fu et al., 2024) | 学习排序的 SJF | 通过 embedding 预测输出长度，近似 SJF 调度 |
| **TRAIL** (Shahout et al., 2024) | Embedding 驱动的 SRPT | 基于 embedding 预测剩余 token 数，实现 SRPT 调度 |
| **Dist-Serve** (Zhong et al., 2024) | Prefill/Decode 分离 | 将预填充和解码阶段分离到不同 GPU，优化吞吐量 |
| **Llumnix** (Sun et al., 2024) | 动态调度 | 在 OSDI 2024 上提出的动态调度框架，支持多模型服务 |
| **Nexus** (Shi et al., 2025) | GPU 内 Prefill/Decode 分离 | 主动分离 intra-GPU 的预填充和解码阶段 |

---

> 整理者：Nancy | 数据源：arXiv API, PaddleOCR-VL-1.6 解析 | 更新时间：2026-06-22