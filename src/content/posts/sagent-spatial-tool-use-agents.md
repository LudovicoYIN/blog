---
author: Ludovico
pubDatetime: 2026-06-22T09:00:00Z
title: "[论文精读] S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence"
featured: false
draft: false
tags:
  - 论文精读
  - Agent
  - 空间智能
  - VLM
  - 多模态
description: NTU 提出的空间工具调用智能体框架，将空间推理重构为时空证据积累过程，零-shot 超越 GPT-5.4，蒸馏出 8B 模型媲美 Gemini 3。
---

## 论文信息

- **标题**: S-Agent: Spatial Tool-Use Elicits Reasoning for Spatial Intelligence
- **作者**: Yalun Dai, Hao Li, Shulin Tian, Runmao Yao, Yuhao Dong, Fangzhou Hong, Zhaoxi Chen, Fangfu Liu, Baoliang Tian, Dingwen Zhang, Tao Wang, Kim-Hui Yap, Ziwei Liu
- **机构**: NTU (南洋理工大学), THU (清华大学), ByteDance, NWPU (西北工业大学)
- **arXiv**: [2606.20515](https://arxiv.org/abs/2606.20515) | [PDF](https://arxiv.org/pdf/2606.20515)
- **代码**: [Ropedia/S-Agent](https://github.com/Ropedia/S-Agent)
- **一句话总结**: 将 VLM 作为语义规划器 + 层次化工具链 + 双记忆系统，把空间推理从"单帧直觉猜测"升级为"时空证据积累"的 Agent 范式。

![论文总览](/blog/papers/2606.20515/figure1_overview.jpg)

> Figure 1: S-Agent 总览。VLM 作为语义规划器，层次化工具提供 2D→3D 证据，双记忆系统维持场景状态与推理历史。

---

## 研究背景与动机

### 问题定义

空间智能（Spatial Intelligence）是 AGI 的关键能力之一——理解物体间的几何关系及其 3D 环境。当前 VLM 的核心缺陷：

1. **语义-几何鸿沟**: VLM 在 2D 视觉-文本语料上训练，缺乏显式 3D 监督，推理依赖语义先验而非几何证据
2. **静态无状态**: 现有工具增强 Agent（VADAR, SpaceTools）处理单帧静态图像，无法维持跨视角/跨时间的对象状态
3. **隐式内部化**: 要求 VLM 在单次前向传播中编码全部空间能力，而非主动获取场景特定证据

### 核心洞察

> 视频空间智能缺少的不是更强的 2D/3D 识别，而是**时空维度的证据积累机制**。

每一帧只是场景的部分且短暂的观测。真正的空间智能需要将这些碎片化观测连接为空间结构化、时间持久的 3D 理解。

---

## 核心方法

### 整体框架

S-Agent 将空间推理公式化为**迭代证据搜索过程**：

![推理流程](/blog/papers/2606.20515/figure2_pipeline.jpg)

> Figure 2: S-Agent 推理流程。VLM 规划器 → 空间工具/专家 → 双记忆更新 → 迭代直至证据充分。

**数学公式化**:

在推理步骤 $t$，S-Agent 维护两个记忆状态：

- **Scene Memory** $S_t$: 存储 grounding 实体及其累积空间属性
- **Agent Memory** $\mathcal{H}_t$: 记录历史工具调用、观测和推理决策

VLM 规划器 $\pi_\theta$ 映射问题 $q$、观测 $\mathcal{F}$、当前记忆 $(S_t, \mathcal{H}_t)$ 到证据请求 $r_t$：

$$r_t = \pi_\theta(q, \mathcal{F}, S_t, \mathcal{H}_t)$$

空间工具执行 $r_t$ 返回观测 $o_t$，更新双记忆：

$$(S_{t+1}, \mathcal{H}_{t+1}) = \text{Update}(S_t, \mathcal{H}_t, r_t, o_t)$$

### 三层空间证据体系

S-Agent 的核心创新在于**层次化空间证据获取**，将原始 2D 观测转化为显式场景特定空间知识：

| 层级 | 功能 | 工具示例 | 输出 |
|------|------|---------|------|
| **Level 1** | 2D 视觉证据获取 | GroundingDINO 检测, VLM grounding, 深度估计 | 边界框、置信度、深度图 |
| **Level 2** | 2D→3D 几何提升 | Depth-Anything-3 度量深度 | 3D 坐标、相机位姿、俯视图 |
| **Level 3** | 空间知识聚合 | 5 个专业专家 | 结构化空间知识 |

**Level 3 五大专家**:

1. **Metric Measurement Expert**: 几何 grounding 测量专家，计算相机-物体距离、物体-物体距离、物理尺寸
2. **Counting Expert**: 检测 grounding 聚合专家，支持单物体计数和条件感知多帧计数
3. **Visual Orientation Expert**: 外观 grounding 方向专家，判断物体朝向
4. **Relative Position Expert**: 3D 关系专家，判断左右/前后/方位
5. **Object-Centric View Expert**: 视角感知专家，处理围绕同一目标的不同视角

### 双记忆系统

**Scene Memory**（场景记忆）:

- 存储 grounding 实体（文本别名、支撑帧、局部化视觉证据、累积几何属性）
- 存储派生空间事实（空间关系、测量值及推导证据）
- 跨帧/视角合并同一实体，避免重复证据和不稳定身份
- 更新操作: $S_{t+1} = \text{Merge}(S_t, e_t)$

**Agent Memory**（智能体记忆）:

- 存储规划器中间思考、工具调用、返回观测、失败信息、中间结论
- 记录"尝试过什么"、"什么不确定"、"下一步需要什么证据"
- 更新操作: $\mathcal{H}_{t+1} = \text{Append}(\mathcal{H}_t, c_t)$

### 训练时蒸馏（S-300K）

从 SenseNova-SI-800K 构建训练数据：

1. **数据生成**: GPT-5.4 作为教师规划器，生成完整推理轨迹
2. **质量过滤**: 仅保留执行成功且答案正确的轨迹（过滤率 ~48.4% 被拒绝）
3. **轨迹分解**: 单条教师轨迹分解为三种监督信号

![数据质量分布](/blog/papers/2606.20515/figure3a_quality_filtering.jpg)
![工具调用分布](/blog/papers/2606.20515/figure3b_tool_distribution.jpg)

> Figure 3: S-300K 数据组成与工具调用统计。100K 原始轨迹 → 51,596 质量过滤 → 292,391 SFT 样本。

**三种监督格式**:

| 格式 | 数量 | 用途 |
|------|------|------|
| Final-answer trajectories | 51,596 | 端到端空间推理 |
| Turn-level trajectories | 154,590 | 迭代工具使用决策 |
| Nontrivial tool/expert trajectories | 86,205 | 空间工具使用策略 |

在 Qwen3-VL-8B-Instruct 上 SFT → **S-Agent-8B**（8×B200, lr=5e-5, 1 epoch, seq_len=8192）

---

## 实验结果

### 零-shot 性能

**MMSI-Bench**（多图像空间智能基准）:

| 模型 | C-C | O-O | R-R | C-O | O-R | C-R | Meas. | Appr. | Cam. | Obj. | MSR | **Avg.** |
|------|-----|-----|-----|-----|-----|-----|-------|-------|------|------|-------|----------|
| Gemini 3 Pro | 47.3 | 48.9 | 42.0 | ... | ... | ... | ... | ... | ... | ... | ... | 45.2 |
| GPT-5.4 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | 41.9 |
| **S-Agent (GPT-5.4)** | ... | ... | ... | ... | ... | ... | ... | ... | **46.0** | **48.7** | **44.4** | **46.4** |

- S-Agent 以 **46.4%** 平均得分登顶 MMSI-Bench
- 超越 GPT-5.4 **4.5%**，超越 Gemini 3 Pro **1.2%**
- 运动感知子任务（相机运动 46.0%，物体运动 48.7%）和多步推理（44.4%）表现尤其突出

**ViewSpatial-Bench**（视角感知空间定位）:

| 模型 | C-OVO | C-RD | P-OVO | P-RD | P-SSRD | **Avg.** |
|------|-------|------|-------|------|--------|----------|
| Gemini 3 Pro | 31.6 | 61.9 | 41.1 | 74.4 | 38.9 | 50.4 |
| GPT-5.4 | 27.9 | 60.2 | 41.0 | 48.5 | 40.1 | 45.6 |
| **S-Agent** | **55.5** | ... | ... | **81.1** | ... | **60.0** |

- 平均 **60.0%**，超越 GPT-5.4 **14.4%**
- P-SSRD（人视角场景模拟相对方向）提升 **20.5%**

**ReVSI**（视频空间推理）:

- 平均 **58.8**，排名第二
- 相对方向和路线规划子任务排名第一
- 超越所有开源通用模型和空间专用基线

### 蒸馏模型 S-Agent-8B

| 模型 | MMSI | ViewSpatial | ReVSI |
|------|------|-------------|-------|
| Qwen3-VL-8B-Instruct | 31.1 | 42.2 | 49.1 |
| S-Agent (Qwen3-VL-8B) | 30.7 | 44.1 | 49.5 |
| **S-Agent-8B** | **41.6** | **56.3** | **57.4** |

关键发现：直接给 8B 模型加 S-Agent 框架**不总是提升**（8B 规划器工具选择能力不足），但蒸馏后的 S-Agent-8B 全面超越。蒸馏不仅教会了空间答案，还教会了**可复用的工具使用和证据整合模式**。

### 消融实验

ViewSpatial 上 GPT-5.4 规划器的组件消融：

| 配置 | L1 2D | L2 3D | L3 专家 | Scene Mem | Agent Mem | **Avg.** |
|------|-------|-------|---------|-----------|-----------|----------|
| VLM-only | | | | | | 45.6 |
| + Level-1 2D 证据 | ✓ | | | | | 49.0 |
| + Level-2 3D 证据 | ✓ | ✓ | | | | 提升有限 |
| + Level-3 专家 | ✓ | ✓ | ✓ | | | **56.7** |
| + Scene Memory | ✓ | ✓ | ✓ | ✓ | | 58.2 |
| + Agent Memory | ✓ | ✓ | ✓ | ✓ | ✓ | **60.0** |

**关键洞察**: 原始 L2 3D 证据（相机位姿、深度值、噪声重建点）对 VLM 难以解释，甚至产生干扰。必须通过 L3 专家过滤为任务导向的测量值、相对位置或空间结论，3D 证据才真正有用。

---

## 定性分析

![定性推理例](/blog/papers/2606.20515/figure4_qualitative_reasoning.jpg)

> Figure 4: 工具 grounding 空间推理示例。VLM 直接回答因遮挡失败，S-Agent 通过工具链恢复 3D 关系。

**案例解析**: 第一人称视频中判断"书架"和"电话"的相对位置。VLM 因两物体部分遮挡、目标帧中不可同时清晰可见，依赖 2D 布局错误猜测"书架在右前方"。

S-Agent 的推理轨迹：
1. 初始 grounding 工具未能同时定位两物体
2. **不立即回答**，发出针对性检测调用（包括语义相关查询"desk phone"）
3. 恢复可用边界框
4. 相对位置专家通过深度工具提升为度量 3D 表示，构建俯视图布局
5. 书架估计在 (-0.52, 1.21)，电话在 (-0.34, 1.46) → 书架在电话左侧后方 → 正确答案

![多任务可视化](/blog/papers/2606.20515/figure5_qualitative_tasks.jpg)

> Figure 5: S-Agent 跨代表性空间推理任务的定性可视化。

![附录定性示例1](/blog/papers/2606.20515/figure6_qualitative_appendix.jpg)

> Figure 6: 附录中更多定性示例。

![附录定性示例2](/blog/papers/2606.20515/figure7_qualitative_appendix.jpg)

> Figure 7: 更多证据驱动空间推理示例。

---

## 个人评价

### 创新点

1. **时空证据积累范式**: 将空间推理从"单帧直觉"升级为"主动证据搜索+积累"的 Agent 范式，这是概念层面的重要转变
2. **三层工具体系设计合理**: L1 2D grounding → L2 3D 提升 → L3 专家聚合，每层职责清晰，L3 专家作为"翻译层"解决原始 3D 数据对 VLM 不可读的问题
3. **双记忆系统**: Scene Memory 维持场景状态，Agent Memory 维持推理过程，分离设计避免信息混淆
4. **蒸馏效果显著**: S-300K 数据从 100K 原始轨迹扩展到 292K SFT 样本（三种分解格式），8B 模型性能接近 GPT-5.4/Gemini 3

### 局限性

1. **依赖外部工具**: 推理延迟取决于工具链执行速度，Depth-Anything-3 等工具本身有计算开销
2. **小模型规划器能力不足**: 8B 模型直接作为 S-Agent 规划器反而可能下降，蒸馏虽缓解但未根本解决
3. **场景覆盖有限**: 主要在室内/结构化场景测试，开放世界复杂场景（如户外大场景、动态遮挡）未充分验证
4. **训练数据依赖强模型**: S-300K 依赖 GPT-5.4 生成教师轨迹，存在教师模型偏差传递风险

### 对后续研究的影响

1. **Agent 范式向空间推理延伸**: 证明了"工具使用+记忆"范式在空间推理中的有效性，可能推动更多空间 Agent 研究
2. **数据蒸馏策略**: 多粒度轨迹分解（final-answer / turn-level / expert-level）为 Agent 训练提供了可复用的数据构造方法
3. **专家中介层设计**: L3 专家作为 3D 原始数据与 VLM 之间的"翻译层"，这一设计可推广到其他需要几何计算的多模态任务

---

## 相关论文

| 论文 | 核心贡献 | 关联 |
|------|---------|------|
| **VADAR** (arXiv:2502.06787) | 动态构建 Python API 合成 3D 推理程序 | 前代工具增强空间推理方法，S-Agent 的对比基线 |
| **SpaceTools** (arXiv:2512.04069) | 通过交互式 RL 训练 VLM 协调视觉/机器人工具 | 强化学习驱动工具协调，S-Agent 采用零-shot 规划器 |
| **Think3D** (arXiv:2601.13029) | 为 VLM Agent 装备 3D 重建和相机操控工具 | 类似工具增强思路，S-Agent 更强调时空证据积累 |
| **Cambrian-S** (arXiv:2511.04670) | 大规模空间指令数据训练视频空间超感知 | 数据驱动空间能力提升，S-Agent 为推理时增强+蒸馏 |
| **Depth-Anything-3** (arXiv:2511.10647) | 从任意视角恢复视觉空间的度量深度估计 | S-Agent Level-2 核心工具 |
| **SenseNova-SI** (arXiv:2511.13719) | 多模态基础模型扩展空间智能 | S-300K 训练数据来源 |

---

> 整理者：Nancy | 数据源：arXiv + PaddleOCR-VL-1.6 | 更新时间：2026-06-22
