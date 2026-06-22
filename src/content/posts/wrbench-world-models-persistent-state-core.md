---
author: Ludovico
pubDatetime: 2026-06-22T13:30:00Z
title: "[论文精读] Current World Models Lack a Persistent State Core — WRBench: 世界模型缺少持久状态核"
featured: false
draft: false
tags: [论文精读, 世界模型]
  - 论文精读
  - 世界模型
  - 评测基准
  - Video Generation
description: "WRBench 首次系统性诊断世界模型在不可观测期间的状态演化能力，发现 23 个模型无一能在摄像头移开后保持事件终态一致性。"
---

## 论文信息

- **标题**: Current World Models Lack a Persistent State Core
- **作者**: Jinpeng Lu, Dexu Zhu, Haoyuan Shi, Linghan Cai, Guo Tang, Yinda Chen, Jie Cao, Duyu Tang, Yi Zhang, Yong Dai, Xiaozhu Ju
- **arXiv**: [2606.20545](https://arxiv.org/abs/2606.20545)
- **PDF**: [https://arxiv.org/pdf/2606.20545](https://arxiv.org/pdf/2606.20545)
- **一句话总结**: 提出 **WRBench**——首个将"摄像头运动作为可观测性干预"的系统性诊断基准，评测 23 个视频世界模型在目标被遮挡/不可观测期间的状态演化能力，发现当前所有模型均无法在不可观测期间保持事件终态一致性。

---

## 研究背景

### 核心问题

世界模型（World Models）被视为通向 AGI 的关键一步。但现有基准（VBench、WorldModelBench、WorldScore 等）只评测**可视化维度**：画面质量、运动流畅度、摄像头可控性。它们从不问一个问题：

> **当摄像头移开、目标不可见时，生成的世界是否仍在持续演化？**

就像月亮在没人看的时候是否还在轨道上运行——现有基准从不验证这一点。

### 现有基准的盲区

论文通过对比表揭示了现有基准的覆盖缺口：

| 基准 | 世界动态 | 统一控制 | 视觉质量 | **状态鲁棒性** | **演化一致性** | **路径诊断** |
|------|---------|---------|---------|-------------|-------------|-----------|
| VBench | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| WorldModelBench | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| WorldScore | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| WorldMark | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| WBench | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| **WRBench (本文)** | ✅ | ✅ | ✅ | **✅** | **✅** | **✅** |

WRBench 是唯一覆盖**状态鲁棒性**、**演化一致性**和**路径诊断**三个维度的基准。

---

## 核心方法

### 3.1 设计理念：摄像头运动 = 可观测性干预

WRBench 的核心洞察是将**摄像头运动**重新定义为对可观测性的干预，而不是单纯的视角控制：

1. **初始观察**：建立场景和事件的初始状态
2. **事件触发**：让某个事件开始发生（物体移动、状态变化等）
3. **视角干预**：摄像头移开，目标暂时不可见
4. **重新观察**：摄像头返回，验证目标是否保持了事件的正确终态

### 3.2 Natural-25 事件-视角记录集

- **25 个场景族** × **4 级事件设计**（空间位移 × 状态变化）
- 每个测试固定相同的初始观察和事件，变视角条件
- 提示词指定初始场景、事件和视角请求，**不透露返回端点状态**

### 3.3 六维评测体系

WRBenchLib 对每个模型生成视频，从六个维度评分：

| 维度 | 含义 |
|------|------|
| **CamPrec** | 严格局部请求控制精度 |
| **CamAlign** | 公共偏航意图对齐 |
| **Integ.** | 视觉完整性 |
| **Reobs. Support** | 可评判的"隐藏-返回"证据比例 |
| **Reobs. Spatial** | 重新观察时的空间一致性 |
| **Reobs. State** | 重新观察时的状态一致性 |

### 3.4 评测流水线

```mermaid
graph LR
    A[Natural-25 事件-视角记录] --> B[WRBenchLib]
    B --> C[23 个模型生成视频]
    C --> D[六维自动评测]
    D --> E[人工偏好标注校准]
    E --> F[诊断报告]
```

![WRBench 方法总览](/blog/papers/2606.20545/img_in_image_box_101_624_1113_973.jpg)

> Figure 2: WRBench 方法总览。Natural-25 提供场景、事件和视角干预；WRBenchLib 为每个模型生成视频和溯源记录；评测套件从控制执行到重新观察状态一致性评分六个诊断维度；人工偏好标注独立校准每个维度。

---

## 实验结果

### 4.1 核心发现：所有模型均无法保持不可观测期间的状态演化

论文评测了 **23 个模型**，跨越 **4 种控制范式**，共 **9,600 个视频**。核心发现：

> **当前系统维持已观察世界如同跟拍镜头，当目标重新出现时，恢复的是被放弃时的状态，而非在不可见期间推进事件。**

这个失败跨控制范式、模型族和规模增量重复出现。

### 4.2 模型级诊断剖面

| 模型 | CamPrec | CamAlign | Integ. | Reobs. Support | Reobs. Spatial | Reobs. State | Avg. |
|------|---------|----------|--------|---------------|---------------|-------------|------|
| ReCamMaster | 0.717 | 0.729 | 0.740 | 58.5% | 0.665 | 0.616 | 0.667 |
| HyDRA | 0.822 | 0.855 | 0.691 | 33.2% | — | — | — |
| InSpatio-14B | — | — | — | 62.3% | 0.734 | — | — |
| LiveWorld | — | — | — | 39.6% | 0.661 | 0.600 | — |
| Gen3C | — | — | — | 73.0% | 0.681 | 0.640 | — |

![模型级诊断剖面](/blog/papers/2606.20545/img_in_image_box_89_42_1123_537.jpg)

> 高重新观察支持率 ≠ 高状态一致性。Gen3C 支持率最高（73%），但状态一致性仅 0.640。

### 4.3 规模升级不带来状态演化能力

论文对 Wan 系列做了细致的规模诊断：

| 指标 | Wan2.1 (1.3B→14B) | Wan2.2 (5B→A14B) | 总计 |
|------|-------------------|-------------------|------|
| Cam Prec. | -0.014 | +0.034 | +0.047 |
| Integ. | +0.004 | +0.036 | +0.049 |
| Vis. Spatial | +0.008 | +0.006 | +0.014 |
| Vis. State | +0.018 | +0.018 | +0.036 |
| **Ret. Spatial** | **-0.051** | **-0.011** | **-0.062** |
| **Ret. State** | **-0.036** | **-0.015** | **-0.051** |
| Reobs. Support | +0.044 | +0.056 | +0.059 |

**关键发现：可见维度改善，但重新观察维度反而下降。**

![Wan 系列规模诊断](/blog/papers/2606.20545/img_in_image_box_100_340_1089_477.jpg)

> Figure 7: Wan 系列规模/版本/架构诊断。共同模式是"可访问性改善比端点持久化更容易"。

### 4.4 失败案例分析

论文对典型失败模式做了详细分类：

**LiveWorld：可信的重新出现，错误的世界**

LiveWorld 在简单场景下重新出现干净，可见分数高（visible spatial >0.9）；但在原地变化事件中，重新观察状态降至 0.31-0.40。其 monitor-agent 设计的缺陷暴露无遗：

> 向前滚动 backbone 来**幻觉**不可见主体的动态，而非检索存储的状态。单个就座的人在重新出现时被合成为两个蹲伏的人，或在原始位置附近生成额外的位移身体，而静态房间保持清晰。

![LiveWorld 失败案例](/blog/papers/2606.20545/img_in_image_box_103_53_1092_407.jpg)

> Figure 14: 重新观察可访问性由摄像头通道决定。同一场景在不同方向摄像头扫掠下产生不同的隐藏-返回事件。

### 4.5 架构类型对比

| 模型 | 类型 | 重新观察支持率 | 返回状态一致性 |
|------|------|-------------|-------------|
| Lingbot World | Memory/action | 5% | 0.66 |
| Wan-Fun 2.1-14B | Camera fine-tune | 19% | 0.62 |
| LiveWorld | Memory/action | 40% | 0.60 |
| ReCamMaster | Geometry carrier | 60% | 0.61 |
| VerseCrafter | Geometry carrier | 28% | 0.58 |
| Spatia | Geometry carrier | 25% | 0.58 |
| Hydra | Camera fine-tune | 35% | 0.44 |

**发现：无论架构类型（几何缓存、记忆/动作、摄像头微调），状态一致性均卡在 0.44-0.66 区间。**

---



### Finding 1: 现有基准奖励表面属性，不测试隐藏状态演化

所有现有基准从生成像素和文本对齐评分，从未从遮挡后的隐藏状态持久化评分。

### Finding 2: 鲁棒的世界状态演化不从更清晰的图像、更紧的控制、更丰富的几何先验或参数规模中自动涌现

失败跨控制范式、模型族和规模增量重复出现。

### Finding 3: 重新观察可访问性 ≠ 端点持久化

更高的重新观察支持率改变了可评判的内容，但不自动使重新观察一致性成立。

### Finding 4: 常规视频监督的缩放不会带来世界状态演化，必须设计进去

> "Larger Wan backbones add re-observation access while the 2.1→2.2 upgrade adds visible fidelity, which are the observable axes such training already optimizes, yet conditional re-observed state stays in a fixed band."

### Finding 5: 世界模型需要"what-memory"，而不仅是"where-memory"

> "Every architecture caches geometry, appearance, or motion to re-render where the scene was, so how the camera is encoded is second-order, while the missing, highest-leverage component is a state writer that records what changed while hidden."

![What vs Where Memory](/blog/papers/2606.20545/img_in_image_box_103_73_1092_522.jpg)

---

## 个人评价

### 创新点

1. **评测视角的范式转换**：将摄像头运动从"视角控制"重新定义为"可观测性干预"，这是一个深刻的概念转变。
2. **六维诊断体系**：不仅评测"看得见的时候对不对"，更评测"看不见的时候发生了什么"。
3. **大规模系统性验证**：23 个模型、9,600 个视频，跨 4 种控制范式，结论具有说服力。
4. **人类偏好校准**：自动评估器经过人工偏好标注校准，确保分数有实际意义。

### 局限性

1. **Natural-25 场景有限**：25 个场景族可能不足以覆盖所有物理交互类型。
2. **未直接评测机器人控制场景**：基准聚焦视频生成模型，未扩展到具身智能/机器人策略学习。
3. **未提出解决方案**：论文是诊断性研究，指出了"what-memory"的缺失，但未给出具体架构建议。

### 对后续研究的影响

这篇论文为后续研究指明了方向：

- **端点持久化（Endpoint Persistence）**应成为世界模型设计的一级目标
- **状态写入器（State Writer）**——记录隐藏期间发生的变化——是最具杠杆率的缺失组件
- **奖励/策略训练**应针对不可观测事件端点设计，而非仅匹配可见分布

论文附录 H 进一步提出了三种端点掩码控制策略，为后续奖励设计提供了具体路径。

---

## 相关论文

1. **[WorldModelBench](https://arxiv.org/abs/2606.19531)** (NeurIPS 2026) — 将视频生成模型作为世界模型评判，但仅评测可视化维度。
2. **[ImageWAM](https://arxiv.org/abs/2606.19531)** (2026-06-17) — 质疑世界动作模型是否真的需要视频生成，或许图像编辑就够了。与 WRBench 形成有趣对比：一个问"需要多少视觉信息"，另一个问"隐藏了多少状态信息"。
3. **[Mem-World](https://arxiv.org/abs/2606.18960)** (2026-06-17) — 记忆增强动作条件世界模型，尝试解决持久机器人操作问题。其记忆机制正是 WRBench 所缺少的"what-memory"候选方案。
4. **[MemoryWAM](https://arxiv.org/abs/2606.20562)** (2026-06-18) — 高效世界动作建模与持久记忆，同样关注持久化问题。
5. **[SurgVista](https://arxiv.org/abs/2606.19889)** (2026-06-18) — 长视界手术世界建模，关注器械-组织动力学。手术场景中不可见状态演化尤为重要。

---

> 整理者：Nancy | 数据源：arXiv (2606.20545) + PaddleOCR-VL-1.6 解析 | 更新时间：2026-06-22
