---
author: Ludovico
pubDatetime: 2026-07-09T09:50:00+08:00
title: "[论文精读] Dinomaly：多类别无监督异常检测里的 Less Is More"
featured: false
draft: false
tags:
  - 论文精读
  - 异常检测
  - UAD
description: Dinomaly 用强预训练 ViT 特征、Noisy Bottleneck、Linear Attention 和 Loose Reconstruction，缓解多类别无监督异常检测中的 identity mapping 问题。
---

Dinomaly 讨论的是 **multi-class unsupervised anomaly detection, MUAD**：训练集中只有正常样本，但这些正常样本来自多个类别；测试时，一个统一模型同时判断不同类别图像是否异常，并定位异常区域。

这篇论文的核心不是把异常检测模型做得更复杂，而是反过来控制重建模型的复制能力：

```text
强 frozen ViT encoder 提供好特征
+ Dropout bottleneck 破坏直接复制
+ Linear Attention 降低局部精确搬运
+ Loose Reconstruction 放松逐层逐点蒸馏
= decoder 更像“正常特征修复器”，而不是“输入复读机”
```

![Dinomaly 总览：多类别设定、性能对比和 scaling](/blog/images/dinomaly/overview.jpg)

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection |
| 任务 | Multi-Class Unsupervised Anomaly Detection |
| 方法范式 | Frozen foundation ViT feature reconstruction |
| 代码 | [github.com/guojiajeremy/Dinomaly](https://github.com/guojiajeremy/Dinomaly) |
| 默认骨干 | DINOv2-Register ViT-Base/14 |
| 主要数据集 | MVTec-AD、VisA、Real-IAD |

一句话总结：Dinomaly 用一个纯 Transformer 的特征重建框架，在不引入伪异常生成、扩散模型、向量量化等复杂模块的情况下，把多类别 UAD 的性能推进到接近甚至超过部分 class-separated 方法的水平。

## 问题：MUAD 为什么难

传统 UAD 通常是 **one-class-one-model**：

```text
bottle 一个模型
cable 一个模型
hazelnut 一个模型
```

MUAD 则希望：

```text
所有类别共用一个模型
```

统一模型在工程上更有吸引力，但它会带来一个核心问题：多类别正常样本的分布非常复杂，decoder 为了重建这些多样化正常模式，会学到过强泛化能力。测试时遇到异常，它也可能把异常区域重建得很好。

论文把这个问题称为 **identity mapping / identical shortcut**，也可以理解成 **over-generalization**：

```text
理想情况：
正常区域 -> 重建好 -> anomaly score 低
异常区域 -> 重建差 -> anomaly score 高

MUAD 中的坏情况：
正常区域 -> 重建好
异常区域 -> 也重建好
```

因此 Dinomaly 的主线不是“让 decoder 越强越好”，而是控制 decoder 的能力边界：对正常模式足够强，对异常模式不要强到直接复制。

## 框架：Encoder、Bottleneck、Decoder

Dinomaly 是一个特征重建框架，由三部分组成：

![Dinomaly 框架图](/blog/images/dinomaly/framework.jpg)

```text
image
  -> frozen pretrained ViT encoder
  -> 8 个中间层特征
  -> MLP bottleneck
  -> 8 层 Transformer decoder
  -> 重建中间层特征
  -> cosine distance 作为异常分数
```

默认 encoder 是 DINOv2-Register ViT-Base/14。输入 392x392 时，patch size 为 14，对应的空间特征图是：

```text
392 / 14 = 28
28 x 28 = 784 tokens
```

论文取 encoder 的 8 个中间层特征，而不是只取最后一层。这符合异常检测的需求：局部纹理、结构缺陷和较高层语义都可能提供异常线索。训练时 encoder 冻结，decoder 学习重建这些中间层特征；推理时，encoder 特征和 decoder 重建特征之间的 cosine distance 构成 anomaly map。

这里的关键是二者分工：

```text
frozen encoder: 忠实表达输入，异常区域也会被编码成异常特征
trained decoder: 只见过正常样本，倾向重建正常模式
encoder-decoder discrepancy: 异常证据
```

## 组件一：Foundation Transformer

论文把强预训练 ViT 放在第一位。原因很直接：UAD 没有异常监督，encoder 特征质量基本决定了上限。

Dinomaly 默认使用 DINOv2-Register ViT-B/14，并系统比较了 DeiT、MAE、D-iGPT、MoCoV3、DINO、iBOT、DINOv2、DINOv2-R 等基础模型。

![不同 ViT foundation 的异常检测表现](/blog/images/dinomaly/foundations.jpg)

论文观察到两个点：

- 大多数 foundation ViT 都能让 Dinomaly 达到较强结果，说明框架对 backbone 有一定鲁棒性。
- MAE 相对较弱，作者认为这和它在无微调场景下的 frozen representation 判别性不足有关。

这说明 Dinomaly 的 “Less Is More” 不是“小模型”，而是“少设计复杂模块”。默认 ViT-B/14 本身并不小，真正的基础能力来自高质量 frozen representation。

## 组件二：Noisy Bottleneck

Noisy Bottleneck 是论文最重要的设计之一。作者没有手工合成伪异常，而是在 bottleneck MLP 中启用 Dropout。

普通重建目标容易变成：

```text
clean normal feature -> decoder -> clean normal feature
```

这很容易学成复制。Dinomaly 让训练更像 denoising：

```text
normal feature
  -> MLP + Dropout
  -> corrupted feature
  -> decoder
  -> clean normal feature
```

Dropout 在这里不是单纯防过拟合，而是充当 **feature-level pseudo anomaly**。它随机破坏正常特征，迫使 decoder 学“从受扰动特征恢复正常模式”，而不是逐 token 原样搬运输入。

这个设计打在 MUAD 的痛点上：多类别正常样本会让 decoder 泛化很强，而 Dropout 把学习目标从 reconstruction 推向 restoration。

论文的 Dropout rate 消融显示：

| Dropout rate | Image AUROC | Pixel AP | 观察 |
|---:|---:|---:|---|
| 0 | 99.19 | 63.11 | 无噪声，定位明显弱 |
| 0.1 | 99.54 | 69.46 | 定位大幅提升 |
| 0.2 | 99.60 | 69.29 | 默认设置，整体均衡 |
| 0.3 | 99.65 | 68.46 | 图像级仍强，定位略降 |
| 0.5 | 99.56 | 67.43 | 噪声过强，定位继续下降 |

结论是：适度 Dropout 能明显提升定位；过强 Dropout 会让正常区域也变得更难稳定重建。

## 组件三：Unfocused Linear Attention

Softmax Attention 的能力很强，可以学到非常尖锐的注意力：

```text
第 i 个输出 token
  -> attend 到第 i 个输入 token
  -> 直接复制局部信息
```

这对分类任务是优势，对异常检测重建 decoder 却可能是风险。因为异常区域也可能被精确传递到下一层。

Dinomaly 使用 Linear Attention：

```text
Softmax Attention: Softmax(QK^T)V
Linear Attention:  phi(Q)(phi(K)^T V)
```

Linear Attention 原本是为了降低复杂度，从 `O(N^2 d)` 降到 `O(N d^2)`。但论文看中的不是省算力本身，而是它“不够会聚焦”的副作用。

![Softmax Attention 与 Linear Attention 的注意力图](/blog/images/dinomaly/attention_map.jpg)

![不同距离上的注意力分布](/blog/images/dinomaly/attention_dist.jpg)

从论文可视化看，Softmax Attention 更容易聚焦在 query 对应的局部区域；Linear Attention 的注意力更分散。Dinomaly 利用这种分散性，让 decoder 更多依赖长程上下文恢复特征，减少对局部异常信息的直接复制。

这个点要读得克制一些：Linear Attention 不是天然更强，它在这里更像一种有意的表达能力约束。

## 组件四：Loose Reconstruction

Loose Reconstruction 包含两部分：Loose Constraint 和 Loose Loss。

![不同重建约束方式](/blog/images/dinomaly/loose_reconstruction.jpg)

### Loose Constraint

很多特征重建方法受知识蒸馏启发，会做严格的 layer-to-layer 对齐：

```text
encoder layer 1 -> decoder layer 1
encoder layer 2 -> decoder layer 2
...
```

监督越密，decoder 越容易学得像 encoder。但异常检测恰恰依赖 encoder-decoder discrepancy：如果 decoder 过度模仿 encoder，异常区域也会被重建好。

Dinomaly 改成 group-to-group：

```text
多个 encoder 层相加/分组
多个 decoder 层相加/分组
组与组之间做重建约束
```

论文还进一步使用 2-group 方案，把低语义层和高语义层分开。低层信息有利于精确定位，高层信息有利于结构和语义判断。

### Loose Loss

Loose Loss 则放松逐点 loss。论文使用 hard-mining global cosine loss：对已经重建得很好的 feature points 缩小梯度，只让更难重建的位置承担主要优化压力。

核心公式是 cosine distance：

```text
Dcos(a, b) = 1 - (a^T b) / (||a|| ||b||)
```

被判定为“重建较好”的位置不会完全丢弃，而是把梯度缩小到原来的 0.1。这样做的目标是防止 decoder 在容易区域上继续精修，逐渐学成过强的逐点复制器。

一句话概括：

```text
监督太紧 -> decoder 像 encoder -> discrepancy 变小
监督稍松 -> decoder 保持正常模式恢复倾向 -> 异常差异保留
```

## 实验结果

论文主实验覆盖 MVTec-AD、VisA、Real-IAD。基础版 Dinomaly 的 image-level AUROC 是：

| 数据集 | 类别规模 | Image AUROC | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|
| MVTec-AD | 15 类 | 99.6 | 98.4 | 69.3 | 94.8 |
| VisA | 12 类 | 98.7 | 98.7 | 53.2 | 94.5 |
| Real-IAD | 30 类，多视角 | 89.3 | 98.8 | 42.8 | 93.9 |

MVTec-AD 的图像级指标已经接近饱和；更值得看的是 VisA 和 Real-IAD。Real-IAD 类别更多、视角更复杂，Dinomaly 仍然超过前序 MUAD 方法，说明它不只是吃了小数据集饱和红利。

论文还比较了 class-separated UAD。结论是：在 MVTec-AD 和 VisA 上，多类别 Dinomaly 与逐类训练的 Dinomaly 几乎没有明显性能差距；在 Real-IAD 上，多类别模型有一定下降，但仍能接近 class-separated SoTA。

## 消融实验怎么看

论文消融验证四个组件：

```text
NB: Noisy Bottleneck
LA: Linear Attention
LC: Loose Constraint
LL: Loose Loss
```

MVTec-AD 上的关键结果可以压缩成：

| 组合 | Image AUROC | Pixel AP | 说明 |
|---|---:|---:|---|
| baseline | 98.41 | 62.96 | 无噪声、Softmax Attention、dense layer-to-layer、global loss |
| + LL | 99.06 | 66.22 | Loose Loss 单独提升明显 |
| + NB + LC | 99.50 | 68.16 | Noisy Bottleneck 与 Loose Constraint 组合有效 |
| + NB + LC + LL | 99.52 | 68.25 | 加 LL 后继续稳定 |
| + NB + LA + LC | 99.57 | 67.93 | Linear Attention 主要在组合中发挥作用 |
| + NB + LA + LC + LL | 99.60 | 69.29 | 完整 Dinomaly |

这张表的读法很重要：

- NB 和 LL 是最直接的增益来源。
- LA 单独不是 magic trick，它更像是在 NB 存在时进一步降低局部复制。
- LC 单独可能让重建变得太容易，必须和噪声机制配合。

所以 Dinomaly 的贡献不是某一个孤立技巧，而是一组围绕 identity mapping 的约束共同生效。

## Scaling 结果

论文强调 Dinomaly 具有 scaling property。ViT 从 Small 到 Base 再到 Large，MVTec-AD 上性能逐步提升，但计算也显著增加：

| Backbone | Params | MACs | Im/s | Image AUROC | Pixel AP |
|---|---:|---:|---:|---:|---:|
| ViT-Small | 37.4M | 26.3G | 153.6 | 99.26 | 68.29 |
| ViT-Base | 148.0M | 104.7G | 58.1 | 99.60 | 69.29 |
| ViT-Large | 275.3M | 413.5G | 24.2 | 99.77 | 70.53 |

输入尺寸也会影响定位。默认 392x392 对应 28x28 token feature map；较小输入已经能保持很强 image-level 检测，但 pixel-level 定位通常受益于更大的特征图。

| 输入 | Image-level | Pixel-level |
|---|---|---|
| 280x280 | 99.6 / 99.8 / 99.3 | 98.2 / 65.2 / 66.3 / 93.6 |
| 336x336 | 99.6 / 99.8 / 99.2 | 98.3 / 67.2 / 67.8 / 94.2 |
| 392x392 | 99.6 / 99.8 / 99.0 | 98.4 / 69.3 / 69.2 / 94.8 |

这里的指标顺序分别是：

```text
Image-level: AUROC / AP / F1-max
Pixel-level: AUROC / AP / F1-max / AUPRO
```

## 定性结果

论文附录展示了随机选择样本的 anomaly maps。下面是 MVTec-AD 上的可视化：

![Dinomaly 在 MVTec-AD 上的 anomaly map](/blog/images/dinomaly/anomaly_maps_mvtec.jpg)

从图中可以看到，Dinomaly 的热力图通常能覆盖缺陷区域，但边界精度仍受 ViT patch 粒度和后处理上采样影响。它是 feature-level localization，不是直接预测高分辨率 mask。

## 论文结论

Dinomaly 的核心判断是：

```text
MUAD 的失败主要来自 decoder 过度泛化和 identity mapping。
解决方向不是继续堆复杂模块，而是让 decoder 学正常恢复，而不是异常复制。
```

四个组件各自对应一个约束：

| 组件 | 作用 |
|---|---|
| Foundation Transformer | 提供强、通用、判别性好的 frozen feature |
| Noisy Bottleneck | 用 Dropout 把 reconstruction 变成 restoration |
| Linear Attention | 降低局部逐点复制能力 |
| Loose Reconstruction | 避免 decoder 被训练成 encoder 的精确学生 |

这篇论文最值得记住的一句话是：

```text
异常检测里的 decoder 不是越强越好；关键是控制它对异常模式的泛化边界。
```

因此 Dinomaly 的 “Less Is More” 更准确地说是：少做复杂异常先验设计，保留强 foundation representation，同时有意识地削弱 decoder 的复制通道。
