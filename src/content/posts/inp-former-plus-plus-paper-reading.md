---
author: Ludovico
pubDatetime: 2026-07-09T10:16:00+08:00
title: "[论文精读] INP-Former++：内在正常原型、Soft Coherence 与残差学习"
featured: false
draft: false
tags:
  - 论文精读
  - 异常检测
description: INP-Former++ 在 INP-Former 的单图内在正常原型上加入 Soft INP Coherence Loss 和 Residual Learning，把方法扩展到 semi-supervised、few-shot、multi-class 和一定 zero-shot 场景。
---

INP-Former++ 是 INP-Former 的扩展版。原版 INP-Former 的核心想法是：不要只依赖训练集里的 normal prototypes，而是从当前测试图像自身提取 **Intrinsic Normal Prototypes, INPs**。

INP-Former++ 继续沿着这条路走，但补上了两个关键问题：

```text
问题 1:
原始 INP Coherence Loss 可能让很多 patch 都塌缩到同一个 INP。

升级:
Soft INP Coherence Loss，让每个 patch 对齐到多个 INP 的加权组合。

问题 2:
只用正常样本训练时，重建残差有时还不够“尖锐”，定位图有背景噪声。

升级:
Residual Learning + Segmentation Head，用伪异常或少量真实异常放大残差边界。
```

![INP-Former++ 总体框架](/blog/images/inp-former-plus-plus/framework.jpg)

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | INP-Former++: Advancing Universal Anomaly Detection via Intrinsic Normal Prototypes and Residual Learning |
| 方法 | INP-Former++ |
| 任务 | Universal Anomaly Detection |
| 覆盖设定 | Single-class、Multi-class、Few-shot、Semi-supervised、一定 Zero-shot |
| 默认骨干 | DINOv2-Register ViT-Base/14 |
| 主要升级 | Soft INP Coherence Loss、Residual Learning、Segmentation Head |

一句话总结：INP-Former++ 保留“从单张图提取正常原型”的思想，用 Soft INP Coherence 避免 prototype collapse，再用残差学习和分割头把 reconstruction residual 转成更干净的 anomaly map。

## 先回到 INP 的动机

传统 prototype 或 memory-bank 异常检测，通常是：

```text
训练集正常样本
  -> 提取 normal prototypes / memory bank
测试图像
  -> 和外部 normality 比较
```

这个逻辑的问题是 normality 可能不对齐。

few-shot 场景下，训练样本太少，prototype 覆盖不了所有正常外观。multi-class 场景下，一个类别的正常背景可能像另一个类别的异常区域。

INP-Former 的反向思路是：很多异常是局部的，所以异常图里仍然有大量正常区域。这些来自同一张图的正常 patch，可能比训练集 prototype 更适合当当前图像的参照。

![INP 动机：从当前测试图像内部提取正常原型](/blog/images/inp-former-plus-plus/motivation.jpg)

于是方法变成：

```text
test image
  -> 提取当前图像自己的 INPs
  -> 用 INPs 指导 decoder 重建正常模式
  -> encoder-decoder discrepancy 作为异常分数
```

## INP-Former++ 的整体流程

INP-Former++ 有两条训练线：

```text
Normal Pattern Modeling:
只用正常图像训练 INP Extractor + INP-guided Decoder

Residual Learning:
用伪异常，或者少量真实异常，训练 Segmentation Head
```

这两条线是解耦的。第一条线学习“怎么从图像里提取正常原型并重建正常模式”；第二条线学习“怎么把 feature residual 放大成更清晰的异常 mask”。

推理时，最终 anomaly map 来自两部分平均：

```text
final anomaly map =
  reconstruction error map
  + segmentation head predicted mask
```

## INP Extractor：少量 token 表示当前图的正常模式

INP Extractor 用 M 个 learnable tokens 作为 query，对当前图像的 ViT patch features 做 cross attention：

```text
patch features: K, V
learnable INP tokens: Q
cross attention
  -> M 个 INPs
```

默认 M=6。也就是说，模型试图用 6 个内在正常原型描述当前图像里的主要正常模式。

这个设计有两个好处：

```text
1. INPs 来自当前图像，天然更对齐测试样本。
2. INPs 数量很少，形成信息瓶颈，不容易完整携带异常细节。
```

## 原版 Coherence Loss 的塌缩问题

原版 INP Coherence Loss 的想法是：每个正常 patch 应该能被某个最近的 INP 表示。直觉没问题，但 INP-Former++ 发现它可能出现 shortcut：

```text
所有 patch token
  -> 都被分配给同一个 INP
```

这样看似 loss 能降，但多个 INP 没有学出多样的语义分工。

![原始 INP Coherence Loss 可能导致 shortcut](/blog/images/inp-former-plus-plus/shortcut.jpg)

这就是 INP-Former++ 的第一个关键升级：**Soft INP Coherence Loss**。

## Soft INP Coherence Loss

Soft INP Coherence 不再让每个 patch 只找最近的一个 INP，而是让 patch 对齐到多个 INP 的加权组合：

```text
patch_i
  -> compute similarity to all INPs
  -> soft weights
  -> weighted sum of INPs
  -> align weighted sum with patch_i
```

可以理解成从 hard assignment 变成 soft reconstruction。

![Soft INP Coherence Loss 让 INP 表达更细，距离图更干净](/blog/images/inp-former-plus-plus/soft-coherence.jpg)

这样有两个直接效果：

```text
1. 不容易让所有 token 塌缩到同一个 INP。
2. 多个 INP 可以共同描述一个 patch，表达能力更平滑。
```

这点对 zero-shot 也有帮助，因为模型在未见类别上更依赖 INP Extractor 的泛化能力。更稳定的 INPs 会带来更稳定的跨域距离图。

## INP-Guided Decoder：用正常原型限制重建

INP-guided Decoder 的注意力结构和普通 self-attention 不一样。

普通 self-attention 是：

```text
Q: patch tokens
K,V: patch tokens
```

这容易把异常 patch 自己的信息搬回输出。

INP-guided attention 是：

```text
Q: decoder patch tokens
K,V: INPs
```

也就是说，decoder 每一步重建都只能从 INPs 里取信息。因为 INPs 被约束为当前图像的正常原型，异常 patch 就更难被原样重建。

复杂度也更低。普通 self-attention 是 `O(N^2 C)`，INP-guided attention 是 `O(N M C)`。由于 `M` 通常只有 6，而 patch token `N` 是几百级别，所以计算和显存都明显下降。

## Soft Mining Loss：把难重建的正常区域照顾好

INP-Former++ 继续使用类似 hard mining / focal 的思想：正常图里有些区域比其他区域更难重建，如果这些正常区域长期重建不好，推理时就会产生背景噪声。

Soft Mining Loss 用区域 reconstruction error 相对 batch 平均 error 的比例，调整梯度大小：

```text
重建误差高的正常区域
  -> 梯度更大
  -> 训练时更受关注

重建已经很好的区域
  -> 梯度相对小
```

它同时考虑 cosine distance 和 MSE。MSE 在这里还为后面的 residual learning 提供更稳定的幅值信息。

## Residual Learning：++ 里最显著的定位增强

INP-Former++ 的第二个大升级是 **Residual Learning**。

只靠正常样本训练的 reconstruction residual 有时能检测异常，但 anomaly map 会带背景噪声。于是作者加了一个 segmentation head，让它学习从 feature residual 预测 anomaly mask。

训练 residual learning 时使用伪异常：

```text
normal image
  + Perlin mask / texture
  -> pseudo anomaly image
  -> encoder + INP-guided decoder
  -> feature residual
  -> segmentation head
  -> pseudo anomaly mask supervision
```

![Residual Learning 让异常定位更清晰](/blog/images/inp-former-plus-plus/residual.jpg)

这里非常关键的一点是 **stop-gradient**：

```text
feature residual stop-gradient
只训练 segmentation head
不反向污染 encoder / INP extractor / decoder
```

如果不 stop-gradient，伪异常会影响 reconstruction model 本身，使它偏向伪异常分布，反而破坏“只建模正常模式”的主线。论文的 ablation 也显示，去掉 stop-gradient 后 pixel-level 性能明显下降。

## Semi-Supervised 为什么自然接上了

INP-Former++ 因为已经有 residual learning 线，所以很容易接入少量真实异常。

做法是把真实异常区域增强后贴到正常图上，和伪异常一起训练 segmentation head。这样模型不需要改变正常模式建模部分，只是在 residual-to-mask 这条线上获得更真实的异常监督。

这也是它比原版 INP-Former 多出来的一个重要任务覆盖：

```text
INP-Former:
single-class / multi-class / few-shot / some zero-shot

INP-Former++:
上面这些
+ semi-supervised anomaly detection
+ 更强 pixel-level localization
```

## Zero-Shot 到底算不算

INP-Former++ 论文确实展示了一定 zero-shot 能力：例如在 Real-IAD 上训练，然后到未见过的 MVTec-AD 上测试，通过 INP Extractor 提取当前图像的 INPs 并生成距离图。

![INP-Former++ 的 zero-shot 定性结果](/blog/images/inp-former-plus-plus/zero-shot.jpg)

但要注意，它不是 CLIP prompt 方法那种主打 zero-shot 的路线。更准确的表述是：

```text
INP-Former++ 有一定 zero-shot 泛化能力，
且比 INP-Former 更好；
但论文也承认，不借助语言提示时，
zero-shot 仍不是它最强的设定。
```

它的强项仍然是 universal AD：multi-class、few-shot、semi-supervised，以及从当前图像动态抽取 normality 的能力。

## 实验重点

论文的 multi-class 结果显示，INP-Former++ 在 MVTec-AD、VisA、Real-IAD 上都取得很强表现。一个值得注意的趋势是：image-level 本来已经很高，++ 的提升更多体现在 pixel-level localization 上。

![INP-Former 与 INP-Former++ 的异常定位对比](/blog/images/inp-former-plus-plus/qualitative.jpg)

这和方法设计一致：Soft INP Coherence 让 INPs 更稳定，Residual Learning 让 residual map 更像分割 mask。

论文还做了不同 setting 的汇总，性能随着可用信息增加而上升：

```text
zero-shot
  -> few-shot
  -> multi-class full normal training
  -> semi-supervised with few anomalies
```

![INP-Former++ 在不同设定下的性能变化](/blog/images/inp-former-plus-plus/scalability.jpg)

这个图的重点不是某个单点数字，而是方法形态：同一个框架可以随着数据条件变好自然增强，而不是每种数据条件换一套 pipeline。

## 和 Dinomaly2 怎么区分

这两篇都使用强 ViT 特征和重建误差，但着力点不同。

| 方法 | 核心抓手 | 主要解决 |
|---|---|---|
| Dinomaly2 | Context-Aware Recentering + minimalist reconstruction constraints | 多类别上下文混淆和 full-spectrum 统一 |
| INP-Former++ | Intrinsic Normal Prototypes + Soft Coherence + Residual Learning | 外部 normality 不对齐、INP 塌缩、定位噪声 |

Dinomaly2 更像是把 Dinomaly 做成一个强统一基座；INP-Former++ 更像是把“当前图像内部 normality”这个思想做完整，并补上 segmentation-style localization。

## 读完后的判断

INP-Former++ 最值得记住的是两个升级：

```text
Soft INP Coherence:
让 INP 不塌缩，让多个原型更稳定地表达当前图像正常模式。

Residual Learning:
在不污染正常重建模型的前提下，
把 feature residual 放大成更清晰的 anomaly mask。
```

如果原版 INP-Former 的亮点是“从单张图提取正常原型”，那么 INP-Former++ 的亮点就是把这个想法做得更稳、更适合定位，也更容易接入少量真实异常。
