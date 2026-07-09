---
author: Ludovico
pubDatetime: 2026-07-09T09:58:00+08:00
title: "[论文精读] INP-Former：从单张测试图里提取内在正常原型"
featured: false
draft: false
tags:
  - 论文精读
  - 异常检测
description: INP-Former 从测试图像自身动态提取 Intrinsic Normal Prototypes，并用这些正常原型指导特征重建，缓解训练集正常原型与测试图不对齐的问题。
---

INP-Former 的核心问题意识很直接：很多异常检测方法都在拿测试图像和训练集里的正常样本、memory bank 或 prototypes 做比较，但这些外部正常参照不一定和当前测试图像对齐。

这篇论文换了一个角度：

```text
既然异常通常是局部的，
那么异常图里仍然有大量正常区域。
这些来自同一张图的正常区域，
可能比训练集里的正常样本更适合作为当前图像的正常参照。
```

于是作者提出 **Intrinsic Normal Prototypes, INPs**，也就是从单张测试图像自身动态提取出来的内在正常原型。

![INP 动机：外部 prototype 可能不对齐，图像内部 normal prototype 更贴近当前测试图](/blog/images/inp-former/motivation.jpg)

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | Exploring Intrinsic Normal Prototypes within a Single Image for Universal Anomaly Detection |
| 方法 | INP-Former |
| 任务 | Universal Anomaly Detection |
| 覆盖设定 | Multi-class、Few-shot、Single-class、部分 Zero-shot |
| 默认骨干 | DINOv2-Register ViT-Base/14 |
| 代码 | [github.com/luow23/INPFormer](https://github.com/luow23/INPFormer) |

一句话总结：INP-Former 在 Dinomaly 式的特征重建框架上，引入从测试图像自身提取的少量正常原型，用这些 INPs 指导 decoder 只重建正常模式，从而在 multi-class、few-shot 和 single-class anomaly detection 中取得强结果。

## 问题：外部正常性不一定对齐

传统异常检测的基本逻辑是：

```text
训练集正常样本 -> 建模 normality
测试图像 -> 判断是否符合 learned normality
```

这个逻辑在 single-class 且训练数据充分时通常有效。但在更现实的场景里，它会遇到 **misaligned normality**。

few-shot 场景中，正常样本很少：

```text
少量 hazelnut 正常图
  -> 预存 prototypes
测试 hazelnut 的角度、纹理、位置不同
  -> prototypes 覆盖不全
```

multi-class 场景中，类别更多：

```text
某类正常背景
可能像另一类异常区域
```

因此，训练集里的 normal prototypes 不一定是当前测试图最合适的参照。作者的关键观察是：大多数工业异常是局部变化，异常图本身仍包含大量正常 patches。与其从训练集中找对齐的 normality，不如直接从当前图中提取。

这就是 INP 的出发点：

```text
外部 normality: 来自训练集，可能不对齐
内部 normality: 来自同一张测试图，更贴近当前上下文
```

## 总体框架

INP-Former 由四个模块组成：

![INP-Former 总体框架](/blog/images/inp-former/framework.jpg)

```text
image
  -> fixed pretrained Encoder Q
  -> multi-scale ViT features f_Q
  -> INP Extractor E 提取 M 个 INPs
  -> Bottleneck B 融合多层特征
  -> INP-Guided Decoder D 用 INPs 指导重建
  -> encoder-decoder reconstruction error 作为 anomaly score
```

默认实现：

```text
encoder: DINOv2-R ViT-Base/14
input: resize 448x448, center crop 392x392
INP number M: 6
decoder layers: 8
optimizer: StableAdamW
epochs: 200
gamma: 3.0
lambda: 0.2
```

论文继承了 Dinomaly 的 group-to-group feature reconstruction。附录里给出的分组是：

```text
ViT layers 3-6: group 1
ViT layers 7-10: group 2
```

异常图由 encoder feature group 和 decoder feature group 的 regional cosine distance 得到，图像级分数使用 anomaly map top 1% 的均值。

## INP Extractor：从当前图里聚合正常原型

INP Extractor 的输入是 encoder 的多层特征。论文先把感兴趣的层做 element-wise sum：

```text
F_Q = sum({f_Q^1, ..., f_Q^L})
```

然后引入 `M` 个 learnable tokens：

```text
T = {t_1, ..., t_M}
```

这些 learnable tokens 作为 query，当前图像的 patch features 作为 key/value，通过 cross attention 聚合出 `M` 个 INPs：

```text
Q = T W^Q
K = F_Q W^K
V = F_Q W^V

T' = Attention(Q, K, V) + T
P  = FFN(T') + T'
```

其中：

```text
P = {p_1, ..., p_M}
```

就是从当前图像中动态提取的内在正常原型。

这一步的关键不是简单挑出几个局部 patch，而是让 learnable tokens 以全局视角聚合当前图像的正常语义。它有点像用少量查询 token 给整张图做“正常成分摘要”。

## INP Coherence Loss：避免 INP 抓到异常

INP 是从测试图自身提取的，这带来一个风险：如果图里有异常，INP Extractor 会不会把异常也提成 prototype？

论文用 **INP coherence loss** 约束 INPs 表示正常特征。训练时只有正常图，所以它让每个正常 patch feature 都能被最近的 INP 表示：

```text
d_i = min_m S(F_Q(i), p_m)
L_c = (1 / N) sum_i d_i
```

这里 `S` 是 cosine distance。直觉是：

```text
正常训练图里的所有 normal tokens
都应该能被少量 INPs 覆盖
```

这样训练出来的 INP Extractor 在测试时更倾向于提取 coherent normality，而不是被异常局部吸走。

论文用距离图展示了 `L_c` 的作用：

![INP coherence loss 的影响](/blog/images/inp-former/coherence_loss.jpg)

不过这里要注意一个细节：`L_c` 本身并不知道测试时哪里异常。它能起作用，是因为训练阶段只见正常图，INP tokens 学到的是“如何从图里聚合正常模式”。

## INP-Guided Decoder：用正常原型指导重建

只拿 patch feature 到最近 INP 的距离，也可以做 anomaly map。但论文指出，少量离散 INPs 很难覆盖所有低代表性的正常区域，直接距离图会有噪声。

于是作者把 INPs 放进 reconstruction 框架，让 decoder 通过多个 INPs 的组合重建正常区域。

普通 self attention 是：

```text
patch tokens as Q, K, V
```

INP-Guided Attention 改成：

```text
Q: decoder patch tokens
K,V: INPs
```

公式是：

```text
Q_l = f_D^{l-1} W_l^Q
K_l = P W_l^K
V_l = P W_l^V

A_l = ReLU(Q_l K_l^T)
f_D' = A_l V_l
f_D^l = FFN(f_D') + f_D'
```

这个设计非常关键：

```text
decoder 输出是 INPs 的线性组合
INPs 被约束为正常原型
因此 decoder 更倾向输出正常特征
```

如果输入 query 来自异常区域，decoder 也只能从正常 INPs 中取信息来重建，于是异常特征难以被原样复制。这是它缓解 identical mapping 的主要机制。

论文还去掉了 decoder 中第一条 residual connection。原因是原始异常 query 如果通过 residual 直接传下去，会把异常特征绕过 INP bottleneck，重新引入复制路径。

从复杂度看，INP-Guided Attention 也更轻：

```text
Vanilla Self Attention: O(N^2 C), memory O(N^2)
INP-Guided Attention:  O(N M C), memory O(N M)
```

因为 `M << N`，默认 `M = 6`，而 392x392、patch size 14 时 `N = 28 x 28 = 784`。注意力矩阵从 `N x N` 变成 `N x M`，这也是 INP-Former 比普通 decoder attention 更省的原因。

## Soft Mining Loss：让模型关注难重建区域

INP-Former 的训练目标包含：

```text
L_total = L_sm + lambda L_c
```

其中 `L_sm` 是 Soft Mining Loss。它借鉴 Focal Loss 和 Dinomaly 的思想：不同区域重建难度不同，训练时应该更关注难优化的正常区域。

论文用某个位置的 reconstruction error 相对 batch 平均 error 的比例来估计难度：

```text
w^l(h,w) = [ M^l(h,w) / u(M^l) ]^gamma
```

其中：

```text
M^l(h,w): 第 l 层在位置 (h,w) 的 regional cosine distance
u(M^l): batch 内平均 regional cosine distance
gamma: 温度超参数，默认 3.0
```

然后它不是直接 reweight loss，而是调整 decoder feature 的梯度：

```text
f_D_hat^l(h,w) = cg(f_D^l(h,w))_{w^l(h,w)}
```

作者这样做是为了保留 feature point manifold 的整体结构，同时让难重建区域获得更大的优化关注。

论文用 KDE 展示了 soft mining 的效果：加入 `L_sm` 后，正常像素和异常像素的 anomaly score 分布重叠更小。

![Soft Mining Loss 在 chewing-gum 类别上的影响](/blog/images/inp-former/soft_mining_chewing.jpg)

![Soft Mining Loss 在 cashew 类别上的影响](/blog/images/inp-former/soft_mining_cashew.jpg)

## 主实验：Multi-Class AD

在 multi-class anomaly detection 上，INP-Former 与 RD4AD、UniAD、SimpleNet、DeSTSeg、DiAD、MambaAD、Dinomaly 等方法比较。

核心结果如下：

| 方法 | MVTec-AD Image | MVTec-AD Pixel | VisA Image | VisA Pixel | Real-IAD Image | Real-IAD Pixel |
|---|---|---|---|---|---|---|
| MambaAD | 98.6/99.6/97.8 | 97.7/56.3/59.2/93.1 | 94.3/94.5/89.4 | 98.5/39.4/44.0/91.0 | 86.3/84.6/77.0 | 98.5/33.0/38.7/90.5 |
| Dinomaly | 99.6/99.8/99.0 | 98.4/69.3/69.2/94.8 | 98.7/98.9/96.2 | 98.7/53.2/55.7/94.5 | 89.3/86.8/80.2 | 98.8/42.8/47.1/93.9 |
| INP-Former | 99.7/99.9/99.2 | 98.5/71.0/69.7/94.9 | 98.9/99.0/96.6 | 98.9/51.2/54.7/94.4 | 90.5/88.1/81.5 | 99.0/47.5/50.3/95.0 |

指标顺序：

```text
Image: AUROC / AP / F1-max
Pixel: AUROC / AP / F1-max / AUPRO
```

最值得看的不是 MVTec-AD 图像级，因为已经接近饱和，而是 Real-IAD。Real-IAD 更复杂，包含 30 个对象和多视角，INP-Former 相比 Dinomaly 在 image-level 和 pixel-level 都有比较明确的提升。

论文的定性结果如下：

![INP-Former 在多类别异常检测中的定位结果](/blog/images/inp-former/qualitative.jpg)

## Few-Shot 与 Single-Class

INP-Former 自称 universal AD solution，因为它不仅做 multi-class，也在 few-shot 和 single-class 上验证。

few-shot 设定下，传统 prototype 方法特别容易受正常样本覆盖不足影响。INP-Former 的优势正好来自它不完全依赖 few-shot normal references，而是从测试图本身提取 INPs。

4-shot 结果中，INP-Former 在三个数据集上整体领先：

| 数据集 | Image-level | Pixel-level |
|---|---|---|
| MVTec-AD | 97.6/98.6/97.0 | 97.0/65.9/65.6/92.9 |
| VisA | 96.4/96.0/93.0 | 97.7/49.3/54.3/93.1 |
| Real-IAD | 76.7/72.3/71.7 | 97.3/32.2/36.7/89.0 |

single-class 设定中，它在 MVTec-AD 和 Real-IAD 上达到新的强结果，在 VisA 上也保持竞争力。这个结果说明 INP 并不是只为 multi-class 设计，它对常规单类建模也有帮助。

## 消融实验

论文消融了三个核心部分：

```text
INP: INP Extractor + INP-Guided Decoder
L_c: INP Coherence Loss
L_sm: Soft Mining Loss
```

MVTec-AD 和 VisA 上的结果可以压缩为：

| 组合 | MVTec Image | MVTec Pixel | VisA Image | VisA Pixel |
|---|---|---|---|---|
| baseline | 98.59/99.18/97.63 | 97.19/61.73/62.94/92.73 | 96.58/97.18/92.89 | 97.50/47.24/51.90/82.85 |
| + INP | 99.53/99.80/98.81 | 98.32/69.82/69.38/94.69 | 98.11/98.23/95.22 | 98.41/50.34/54.23/93.63 |
| + INP + L_c | 99.61/99.83/99.02 | 98.39/70.01/69.53/95.10 | 98.16/98.30/95.47 | 98.46/51.09/54.46/93.71 |
| + INP + L_c + L_sm | 99.67/99.88/99.20 | 98.48/71.02/69.65/94.87 | 98.90/99.02/96.57 | 98.90/51.22/54.74/94.36 |

读法很清楚：

- `INP` 本身贡献最大，说明从当前图提取 prototypes 并指导 decoder 是主要增益来源。
- `L_c` 进一步提高 INP 的正常一致性，减少 INP 捕获异常的风险。
- `L_sm` 继续拉开正常和异常区域的分数分布，尤其提升 VisA image-level。

## INP 数量 M

论文默认 `M = 6`。数量太少时，INPs 不能覆盖足够多的正常模式；数量太多时，INPs 可能开始吸收异常 token 信息，性能反而略降。

![INP 数量 M 的影响](/blog/images/inp-former/inp_count.jpg)

作者的结论是：当 `M > 4` 后性能基本稳定，因此选择 6 个 INPs 作为默认配置。

这个数字也强化了论文的一个主张：一张图的正常模式可以被少量 compact prototypes 表示。

## INP 可视化与 Zero-Shot 能力

论文可视化了 6 个 INPs 的 attention maps。它们会关注不同正常区域，例如物体区域、边缘、背景等。

![INP attention maps 可视化](/blog/images/inp-former/inp_visual.jpg)

这说明 INPs 不是 6 个重复的平均特征，而是在分工覆盖不同类型的 normality。

更有意思的是 zero-shot 实验：INP-Former 在 Real-IAD 上训练，然后测试 MVTec-AD 的未见类别，仍能提取 INPs 并形成可用的 distance map。

![Zero-shot anomaly detection：Real-IAD 训练，MVTec-AD 测试](/blog/images/inp-former/zero_shot.jpg)

论文报告在未专门为 zero-shot 训练的情况下，它能在 MVTec-AD 和 VisA 上取得 88.0 和 88.7 的 pixel-level AUROC，并超过指定 zero-shot 方法 WinCLIP 的部分结果。

这里的核心原因仍然是：INP Extractor 学到的是“如何从单张图中找 normality”，而不是记住某个固定类别的 prototypes。

## 局限

论文明确提到一个局限：当逻辑异常和背景分布非常相似时，INP Extractor 可能错误地把异常区域也提成 INPs。

典型例子是 MVTec-AD Transistor 类里的 misplaced anomaly。某些错位区域在视觉上接近背景，INP Extractor 可能将其视为正常原型，导致检测失败。

这说明 INP-Former 强在图像内部对齐，但也有天然风险：

```text
如果异常本身很像图内正常背景，
从图内提取 normality 可能会把异常也吸进去。
```

作者未来方向是把 INPs 与 pre-stored prototypes 结合：

```text
INPs: 当前图像内部对齐强
pre-stored prototypes: 训练集语义覆盖更完整
```

这也是论文对自身方法边界的清醒补充。

## 和 Dinomaly 的关系

INP-Former 和 Dinomaly 不是对立关系。更准确地说，INP-Former 是在 Dinomaly 的重建范式上引入图像内部 prototypes。

```text
Dinomaly:
强 ViT feature + 限制 decoder 复制能力

INP-Former:
强 ViT feature + 当前图像 INPs + INP-guided reconstruction
```

Dinomaly 主要解决：

```text
decoder 不要把异常也重建好
```

INP-Former 进一步解决：

```text
用什么 normality 来指导当前图像的正常重建
```

它的答案是：优先利用测试图像内部的正常区域。

## 论文结论

INP-Former 的贡献可以压成三句话：

```text
1. 单张测试图像内部存在可用的正常原型 INPs。
2. 用 learnable tokens + cross attention 可以动态提取这些 INPs。
3. 用 INPs 作为 decoder 的 key/value，可以让重建输出更偏向正常模式，从而放大异常区域的重建误差。
```

最值得记住的是它对 normality 来源的重新定义：

```text
异常检测不一定只能依赖训练集 normality。
当前测试图像自身也包含可利用的 normality。
```

这就是 INP-Former 相比普通 prototype / memory bank / reconstruction 方法最有辨识度的地方。
