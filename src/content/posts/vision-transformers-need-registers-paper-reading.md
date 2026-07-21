---
author: Ludovico
pubDatetime: 2026-07-21T16:00:00+08:00
title: "[论文精读] Vision Transformers Need Registers：给 ViT 加几个工作寄存器"
featured: false
draft: false
tags:
  - 论文精读
  - Vision Transformer
  - 自监督学习
description: Vision Transformers Need Registers 解释 ViT 中背景高范数 token 的来源，并用额外的 register tokens 把模型的隐式工作空间显式化。
---

论文：**Vision Transformers Need Registers**，ICLR 2024。

这篇论文讲了一个很有意思、也很容易用工程直觉理解的问题：Vision Transformer 在处理图片时，会把一些本来代表背景 patch 的 token 变成异常高范数的 token。它们经常出现在天空、墙面、地面这类低信息区域，看起来像特征图里的几个“亮点”。

作者的解释是：这些 token 被模型借去做中间计算了。模型缺少专门的临时存储位置，于是把不太重要的背景 patch 当成了工作空间。

所以，用户说它们像“垃圾桶”，方向是对的，但更准确的说法是：**背景 patch 被临时当成了工作寄存器**。它们不是无意义地丢垃圾，而是在保存模型内部计算需要的中间状态；真正的问题是，这些中间状态污染了本该表示图像内容的 token。

## 论文信息

| 字段 | 内容 |
| --- | --- |
| 标题 | Vision Transformers Need Registers |
| 作者 | Timothée Darcet、Maxime Oquab、Julien Mairal、Piotr Bojanowski |
| 会议 | ICLR 2024 |
| 研究对象 | supervised 和 self-supervised ViT |
| 代码 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| 论文 | [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) |

一句话总结：**给 ViT 增加几个不对应图像位置的可学习 token，让模型把中间计算放到专用位置，从而清理 patch feature 和 attention map。**

## 先看现象：背景里的高范数 token

标准 ViT 把图片切成 patch，再把每个 patch 映射成 token：

```text
image
  -> image patches
  -> patch tokens
  -> Transformer blocks
  -> visual representation
```

理想情况下，一个 patch token 应该主要描述对应的图像区域。但作者观察到，很多预训练 ViT 会产生一些异常 token：

```text
token norm 很高
位置却在背景区域
语义上没有明显物体
```

把 token 的范数画回图像空间后，就会看到几个稳定的热点。这些热点不是输入图像本身的显著目标，也不是简单的噪声。它们更像是模型在计算过程中主动制造出来的“特殊位置”。

这会影响下游视觉任务。分类只需要一个全局表示，模型可能还能正常工作；但目标分割、深度估计、目标发现等 dense prediction 任务需要每个 patch 都保持干净的空间语义，这些异常 token 就会变得明显。

论文统计了正常 patch 和 artifact patch 的范数分布。异常位置的长尾非常明显，这也是“背景里冒出几个尖峰”的定量证据。

![正常 patch 与 artifact patch 的 token 范数分布](/blog/images/vision-transformers-registers/token-norm-histogram.png)

## 为什么模型会借用背景 patch

Transformer 的 self-attention 允许任意 token 互相读写。某个 token 不一定只保存“它所在 patch 的信息”，也可以成为全局信息交换的中转站。

如果序列中没有专门的工作空间，模型可能会选择低信息背景 patch 作为存储位置：

```text
正常背景 patch
  -> 被模型选中
  -> 范数被放大
  -> 承担中间计算
  -> 不再纯粹表示原始背景
```

这就是论文里 register 的动机。这里的 register 可以类比为 CPU 里的寄存器：它不代表外部输入中的某个实体，而是给计算过程提供临时存储。

不过这个类比需要加一个限定：ViT 中的高范数 patch 并不是硬件寄存器，也不是一个预先定义好的离散缓存。它是模型在训练后自己形成的行为。论文做的是提供更合适的 token，让这种行为有专门的落脚点。

## 方法：增加 register tokens

普通 ViT 的输入序列大致是：

```text
[CLS] + patch_1 + patch_2 + ... + patch_N
```

加入 registers 后变成：

```text
[CLS] + register_1 + ... + register_R + patch_1 + ... + patch_N
```

register token 是额外的可学习 token。它们没有对应的图像区域，也不参与最终的空间特征图。Transformer 可以通过 attention 读取和更新它们，把它们作为中间计算的工作区。

关键点是：**register 不负责识别某个物体，也不是新的类别 token。它只是把模型原本隐式使用的工作空间显式提供出来。**

下图是论文对输入序列的示意：register tokens 插在特殊 token 和图像 patch tokens 之间，训练时和 Transformer 一起参与计算，输出空间特征时则可以把它们排除。

![register tokens 在 ViT 输入序列中的位置](/blog/images/vision-transformers-registers/register-placement.png)

推理时，做 dense prediction 通常只取 patch tokens，去掉 `[CLS]` 和 register tokens，再把 patch token 还原到二维网格。这样输出的空间特征就不会被那些“借来的背景位置”污染。

## 它不是简单地多加几个 `[CLS]`

`[CLS]` token 和 register token 的用途不同。

`[CLS]` 通常承担全局聚合任务，最后用于分类或图像级表示；register tokens 则是 Transformer 内部可以反复读写的工作空间。它们不需要对应一个最终输出，也不需要代表整张图的语义。

可以粗略地这样区分：

```text
[CLS]    -> 汇总整张图，服务于全局任务
register -> 保存中间计算，服务于模型内部
patch    -> 表示图像空间中的局部内容
```

这也是为什么 register token 最终可以从 dense feature map 中排除：它们从一开始就不是空间采样点。

## 实验观察

作者在监督和自监督 ViT 上都观察到了类似现象，并验证了加入 registers 的效果。

主要结果可以归纳为四点。

### 1. 高范数背景 token 消失

加入 registers 后，原来集中在背景位置的高范数 token 大幅减少。模型不再需要把天空、墙面或地面上的 patch 强行改造成工作空间。

### 2. 特征图更平滑

patch feature 的空间分布更加连续，局部区域之间的变化更符合图像内容，而不是被几个异常热点打断。

### 3. 注意力图更自然

没有 registers 时，部分 attention head 会在背景中形成非常突出的异常位置。增加 registers 后，一部分原本指向背景 patch 的注意力转移到了 register tokens 上。

### 4. Dense prediction 更好

作者在语义分割、深度估计和目标发现等任务上验证了更干净的特征带来的收益。尤其是目标发现：当模型规模变大时，register 能让 object discovery 方法继续工作得更稳定。

这篇论文的重点不是宣称 registers 会让所有分类准确率都大幅上升，而是说明：**它修复了预训练视觉特征的结构性问题，让同一套特征更适合空间密集型下游任务。**

论文在多个 dense prediction 设置中比较了加入 registers 前后的结果。收益并不只是视觉上的特征图变干净，也会反映到分割和深度估计等下游指标上。

![加入 registers 后的下游任务结果](/blog/images/vision-transformers-registers/downstream-results.png)

## “垃圾桶”这个比喻到底对不对

可以把背景 token 看成垃圾桶，但最好分成两层意思：

```text
直观说法：模型把不重要的背景 patch 当成垃圾桶
准确说法：模型把背景 patch 当成临时工作寄存器
```

“垃圾桶”强调的是这些 token 不再忠实表示所在位置；“寄存器”强调的是它们实际上承担了有用的计算功能。后者更接近论文结论。

如果模型真的只是把无用噪声塞进背景 token，清理它们可能只需要正则化。但论文的做法是增加可用的 register tokens，这说明作者认为问题不是模型产生了多余计算，而是**模型有计算需求，却没有合适的存储位置**。

## 对模型使用者有什么意义

如果只做图像分类，异常 token 可能不容易暴露，因为分类主要依赖全局表示。但如果使用 DINO、DINOv2 或其他 ViT foundation model 做以下任务，就应该关注这个问题：

- 语义分割和实例分割；
- 深度估计；
- 特征匹配和图像检索；
- 目标发现和无监督目标定位；
- 任何需要把 token 还原为空间特征图的任务。

实际使用时，优先选择已经训练了 registers 的模型版本。不能简单地给一个已经训练好的普通 ViT 在推理阶段拼上几个随机 token，因为 register 的行为需要在训练过程中形成，新增 token 的参数和模型内部协作方式都需要被学习。

这也是后来很多视觉模型名称里出现 `Register` 或 `DINOv2 with registers` 的原因：它不是为了增加一个新的视觉模块，而是把 Transformer 原本隐式形成的工作空间显式化。

## 局限和边界

这篇论文并没有证明所有 ViT 都必须使用 registers，也没有说任何异常高范数 token 都是坏的。

首先，某些高范数 token 可能确实承载了有用的全局信息。是否应该移除它们，要看下游任务如何使用 feature。其次，register 数量和插入位置仍然是架构与训练配置的一部分，不是一个完全免费的推理补丁。最后，register 主要解决的是 token 空间被内部计算污染的问题，它不能替代更好的数据、预训练目标或下游任务设计。

## 最后总结

这篇论文最值得记住的不是“加了几个 token”，而是它对 Transformer 行为的解释：

```text
ViT 不只是处理 patch token。
它还需要一个内部工作空间。
没有工作空间时，它会占用低信息的背景 patch。
给它 register tokens 后，空间 token 可以回归空间语义。
```

可视化上，加入 registers 后，patch feature 的空间响应也会更规律；原本集中在少数位置的异常响应被削弱，模型的内部工作量转移到 register token 上。

![加入 registers 前后的 patch feature 对比](/blog/images/vision-transformers-registers/feature-comparison.png)

所以，“垃圾桶”是一个很好的第一印象；从论文精度来说，它们更像是**被误用的内存位置**。register tokens 做的事情，就是给模型配了一组明确的临时寄存器。
