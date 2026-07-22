---
author: Ludovico
pubDatetime: 2026-07-22T16:00:00+08:00
title: "ViT PTQ 论文与代码地图：从 PTQ4ViT、FQ-ViT、RepQ-ViT 到 RegCache，再落到 DINOv3"
featured: false
draft: false
tags:
  - 论文精读
  - 量化
  - Vision Transformer
  - DINOv3
description: "一张以问题为线索的 ViT 训练后量化地图：各论文实际解决什么、官方仓库能复现什么，以及为什么 DINOv3 的 learned storage token 需要不同的工程路径。"
---

ViT 的 W8A8 不能只看一份 ImageNet 表格。不同论文处理的是不同的数值故障：GELU/Softmax 的非均匀分布、LayerNorm 的通道尺度差、线性层输入的 channel outlier，或者视觉 encoder 中少数 token 的超大范数。把它们都叫作“ViT PTQ”会掩盖关键的适用条件。

本文按问题而非发表时间梳理公开论文和官方仓库，并将其与 `research` 里的 DINOv3 ViT-B/16 ONNX 实验对照。链接和仓库在 2026-07-22 逐一核对；论文中的精度均是作者在其任务、模型和后端下的结果，不能横向当作同一基准。

## 先给选择结论

| 你遇到的主要问题                               | 优先读/试                | 为什么                                                       | 对现有 DINOv3 的判断                                      |
| ---------------------------------------------- | ------------------------ | ------------------------------------------------------------ | --------------------------------------------------------- |
| GELU、Softmax 的长尾和不对称，想做通用静态 PTQ | PTQ4ViT                  | Twin Uniform Quantization (TUQ) 和 Hessian 引导的 scale 搜索 | 有价值，但没有处理 token group 共用一个 activation scale  |
| 要把 LayerNorm、Softmax 也放入整数图           | FQ-ViT                   | PTF LayerNorm 和 Log-Int-Softmax (LIS) 是算子级数值格式      | 方向正确；普通 QNN primitive 图需要等价融合/内核支持      |
| LayerNorm 输出的通道尺度不平衡                 | RepQ-ViT                 | 等价 scale reparameterization，把难度移到权重                | 已移植到本地 ONNX，适合与其他办法叠加                     |
| Linear 输入有 channel outlier                  | SmoothQuant              | 以等价变换在 activation 与 weight 之间重分配范围             | 已在 48 个 DINOv3 projection 上试验，统一 A8 明显改善     |
| 少数图像 patch 成为中层 sink/outlier           | RegCache                 | 缓存 KV prefix，并删除残留 patch sink                        | 对本模型不直接成立：最大 outlier 是训练好的 storage token |
| 训练好的 register 与 patch 共用一把 A8 尺      | 本地 register-aware 分支 | CLS/register/patch 分开编码，register 用 A16                 | 当前最接近模型结构的方案，但尚不是全图整数部署证明        |

这里的最后一行是本文最重要的结论：**论文方法先解决它的假设所描述的问题；先 profile token 与算子边界，再选择方法。**

## 论文与仓库索引

| 工作                                      | 核心机制                                       | 论文/官方实现                                                                                                    | 代码实际覆盖                                                               | 不能直接承诺的事                                                             |
| ----------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| PTQ4ViT (2021)                            | TUQ + Hessian-guided scale search              | [论文](https://arxiv.org/abs/2111.12293) / [hahnyuan/PTQ4ViT](https://github.com/hahnyuan/PTQ4ViT)               | ViT、DeiT、Swin 的 PyTorch/ImageNet PTQ；README 给出 `example/test_all.py` | 不会自动导出任意 ONNX/QNN 图，也不解决运行时 token-axis 分组                 |
| FQ-ViT (2021)                             | PTF LayerNorm + LIS Softmax                    | [论文](https://arxiv.org/abs/2111.13824) / [megvii-research/FQ-ViT](https://github.com/megvii-research/FQ-ViT)   | 全量化 ViT 的 PyTorch 实验；可用 `--ptf --lis` 运行                        | PTF/LIS 需后端有相应整数算子，并非 ONNX `Softmax` 自动变成整数实现           |
| RepQ-ViT (2023)                           | LayerNorm/Softmax scale reparameterization     | [论文](https://arxiv.org/abs/2212.08254) / [zkkli/RepQ-ViT](https://github.com/zkkli/RepQ-ViT)                   | 官方 PyTorch 分类与检测目录                                                | 不会消除 token 间的整体范数差；等价变换也会移动 weight 的量化负担            |
| SmoothQuant (2022)                        | channel-wise smoothing                         | [论文](https://arxiv.org/abs/2211.10438) / [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) | 原仓库重点是 LLM；核心等式可迁移到 Linear/MatMul                           | 不是专为 ViT token outlier 提出的，alpha 必须在目标图上校准                  |
| RegCache (2025)                           | middle-layer cached KV prefix + token deletion | [论文](https://arxiv.org/abs/2510.04547)                                                                         | 论文宣称可叠加 PTQ4ViT、RepQ-ViT 等                                        | prefix 与删除依赖“可牺牲的 patch sink”假设；不能机械套到已有 register 的模型 |
| Vision Transformers Need Registers (2024) | 训练时加入 register token                      | [论文](https://arxiv.org/abs/2309.16588) / [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | DINOv2 `_reg` 模型与训练实现                                               | 是架构/训练方案，不是给现成 checkpoint 的 PTQ 补丁                           |

RegCache 截至本文核对时只列论文页，没有在论文中给出可核验的官方 GitHub 链接。因此不把第三方同名仓库称为“官方实现”；可复现的本地 ONNX 探索另行说明。

## 1. PTQ4ViT：先承认非线性不是高斯分布

PTQ4ViT 的出发点很朴素：ViT 的 Softmax 输出和 GELU 输出不适合用一个普通均匀量化区间描述；而只以 activation MSE/cosine 选 scale，未必与网络输出误差相关。它因此做两件事：

- **TUQ**：把一个 activation 区间拆为两个均匀子区间，以更细的区间留给密集区域；
- **Hessian-guided metric**：用二阶敏感性近似给候选 scale 打分，而非只量单层 reconstruction error。

官方 README 的一个实践信息常被忽略：它使用并行（parallel）而不是逐层串行量化，称小校准集上更稳定；表中 32 与 128 张 ImageNet calibration 的 W8A8 结果接近。这是其框架的经验结论，不应推导为“任何新 ViT 用 32 张图都够”。

它适合把 `MatMul/Linear`、GELU、Softmax 作为量化对象、并可控制 PyTorch 模型的研究环境。若部署约束是 QNN/ONNX 的标准量化算子，应先确认 TUQ 的双区间编码如何落在后端，而不是只复用校准脚本。

## 2. FQ-ViT：整数化的难点在 LayerNorm 和 Softmax

许多早期 ViT PTQ 为保精度让 LayerNorm、Softmax 留在浮点；FQ-ViT 明确把这两个“空白区”作为方法本体。

LayerNorm 输入存在严重的 **inter-channel variation**。逐 channel scale 过重，纯 tensor scale 又太粗；PTF 让各 channel 选择 2 的幂次因子，借助位移逼近多尺度。Softmax 则是“多数很小、极少接近 1”的分布，LIS 以对数整数格式把更多码字留给小值区间，并用整数近似实现相关运算。

这篇论文对端侧的启发不是“把 `Softmax` 打上 int8 标记”。真正的要求是：量化图必须实现 PTF 的逐通道 shift 和 LIS 的数值语义。没有融合内核时，Q/DQ 包围一个浮点 Softmax 并不等于 FQ-ViT。

## 3. RepQ-ViT 与 SmoothQuant：等价重参数化是可迁移的工具

两者都不训练模型，利用线性层的等价变换改变量化时看见的分布。对 `Y = XW + b`，取逐通道正 scale `s`，可写成：

\[
Y=(X / s)\,(\operatorname{diag}(s)W)+b.
\]

SmoothQuant 用 activation maxima 与 weight maxima、以及 alpha 计算 `s`，在二者间分摊难度。RepQ-ViT 更针对 ViT：对 post-LayerNorm 的逐 channel 变化做 scale/shift 重参数化，并对 post-Softmax 处理极端分布。二者的价值是 FP32 图保持等价，因此可以把“改模型”变成可验证的图重写。

但它们有清晰边界：如果 tensor `[B, T, C]` 中仅四个 token 的整体范数比其余 token 高数百倍，channel scaling 不会让这四个 token 自动拥有另一把 A8 尺。它能改善每个 token 的 channel 不平衡，却不能表示 token-group 的不同动态范围。

## 4. RegCache：把 outlier 当作模型占用的工作空间

RegCache 是这条脉络中最新且最容易被误用的一篇。它发现 CLIP、SigLIP、DINOv2 等视觉 encoder 的量化敏感 outlier 常出现在中间层的少数背景 patch；这些 token 跨图像的表示相似，像模型临时占用的工作寄存器。

方法可记为三个阶段：

```text
Curate: 在参考图像中定位敏感层和高范数 token
Cache:   平均这些 token 的未量化 K/V，从中层起作为 attention prefix
Delete:  在敏感层删除仍在当前图像内形成的 patch sink
```

它不是“给 ViT 多拼几个 token”这么简单。只 cache 可能仍留下原始 sink；只删除会伤害模型。论文的关键是二者联用，并为不同 encoder 搜索敏感层、prefix 数和删除数。它特别适合不能重训、而 outlier 的确是语义稀薄 patch 的视觉 encoder；对 dense task，token deletion 还必须恢复空间位置或单独验证。

与其紧密相连的《Vision Transformers Need Registers》则更早在训练期给模型无语义的 register token，让它不必挪用 image patch 做内部计算。前者是已有 checkpoint 的测试时补救，后者是架构训练选择，二者不是同一方法。

## 5. 用 DINOv3 实验校验适用性，而不是照搬论文名字

本地研究模型为 DINOv3 ViT-B/16 ONNX，序列由 `1 CLS + 4 storage/register + 64 patch` 组成。以相同前处理、64 张 calibration 与 64 张 evaluation 做 encoder 输出 reconstruction（不是分类 Top-1）后，结论如下：

| 配置                                        | Mean cosine | 说明                                             |
| ------------------------------------------- | ----------: | ------------------------------------------------ |
| W8 weight-only                              |     0.99950 | 权重量化基本不是瓶颈                             |
| 默认全图 W8A8 MinMax                        |    -0.00788 | 全图 primitive activation fake-quant 失效        |
| 线性路径统一 A8                             |     0.80623 | 将非线性/残差边界排除后仍有严重 token range 问题 |
| SmoothQuant alpha=0.5，统一 A8              |     0.98517 | channel smoothing 显著有效，但仍不是最佳         |
| CLS/register/patch 三支路独立 A8 scale      |     0.99333 | 直接证明主矛盾是 token group 的 scale 污染       |
| 三支路，register A16，SmoothQuant alpha=0.5 |     0.99613 | 当前线性路径 QuantSim 候选                       |

profile 定位到 block 2：register token 2/1 的 FC2 输入平均 L-infinity 为 `804.45/666.67`，patch 平均约 `2.80`。这与 RegCache 的典型前提不同：异常值不是可删背景 patch，而是模型已经训练出的 `storage_tokens`。

本地 `build_regcache_model.py` 因此做了反证。cache-only 的 FP32 近似尚可（8 图 mean cosine `0.99765`），但线性 A8 重建仍只有 `0.86972`；若删除 block 2 的 register，FP32 cosine 已降至 `0.81647`。原因并不神秘：原 register 仍在 residual/MLP 流中污染 scale；删除它又删掉了模型确实依赖的状态。**这不是 RegCache 论文失败，而是模型结构不满足其删除假设。**

对应的 ONNX 工具在 `research` 中：`smoothquant_dinov3.py` 重写 48 个 QKV/proj/FC1/FC2 MatMul，`repq_vit_onnx.py` 重写 24 个 LayerNorm-to-linear 边界，`split_register_linear_paths.py` 将每个 projection 拆为 CLS/register/patch 支路。所有图重写都应先检查 FP32 等价，再做量化比较。

## 6. 面向 QNN 的实际落地顺序

1. **先测 FP32 token profile 与逐层量化敏感性。** 区分 channel outlier、非线性分布和 token outlier，记录 calibration/evaluation 划分。
2. **用可导出的等价变换处理 channel 问题。** SmoothQuant/RepQ 类重写先验证 ONNX checker 和 FP32 reconstruction；alpha/截断不能跨模型照抄。
3. **为 token-group 问题选择结构策略。** patch sink 可评估 RegCache；已有且必要的 register 则应隔离 scale/精度，而不是删除。
4. **单独实现 LayerNorm、Softmax、RoPE、residual Add 和 Concat 的整数语义。** 线性路径高 cosine 不是全图定点精度；这些边界的 requantization 仍可主导误差。
5. **最后在目标 HTP 验证。** 当前 QNN conversion 已能接受 48 个 register A16 encoding override，但这只说明 converter 表达了边界；必须检查 V68 的实际 placement、A8/A16 Convert 和设备端输出。

## 参考与复现入口

- [PTQ4ViT 论文](https://arxiv.org/abs/2111.12293) 与 [官方仓库](https://github.com/hahnyuan/PTQ4ViT)
- [FQ-ViT 论文](https://arxiv.org/abs/2111.13824) 与 [官方仓库](https://github.com/megvii-research/FQ-ViT)
- [RepQ-ViT 论文](https://arxiv.org/abs/2212.08254) 与 [官方仓库](https://github.com/zkkli/RepQ-ViT)
- [SmoothQuant 论文](https://arxiv.org/abs/2211.10438) 与 [官方仓库](https://github.com/mit-han-lab/smoothquant)
- [RegCache 论文](https://arxiv.org/abs/2510.04547)；本文不将未被论文链接核验的仓库列为官方代码
- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) 与 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)
- 本地实验细节：`research/deliverables/dinov3_w8a8_register_a16_reproduction/docs/ABLATION.md` 与 `research/EXPERIMENTS.md`

真正可迁移的不是某个 observer 的名字，而是诊断顺序：先确定 outlier 是落在 channel、算子还是 token 身上，再让量化格式、图重写和硬件内核匹配同一个事实。
