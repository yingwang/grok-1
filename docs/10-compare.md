# 第 10 章 与其他 MoE 对比

本章把 Grok-1 放进同代 MoE 的横向对照里看。覆盖 Mixtral 系、DeepSeek 系、Qwen-MoE。所有数字尽量来自各模型的公开技术报告或 HuggingFace config，未公开则注明。

本章面向已经读完前 9 章的读者：你已经了解 Grok-1 内部的具体张量流转，现在需要的是一份"地图"，把 Grok-1 的每一项设计放进 2024-2025 年 MoE 发展的整体坐标系。每节都先列对照表，再展开为什么有这样的差异、差异在工程上意味着什么。如果你只是想快速翻阅，10.1 和 10.13 两个汇总表已经覆盖核心事实；想理解"为什么是这样"，再按节往下读。

## 10.1 速查总表

| 模型 | 发布 | 总参 | 激活参 | 激活比 | 专家数 | top-k | 路由策略 | hidden | layers | head Q/KV | head dim | $d_{\text{ffn}}/\text{expert}$ | tokenizer 词表 | 上下文 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Grok-1** | 2024-03 | 314 B | ~86 B | 27.4% | 8 | 2 | token-choice, softmax→top-k, 不归一化 | 6144 | 64 | 48/8 | 128 | 32768 | 131072 | 8192 |
| Mixtral 8x7B | 2023-12 | 46.7 B | 12.9 B | 27.6% | 8 | 2 | token-choice, top-k→softmax, 归一化 | 4096 | 32 | 32/8 | 128 | 14336 | 32000 | 32768 |
| Mixtral 8x22B | 2024-04 | 141 B | 39 B | 27.7% | 8 | 2 | 同 8x7B | 6144 | 56 | 48/8 | 128 | 16384 | 32000 | 65536 |
| DeepSeek-V2 | 2024-05 | 236 B | 21 B | 8.9% | 160 + 2 shared | 6 | token-choice, fine-grained + shared expert | 5120 | 60 | 128/128 (MLA) | 128 | 1536 | 102400 | 128000 |
| DeepSeek-V3 | 2024-12 | 671 B | 37 B | 5.5% | 256 + 1 shared | 8 | token-choice, aux-loss-free | 7168 | 61 | 128/128 (MLA) | 128 | 2048 | 129280 | 131072 |
| Qwen1.5-MoE-A2.7B | 2024-03 | 14.3 B | 2.7 B | 19% | 60 + 4 shared | 4 | shared + routed, fine-grained | 2048 | 24 | 16/16 | 128 | 1408 | 151936 | 8192 |
| Qwen2-57B-A14B | 2024-06 | 57.4 B | 14 B | 24.4% | 64 + 8 shared | 8 | shared + routed | 3584 | 28 | 28/4 | 128 | 2560 | 151936 | 65536 |
| Qwen3-235B-A22B | 2025-04 | 235 B | 22 B | 9.4% | 128 | 8 | token-choice | 4096 | 94 | 64/4 (GQA) | 128 | 1536 | 151936 | 128000 |

数字来源：

- Grok-1：本仓库 + 官方博客
- Mixtral：[Mixtral 论文](https://arxiv.org/abs/2401.04088) + HF config
- DeepSeek-V2/V3：[V2 论文](https://arxiv.org/abs/2405.04434)、[V3 报告](https://arxiv.org/abs/2412.19437)
- Qwen MoE：各自 HF 模型卡（Alibaba Cloud 报告）

!!! note "shared expert（共享专家）"
    标准 MoE 里每个 token 只走路由器挑出来的几个 expert，剩下的不参与；但有些通用知识（基础语法、常识词汇）几乎所有 token 都需要，强行让路由器在每个 token 上都挑同一个 expert 又浪费容量。DeepSeek-V2 引入 shared expert：在路由的 N 个 expert 之外，额外放 1~2 个"每个 token 都必走"的专家，路由的 expert 专攻特化能力。

    Grok-1 没有 shared expert - 8 个 expert top-2 全部走路由。DeepSeek-V2 是 160 routed + 2 shared，DeepSeek-V3 是 256 routed + 1 shared，Qwen-MoE 系列也用 shared expert。这是细粒度 MoE 路线的标配。

!!! note "fine-grained expert（细颗粒专家）"
    "细颗粒"与"胖"是同一根坐标轴的两端。胖专家路线（Grok / Mixtral）让单个 expert 的 FFN 中间维度保持在 1.4 万到 3.3 万的量级，每个 expert 自身就是一个"小型完整 FFN"；细颗粒路线（DeepSeek / Qwen-MoE）则把单个 expert 缩到中间维度 1.4K-2.5K，但同时把专家总数从 8 扩大到 60-256。

    总参不变的前提下，"专家少而胖"与"专家多而瘦"在数学上可以等价（参数预算只是被切成 8 块还是切成 256 块的差别），但路由的语义不同：top-2 of 8 一共只有 $\binom{8}{2}=28$ 种激活组合，top-8 of 256 有 $\binom{256}{8} \approx 4 \times 10^{14}$ 种激活组合。组合空间大几个数量级，意味着模型在不同 token 上可以走非常不同的路径，专家之间形成的"特征分工"也更细。代价是路由器输出维度变大、负载均衡更难、稀疏调度的 kernel 实现更复杂。

!!! note "MLA（Multi-head Latent Attention，多头潜在注意力）"
    传统 MHA 给每个 head 各存一份完整的 K 和 V 向量，KV cache 总大小与 head 数成正比；GQA 让多个 Q head 共享同一组 KV，把 cache 缩小到 $H_{kv}/H_q$ 倍。MLA 走得更彻底：每个 token 在 cache 里只存一个**低维 latent 向量**（DeepSeek-V2 用 512 维），实际算 attention 时再用一组解压投影矩阵把 latent 还原成每个 head 的 K 和 V。cache 占用只跟 latent 维度有关，与 head 数完全解耦。

    DeepSeek-V2 的 MLA 把 KV cache 压到 GQA 的约一半，DeepSeek-V3 沿用并进一步优化。代价是 attention 每一步要多算一次"latent → 各 head K/V"的解压，但解压矩阵很小，整体收益远大于代价。Grok-1 选择更传统的 GQA（48 Q : 8 KV），实现简单、与 LLaMA-2 / Mixtral 等同代模型的工具链一致，但在 8K 上下文之外的长序列推理上，cache 占用会成为明显瓶颈。

!!! note "token-choice 路由 vs expert-choice 路由"
    token-choice（token 选 expert）是最常见的写法：每个 token 让 router 打分，挑出分数最高的 k 个 expert 走。Grok-1、Mixtral、DeepSeek、Qwen-MoE 全部都是 token-choice。

    expert-choice（expert 选 token）反过来：让每个 expert 自己挑出"我最想处理的" T 个 token。这种写法天然保证每个 expert 拿到的 token 数都是 T，不需要任何 aux loss 就能做到完美负载均衡。Google 在 2022 年的 expert-choice 论文里提出过，少数 Pathways 内部模型使用，但因为它打破了"每个 token 的算法行为对称"这一前提（同一个 batch 里有些 token 可能完全不被任何 expert 选中），主流开源 MoE 都没有采用。本书后续仍以 token-choice 为默认假设。

### 10.1.1 怎么读这张表

如果只读本章一遍，速查表已经覆盖关键事实。但每个数字背后都有具体的设计动机，接下来逐节展开。

读法建议：

1. **横向比较激活比**：胖专家派（Grok / Mixtral）激活参占总参约 27%，细粒度派（DeepSeek / Qwen3-MoE）的激活比下降到 5-10%。这是 MoE 设计上最重要的分水岭，直接决定推理硬件需求与训练算法选择
2. **横向比较 head Q/KV**：注意 DeepSeek 系列的 128/128 是 MLA 而非传统 MHA，KV 在 cache 里被压缩到 latent 维度，因此与 Mixtral / Grok 的 GQA 数字不可直接对比
3. **横向比较 expert FFN 大小**：胖专家路线的单 expert 中间维度在 14K-33K，细粒度专家路线的中间维度只有 1.4K-2K，二者相差一个数量级，体现的是单 expert 是否需要"自成一个完整子模型"
4. **纵向看时间**：胖专家派集中出现在 2023-2024 年，细粒度路线从 2024 年 5 月 DeepSeek-V2 开始铺开。整个 MoE 领域用约一年时间，完成了"从 8 个胖专家到 256 个小专家"的范式转向

## 10.2 设计哲学：胖专家 vs 细粒度

把这些模型按"专家粒度"分成两派：

### 胖专家派（Mixtral / Grok-1）

- 8 个 expert，每 expert FFN 中间维度 14K-33K
- 每个 expert 是个"小完整模型"
- top-2 of 8 选择空间小（28 种组合）
- 优点：实现简单，路由开销小
- 缺点：专家粒度粗，难以精细分工

### 细粒度派（DeepSeek / Qwen-MoE）

- 60-256 个 expert，每 expert FFN 中间维度 1.4K-2K
- 每个 expert 是个"小特征提取器"
- top-6/8 of 64-256，选择空间大（数千-数十万种组合）
- 优点：组合多，专家专业化更细
- 缺点：路由开销大、需要更好的负载均衡机制（aux-loss-free / shared expert）

### Grok-1 站在胖专家派

Grok-1 是胖专家派里最极端的：单 expert 0.6 B 参数 - 比 LLaMA-2 7B 的 FFN 还大（LLaMA-2 7B 的 FFN 单层只有 0.1 B）。

这意味着：

1. **训练时单 expert 收敛快**（参数多，容量足）
2. **推理时激活参数高**（2 × 0.6 B × 64 = 76 B，加 attention 后 ~86 B）
3. **MoE 的"稀疏优势"不大**（86 / 314 = 27% 激活比，比 DeepSeek-V3 的 5.5% 高得多）

第 11 章会进一步分析：xAI 选择这条路线，可能与时间约束直接相关 - 2023 年中段细粒度 MoE 的工程实现尚不成熟，相关的负载均衡 loss 与稀疏调度技术也还在快速演进中，先用结构最简单、训练最可控的胖专家设计完成一次完整的 300 B 级训练，是更稳妥的工程选择。

### 10.2.1 一张图理解两条路线的几何含义

把"专家粒度"映射到几何空间里看会更直观。假设语义空间是一个高维球面，每个 token 在某一层的"语义身份"对应球面上的一个点。

- **胖专家路线**把球面切成 8 个大瓣，每个瓣由一个胖 expert 负责。Top-2 路由的含义是"这个 token 落在两个瓣的交界处，需要两个 expert 的混合"。瓣的边界粗、表达分辨率受限于 8 个瓣的几何关系
- **细颗粒路线**把球面切成 256 个小瓣，每个瓣由一个瘦 expert 负责。Top-8 路由意味着"用 8 个小瓣的线性组合来逼近这个 token 的语义点"。本质上更像分布式向量基（distributed basis）

这个视角下，细颗粒路线相当于把 MoE 从"硬分配"走向"软分解"：单个专家的语义角色变得不再明确，取而代之的是一组协作专家在共同表示语义。这与稠密 FFN 的内部行为（一组 neuron 的线性组合）更接近，因此细颗粒路线的稀疏 MoE 在结构上更"像"一个被人为加上稀疏约束的稠密网络。这也解释了为什么细颗粒 MoE 的训练动力学更平滑、负载更易均衡，因为它没有强迫 router 在 8 个语义大类之间做硬选择。

## 10.3 路由策略对比

| 模型 | softmax 位置 | gate 归一化 | aux loss | capacity | drop token | noise |
| --- | --- | --- | --- | --- | --- | --- |
| **Grok-1** | 在 top-k 之前 | 否 | 推理代码无（训练未知） | 字段定义但不用 | 否 | 否 |
| Mixtral 8x7B | 在 top-k 之前 | **是** | 训练时有 | 不用 | 否 | 否 |
| Mixtral 8x22B | 同 | 同 | 同 | 同 | 同 | 同 |
| DeepSeek-V2 | 在 top-k 之前 | 是 | **多种**（device-balanced + communication-balanced + expert-balanced） | 是 | 是 | 否 |
| DeepSeek-V3 | 在 top-k 之前 | 是 | **aux-loss-free** + bias term | 是 | 是 | 否 |
| Qwen-MoE | 在 top-k 之前 | 是 | aux loss | 是 | 是 | 否 |

### 10.3.1 Grok-1 vs Mixtral：归一化的差异

这是第 5 章重点讨论的。让我们再正式列一遍：

**Mixtral：**

```python
routing_weights = softmax(logits, dim=-1)            # 所有 8 个 expert 归一化
routing_weights, idx = top_k(routing_weights, k=2)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)  # 重新归一化
# gate sum = 1
```

**Grok-1：**

```python
routing_probs = softmax(routing_logits, axis=-1)     # 所有 8 个 expert 归一化
expert_gate, expert_index = top_k(routing_probs, k=2)
# expert_gate.sum < 1, 不再归一化
```

后果：

- Mixtral 的 expert 输出量级稳定，sandwich norm 不需要
- Grok-1 的 expert 输出量级随路由确定性变化，sandwich norm 的 post-RMSNorm 起兜底作用

### 10.3.2 aux-loss-free：DeepSeek-V3 的新方法

DeepSeek-V3 引入了"无辅助损失负载均衡"：

$$
\text{logit}_i = e_i^T h + b_i
$$

每个 expert 有一个**可学习的 bias** $b_i$，当某个 expert 被过度选用时，训练动态地降低它的 bias - 而不用任何额外 loss 项。这避免了 aux loss 与主 loss 的张力。

!!! note "aux loss（辅助损失）"
    标准 MoE 训练里，主 loss 是预测下一个 token 的交叉熵，但只用主 loss 训练会出现严重的"赢者通吃"现象：少数 expert 被反复挑中、其余 expert 完全得不到梯度更新，最终大部分专家退化成无用参数。aux loss（辅助损失）的目的是在主 loss 之外强加一项"负载均衡惩罚"，常见形式是计算"每个 expert 被选中的频率"与"每个 expert 在 softmax 后的平均权重"的乘积，鼓励两者都接近均匀分布。
    
    aux loss 的副作用是"分裂主 loss 的优化方向"：模型一方面要把 token 路由到最合适的 expert（语义最优），另一方面又要让负载平均（结构最优），两个目标常常打架，调权重很难。这就是 aux-loss-free 方案出现的动机 - 用直接调整 bias 的方式代替惩罚项，模型在训练过程中自动调整路由偏好以达到均衡，且不再扰动主 loss 的优化方向。

Grok-1 在训练时不可能使用这一方法，原因是其训练发生在 2023 年，而 DeepSeek-V3 的 aux-loss-free 方案直到 2024 年底才正式公开发表。Grok-1 训练阶段大概率沿用了当时的标准做法 - 传统辅助损失（load-balance loss）加上 capacity factor 控制 - 但开源仓库中的推理代码没有保留训练侧的负载均衡逻辑，因此外界无法从仓库直接验证其训练时的具体配置。

!!! note "capacity factor（容量因子）"
    capacity factor 是 GShard / Switch Transformer 时代留下的工程概念。假设一个 batch 里有 $N$ 个 token，按 top-k 路由理论上每个 expert 平均会拿到 $Nk/E$ 个 token。但实际分布几乎不可能是均匀的，某些 expert 拿到的 token 数会显著超过平均值。capacity factor $c$ 定义"每个 expert 最多接收 $c \cdot Nk/E$ 个 token"，超出部分的 token 会被**drop**（直接跳过当前层、走残差直连，等价于这一层对该 token 无效）。
    
    Grok-1 的 model.py 里定义了 capacity factor 字段（具体是 `model.py` 的 TransformerConfig），但推理代码并未启用任何 drop 逻辑，所有路由结果都被完整执行。这种"字段保留但不使用"的写法多半是从训练代码里直接搬过来的，反映了训练侧曾经使用过 capacity 控制。

## 10.4 专家结构对比

| 模型 | FFN 类型 | 激活函数 | 中间维度 |
| --- | --- | --- | --- |
| **Grok-1** | GeGLU | GELU | 32768 |
| Mixtral 8x7B | SwiGLU | SiLU | 14336 |
| Mixtral 8x22B | SwiGLU | SiLU | 16384 |
| DeepSeek-V2 routed | SwiGLU | SiLU | 1536 |
| DeepSeek-V2 shared | SwiGLU | SiLU | 1536 × 2 = 3072 |
| Qwen-MoE | SwiGLU | SiLU | 1408 / 2560 |

Grok-1 是唯一用 GeGLU 的（Gemma 也用 GeGLU，但 Gemma 是 dense）。SiLU vs GELU 在中间区段几乎相同，但尾部行为略有差异 - SiLU 在大负值有微小负输出，GELU 几乎 0。

!!! note "GeGLU vs SwiGLU"
    两者都是 GLU 家族变体，结构都是 $\text{down}(\text{act}(\text{up}_1(x)) \odot \text{up}_2(x))$，差别只在 act 选哪个：GeGLU 用 GELU $\big(x \cdot \Phi(x)\big)$，SwiGLU 用 SiLU $\big(x \cdot \sigma(x)\big)$，其中 $\Phi$ 是高斯累积分布函数、$\sigma$ 是 sigmoid。两者在主激活区域形状几乎一致，差异主要在尾部：SiLU 在大负值仍有微小负输出，GELU 在大负值几乎压到 0。
    
    Grok-1 的 expert FFN 中间维度 32768 是 Mixtral 8x22B 的 2 倍、DeepSeek-V2 routed expert 的 21 倍。从总参分配看，Grok-1 把绝大部分参数预算（约 309 B / 314 B）都投在 expert 的三个矩阵上，attention 部分相对很轻。这是胖专家路线的标志性几何特征。

实际训练的最终效果差异通常很小，可在 loss 曲线上的影响往往被其他超参的噪声盖过。但**ckpt 不能跨激活函数互换** - 把按 SiLU 训出来的权重直接载入用 GELU 的 graph，GLU 那一支的几何含义已经变了，必须重训而不能简单移植。

## 10.5 Attention 对比：GQA / MQA / MLA

| 模型 | Attention | head Q | head KV | 比例 | head dim |
| --- | --- | --- | --- | --- | --- |
| Grok-1 | GQA | 48 | 8 | 6:1 | 128 |
| Mixtral 8x7B | GQA | 32 | 8 | 4:1 | 128 |
| Mixtral 8x22B | GQA | 48 | 8 | 6:1 | 128 |
| LLaMA-2 70B | GQA | 64 | 8 | 8:1 | 128 |
| DeepSeek-V2 | **MLA** | 128 | 128 (KV compress) | - | 128 |
| DeepSeek-V3 | **MLA** | 128 | 128 (KV compress) | - | 128 |
| Qwen1.5-MoE | MHA | 16 | 16 | 1:1 | 128 |
| Qwen2-57B-A14B | GQA | 28 | 4 | 7:1 | 128 |

MLA（Multi-Head Latent Attention）是 DeepSeek-V2 引入的全新方法 - 把 KV 压缩到一个低维 latent space，大幅减少 KV cache 占用（比 GQA 还省）。这是 DeepSeek 系列的关键创新。

!!! note "MLA（Multi-Head Latent Attention）"
    MHA 给每个 head 存一份完整 K/V，GQA 让多个 Q head 共享一组 K/V 来省 cache，MLA 走得更狠：每个 token 只在 cache 里存一个低维 latent 向量（比如 512 维），实际算 attention 时再用一个投影矩阵把它"解压"回每个 head 的 K 和 V。所以 cache 占用只跟 latent 维度有关，跟 head 数无关。

    DeepSeek-V2/V3 用 MLA，latent 大约 512 维，相比 LLaMA-2 70B 那种 8 KV head × 128 dim = 1024 的 GQA cache 还能再省一倍以上。代价是 attention 算每一步多一次解压投影。Grok-1 用 GQA 48 Q : 8 KV（每个 token 在 cache 里存 8 × 128 = 1024 维 K + 同维 V），比 MLA 占用大，但实现简单，shard 起来直接。

Grok-1 与 Mixtral 选择的是 GQA：实现简单、运行可靠、ckpt 在多卡之间切分时只需要按 head 维直接拆分，不涉及任何额外的解压投影。这种方案虽然在 KV cache 占用上不如 MLA，但工程门槛低、与既有训练 / 推理代码兼容性最好。

### 10.5.1 KV cache 占用的实际计算

把三种方案在 Grok-1 这种规模（64 层、batch 1、序列 8192）下的 KV cache 占用算清楚，差异会变得非常具体。所有数字按 bf16 (每个值 2 字节) 计算。

- **MHA（假设 48 Q : 48 KV）**：$64 \times 8192 \times 2 \times 48 \times 128 \times 2 = 12.9 \text{ GB}$
- **GQA（Grok-1 实际，48 Q : 8 KV）**：$64 \times 8192 \times 2 \times 8 \times 128 \times 2 = 2.15 \text{ GB}$
- **MLA（按 latent = 512）**：$64 \times 8192 \times 512 \times 2 = 0.54 \text{ GB}$

GQA 在 MHA 的基础上节省了 6 倍 cache，MLA 在 GQA 基础上再节省约 4 倍。把这些数字代到第 7 章的显存账里：Grok-1 八卡部署中，权重占用约 314 GB（int8）、KV cache 单 batch 仅约 2 GB，看起来 cache 占比很小。但**当上下文从 8K 扩到 64K、batch 从 1 扩到 32 时**，cache 会涨到 $2.15 \times 8 \times 32 = 550\text{ GB}$，超过权重本身。这就是为什么 DeepSeek 系列在主打长上下文与高吞吐推理的产品形态下必须用 MLA - GQA 在这种场景下会让 cache 直接撑爆显存。

Grok-1 在 8K 上下文的硬限下，GQA 的 cache 占用问题被"按下不表"了；但反过来说，**正因为 cache 不可能放下更长上下文，xAI 也就没有必须切换到 MLA 的动力**。这是一种"短上下文 + 简单 cache 方案"自洽的设计闭环，但闭环之外的工程余地非常窄。

## 10.6 上下文长度

| 模型 | 上下文 | RoPE base |
| --- | --- | --- |
| Grok-1 | 8192 | 10000 |
| Mixtral 8x7B | 32768 | 1000000 |
| Mixtral 8x22B | 65536 | 1000000 |
| DeepSeek-V2 | 128000 | 10000 + YaRN |
| DeepSeek-V3 | 131072 | 10000 + YaRN |
| Qwen3-235B | 128000 | YaRN |

**Grok-1 的上下文窗口仅为 8K，是同代 MoE 中最短的**。可能的原因包括：

1. 训练阶段将 sequence length 限定在 8K：在 314 B 规模下，每倍增上下文长度都会带来近线性甚至超线性的算力代价，把 max sequence 控制在 8K 是当时较稳妥的工程选择
2. RoPE base 仍然保持在 10000：没有像 Mixtral 系列那样把 RoPE base 上调到 1e6，也没有引入 YaRN 等显式的长上下文位置编码方案，导致位置编码本身不具备外推能力
3. 整体上沿用 LLaMA-2 时代的设计风格：LLaMA-2 当时使用 4K 上下文，Grok-1 在此基础上仅倍增到 8K，并未把长上下文作为独立的设计目标

社区曾尝试用 NTK-aware scaling 或 position interpolation 等方法把 Grok-1 外推到 32K，但实测效果一般 - 模型在训练时根本没有见过长上下文样本，位置编码即便外推到更远位置，attention 也无法学到合理的长程依赖。

### 10.6.1 上下文长度的演进路径

从短上下文向长上下文扩展，是 2024 年 LLM 演进过程中最显著的一个方向。Grok-1 的 8K 上下文在 2024 年 3 月发布时属于同期合理水平（同期 LLaMA-2 仅 4K，Grok-1 已经是它的两倍），但接下来同代模型与 Grok-1 之间的上下文长度差距迅速拉大：2024 年 5 月 DeepSeek-V2 已经做到 128K，到 2024 年底 Gemini 1.5 Pro 进一步推到 1M 级别。Grok-1 在长上下文这条赛道上仅用半年就从合理水平退到落后于主流。

长上下文实现的关键技术：

1. **训练时直接使用更长 context**：最直接的方法，但 attention 的计算量按序列长度 T 的平方增长，单 step 算力开销随之放大，是当前最贵的一种做法
2. **将 RoPE base 调大**：把 RoPE 的 base 频率从 10000 调到 1e5 甚至 1e6 量级，让位置编码在训练时未见过的长度上仍能保持平滑的相对位置区分能力
3. **使用 YaRN / NTK-aware scaling 等位置编码外推方法**：显式调整 RoPE 不同频率分量的缩放，让模型不需要重新训练就能在更长上下文上保留可用的长程位置信号
4. **post-training 阶段做长 context fine-tune**：预训练阶段使用较短上下文以节省算力，再在 fine-tune 阶段把窗口拉长，重点训练长程注意力

Grok-1 除了使用标准 RoPE 之外没有采纳上述任何一项手段，因此 8K 实际上是其推理时的硬上限。要把 Grok-1 外推到 32K 一般需要先做长上下文 fine-tune，而对一个 314 B 规模、缺乏配套训练栈的模型来说，这一步在工程上并不现实。

## 10.7 推理成本与性能权衡

近似 FLOPs / token（forward 一次）：

$$
\text{FLOPs/tok} \approx 2 \cdot P_{\text{active}}
$$

| 模型 | $P_{\text{active}}$ | FLOPs/tok | 推理硬件最低需求（bf16） |
| --- | --- | --- | --- |
| Grok-1 | 86 B | 172 G | 8 × A100 80GB |
| Mixtral 8x7B | 13 B | 26 G | 1 × H100 80GB |
| Mixtral 8x22B | 39 B | 78 G | 4 × A100 80GB |
| DeepSeek-V2 | 21 B | 42 G | 2 × A100 80GB |
| DeepSeek-V3 | 37 B | 74 G | 4 × A100 80GB |
| Qwen1.5-MoE | 2.7 B | 5.4 G | 1 × A100 40GB |

Grok-1 的"激活参数偏大"在推理侧是明显的劣势 - 在大致同等的 base 模型质量假设下，DeepSeek-V3 每 token 的 FLOPs 仅为 Grok-1 的 43%，最低推理硬件需求也只需要 Grok-1 的一半。这意味着把 Grok-1 部署为线上服务的成本，比同等质量目标的细粒度 MoE 高出近一倍。

但 Grok-1 的总参数较大同时意味着潜在的模型容量更大 - 在 base 模型表征能力上，理论上可能优于激活参更小的同类。遗憾的是社区始终没有完成一次权威的横向基准测试，因为 Grok-1 base 没有官方的 chat / instruct 模板，无法直接对话评测，而要做有意义的 SFT 又受限于硬件门槛，最终缺少能与 Mixtral 8x22B-base、DeepSeek-V2/V3-base 在同一刻度下比较的数据。

## 10.8 训练数据规模

| 模型 | 训练 token | 来源 |
| --- | --- | --- |
| Grok-1 | **未公开**（据 xAI 博客"约万亿级 token"，但未给具体数字） | xai.com/blog |
| Mixtral 8x7B | 未公开（估计 ~2 T） | - |
| LLaMA-2 70B | 2 T | LLaMA-2 paper |
| DeepSeek-V2 | 8.1 T | DeepSeek-V2 报告 |
| DeepSeek-V3 | 14.8 T | DeepSeek-V3 报告 |
| Qwen2.5 系列 | 18 T | Qwen 报告 |

按"激活参 vs 训练 token"的 Chinchilla 估计：

- Grok-1 的 86B 激活按 Chinchilla 应训 ~1.7 T token，如果真训了 1-2 T，是合理的
- DeepSeek-V3 用 14.8 T 训 37 B 激活，远超 Chinchilla 最优 - 是"over-training"策略，让小激活模型质量逼近大模型

**Grok-1 大概率属于 under-trained**：若按总参 314B 套用 Chinchilla 1:20 配比应训约 6 T token，若按激活 86 B 估算也至少需要约 1.7 T token，而 xAI 公开口径只说到"万亿级"且没有具体数字，结合训练只用了 4 个月这一事实，token 总量超过 2 T 的可能性较低。训练 token 不足，是 Grok-1 base 在直观印象上质量明显弱于同代 Mixtral 8x22B-base 的可能原因之一。

!!! note "scaling law / Chinchilla"
    scaling law 是一系列经验公式，描述"模型大小、训练 token 数、最终 loss"之间的关系。OpenAI 2020 年的 Kaplan 论文是第一版，DeepMind 2022 年的 Chinchilla 论文是后续修正。Chinchilla 的结论是：给定一笔训练算力预算，最优分配是让"参数量"和"训练 token 数"按 1:20 配比 - 即每参数训 20 个 token 时 loss 最低。

    按 Chinchilla 看 Grok-1：314 B 总参 × 20 = 6.3 T token 是"算力最优"配比，但 MoE 通常按激活参算（86 B × 20 ≈ 1.7 T）。xAI 没公开训练 token 数，博客只说"万亿级"。对比 DeepSeek-V3 用 14.8 T 训 37 B 激活参 - 远超 Chinchilla 最优，属于 over-training，用额外训练换额外质量。这是 2024-2025 年的明显趋势。

## 10.9 Sharding 与并行策略

| 模型 | 训练框架 | 主要并行策略 |
| --- | --- | --- |
| Grok-1 | JAX + Rust 自研 | tensor parallel（数据轴 + 模型轴） |
| Mixtral | (未公开，推测 PyTorch + Megatron) | TP + PP + Expert parallel |
| DeepSeek-V2/V3 | DeepSeek 自研 (HAI-LLM, PyTorch) | TP + PP + EP，重通信优化 |
| Qwen | 阿里飞天 / Megatron-LM | 标准 Megatron |

Grok-1 选 JAX 在业界属于少数派 - 同代大模型的训练栈主流是 PyTorch + Megatron-LM 体系。这种选型在开源后带来一系列连锁影响：

- Grok-1 的训练栈不能被外部团队直接 fork 复用，要在自己的工程体系里跑起来需要先理清 JAX 与 Haiku 的并行抽象
- Grok-1 的 ckpt 是 JAX pickle 而非 PyTorch state_dict 格式，加载到 PyTorch 必须先写一份完整的权重转换器
- 上述两点叠加，使社区做微调的工程门槛显著高于同期纯 PyTorch 模型

HuggingFace Hub 上虽然有 `xai-org/grok-1` 的官方 ckpt，但**始终没有官方 PyTorch 版本**。理论上社区可以自发实现转换器，但 314 B 规模下的转换、验证、再切分都需要大量人力，而 base 模型在终端用途上又有限，投入产出比不划算，因此到本书写作时也没有出现一份被广泛采用的转换实现。

## 10.10 谁的设计被后来吸收了

按时间线看，Grok-1 的几个设计选择被吸收 / 被否定的情况：

| Grok-1 设计 | 后来被吸收？ |
| --- | --- |
| 8 个胖 expert top-2 | **被否定** - DeepSeek/Qwen 都转向细粒度 |
| Sandwich norm | 部分吸收（Cohere、Gemma 2 用类似） |
| Attention logit soft-cap (30·tanh) | **被吸收** - Gemma 2 直接采用 |
| GeGLU | 部分吸收（Gemma 用） |
| 大词表（131k） | **被吸收** - LLaMA-3 升到 128k，DeepSeek-V3 升到 129k，Qwen 升到 152k |
| GQA 6:1 | 部分吸收（Mixtral 8x22B 同样配置） |
| 8K 上下文 | **被淘汰** - 主流都到 32K+ |
| JAX + Haiku | **没被吸收** - 业界依然主流 PyTorch |
| Unnormalized gating | **被否定** - 大家都归一化 |
| 无 capacity / aux loss（推理代码看到的） | **被否定** - 大家都做负载均衡 |

总结起来，Grok-1 留给后续工作的技术遗产主要集中在 **"大词表 + attention logit soft-cap + GeGLU + sandwich norm"** 这四点：它们或者被 Gemma 2 等模型直接吸收，或者作为大词表潮流的早期范例被引用。除此之外，Grok-1 在 MoE 路由、专家粒度、稀疏部署等设计上的选择，都在后来的工作中被改写或替代，并没有形成持续影响力。

## 10.11 各家模型的细节速记

接下来对每个模型展开一段说明，覆盖架构亮点、训练取向与工程选型，方便建立更具体的横向印象。

### 10.11.1 Mixtral 8x7B

Mistral 在 2023 年 12 月以磁力链方式发布的首个开源 MoE，是后续 MoE 讨论的事实基准。整体设计在当时极为节制：

- hidden 4096、32 层、32 Q / 8 KV head，与 Mistral 7B（dense 基线）保持一致，等于"把 Mistral 7B 的 FFN 拆成 8 个 expert，其它都不变"
- top-2 of 8 路由、SwiGLU FFN、SiLU 激活、RoPE base 调到 1e6 直接支持 32K 上下文
- gate 在 top-k 之后重新归一化到 1，使 expert 输出量级稳定，便于训练
- tokenizer 沿用 LLaMA-2 系列的 32K BPE 词表，没有扩词

Mixtral 8x7B 的工程意义不在于它有多少创新，而在于它**第一次把"开源 MoE 可以与同尺寸稠密模型直接竞争"这件事打实**。它的存在直接定义了 Grok-1 进入开源生态时的对照基线。

### 10.11.2 Mixtral 8x22B

Mistral 在 2024 年 4 月发布的等比放大版本，与 Grok-1 公开时间相距仅一个月，配置高度相似：hidden 6144、48 Q / 8 KV、56 层、SwiGLU、top-2 of 8。差异主要在以下几点：

- FFN 中间维度 16384，明显小于 Grok-1 的 32768，因此单 expert 参数约 0.3 B，是 Grok-1 单 expert 的一半
- 总参 141 B、激活 39 B，激活比同为约 27%，但绝对算力门槛明显低于 Grok-1
- RoPE base 1e6、上下文 65536，比 Grok-1 的 8K 长一个数量级
- 32K BPE 词表沿用，没有像 Grok-1 那样上推到 128K+

Mixtral 8x22B 与 Grok-1 几乎是同一类设计的两个不同尺寸点，可以视为胖专家路线在 2024 年初的标准答案。两者在配置上相似度极高这一事实，也旁证了"48 Q / 8 KV / hidden 6144"在 100-300 B 规模下确实是一个被多支团队独立发现的自然 sweet spot。

### 10.11.3 DeepSeek-V2

2024 年 5 月发布，是细粒度 MoE 路线第一份完整的公开实现：

- 160 routed expert + 2 shared expert，top-6 路由，激活比仅约 8.9%
- MLA 注意力，KV cache latent 维度 512，相对同级 GQA 节省约 4 倍 cache
- 多种 aux loss 组合：device-balanced loss、communication-balanced loss、expert-balanced loss，分别针对设备、通信、专家三层面的不均衡施加惩罚
- 训练 token 8.1 T，已经超过 Chinchilla 最优配比一倍，进入 over-training 区间

DeepSeek-V2 第一次把"细粒度专家 + MLA + 精细 aux loss"组合为一个可复现的工程方案，并通过完整的技术报告把每一项的实现细节披露出来。它直接催生了 2024 年下半年开源 MoE 的范式转向。

### 10.11.4 DeepSeek-V3

2024 年 12 月发布，是细粒度路线的集大成之作：

- 256 routed expert + 1 shared expert，top-8 路由，激活比降到 5.5%
- aux-loss-free 路由（每个 expert 配可学习 bias，按使用频率自动调节），把负载均衡从 loss 项转为 bias 调节
- MLA 注意力延续 V2 设计，KV cache 进一步减小
- 671 B 总参 + 37 B 激活，训练 14.8 T token，FLOPs/token 仅为 Grok-1 的 43%
- 国内自研训练栈 HAI-LLM，通信优化做到了 DualPipe 等专门为 MoE 设计的流水线方案

DeepSeek-V3 在公开模型里把"总参越来越大、激活越来越小、训练 token 越来越多"这条三轴趋势推到了同期最远的位置。它在多项开源 base 评测上的成绩，是 Grok-1 这种胖专家路线在 2025 年之后明显失去吸引力的直接原因。

### 10.11.5 Qwen-MoE 系列

阿里通义实验室的 MoE 系列，从 2024 年 3 月的 Qwen1.5-MoE-A2.7B 起步，到 2025 年的 Qwen3-235B-A22B 共发布多个版本，呈现稳健的工程演进：

- 早期版本沿用 shared expert 设计（4-8 个 shared + 60-64 个 routed），后期版本（Qwen3-235B）回归纯 routed
- 词表 152K，对中英文混排场景的压缩率明显好于 LLaMA 系的 32K
- GQA 比例从 1:1（Qwen1.5-MoE）逐步收紧到 7:1（Qwen2-57B-A14B），与同期主流保持一致
- 训练栈基于阿里飞天体系与 Megatron-LM 改造，工程化程度高

Qwen-MoE 没有 DeepSeek 那样的单点创新，但每一项设计都经过工业验证且配套工具链完整，是国内开源 MoE 路线中"稳健工程优先于研究激进"的代表。

### 10.11.6 一句话总结

- **Grok-1**：胖专家路线的极端代表，工程上以"先把模型完整训出来"为第一目标
- **Mixtral 8x7B**：首个被社区广泛使用的开源 MoE 模型，为后续 MoE 工作建立了参照基准
- **Mixtral 8x22B**：在 Mixtral 8x7B 基础上等比放大，hidden / heads 配置与 Grok-1 高度一致
- **DeepSeek-V2**：细粒度专家 + MLA + 多 aux loss 的精细化设计代表
- **DeepSeek-V3**：aux-loss-free 路由配合大规模 over-training 的集大成之作
- **Qwen-MoE 系列**：shared expert 与大词表组合的稳健工程选择

## 10.12 谁影响了 Grok-1，Grok-1 又影响了谁

Grok-1 的设计灵感看起来直接来自：

- **GLaM**：8 选 2 的 top-k 路由结构（GLaM 也用 8/2）
- **Switch Transformer**：fp32 路由的稳定性建议
- **PaLM**：sandwich norm 的早期实验
- **Vaswani 2017 原始 Transformer**：sqrt(d) embedding scaling

Grok-1 直接影响：

- **Gemma 2**：logit soft-cap、GeGLU
- **Mixtral 8x22B**：在 hidden=6144 与 heads=48/8 上与 Grok-1 完全一致（两者发布时间相距仅一个月，不太可能是直接借鉴，更可能是两支团队在相近规模下独立得出的相同最优配比）

没有影响到的：

- DeepSeek 系列（路线完全不同）
- LLaMA 系列（继续走 dense）
- Qwen MoE（细粒度路线）

## 10.13 总结表：MoE 设计选项谱

| 选项 | A 极简 | B 中间 | C 精细 |
| --- | --- | --- | --- |
| 专家数 | 8（Grok, Mixtral） | 32-64（Qwen-MoE） | 128-256（DeepSeek-V3） |
| top-k | 1-2（Switch, Grok, Mixtral） | 4（Qwen1.5-MoE） | 6-8（DeepSeek） |
| Capacity | 不用（Grok, Mixtral） | 1.0-1.25（GLaM） | 动态调整（DeepSeek） |
| Aux loss | 推理不用（Grok 看到的） | 单 aux loss（Mixtral） | 多 aux loss / 无 aux loss（DeepSeek） |
| Shared expert | 无（Grok, Mixtral） | 1-2 个（Qwen-MoE, DeepSeek-V2） | - |

Grok-1 在每一项上都落在 A 极简列。这种组合在 2024 年的时点上能够支撑可用的训练与推理流程，但到了 2025 年，随着 DeepSeek 系列陆续证明"精细化路由可以在更小的激活参数下达到接近大模型的质量"，A 极简路线在新模型设计中的吸引力明显下降，胖专家方案在新立项中越来越少见。

下一章是收尾，讨论 Grok-1 的工程教训与遗产。

## 延伸阅读

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - aux-loss-free 路由的详细推导
- [Qwen Technical Report](https://qwenlm.github.io/blog/qwen2.5/)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
