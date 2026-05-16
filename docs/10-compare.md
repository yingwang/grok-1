# 第 10 章 与其他 MoE 对比

本章把 Grok-1 放进同代 MoE 的横向对照里看。覆盖 Mixtral 系、DeepSeek 系、Qwen-MoE。所有数字尽量来自各模型的公开技术报告或 HuggingFace config，未公开则注明。

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

### 10.1.1 怎么读这张表

如果你只看一遍这章，看这张表就够。但每个数字背后都有故事，我接下来一节一节展开。

读法建议：

1. **横向比较激活比**：胖专家派（Grok / Mixtral）激活比 ~27%，细粒度派（DeepSeek / Qwen3-MoE）激活比 5-10%。这是最重要的设计分水岭
2. **横向比较 head Q/KV**：注意 DeepSeek 系列的 128/128 是 MLA 不是 MHA，与其他模型不可直接对比
3. **横向比较 expert FFN 大小**：胖专家的 expert 中间维 14K-33K，细粒度专家的中间维 1.4K-2K，差一个数量级
4. **纵向看时间**：胖专家派出现在 2023-2024，细粒度派从 2024-05 开始（DeepSeek-V2）。整个领域用 1 年时间完成了"从 8 个胖专家到 256 个小专家"的转向

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

第 11 章会讨论：xAI 选这个路线可能是因为时间紧迫（2023 年中 MoE 工程还不成熟，细粒度路由的训练难度高），先用简单设计验证可行性。

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

Grok-1 不可能用这个，因为它 2023 年训练时 DeepSeek 这个方法还没发表（2024 年底才公开）。Grok-1 训练时大概率用了传统 aux loss + capacity factor，但开源代码里没保留这部分。

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

实践影响很小，但**ckpt 不能跨激活函数互换**。

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

Grok-1 / Mixtral 走的是 GQA - 简单、可靠、ckpt 易于切分。

## 10.6 上下文长度

| 模型 | 上下文 | RoPE base |
| --- | --- | --- |
| Grok-1 | 8192 | 10000 |
| Mixtral 8x7B | 32768 | 1000000 |
| Mixtral 8x22B | 65536 | 1000000 |
| DeepSeek-V2 | 128000 | 10000 + YaRN |
| DeepSeek-V3 | 131072 | 10000 + YaRN |
| Qwen3-235B | 128000 | YaRN |

**Grok-1 上下文只有 8K，是同代 MoE 里最短的**。原因可能是：

1. 训练时 sequence length 限制在 8K（更长 context 训练成本高）
2. RoPE base 仍是 10000，没做长 context 适配
3. 与 LLaMA-2 同代风格（LLaMA-2 也是 4K，Grok-1 翻倍到 8K）

社区有尝试用 NTK-aware scaling / position interpolation 把 Grok-1 外推到 32K，但效果一般 - 模型本身就不是为长 context 训练的。

### 10.6.1 上下文长度的演进路径

短上下文 → 长上下文是 2024 年 LLM 演进最显著的方向。Grok-1 8K 在 2024 年 3 月看正常（同期 LLaMA-2 4K），但很快被甩开 - 2024 年 5 月 DeepSeek-V2 已经 128K，年底 Gemini 1.5 Pro 已经 1M。

长上下文实现的关键技术：

1. **训练时直接训长 context** - 最直接但最贵（每 step 计算量按 T² 增长）
2. **RoPE base 调大** - 让位置编码能外推到训练时未见过的长度
3. **YaRN / NTK-aware scaling** - 显式调整 RoPE 频率分布
4. **post-training 长 context fine-tune** - 训练时短 context、fine-tune 时拉长

Grok-1 没用任何上述技术（除了标准 RoPE），所以 8K 是硬上限。要外推到 32K 需要先 fine-tune，对 314B 来说不现实。

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

Grok-1 的"激活参数大"是劣势 - 同样质量下，DeepSeek-V3 的 FLOPs 是 Grok-1 的 43%，硬件需求也低一半。

但 Grok-1 的总参数大也意味着模型容量大 - 在 base 模型质量上可能优于激活参更小的同类。可惜社区没做过权威横向 benchmark（因为 Grok-1 base 不能直接对话）。

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

**Grok-1 大概率 under-trained**（按总参 314B 算应该训 ~6 T token，按激活 86 B 算应该训 ~1.7 T，xAI 没说具体）。这可能是 Grok-1 base 质量比同代 Mixtral 8x22B-base 弱的原因之一。

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

Grok-1 的 JAX 选择是少数派 - 业界主流是 PyTorch + Megatron。这导致：

- Grok-1 的训练栈不能直接 fork
- Grok-1 ckpt 加载到 PyTorch 需要写转换器
- 社区微调极其困难

HF Hub 上有 `xai-org/grok-1` 的官方 ckpt 但**没有 PyTorch 版本**。理论上有人可以写转换器，但工程量大、收益小（base 模型用处有限），所以一直没出现。

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

总结：Grok-1 留下的技术遗产主要是 **"大词表 + soft-cap + GeGLU + sandwich norm"**这 4 件事，其他的 MoE 设计选择都被后来的工作改写了。

## 10.11 一句话总结每个模型

- **Grok-1**：胖专家的极端，先做出来再说
- **Mixtral 8x7B**：MoE 开源的破冰者
- **Mixtral 8x22B**：把 Mixtral 8x7B 等比放大，结构上 Grok-1 的"小弟"
- **DeepSeek-V2**：细粒度 + MLA + 多 aux loss 的精细化设计
- **DeepSeek-V3**：aux-loss-free + 大规模 over-training 的集大成
- **Qwen-MoE 系列**：shared expert + 大词表的稳健选择

## 10.12 谁影响了 Grok-1，Grok-1 又影响了谁

Grok-1 的设计灵感看起来直接来自：

- **GLaM**：8 选 2 的 top-k 路由结构（GLaM 也用 8/2）
- **Switch Transformer**：fp32 路由的稳定性建议
- **PaLM**：sandwich norm 的早期实验
- **Vaswani 2017 原始 Transformer**：sqrt(d) embedding scaling

Grok-1 直接影响：

- **Gemma 2**：logit soft-cap、GeGLU
- **Mixtral 8x22B**：相同的 hidden=6144 / heads=48/8（虽然不一定是直接抄，但碰巧一样）

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

Grok-1 处于 A 极简栏。这种设计在 2024 年看是"够用"，但到 2025 年随着 DeepSeek 显示"精细化路由能在小激活参下达到大模型质量"，A 派的吸引力下降了。

下一章是收尾，讨论 Grok-1 的工程教训与遗产。

## 延伸阅读

- [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - aux-loss-free 路由的详细推导
- [Qwen Technical Report](https://qwenlm.github.io/blog/qwen2.5/)
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
