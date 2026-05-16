# 第 2 章 总体架构

本章是后续章节的"地图" - 我们先用一张图、一张表、几段叙述，把 Grok-1 的整体形状描绘清楚，让读者在进入逐行精读之前对全模型的拓扑结构有一个完整的印象。如果你已经熟悉同代 MoE 模型的整体形态，这一章可以快速跳过，直接进入第 3 章的配置精读。

## 2.1 一句话先把骨架立起来

Grok-1 是 **64 层 decoder-only transformer**，每一层把 FFN 子层换成 **8 个 SwiGLU 专家中 top-2 路由**的 MoE 块，其余部分（GQA 注意力、RoPE、RMSNorm、残差）与 LLaMA-2 同代设计基本一致。模型整体只接受 token 输入、输出 next-token logits，没有 cross-attention，没有 encoder。

但"基本一致"里藏着几个少见的设计选择，本章先列出来，后面的精读章节再细看：

1. **GQA：48 Q 头对 8 KV 头**，比例为 6:1。同代的 LLaMA-2 70B 配置是 64 Q 头对 8 KV 头，比例为 8:1。Grok-1 在 GQA 的"压缩比"上比 LLaMA-2 略保守，每组 K/V 服务 6 个 Q 而不是 8 个，理论上保留了更多 attention 表达能力，但 KV cache 体积也相应增大了一点
2. **每个 sub-layer 用了两次 RMSNorm**：进入子层之前做一次 pre-norm、子层输出加回 residual 之前再做一次 post-norm，两次 norm 把子层输入和输出都钳制住。这种布局被称为"sandwich norm"，与 Cohere Command R / Command R+ 的设计一致（`model.py:1056-1060`，下文 2.5 详述）

!!! note "sandwich norm（pre + post 双 norm 布局）"
    早期 transformer 是 post-norm：`y = LayerNorm(x + SubLayer(x))`，深网络梯度反传时常炸。主流后来切到 pre-norm：`y = x + SubLayer(LayerNorm(x))`，深层稳很多但 residual 流的量级会随层数累积上涨。

    sandwich norm 把两者夹在一起：`y = x + LayerNorm(SubLayer(LayerNorm(x)))` - 进子层前 norm 一次（pre）、出子层加回 residual 前再 norm 一次（post）。两次 norm 把 residual 流和子层输出都钳住，深网络训练更稳。代价是每层多一次 RMSNorm 计算（在 314B 里几乎免费）。Grok-1 64 层 + 4 个 RMSNorm/层（attention pre/post + FFN pre/post），共 256 次 norm，是 Cohere Command R 同款选择。
3. **embedding 用 sqrt(d) 量级放大**：token 查表得到 embedding 之后，整体乘以约 $\sqrt{6144} \approx 78.38$ 的常数；输出端 logits 再乘以 $1/\sqrt{3} \approx 0.577$。这一对乘法配合在一起，是 µ-Transfer / DeepNet 风格的"输入/输出 scale 控制"思路，目的是让训练时不同层的激活量级保持稳定
4. **attention logit 用 `30 * tanh(x/30)` 软裁剪**：Q 和 K 点积、再除以 $\sqrt{d_h}$ 之后，attention logit 还要再经过一个 `30 * tanh(x/30)` 的软裁剪函数（`model.py:864-865`）。这个函数对小输入几乎是恒等映射，对大输入则平滑地饱和到 $\pm 30$，目的是防止 attention logit 在 bf16 下溢出导致 softmax 出 NaN
5. **MoE 路由没有 auxiliary loss、没有 capacity drop**：路由器只做纯 softmax + top-2 的简单选择，没有 Switch Transformer / GShard 那样的 auxiliary load-balancing loss，也没有"专家容量超限就丢弃部分 token"的 capacity drop 机制。所有 token 一律走两个专家，路由实现因此非常简洁

下面用一张数据流图把整体串起来。

值得在进入图之前先确认一下"什么是 decoder-only"。Transformer 原始论文（2017）的架构是 encoder + decoder，做翻译时 encoder 编码源语言、decoder 生成目标语言。decoder 里有两种 attention - self-attention（看自己已生成的部分）和 cross-attention（看 encoder 的输出）。

2018-2020 年的研究发现：纯 decoder 训练 language modeling 任务，scale 起来效果更好。GPT 系列从一开始就是 decoder-only，Llama、Mistral、Grok 全部是 decoder-only。decoder-only 的好处：架构统一、可以做 autoregressive 推理、训练效率高。Grok-1 的"decoder-only Transformer"就是这个意思 - 不带 encoder，只有 self-attention 和 FFN 的层堆叠。

## 2.2 数据流总览

```mermaid
graph TD
    A[tokens shape B,T] --> B[InOutEmbed]
    B --> C[*embedding_multiplier_scale]
    C --> D{Layer i in 0..63}
    D --> E[RMSNorm]
    E --> F[GQA Attention<br/>48 Q heads, 8 KV heads]
    F --> G[RMSNorm post]
    G --> H[+ residual]
    H --> I[RMSNorm]
    I --> J{MoE}
    J --> K[Router softmax fp32]
    K --> L[top-2 indices, gates]
    L --> M[Expert 1 SwiGLU]
    L --> N[Expert 2 SwiGLU]
    M --> O[weighted sum by gate]
    N --> O
    O --> P[RMSNorm post]
    P --> Q[+ residual]
    Q --> D
    Q --> R[Final RMSNorm]
    R --> S[in_out_embed.decode]
    S --> T[*output_multiplier_scale]
    T --> U[logits shape B,T,V]
```

几点需要特别注意：

- **InOutEmbed**（`model.py:1110-1143`）输入和输出权重共享 - 输入是 `embed_mat[token_id]`，输出是 `embeddings @ embed_mat.T`
- **每个 sub-layer 后做了两次 RMSNorm** - 一次 pre（在子层输入前），一次 post（在子层输出后再加到 residual 前）
- **MoE 的两个专家输出是按 gate 加权求和后再过 post-RMSNorm**，不是各自单独 norm

### 2.2.1 数据流图的几个细节再说明

图里的关键步骤：

- **InOutEmbed**：输入 token id 通过查表得到 hidden 向量。输出阶段同样的 embedding 矩阵转置后做 logit 投影。这种"输入输出共享 embedding"叫 tied embedding，节省 0.81B 参数

!!! note "tied embedding（输入输出共享 embedding 矩阵）"
    Transformer 输入端有个 embedding 矩阵 `(V, d)`，token id 查表拿到 hidden；输出端做 next-token 预测时需要从 hidden 投回 `(V,)` 维 logits，理论上是另一个 `(d, V)` 矩阵。tied embedding 的做法是让这两个矩阵共享同一份参数 - 输出投影直接用输入 embedding 的转置 `hidden @ embed_mat.T`，省掉一整块大参数。

    Grok-1 词表 131072、d=6144，单独一份就是 0.81B 参数 - tying 之后总参省一份。GPT-2、LLaMA 系列都用 tied embedding，Grok-1 在 `InOutEmbed`（`model.py:1110-1143`）里实现的就是这种共享。
- **\*embedding_multiplier_scale**：embedding 取出之后，整体乘以约 78.38 的放大因子，这个数约等于 $\sqrt{d} = \sqrt{6144}$。这是 Vaswani 2017 原版 Transformer 论文里就有的写法，目的是让 embedding 出来的激活量级与后续层匹配。LLaMA 系列已经放弃这一步，Grok-1 还保留着
- **每层 RMSNorm 共出现 4 次**：分别位于 attention 子层之前（pre-attn）、attention 子层之后（post-attn）、FFN/MoE 子层之前（pre-FFN）、FFN/MoE 子层之后（post-FFN）。这种"每个子层前后各做一次 RMSNorm"的布局就是 Grok-1 的 sandwich norm
- **GQA 48Q vs 8KV**：48 个 query head 被分成 8 组，每组包含 6 个 Q 头；同一组内的 6 个 Q 共享一组 K/V。这样 KV cache 的体积按"头数"维度从 48 降到 8，节省约 6 倍存储
- **Router**：路由器是一个 $(d, E) = (6144, 8)$ 的小线性层，输入是当前 token 的 hidden，输出经过 fp32 softmax 之后取 top-2 个专家
- **MoE 选 2 个专家加权求和**：每个 token 只在被 router 选中的 2 个 expert 上做 FFN 计算，两个 expert 的输出按 router 给出的概率（未做额外归一化）加权求和，剩下 6 个 expert 不参与本 token 的计算
- **Final RMSNorm**：64 层 decoder 全部跑完之后，再对最终的 hidden 做一次 RMSNorm，然后才进入输出投影
- **\*output_multiplier_scale**：输出投影得到的 logits 整体乘以约 0.577 的常数，这个数等于 $1/\sqrt{3}$。它与开头的 embedding_multiplier_scale 配对，共同构成 µ-Transfer 风格的输入/输出量级控制

这 8 个步骤是 Grok-1 推理时每个 token 都要走的"宏路径"。第 4-6 章会把每个步骤展开到代码级。

## 2.3 模型规模账：314B 到底从哪里来

精确算一遍。设：

- $d = 6144$（emb_size）
- $L = 64$（num_layers）
- $E = 8$ 专家，$k = 2$ 激活
- $V = 131072$（vocab_size = 128 × 1024）
- $H_q = 48$, $H_{kv} = 8$, $d_h = 128$
- FFN 中间维度由 `ffn_size(d, w)` 计算（`model.py:85-89`）：

```python
# model.py:85-89
def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    return _ffn_size
```

代入 `widening_factor=8`：`int(8 * 6144) * 2 // 3 = 49152 * 2 // 3 = 32768`，已经是 8 的倍数，所以 $d_{\text{ffn}} = 32768$。

### 2.3.1 每层参数

**Attention：**

| 矩阵 | shape | 参数 |
| --- | --- | --- |
| Q 投影 | $(d, H_q d_h) = (6144, 6144)$ | 37.7 M |
| K 投影 | $(d, H_{kv} d_h) = (6144, 1024)$ | 6.3 M |
| V 投影 | $(d, H_{kv} d_h) = (6144, 1024)$ | 6.3 M |
| Output 投影 | $(H_q d_h, d) = (6144, 6144)$ | 37.7 M |
| 合计 | - | **88.1 M** |

注意：Q/K/V 都有 bias（`with_bias=True` 是 `Linear` 的默认值，但 MHA 调用时显式 `with_bias=False`，见 `model.py:887` 的 `final_projection` 与 `_linear_projection` 中 `Linear(num_heads * head_size, with_bias=False, ...)` 的 `model.py:905`）。所以 attention 内部所有 Linear 都没 bias。

**FFN（每个 expert 是一个 SwiGLU）：**

`DenseBlock` 在 `model.py:963-1007` 定义，三个 Linear 都是 `with_bias=False`：

| 矩阵 | shape | 参数 |
| --- | --- | --- |
| linear_v | $(d, d_{\text{ffn}}) = (6144, 32768)$ | 201.3 M |
| linear (gate) | $(6144, 32768)$ | 201.3 M |
| linear_1 | $(32768, 6144)$ | 201.3 M |
| 单 expert 合计 | - | **603.9 M** |
| 8 个 expert | - | **4.83 B** |

**LayerNorm（RMSNorm）：** 每层 4 个 RMSNorm（attention pre/post + FFN pre/post，见 `model.py:137-140` partition rules 中的 `rms_norm` ~ `rms_norm_3`），每个 6144 个 scale，共 24576 参数 - 可忽略。

**Router：** $d \times E = 6144 \cdot 8 = 49152$ 参数 - 可忽略。

**每层总计：** $88.1\text{M} + 4830\text{M} \approx 4.92\,\text{B}$

### 2.3.2 整模型

| 组件 | 参数 |
| --- | --- |
| 64 层 × 4.92 B | 314.9 B |
| InOutEmbed (131072 × 6144) | 0.81 B |
| 最终 RMSNorm | 6144 |
| **总计** | **~315.7 B** |

官方说 "314B"，与上面的估算吻合（差额来自 router、norm、对 ffn_size 取整等小项；权重 tying 让 embedding 不重复计算）。

### 2.3.3 激活参数

每个 token 只激活 2 个 expert：

$$
P_{\text{active}} = 64 \cdot (0.088 + 2 \cdot 0.604) + 0.81 \approx 84\,\text{B}
$$

加上 router（每层 49K，可忽略）后约 86B - 与官方"约 86B 激活"对得上。

### 2.3.4 为什么 widening_factor=8 而 SwiGLU 通常是 widening=2.67

这是个看起来奇怪的细节。SwiGLU/GeGLU 的"中间维度"惯例是 $d_{\text{ffn}} \approx \frac{2}{3} \cdot 4 \cdot d = \frac{8}{3} \cdot d$，即 widening = 8/3 ≈ 2.67。Llama2 70B 的 hidden=8192、ffn=28672，正好 ratio = 3.5（略大于 8/3）。

Grok-1 的 widening_factor=8 看起来是 Llama2 的 3 倍。但 `ffn_size` 函数对它再乘 2/3，最终得到 $8 \cdot 6144 \cdot 2/3 = 32768$，是 $d=6144$ 的 5.33 倍。这比 SwiGLU 的 2.67 倍多了一倍。

为什么 Grok-1 选这么"胖"的 FFN？

可能的解释是：**MoE 里每个 expert 只服务 1/4 的 token**（top-2/8 = 25% 概率激活），如果想让 expert"看到足够多 token 学到特征"，就需要让每个 expert 的容量更大、参数更多，否则容量浪费。胖 expert 路线在 Mixtral 8x22B 也能看到（widening = 32768 / 6144 ≈ 5.33... 不对，Mixtral 8x22B 是 16384 / 6144 ≈ 2.67，标准 SwiGLU）。

所以 Grok-1 比 Mixtral 8x22B 在 FFN 上更胖 - 这又一次印证"Grok-1 是胖专家派的极端"。

## 2.4 与稠密 70B、Mixtral 8x7B 的参数账对比

| 模型 | 总参 | 激活参 | 激活比 | 路由 | $d$ | $L$ | $H_q$/$H_{kv}$ | $d_{\text{ffn}}$ | 专家数 / top-k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LLaMA-2 70B (dense) | 70 B | 70 B | 100% | - | 8192 | 80 | 64 / 8 | 28672 | 1 / - |
| Mixtral 8x7B | 46.7 B | 12.9 B | 27.6% | top-2 of 8 | 4096 | 32 | 32 / 8 | 14336 | 8 / 2 |
| Mixtral 8x22B | 141 B | 39 B | 27.7% | top-2 of 8 | 6144 | 56 | 48 / 8 | 16384 | 8 / 2 |
| **Grok-1** | **314 B** | **86 B** | **27.4%** | **top-2 of 8** | **6144** | **64** | **48 / 8** | **32768** | **8 / 2** |
| DeepSeek-V2 | 236 B | 21 B | 8.9% | top-6 of 160 + 2 shared | 5120 | 60 | 128 / 128 (MLA) | 12288 | 160 / 6 |

看到这张表，几个观察立刻浮现：

1. **Grok-1 几乎是 Mixtral 8x22B 的"放大版"**：两个模型在 hidden 维度 $d=6144$、$H_q/H_{kv}=48/8$、top-2 of 8 的路由配置上完全一致，差别仅在 transformer 层数（64 vs 56）和 FFN 中间维度（32768 vs 16384）这两项。换句话说，xAI 选择的整体结构与 Mistral 在 Mixtral 8x22B 上选择的几乎是同一条线，只是规模上更进一步
2. **激活比都在 27% 上下**：top-2 of 8 的路由设计决定了 FFN 部分的激活比例约为 2/8 = 25%，再加上 attention 这部分始终全激活带来的少量额外贡献，总激活参与总参的比值落在 27% 附近。Mixtral 8x7B、Mixtral 8x22B、Grok-1 都符合这条规律
3. **Grok 用"层多 + 专家容量大"的组合**：Grok-1 有 64 层、单个 expert 参数量约 0.6B，是同代 MoE 模型里单专家容量最大的一档。其他 8 选 2 模型要么层数更少（Mixtral 8x7B 只有 32 层）、要么单专家参数更小（Mixtral 8x22B 单专家约 0.3B）
4. **DeepSeek-V2 走了完全不同的路线**：DeepSeek-V2 把专家数从 8 提升到 160，单个 expert 参数量随之大幅缩小，并额外引入 2 个 shared expert 让所有 token 都经过；同时 attention 改用 MLA 压缩 KV，整体激活比因此降到 9%。这是与 Grok-1 完全对立的设计思路

这里"胖专家 vs 细粒度专家"是 MoE 设计的核心 trade-off。胖专家的优势：每个 expert 容量足，对单一任务学得透；缺点：选择空间小（8 选 2 只有 28 种组合），不同 token 之间的"路由分工"较粗。细粒度专家的优势：选择空间大（160 选 6 是天文数字），不同 token 可以分得很细；缺点：每个 expert 容量小，需要更多 expert 协作才能完成单一任务，路由开销高、负载均衡难。

Grok-1 选了胖专家路线，这是 2023 年初到中期最自然的选择 - 当时 Switch Transformer、GShard、GLaM 都用相对小数量的专家（8-64 个）。直到 2024 年下半年 DeepSeek-V2 用 160 个专家做出好效果，业界才开始大规模转向细粒度。

把表展开看更直观。dense 70B 把所有参数都激活，等同于"无专家的胖网络"；Mixtral 8x7B 是"小胖专家"；Mixtral 8x22B 和 Grok-1 是"大胖专家"；DeepSeek-V2 是"很多小专家 + shared expert"的混合策略。这是一条连续的设计谱，Grok-1 站在最靠近 dense 的那一端。

第 10 章会在这个表的基础上展开详细对比。

## 2.5 Sandwich norm：少见的归一化布局

主流的 pre-norm Transformer 写法是：

```
y = x + SubLayer(LayerNorm(x))
```

Grok-1 不这么写。看 `model.py:1048-1061`：

```python
# model.py:1048-1061
attn_output = MHABlock(...)(layer_norm(h), mask, layer_memory)
h_attn = attn_output.embeddings

h_attn = layer_norm(h_attn)       # 注意：再 norm 一次
h += h_attn
h = with_sharding_constraint(h, sharding)
```

即：

```
y = x + LayerNorm(SubLayer(LayerNorm(x)))
```

输入做 pre-norm 进子层，子层的输出**再做一次 post-norm**才加回 residual。两次 norm 把 residual 流和子层流都钳制住了。

这种"sandwich"布局在 Cohere Command R / Command R+ 里也有，被认为能让深层网络（80+ 层）训练更稳。Grok-1 64 层，使用这个 trick 应该是出于稳定性考虑。代价是每层多了 1 次 RMSNorm 计算 - 在 314B 的 FLOPs 占比里可以忽略。

第 6 章会在 DecoderLayer 精读时再展开。

## 2.5.1 双重 norm 的代价与收益

每层多一次 RMSNorm 的成本：

- **参数**：每 RMSNorm 多 6144 个 scale，4 个 norm 每层 24576 个参数。64 层共 1.57M 参数 - 在 314B 里可以忽略
- **计算**：每个 RMSNorm 需要 sum-of-squares + rsqrt + 乘 scale，约 3·d FLOPs。在 d=6144 时约 18K FLOPs，与一次 6144x6144 matmul（约 76M FLOPs）比小 4000 倍 - 几乎免费
- **内存带宽**：RMSNorm 是 memory-bound 算子（需要把 hidden 张量从 HBM 读一次、再写一次），所以相对其总计算量而言带宽占用偏高，但相对于一层中 attention 和 FFN 的带宽消耗，仍然属于不显著的开销

收益：

- **训练稳定性**：64 层的 transformer 在标准 pre-norm 布局下也有相当大的概率能稳定训练，但 sandwich norm 在 pre-norm 基础上又增加了一道 post-norm，相当于多了一层数值保险，对训练过程中偶发的 loss spike 更加耐受
- **量级控制**：residual stream 的量级在深层网络里有累积上涨的倾向，post-RMSNorm 把每一层子层输出的量级显式约束到 RMS = 1 附近，避免激活在 60 多层之后出现爆炸性增长
- **与 unnormalized MoE gating 互补**：第 5 章会看到 Grok-1 的 expert gate 是不归一化的（路由概率直接乘到 expert 输出上，没有再过 softmax 之外的 normalization），expert 输出的量级因此会随路由概率本身的大小变化；post-RMSNorm 紧跟在 MoE 之后，相当于对这个量级波动做了一次重新校准

这是一个典型的"小代价换大稳定性"工程决策。任何训练过较深网络、遇到过 loss spike 的工程师都能体会这种保守做法的价值。

## 2.6 Sharding 策略概览

`model.py:112-160` 的 `TRANSFORMER_PARTITION_RULES` 列出了每个权重的 partition 方案，分两轴 `data` / `model`。

```python
# model.py:112-160 节选
TRANSFORMER_PARTITION_RULES = [
    (("multi_head_attention", "(query|key|value)", "w"), P("data", "model")),
    (("multi_head_attention", "linear", "w"), P("model", "data")),
    ((r"decoder_layer_[0-9]+", "linear", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear_v", "w"), P("data", "model")),
    ((r"decoder_layer_[0-9]+", "linear_1", "w"), P("model", "data")),
    (("router", "w"), P("data")),
    (("moe", "linear", "w"), P(None, "data", "model")),
    (("moe", "linear_v", "w"), P(None, "data", "model")),
    (("moe", "linear_1", "w"), P(None, "model", "data")),
    ...
]
```

读法：

- 元组前两个元素是路径正则，匹配参数名
- 元组最后是 `PartitionSpec`，指出每个张量维沿哪个 mesh 轴切

整体策略可以总结为：

| 类型 | 张量 shape | partition |
| --- | --- | --- |
| Q/K/V 投影 | $(d, d_{\text{out}})$ | (data, model) |
| Attention output | $(d_{\text{model}}, d)$ | (model, data) |
| MoE up | $(E, d, d_{\text{ffn}})$ | (None, data, model) |
| MoE gate | $(E, d, d_{\text{ffn}})$ | (None, data, model) |
| MoE down | $(E, d_{\text{ffn}}, d)$ | (None, model, data) |
| RMSNorm scale | $(d,)$ | None（复制） |
| Embedding | $(V, d)$ | (None, ("data", "model")) - 第二维同时沿两轴切 |

`run.py:60` 给的 `local_mesh_config=(1, 8)`，即 1 × 8 = 8 个 device，全部都给 model 维。`between_hosts_config=(1, 1)` 表示单机。如果是 16 卡分两机，会变成 `local_mesh_config=(1, 8)` + `between_hosts_config=(1, 2)`。

这种"全 model 不切 data"的策略，意味着 Grok-1 在 8 卡上跑的是**纯 tensor parallel** - 一个 batch 在所有卡上共享同一份计算图，但每张卡只保留一部分参数和激活。和 dense 模型在 8 卡上做 tensor parallel 的拓扑几乎一样。

!!! note "tensor parallel / data parallel / model parallel"
    单机装不下的模型在多卡上有几种切法。**data parallel** 最简单：每张卡都有完整模型副本，把 batch 切几份让各卡同时算自己那份 micro-batch，最后 all-reduce 梯度 - 显存吃紧时根本用不了。**tensor parallel** 是把单个矩阵沿某一维切到多张卡上，每张卡只持有一部分 weight，forward 时 matmul 拆成多卡协作算，结果 all-gather 拼回来 - 显存压力小但通信频繁，对 NVLink 之类的高带宽互联很依赖。**model parallel** 是更宽的概念，pipeline parallel（按层切，一张卡跑前 16 层、另一张跑后 16 层）也算一种 model parallel。

    Grok-1 默认 mesh = (1, 8) - data 维 1 个 shard、model 维 8 个 shard，相当于纯 tensor parallel 8 卡。MoE 还多一个 expert parallel 维度，但 Grok-1 没单独切 expert，是通过 `shard_map` 在 model 维上把 expert 维度顺手切了。

如果想做 data parallel（让 batch 在多卡之间并行），需要把 mesh 改成 (2, 4) 或 (4, 2) 这一类比例。但 314B 的总参不允许在少于 8 个 device 上做 model 维切分 - 4 个 device 一个 model shard 平均要承担约 78 GB 权重，已经超过单卡 80 GB 的显存上限。所以 mesh 的形状实质上被单卡显存上限固定下来了。

注意 **embedding 的 partition 是 `(None, ("data", "model"))`** - 元组形式表示该维同时沿两条 mesh 轴切（即 device_count = data × model 个分片）。这是 JAX 的"嵌套维度"特性，在 `model.py:166` 体现。

```python
# model.py:162-174
LM_PARTITION_RULES = [
    (("language_model", "positional_embeddings"), P(None, ("data", "model"))),
    (("language_model", "in_out_embed", "embeddings"), P(None, ("data", "model"))),
    (("language_model", "rms_norm"), P(None)),
]
```

第 8 章会详细展开 mesh 和 `pjit` 的用法。第 3 章我们先把 config 字段全部说清楚。

## 延伸阅读

- Cohere [Command R Technical Report](https://docs.cohere.com/docs/command-r) - sandwich norm 的另一个实例
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) - 细粒度专家与 shared expert 的对比设计
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - MoE 路由的奠基论文，理解 capacity factor、aux loss 的源头
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) - Google 关于 MoE 路由设计的早期工作
