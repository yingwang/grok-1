# 第 3 章 配置与超参

本章把 `model.py` 中所有 dataclass / NamedTuple 的字段、它们在 `run.py` 里的实参、以及由此推出的 FLOPs 都列清楚。这是后面所有精读章节的"字典"。

## 3.1 三层配置嵌套

Grok-1 的配置是三层嵌套：

```
LanguageModelConfig          # 顶层，包语言模型超参
  └── TransformerConfig      # 包 Transformer 主干超参
        └── (MoE 字段内嵌)   # 没有独立的 MoEConfig
```

下面逐个字段过。

## 3.2 `LanguageModelConfig`

定义在 `model.py:1146-1194`：

```python
# model.py:1146-1163
@dataclass
class LanguageModelConfig:
    model: Optional[TransformerConfig]
    vocab_size: int
    pad_token: int
    eos_token: int
    sequence_len: int
    model_size: int = 0
    embedding_init_scale: float = 1.0
    embedding_multiplier_scale: float = 1.0
    output_multiplier_scale: float = 1.0
    name: Optional[str] = None
    fprop_dtype: Any = jnp.bfloat16
    model_type: Optional[str] = None
    init_scale_override: Optional[float] = None
    shard_embeddings: bool = True
```

| 字段 | 默认 | run.py 实参 | 中文解释 |
| --- | --- | --- | --- |
| `model` | None | `TransformerConfig(...)` | 嵌套的 Transformer 主干 config |
| `vocab_size` | - | `128 * 1024 = 131072` | SentencePiece 词表大小 |
| `pad_token` | - | `0` | padding token id（亦是 `<unk>`） |
| `eos_token` | - | `2` | 句尾 token id |
| `sequence_len` | - | `8192` | 最大上下文长度 |
| `model_size` | 0 | （0，构造时自动设为 `model.emb_size`） | hidden dim，未指定则取主干 emb_size |
| `embedding_init_scale` | 1.0 | `1.0` | embedding 初始化 scale（推理时无用，训练才用） |
| `embedding_multiplier_scale` | 1.0 | `78.38367176906169` | **token embedding 拿出来后乘的常数** |
| `output_multiplier_scale` | 1.0 | `0.5773502691896257` | **decode 出来的 logits 再乘的常数** |
| `name` | None | - | 名字 |
| `fprop_dtype` | `jnp.bfloat16` | （默认） | 前向计算 dtype |
| `model_type` | None | - | 未用 |
| `init_scale_override` | None | - | 仅训练用 |
| `shard_embeddings` | True | - | 是否对 embedding 做 sharding |

### 3.2.1 `embedding_multiplier_scale = 78.38367176906169`

这是个有意思的数字。注意 $\sqrt{6144} \approx 78.38367$ - 正好是 hidden dim 的平方根。

`model.py:1233-1237`：

```python
# model.py:1233-1237
input_embeddings = in_out_embed(tokens).astype(config.fprop_dtype)
input_embeddings = with_sharding_constraint(
    input_embeddings, P("data", None, self.model.model_axis)
)
input_embeddings *= config.embedding_multiplier_scale
```

也就是说，token embedding 被乘了 $\sqrt{d}$ 量级的放大因子。这是经典的"original Transformer"做法（Vaswani 2017）。LLaMA 不这么做（embedding 直接用），Mixtral 也不这么做。

它和后面的 `output_multiplier_scale = 1/\sqrt{3}` 配对，整体起到一个 µ-Transfer 风格的 logit 量级控制。

### 3.2.2 `output_multiplier_scale = 0.5773502691896257`

$0.5773502691896257 = 1/\sqrt{3} \approx 0.5774$。

`model.py:1265-1269`：

```python
# model.py:1265-1269
out = in_out_embed.decode(embeddings)
rank_logger.info(out.shape)
out *= config.output_multiplier_scale
```

logits 整体除以 $\sqrt{3}$。可能的解释：训练时 attention logit 经过 `tanh` 软裁剪到 $\pm 30$，配合 output 这个 $1/\sqrt{3}$，使得最终 softmax 的温度落在一个合适的区间。具体推导 xAI 没公开，这里属于"看到了但不能 100% 解释"的细节。

## 3.3 `TransformerConfig`

定义在 `model.py:420-487`：

```python
# model.py:420-449
@dataclass
class TransformerConfig:
    emb_size: int
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int = 128 * 1024
    widening_factor: float = 4.0

    attn_output_multiplier: float = 1.0

    name: Optional[str] = None

    num_experts: int = -1
    capacity_factor: float = 1.0
    num_selected_experts: int = 1

    init_scale: float = 1.0
    shard_activations: bool = False

    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
```

| 字段 | 默认 | run.py 实参 | 中文解释 |
| --- | --- | --- | --- |
| `emb_size` | - | `48 * 128 = 6144` | hidden / model dim |
| `key_size` | - | `128` | 每个 attention head 的 dim |
| `num_q_heads` | - | `48` | Q 头数 |
| `num_kv_heads` | - | `8` | KV 头数（GQA 关键） |
| `num_layers` | - | `64` | Transformer 层数 |
| `vocab_size` | 131072 | - | 在 LanguageModelConfig 也有一份（重复） |
| `widening_factor` | 4.0 | `8` | FFN 宽化倍数（实际 ffn_size 还经过 2/3 缩放） |
| `attn_output_multiplier` | 1.0 | `0.08838834764831845` | **attention logit 的缩放因子** |
| `name` | None | - | - |
| `num_experts` | -1 | `8` | 专家数 |
| `capacity_factor` | 1.0 | （未传，沿默认） | **代码里没用到这个字段** - 见 3.3.3 |
| `num_selected_experts` | 1 | `2` | top-k 的 k |
| `init_scale` | 1.0 | （未传） | 仅训练用 |
| `shard_activations` | False | `True` | 是否对中间激活做 sharding |
| `data_axis` / `model_axis` | "data"/"model" | （同默认） | mesh 轴名 |

### 3.3.1 `key_size = 128`

每个 attention head 内部 dim 是 128。Q 头数 48 × 128 = 6144 = $d$，所以 Q 投影矩阵是 $(6144, 6144)$。KV 头数 8 × 128 = 1024，K 与 V 投影矩阵都是 $(6144, 1024)$。

`model.py:773` 有 assert：

```python
assert self.num_q_heads % self.num_kv_heads == 0
```

48 % 8 = 0，每 KV 头被 6 个 Q 头共享。

### 3.3.1.1 head_dim = 128 的工程理由

为什么 head_dim 选 128？业内有几个事实：

1. **NVIDIA Tensor Core 在 128 维上有最佳 throughput** - 128 是 16 (TC tile) × 8 的倍数
2. **FlashAttention 在 head_dim ∈ {64, 96, 128, 192, 256} 上有专门优化**，128 是最常用
3. **head_dim 太小**（<64）会让 head 间的特征区分能力下降；太大（>256）会让 attention pattern 难学，且 RoPE 频率分辨率变差

Llama2、Mistral、Mixtral、DeepSeek、Qwen 全部选 128。Grok-1 跟随这个标准。

### 3.3.2 `attn_output_multiplier = 0.08838834764831845`

$0.08838834764831845 = 1/\sqrt{128} \approx 0.08839$ - 注意是 $1/\sqrt{d_h}$ 而不是 $1/\sqrt{d_k}$。

`model.py:860-865`：

```python
# model.py:860-865
attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(
    jnp.float32
)
attn_logits *= self.attn_output_multiplier
max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)
```

也就是说，标准 attention 公式 $\text{softmax}(QK^T/\sqrt{d_k})$ 里的缩放在这里被显式拆出来做：先乘 $1/\sqrt{128}$，再过 `30 * tanh(x/30)` 软裁剪。

软裁剪是个少见的细节，主要功能是把 attention logit 限制在 $[-30, +30]$ 区间内，防止 fp32 → fp16/bf16 转换时溢出。代价是当 logit 真的需要很大（比如非常 confident 的 attention），会被裁掉细微差别。

Gemma 2 后来用了类似的 `soft-cap`，Grok-1 是较早采用这个 trick 的公开模型。

更详细分析这个软裁剪的数值含义：`tanh(x/30)` 在 |x| < 10 时几乎线性，几乎不影响小值；在 |x| > 60 时几乎饱和到 ±1，再大的输入也只输出 30 或 -30。这个"软上限 30"对应到 softmax 后的概率，差值约 $e^{60}$ ≈ $10^{26}$ - 比模型实际需要的对比度大得多。所以软裁剪只在"病态情况"（如训练初期 logit 爆炸、或某些数值异常）时起作用，正常情况下接近无效。

但有这一道软裁剪比没有要稳健得多 - 训练过程中只要出现一次严重的数值溢出，loss 就可能突然变成 NaN，由于优化器状态已被污染，整个训练任务往往只能终止。软裁剪是一种代价极低的数值保险。

### 3.3.3 `capacity_factor = 1.0` 但代码里不用

`TransformerConfig.capacity_factor` 默认 1.0，但搜遍 `model.py` 它没出现在任何 MoE 实现里。`MoELayer._inference_call`（`model.py:293-397`）的实现是：所有 token 一律走两个专家，没有任何"超过容量就 drop"的逻辑。

这是 Grok-1 MoE 实现的一个关键观察：**没有 capacity drop**。第 5 章会详细对比 Mixtral 的实现，Mixtral 同样去掉了 capacity drop，但 Switch Transformer / GLaM 是有 drop 的。

### 3.3.4 `widening_factor = 8` 与实际 FFN 大小

字面上 widening_factor 是 8，但 `ffn_size(6144, 8) = 32768`（见第 2 章 2.3）。这里 ffn_size 函数中的 `* 2 // 3` 是 SwiGLU 派的惯例 - 因为 SwiGLU 有 gate 矩阵，参数比 GELU FFN 多 50%，所以把中间维度乘 2/3 来对齐总参数量。

## 3.4 `MultiHeadAttention` 的参数

不是独立 dataclass，是一个 `hk.Module`。`model.py:694-720`：

```python
# model.py:694-718
class MultiHeadAttention(hk.Module):
    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        *,
        with_bias: bool = True,
        value_size: Optional[int] = None,
        model_size: Optional[int] = None,
        attn_output_multiplier: 1.0,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        name: Optional[str] = None,
    ):
```

注意 `attn_output_multiplier: 1.0` 这一行 - 这是 Python 类型注解里的语法错误（应是 `: float = 1.0`），但 Python 不会报错，因为 `1.0` 被当作 type 处理但实际不影响构造。源码留了这个小 bug，可见这份代码没经过严格 mypy 校验。

字段：

| 字段 | 中文 |
| --- | --- |
| `num_q_heads` | Q 头数（48） |
| `num_kv_heads` | KV 头数（8） |
| `key_size` | 每头 dim（128） |
| `with_bias` | 是否有 bias（实际在 `MHABlock` 调用时只对 final_projection 用 with_bias=False） |
| `value_size` | V 的 head dim，默认与 key_size 相等 |
| `model_size` | 输出投影回到的 dim，默认 `key_size * num_q_heads` = 6144 |
| `attn_output_multiplier` | 见 3.3.2 |

## 3.5 路由参数：藏在 `Router` 里

`Router` 也不是 dataclass，构造在 `model.py:208-224`：

```python
# model.py:208-224
class Router(hk.Module):
    def __init__(
        self,
        num_selected_experts: int,
        data_axis: Union[str, Tuple[str, ...]] = "data",
        model_axis: Union[str, Tuple[str, ...]] = "model",
        shard_activations: bool = False,
        mesh: Any = None,
        name: str = "router",
    ):
```

只有 `num_selected_experts`（=2）一个核心超参。其他都是 sharding 相关。

权重在 `_router_weights`（`model.py:250-269`）：

```python
# model.py:262-264
w = hk.get_parameter(
    "w", [input_size, num_experts], jnp.float32, init=hk.initializers.Constant(0)
)
```

即 router 是一个 $(d, E) = (6144, 8)$ 的线性层，**没有 bias**。

注意 init 是 `Constant(0)` - 这只是个占位 init，真实权重从 ckpt 加载。所以"路由器初始全 0"在推理时不影响。

## 3.6 由 config 推出的 FLOPs

### 3.6.1 每 token 的 FLOPs

按 [Chinchilla scaling](https://arxiv.org/abs/2203.15556) 风格的近似 $\text{FLOPs/token} \approx 2 \cdot P_{\text{active}}$ - 因为前向是 1 次 matmul、反向是 2 次，推理只算前向，每参数 2 次 op。

对推理：

$$
\text{FLOPs/token} \approx 2 \cdot 86\text{B} = 172\,\text{GFLOPs/token}
$$

但这是忽略 attention 的近似。加上 attention 部分（与序列长度成平方）：

$$
\text{FLOPs}_{\text{attn}}/\text{token} \approx 4 \cdot L \cdot H_q \cdot d_h \cdot T = 4 \cdot 64 \cdot 48 \cdot 128 \cdot T \approx 1.57\text{M} \cdot T
$$

在 $T = 8192$ 时为 $\approx 12.9\,\text{GFLOPs/token}$ - 比 FFN 部分小一个量级。

### 3.6.1.1 长上下文下的 attention 占比

当 $T$ 变大，attention 的 FLOPs 占比上升。在 $T = 8192$（Grok-1 上下文上限）：

$$
\frac{\text{FLOPs}_{\text{attn}}}{\text{FLOPs}_{\text{total}}} \approx \frac{12.9}{172 + 12.9} \approx 7\%
$$

attention 占总 FLOPs 的 7%，主体仍是 FFN（MoE）。这意味着 Grok-1 即便扩展到更长上下文（比如 32K），attention 也只占 ~25% - 主体计算仍是 MoE。这与 MLA 系（DeepSeek）"压缩 KV 节省 attention"的优化方向不同，Grok-1 优化方向应该是"减少 expert 浪费"。

### 3.6.2 与 dense 314B 的对比

假想一个 314B dense 模型，每 token 约 628 GFLOPs。Grok-1 MoE 把这个降到 172 GFLOPs/token，节省 73%。

但 **MoE 不能节省内存**：314B 参数全都必须加载到 GPU 显存中，HBM 的带宽在 decode 阶段也会被几乎用尽。MoE 节省的是**算力**和**通信**（如果 expert 在多卡间合理分片，被激活的 2 个 expert 也只需要在持有它们的少数卡之间通信）。

### 3.6.3 推理时的内存带宽限制

推理 decode 阶段（每次只产生 1 token），瓶颈往往不是 FLOPs 而是**内存带宽** - 每生成一个 token，需要从 HBM 读取所有激活的参数。

每 token 读取量：

$$
B_{\text{read}} = P_{\text{active}} \cdot \text{bytes\_per\_param}
$$

bf16 时是 86B × 2 bytes = 172 GB。在 H100（HBM 3.35 TB/s）上单卡理论上限 ~19 token/s；8 卡并行（带 NVLink 通信）实际能达 ~50-80 token/s。

8-bit 量化能把读取量减半到 86 GB，理论上限翻倍。但 KV cache 仍 bf16，还要算上 cache 读取（每步 ~3 GB/层，对长序列影响显著）。

这就是为什么 Grok-1 推理速度在同一代 MoE 模型中明显落后于 Mixtral 8x7B - 激活参数从 13B 上升到 86B，每 token 需要从 HBM 读取的字节数随之上升约 6.6 倍，在带宽受限的 decode 阶段，单 token 时延必然成比例延长。

## 3.7 `MHAttentionConfig`：在 Grok-1 中并不存在

题目里之所以提到 `MHAttentionConfig` 这个名字，是因为很多 MoE 项目（如 Megablocks、tutel）会为 attention 单独拆一个 config dataclass。但 Grok-1 实际上没有这个 dataclass，`MultiHeadAttention` 模块所需的超参全部由 `TransformerConfig` 直接传入。这一节的目的就是澄清这件事：**Grok-1 的配置只有 `LanguageModelConfig` 和 `TransformerConfig` 两层嵌套**，MoE 字段、路由字段、Attention 字段都内嵌在 `TransformerConfig` 里，没有独立的子 config。

这是相对其他 MoE 项目（如 Megablocks、tutel）较简化的设计，但也意味着 config 缺乏正交性 - 想换 attention 实现就得改 TransformerConfig 字段。

## 3.8 `run.py` 实参与 config 字段的对应总表

| run.py 行 | 字段 | 实参 |
| --- | --- | --- |
| L26 | `vocab_size` | 131072 |
| L27 | `pad_token` | 0 |
| L28 | `eos_token` | 2 |
| L29 | `sequence_len` | 8192 |
| L30 | `embedding_init_scale` | 1.0 |
| L31 | `output_multiplier_scale` | 0.5773502691896257 |
| L32 | `embedding_multiplier_scale` | 78.38367176906169 |
| L34 | `emb_size` | 6144 |
| L35 | `widening_factor` | 8 |
| L36 | `key_size` | 128 |
| L37 | `num_q_heads` | 48 |
| L38 | `num_kv_heads` | 8 |
| L39 | `num_layers` | 64 |
| L40 | `attn_output_multiplier` | 0.08838834764831845 |
| L41 | `shard_activations` | True |
| L43 | `num_experts` | 8 |
| L44 | `num_selected_experts` | 2 |
| L46 | `data_axis` | "data" |
| L47 | `model_axis` | "model" |

这张表是后面所有章节都会回头看的"配方"。第 4 章开始我们正式进入 `model.py` 精读。

## 延伸阅读

- [Attention Is All You Need (Vaswani 2017)](https://arxiv.org/abs/1706.03762) - sqrt(d) embedding scaling 的源头
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) - µ-Transfer 的原始论文，理解 output multiplier 的设计意图
- [Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118) - 同样用了 logit soft-cap
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - 同样去掉了 capacity drop
