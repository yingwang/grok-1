# 第 3 章 配置与超参

本章把 `model.py` 中所有 dataclass / NamedTuple 的字段、它们在 `run.py` 里的实参、以及由此推出的 FLOPs 都列清楚。这是后面所有精读章节的"字典"。

## 3.1 三层配置嵌套

Grok-1 的配置是三层嵌套：

```
LanguageModelConfig          # 顶层，包语言模型超参
  └── TransformerConfig      # 包 Transformer 主干超参
        └── (MoE 字段内嵌)   # 没有独立的 MoEConfig
```

!!! note "Python dataclass（声明式配置类）"
    `@dataclass` 是 Python 3.7 引入的装饰器，自动给类生成 `__init__`、`__repr__`、`__eq__` 这些样板方法。写法是先声明字段（带类型注解，可选默认值），dataclass 装饰器读出这些字段然后合成构造函数。

    ```python
    @dataclass
    class TransformerConfig:
        emb_size: int
        widening_factor: float = 4.0  # 带默认值的字段必须放后面
    ```

    PyTorch 项目里同样的角色通常由 HuggingFace 的 `PretrainedConfig` 子类承担，或者直接用 `argparse`/OmegaConf YAML。dataclass 的好处是**自带类型注解、IDE 补全友好、字段一目了然**，缺点是不能像 OmegaConf 那样支持运行时插值。Grok-1 全栈用 dataclass，没有 YAML 配置文件，所有超参都在 `run.py` 里硬编码。

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

这是个有意思的数字。先把它和 hidden dim 对一下：$\sqrt{6144} \approx 78.38367$，正好是 hidden dim 的平方根。

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

为什么 Vaswani 当年要做这一步放大？把推导走一遍。Embedding 矩阵的初始化常用 $\mathcal{N}(0, 1/d)$ - 让每个分量的方差是 $1/d$（这样 hidden 向量的 L2 范数约为 1，是 PyTorch `nn.Embedding` 默认 init 的标准做法）。查表得到的 hidden 每个分量都是 $\mathcal{N}(0, 1/d)$，整体方差为 1、L2 范数为 1 量级。

但后面 attention 里有 $QK^T/\sqrt{d_h}$ 这种"乘 K 再除根号"的步骤，对输入的 scale 假设是"每个分量方差为 1"。如果 embedding 出来的方差只有 $1/d$，attention 中 Q/K/V 的 scale 就会偏小，最终 softmax logit 难以拉开。Vaswani 的做法是：在 embedding 之后整体乘 $\sqrt{d}$，把每个分量方差从 $1/d$ 放回 1。代入 Grok-1 数字：$\sqrt{6144} \approx 78.38$，正好就是这个常数。

LLaMA 派系不再做这个放大，因为后来发现用 `Uniform(-1/sqrt(d), 1/sqrt(d))` 这种 init scheme 时 embedding 出来的方差就已经在 1 附近，不需要再放。Grok-1 用 `embedding_init_scale=1.0` 走的是早期 init 风格 + 显式 sqrt(d) 放大的组合，与 LLaMA 习惯不同。

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

!!! note "µ-Transfer / µP（让"小模型调参直接迁到大模型"的那套理论）"
    µ-Parameterization（简称 µP，相关论文 Tensor Programs V）是一套 init scale + learning rate scale 的"参数化方案"，目标是：**让超参数（学习率、init scale 等）从小模型到大模型可以零样本迁移**。在 µP 下，先用 100M 模型扫一遍最优 lr，再换 100B 模型直接复用同一个 lr，不需要重新调。

    µP 的关键发现之一是：要让 forward signal 在不同 width 下保持一致，需要把 embedding 出来的激活乘 $\sqrt{d}$，把 logit 投影出来的激活乘 $1/\sqrt{d}$ 或类似常数。Grok-1 的 `embedding_multiplier_scale` 和 `output_multiplier_scale` 一对常数就是这种思路的痕迹。

    PyTorch 用户对照：HuggingFace LLaMA 实现里没有这些 multiplier，因为 LLaMA 走的是标准参数化（standard parameterization），调参时每个尺寸都要单独搜索 lr。xAI 用 µP 是合理的工程决定 - 它能让"先用 1B/10B 模型调好配方，再投入 314B 的训练"这条路径更可靠，避免大模型训练中途因为 lr 不对而失败。

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

!!! note "attention head（多头注意力里"头"的实际含义）"
    一个 attention head 不是独立的子网络，而是把 hidden 维度 $d$ 切成 $H$ 段、每段 $d_h = d / H$ 维分别做一次注意力计算的过程。每个 head 独立学一种"关注模式"（比如有的 head 学语法依赖、有的学指代消解、有的学位置关系），最后把所有 head 的输出 concat 回 $d$ 维再过一次输出投影。

    Grok-1 的 $H_q = 48$、$d_h = 128$，所以 Q 投影后 hidden 被切成 48 段每段 128 维，等价于 48 个独立的"小型 attention"在并行跑。GQA 的特殊之处是 KV 头数只有 8，每 6 个 Q head 共享 1 套 K/V head - 这意味着这 6 个 Q "看同样的内容"但用不同的 query pattern 去匹配。

    PyTorch 对照：`nn.MultiheadAttention(embed_dim=6144, num_heads=48)` 是 MHA 标准实现，但它不支持 GQA。HuggingFace `LlamaAttention` 里手写的 `num_attention_heads=48, num_key_value_heads=8` 才是 GQA 等价物。

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

先把标准 attention 公式拆解一遍。Q 和 K 都是 $d_h$ 维向量，假设它们的每个分量独立同分布且方差 ≈ 1，那么 $\langle Q, K \rangle = \sum_{i=1}^{d_h} Q_i K_i$ 是 $d_h$ 个均值 0、方差 1 的乘积之和，根据中心极限定理，整体方差 ≈ $d_h$，标准差 ≈ $\sqrt{d_h}$。在 Grok-1 里 $d_h = 128$，所以 raw $\langle Q, K \rangle$ 的典型量级约 $\sqrt{128} \approx 11.3$。

如果直接把这个量级的 logit 喂给 softmax，对于 8192 长的序列，softmax 在不同 token 间会显著拉开，但又没拉到极端。除以 $\sqrt{d_h}$ 把方差缩回 1，是为了让 logit 在不同 $d_h$ 设置下都有可比的"温度"。Grok-1 把这一步的常数写成显式字段 `attn_output_multiplier = 1/sqrt(128)`，而不是像 HuggingFace 代码那样写 `scores / math.sqrt(head_dim)`，本质是同一件事。

软裁剪是个少见的细节，主要功能是把 attention logit 限制在 $[-30, +30]$ 区间内，防止 fp32 → fp16/bf16 转换时溢出。代价是当 logit 真的需要很大（比如非常 confident 的 attention），会被裁掉细微差别。

Gemma 2 后来用了类似的 `soft-cap`，Grok-1 是较早采用这个 trick 的公开模型。

更详细分析这个软裁剪的数值含义：`tanh(x/30)` 在 |x| < 10 时几乎线性，几乎不影响小值；在 |x| > 60 时几乎饱和到 ±1，再大的输入也只输出 30 或 -30。这个"软上限 30"对应到 softmax 后的概率，差值约 $e^{60}$ ≈ $10^{26}$ - 比模型实际需要的对比度大得多。所以软裁剪只在"病态情况"（如训练初期 logit 爆炸、或某些数值异常）时起作用，正常情况下接近无效。

但有这一道软裁剪比没有要稳健得多 - 训练过程中只要出现一次严重的数值溢出，loss 就可能突然变成 NaN，由于优化器状态已被污染，整个训练任务往往只能终止。软裁剪是一种代价极低的数值保险。

!!! note "为什么 fp32 cast 在 softmax 之前是必要的"
    上面那段代码第 1 行就把 attn_logits cast 到 `jnp.float32`，看似多余（前面计算用的是 bf16），但这是必须的。softmax 公式 $\frac{e^{x_i}}{\sum_j e^{x_j}}$ 中的 $e^x$ 在 bf16 下当 $x > 88$ 时直接 inf；即便加上 "减去 max" 这种数值稳定 trick，bf16 的尾数精度（7 位）也不够区分相近 logit 之间的细微差别 - 一个相差 0.001 的两个 logit 在 bf16 下可能完全相同。fp32 既有足够范围（指数不会溢出）又有足够精度（尾数 23 位）来支持 softmax。这是几乎所有大模型实现的标准做法 - softmax 这一步一律 fp32，结果再 cast 回 bf16 与 V 相乘。

### 3.3.3 `capacity_factor = 1.0` 但代码里不用

`TransformerConfig.capacity_factor` 默认 1.0，但搜遍 `model.py` 它没出现在任何 MoE 实现里。`MoELayer._inference_call`（`model.py:293-397`）的实现是：所有 token 一律走两个专家，没有任何"超过容量就 drop"的逻辑。

这是 Grok-1 MoE 实现的一个关键观察：**没有 capacity drop**。第 5 章会详细对比 Mixtral 的实现，Mixtral 同样去掉了 capacity drop，但 Switch Transformer / GLaM 是有 drop 的。

!!! note "capacity_factor 的语义（即便 Grok-1 不用，知道它含义有助于看其他 MoE 代码）"
    假设 batch 里有 $N$ 个 token，每个走 $k$ 个 expert，总共 $N \cdot k$ 次 expert 调用，平均到 $E$ 个 expert 每个应该被分配 $N k / E$ 个 token。但实际路由不会完美均衡，可能有些 expert 被分到很多、有些很少。

    capacity_factor 定义"每个 expert 最多能接的 token 数 = $\lceil (N k / E) \cdot \text{capacity\_factor} \rceil$"。capacity_factor = 1.0 表示严格按平均值上限，超出部分丢弃；1.25-2.0 表示给点冗余，能容纳路由不均衡，超出才丢。

    drop 的代价：被 drop 的 token 这一层 MoE 不参与 FFN 计算，hidden 直接走 residual，相当于跳过这一层。训练时这种行为可能让某些 token 学不到东西，但配合 aux loss 提高均衡度后，drop 率通常 < 5%，影响有限。Grok-1 推理不做 drop 的代价是：要预留每个 expert 能处理全部 batch token 的显存余地，但推理 batch 一般不大，余地也不大，所以 xAI 选了"代码简单优先"。

### 3.3.4 `widening_factor = 8` 与实际 FFN 大小

字面上 widening_factor 是 8，但 `ffn_size(6144, 8) = 32768`（见第 2 章 2.3）。这里 ffn_size 函数中的 `* 2 // 3` 是 SwiGLU 派的惯例 - 因为 SwiGLU 有 gate 矩阵，参数比 GELU FFN 多 50%，所以把中间维度乘 2/3 来对齐总参数量。

代入数字看推导。`ffn_size` 函数体：

```python
def ffn_size(emb_size, widening_factor):
    _ffn_size = int(widening_factor * emb_size) * 2 // 3
    _ffn_size = _ffn_size + (8 - _ffn_size) % 8  # ensure it's a multiple of 8
    return _ffn_size
```

第一步：`int(8 * 6144) * 2 // 3 = 49152 * 2 // 3 = 32768`。
第二步：检查是否是 8 的倍数。`32768 % 8 = 0`，所以 `(8 - 0) % 8 = 0`，不需要额外补齐。最终 $d_{\text{ffn}} = 32768$。

第二步的"补到 8 的倍数"是为了配合 GPU Tensor Core 在 16/32 维度块上的高效计算。如果 widening 设成奇数比如 7，第一步会得到 $\lfloor 7 \cdot 6144 \rfloor \cdot 2 // 3 = 28672$，已经是 8 的倍数；但如果设成 5.5，第一步是 $\lfloor 5.5 \cdot 6144 \rfloor \cdot 2 // 3 = 22528$，仍是 8 的倍数。这个补齐机制兜底处理那些"算下来不是 8 倍数"的边界 case。

PyTorch 对照：HuggingFace `LlamaConfig` 里的 `intermediate_size` 直接就是 $d_{\text{ffn}}$，不像 Grok-1 这样间接通过 widening_factor 推导。Grok-1 这种写法的好处是想做 scaling law 实验时只需要扫 widening_factor 一个标量；缺点是字段名 widening_factor=8 容易让人误以为 FFN 中间维度是 8 倍 hidden，实际只有约 5.33 倍。

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

!!! note "Haiku Module vs PyTorch nn.Module"
    `hk.Module` 是 Haiku 提供的"看起来像 PyTorch nn.Module"的语法糖。差别在于：

    - **参数声明位置不同**：PyTorch 在 `__init__` 里 `self.linear = nn.Linear(d, d)` 就建好了参数；Haiku 在 `__call__` 里第一次执行 `hk.Linear(d)(x)` 时才通过 `hk.get_parameter("w", ...)` 延迟创建参数（这就是为什么 Haiku 模型必须先 `forward_t.init(rng, dummy_input)` 跑一次 dummy forward 才能拿到参数）。
    - **forward 调用方式不同**：PyTorch 是 `model(x)` 直接调；Haiku 在 transform 之外不能直接调，必须通过 `forward_t.apply(params, rng, x)`。
    - **`with_bias` 的默认值**：Haiku `hk.Linear` 默认 `with_bias=True`，PyTorch `nn.Linear` 也是默认有 bias。但现代大模型几乎都显式关掉 bias（Grok-1 的 MHA `final_projection` 和所有 FFN linear 都是 `with_bias=False`），因为 RMSNorm 后再加 bias 几乎没有意义、还会引入额外参数和数值噪声。

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

!!! note "router dtype 为什么是 fp32"
    上面这行 `hk.get_parameter(..., jnp.float32, ...)` 把 router 权重明确指定成 fp32，而模型其他参数（attention、FFN）都是 bf16。原因和上一节的 softmax 一样：router 输出会经过 fp32 softmax 算 top-k 概率，权重保持 fp32 避免在"线性层 + softmax"这条小通路里出现精度损失导致路由不稳定。

    Mixtral、DeepSeek 系列的 router 也都用 fp32。这是工业界对 MoE 路由的一个共识：**router 数值稳定性比节省一点显存重要得多** - router 微小的数值扰动可能让某个 token 被路由到不同 expert，引起整段输出剧变。fp32 router 是廉价的保险。

## 3.6 由 config 推出的 FLOPs

### 3.6.1 每 token 的 FLOPs

按 [Chinchilla scaling](https://arxiv.org/abs/2203.15556) 风格的近似 $\text{FLOPs/token} \approx 2 \cdot P_{\text{active}}$ - 因为前向是 1 次 matmul、反向是 2 次，推理只算前向，每参数 2 次 op。

!!! note "FLOPs / GFLOPs / TFLOPs 单位说明 + "每参数 2 次"的由来"
    **FLOP**（FLoating-point OPeration）指一次浮点运算（加或乘）。**FLOPs** 是 FLOP 的复数，前缀按千进位：MFLOPs = $10^6$、GFLOPs = $10^9$、TFLOPs = $10^{12}$、PFLOPs = $10^{15}$。注意末尾的 's' 是复数标志，不是"秒"；表示算力的常说 "TFLOPs/s"（每秒浮点运算次数）才是吞吐量单位。

    "每参数 2 次"是经验近似。一个 $(m, n)$ 矩阵乘 $(n,)$ 向量得到 $(m,)$ 向量，做 $m \cdot n$ 次乘法和 $m \cdot (n-1)$ 次加法，合起来约 $2 \cdot m \cdot n$ 次浮点运算 - 正好是矩阵参数数 $m \cdot n$ 的 2 倍。Transformer 推理时绝大部分计算都是这种矩阵乘，所以总 FLOPs ≈ 2 × 总参数。训练时还要加上反向传播（约前向的 2 倍），合起来约 6 × 总参数 - 这就是 Chinchilla 论文里的"$C = 6ND$"公式（C 是总训练 FLOPs、N 是参数数、D 是 token 数）的来源。

对推理：

$$
\text{FLOPs/token} \approx 2 \cdot 86\text{B} = 172\,\text{GFLOPs/token}
$$

但这是忽略 attention 的近似。把 attention 那部分也算进来：

attention 里的两个主要 matmul 是 $QK^T$ 和 $\text{softmax}(QK^T) V$。对一个新生成的 token，Q shape 是 $(H_q, d_h) = (48, 128)$、K 是 $(H_{kv}, T, d_h) = (8, T, 128)$，但 GQA 下 Q head 被分成 8 组每组 6 个，每组共用一个 K head，所以 $QK^T$ 的运算量约为 $H_q \cdot T \cdot d_h = 48 \cdot T \cdot 128$；同理 attention @ V 也是这个量级。两个 matmul 各算两次浮点（乘 + 加），所以每层 attention 的 FLOPs 约 $4 \cdot H_q \cdot d_h \cdot T$。乘 $L$ 层得到：

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

!!! note "compute-bound vs memory-bound（推理阶段两种瓶颈）"
    判断一个 kernel 是 compute-bound 还是 memory-bound，看 **arithmetic intensity**（算术强度）= FLOPs / bytes_read。如果这个比值远大于硬件的 FLOPs/bandwidth 比，说明对每字节数据都做了大量计算 - 瓶颈是算力（compute-bound），典型例子是大 batch 的训练前向，每读一份参数都对它做几百个 token 的矩阵乘。如果比值小，说明计算很简单但数据量大 - 瓶颈是带宽（memory-bound），典型例子是 decode 阶段每次只对每个权重做 1 次 token 的乘加。

    H100 在 bf16 下算力约 989 TFLOPs/s、HBM 带宽约 3.35 TB/s，比值约 989/3.35 ≈ 295 FLOPs/byte。Decode 阶段一个 token 对每个权重只做 2 FLOPs（一次乘一次加），bytes per weight 是 2（bf16），算术强度只有 1 FLOPs/byte，远低于 295 - 完全是 memory-bound。这就是为什么单 batch decode 时 GPU 算力利用率经常只有 1-5%。

每 token 读取量：

$$
B_{\text{read}} = P_{\text{active}} \cdot \text{bytes\_per\_param}
$$

bf16 时是 86B × 2 bytes = 172 GB。在 H100（HBM 3.35 TB/s）上单卡理论上限 ~19 token/s；8 卡并行（带 NVLink 通信）实际能达 ~50-80 token/s。

8-bit 量化能把读取量减半到 86 GB，理论上限翻倍。但 KV cache 仍 bf16，还要算上 cache 读取（每步 ~3 GB/层，对长序列影响显著）。

这就是为什么 Grok-1 推理速度在同一代 MoE 模型中明显落后于 Mixtral 8x7B - 激活参数从 13B 上升到 86B，每 token 需要从 HBM 读取的字节数随之上升约 6.6 倍，在带宽受限的 decode 阶段，单 token 时延必然成比例延长。

### 3.6.4 prefill vs decode：两种推理阶段的算术强度差异

推理还可以再细分成两个阶段。**Prefill 阶段** 是把用户输入的 prompt（比如 500 个 token）一次性塞给模型，算出所有位置的 KV cache 和最后一个位置的 logits；这一步可以把 prompt 当成 batch=500 的输入并行处理。**Decode 阶段** 是从最后一个位置开始一次产生 1 个 token，反复迭代直到 stop 或达到 max_new_tokens。

两阶段的算术强度差异很大：

- **prefill**：每读一份参数，对 prompt 长度 $T_p$ 个 token 都做矩阵乘，算术强度 ≈ $T_p$ FLOPs/byte。在 $T_p = 500$、bf16 时约 250 FLOPs/byte，接近 H100 的 295 上限，基本 compute-bound。
- **decode**：每读一份参数只对 1 个 token 做计算，算术强度 ≈ 1 FLOPs/byte，严重 memory-bound。

这个差异解释了用户体感：用 Grok-1 处理一个 1000 token 的 prompt，prefill 可能只要 1-2 秒（算力跑满），但生成回答的每个 token 都要 ~100-200ms（被带宽卡死），所以"首 token 延迟低、后续吐字慢"。优化推理的关键往往就是把 decode 阶段的算术强度抬上去 - speculative decoding、batch 推理、KV cache 量化都是朝这个方向做的。

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
