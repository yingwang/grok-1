# 第 6 章 model.py 精读·下：Block 组装与切分

本章把 `model.py` 剩下的部分讲完：

- `MHABlock`：把 attention 封装成"输入 -> 输出"的子层
- `DenseBlock`：单个 SwiGLU FFN（也是 MoE 的 layer_fn）
- `DecoderLayer`：sandwich norm + MoE 选择
- `LanguageModel` 与 `Transformer`：顶层组装
- `InOutEmbed`：tied embedding 实现
- causal mask
- KV cache 的接口

读完这一章，你能完整在脑子里跑一遍 `forward(tokens) -> logits`。

## 6.1 `MHABlock`

`model.py:914-960`：

```python
# model.py:914-960
@dataclass
class MHABlock(hk.Module):
    """A MHA Block"""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    attn_output_multiplier: float = 1.0
    mesh: Any = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,  # [B, 1, T, T] or [B, 1, 1, T] or B[1, 1, 1, 1]
        layer_memory: Optional[KVMemory],
    ) -> MHAOutput:
        _, _, model_size = inputs.shape
        assert mask.ndim == 4, f"shape: {mask.shape}"
        assert mask.shape[2] in {1, inputs.shape[1]}, str(mask.shape)
        assert mask.shape[3] in {1, inputs.shape[1]}, str(mask.shape)
        side_input = inputs

        def attn_block(query, key, value, mask, memory) -> MHAOutput:
            return MultiHeadAttention(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                model_size=model_size,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                attn_output_multiplier=self.attn_output_multiplier,
            )(
                query, key, value, mask, memory, mesh=self.mesh,
            )

        attn_output = attn_block(inputs, side_input, side_input, mask, layer_memory)
        h_attn = attn_output.embeddings

        return attn_output._replace(embeddings=h_attn)
```

这层只是个**自注意力包装** - `side_input = inputs` 让 query/key/value 都是同一个张量。可以理解为"self-attention block"的语法糖。

`@hk.transparent` 装饰器让这个 module 的参数 scope **不进入** Haiku 命名空间 - 也就是说 `MultiHeadAttention` 创建的参数名直接挂在父 module（DecoderLayer）下，不会变成 `mha_block/multi_head_attention/query/w`，而是 `decoder_layer_0/multi_head_attention/query/w`。

这种设计让 ckpt 的参数路径短一些，但代价是 `MHABlock` 不能被独立 init/apply。

## 6.2 `DenseBlock`：单专家的 FFN

`model.py:963-1007`：

```python
# model.py:963-1007
@dataclass
class DenseBlock(hk.Module):
    num_q_heads: int
    num_kv_heads: int
    key_size: int
    widening_factor: float = 4.0
    sharding_constraint: bool = False
    mesh: Any = None

    @hk.transparent
    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
    ) -> jax.Array:  # [B, T, D]
        _, _, model_size = inputs.shape
        h_v = Linear(
            ffn_size(model_size, self.widening_factor),
            with_bias=False,
            mesh=self.mesh,
            sharding=P("data", "model"),
            name="linear_v",
        )(inputs)
        h_w1 = jax.nn.gelu(
            Linear(
                ffn_size(model_size, self.widening_factor),
                with_bias=False,
                mesh=self.mesh,
                sharding=P("data", "model"),
            )(inputs)
        )
        h_dense = Linear(
            model_size,
            with_bias=False,
            sharding=P("model", "data"),
            mesh=self.mesh,
            shard_axis=1,
        )(h_w1 * h_v)

        return h_dense
```

数学：

$$
\text{FFN}(x) = W_1 \cdot (\text{GELU}(W_g x) \odot W_v x)
$$

注意 `Linear(...)`（中间那个，没传 name）会得到 `linear` 作为默认 name（来自 `hk.Linear` 的默认）。所以三个矩阵的参数名是：

- `linear_v`：up projection 1（value path）
- `linear`：up projection 2（gate path）
- `linear_1`：down projection

这正好匹配 partition rules 里 `linear`、`linear_v`、`linear_1` 三个名字。

`num_q_heads`、`num_kv_heads`、`key_size` 在 DenseBlock 里**完全没用**（构造参数里有，但 `__call__` 没读）- 这是从一个更大的 module 复制过来的残留代码，不影响功能。

### 6.2.1 SwiGLU 还是 GeGLU？

回到第 5 章的讨论 - 这里 `jax.nn.gelu`，是 **GELU**，所以严格来说是 **GeGLU**。但社区习惯都把这类 gated FFN 叫 SwiGLU，本书后文也偶尔用 SwiGLU 这个泛称。

## 6.3 `DecoderLayer`：sandwich norm 的真身

`model.py:1010-1102`：

```python
# model.py:1010-1102
@dataclass
class DecoderLayer(hk.Module):
    """A transformer stack."""

    num_q_heads: int
    num_kv_heads: int
    key_size: int
    num_layers: int
    # MoE.
    num_experts: int
    layer_index: Optional[int] = None
    num_selected_experts: int = 1
    widening_factor: float = 4.0
    name: Optional[str] = None
    data_axis: Union[str, Tuple[str, ...]] = "data"
    model_axis: Union[str, Tuple[str, ...]] = "model"
    shard_activations: bool = False
    attn_output_multiplier: float = 1.0
    mesh: Any = None

    def __call__(
        self,
        inputs: jax.Array,  # [B, T, D]
        mask: jax.Array,    # [B, 1, T, T] or [B, 1, 1, T]
        padding_mask: Optional[jax.Array],
        layer_memory: Optional[KVMemory],
    ) -> DecoderOutput:
        """Transforms input embedding sequences to output embedding sequences."""

        def layer_norm(x):
            return hk_rms_norm(x)

        if self.shard_activations:
            sharding = P(self.data_axis, None, self.model_axis)
        else:
            sharding = P(self.data_axis, None)
        h = with_sharding_constraint(inputs, sharding)

        attn_output = MHABlock(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            attn_output_multiplier=self.attn_output_multiplier,
            mesh=self.mesh,
            data_axis=self.data_axis,
            model_axis=self.model_axis,
        )(layer_norm(h), mask, layer_memory)
        h_attn = attn_output.embeddings

        h_attn = layer_norm(h_attn)
        h += h_attn
        h = with_sharding_constraint(h, sharding)

        def base_dense_block(h):
            h = DenseBlock(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=self.widening_factor,
                sharding_constraint=False,
                mesh=self.mesh,
            )(h)
            return h

        if self.num_experts > 1:
            rank_logger.debug("Using MoE!")
            router = Router(
                num_selected_experts=self.num_selected_experts,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
                mesh=self.mesh,
            )
            h_dense = MoELayer(
                num_experts=self.num_experts,
                mesh=self.mesh,
                layer_fn=base_dense_block,
                router=router,
                shard_activations=self.shard_activations,
                data_axis=self.data_axis,
                model_axis=self.model_axis,
            )(layer_norm(h), padding_mask)
        else:
            h_dense = base_dense_block(layer_norm(h))

        h_dense = layer_norm(h_dense)
        h += h_dense
        h = with_sharding_constraint(h, sharding)

        return DecoderOutput(
            embeddings=h,
            memory=attn_output.memory,
        )
```

### 6.3.1 sandwich norm 的完整流程

把每一句话拆开：

```
h = inputs                                  # residual stream
h_attn = MHABlock(layer_norm(h), ...)       # pre-norm attention
h_attn = layer_norm(h_attn)                 # post-norm
h += h_attn                                 # residual add

h_dense = MoELayer(layer_norm(h), ...)      # pre-norm FFN
h_dense = layer_norm(h_dense)               # post-norm
h += h_dense                                # residual add
```

每个 sub-layer 用了**两次 RMSNorm**。所有 RMSNorm 调用都是 `hk_rms_norm(x)`，每次都会创建一个新的 RMSNorm module（按 Haiku 的命名规则，name 自动递增）。

按 `model.py:137-140` 的 partition rule：

```python
((r"decoder_layer_[0-9]+", "rms_norm", "scale"), P(None)),
((r"decoder_layer_[0-9]+", "rms_norm_1", "scale"), P(None)),
((r"decoder_layer_[0-9]+", "rms_norm_2", "scale"), P(None)),
((r"decoder_layer_[0-9]+", "rms_norm_3", "scale"), P(None)),
```

每层有 `rms_norm`、`rms_norm_1`、`rms_norm_2`、`rms_norm_3` 四个。对应：

| 位置 | 名字 |
| --- | --- |
| pre-attn | `rms_norm` |
| post-attn | `rms_norm_1` |
| pre-FFN | `rms_norm_2` |
| post-FFN | `rms_norm_3` |

每个 RMSNorm 有 6144 个 scale 参数。

### 6.3.1.1 命名规则的细节

Haiku 的命名规则是：每次创建同名 module，名字自动加后缀 `_1`、`_2`、...。`DecoderLayer.__call__` 里调用 `hk_rms_norm(x)` 4 次，每次都创建一个新的 `RMSNorm`，按顺序得到：

1. `rms_norm` (第一次)
2. `rms_norm_1` (第二次)
3. `rms_norm_2` (第三次)
4. `rms_norm_3` (第四次)

这要求**调用顺序固定** - 如果有 if 分支让某些调用被跳过，name 会错乱。Grok-1 的 DecoderLayer 没有这种分支（`if num_experts > 1` 只影响 MoE/DenseBlock，不影响 RMSNorm 调用），所以是安全的。

但这种"靠隐式 name counter 维护参数命名"的风格比较 fragile。Flax 后来引入显式 name 必填的设计部分是为了避免这种陷阱。

### 6.3.2 与传统 pre-norm / post-norm 的对比

**Post-norm（原始 Transformer）：**

```
y = LayerNorm(x + SubLayer(x))
```

训练困难，深层时残差被反复 norm，梯度难以传播。

**Pre-norm（GPT-2 起的主流）：**

```
y = x + SubLayer(LayerNorm(x))
```

训练稳定，深层 OK，但深度过大（80+ 层）时残差 stream 量级会爆炸。

**Sandwich norm（Grok-1、Cohere Command R）：**

```
y = x + LayerNorm(SubLayer(LayerNorm(x)))
```

子层输入和输出都被 norm，残差量级被双重约束。代价：每层多一次 norm，参数多一倍。

64 层在 pre-norm 下也未必爆炸，所以 Grok-1 选 sandwich 不是必需，可能是 xAI 经验上倾向更保守的设计。

### 6.3.3 `if self.num_experts > 1` 分支

```python
if self.num_experts > 1:
    h_dense = MoELayer(...)(layer_norm(h), padding_mask)
else:
    h_dense = base_dense_block(layer_norm(h))
```

`num_experts = 8 > 1`，所以走 MoE 分支。但这个分支保留了 dense 路径，意味着可以用同一份代码训练 dense baseline - 这是研究友好的设计。

### 6.3.4 `with_sharding_constraint`

每次 residual add 后都做一次 sharding constraint，把张量的 partition 显式约束到指定形状。当 `shard_activations=True`：

```python
sharding = P("data", None, "model")
```

即 batch 沿 data 切、sequence 不切、hidden 沿 model 切。这是激活的"标准布局"。

XLA 编译时会用这些 constraint 决定通信策略 - 比如 attention 输出后 hidden 需要 all-gather 还是不需要。

## 6.4 `Transformer`：64 层堆叠

`model.py:1291-1398`：

```python
# model.py:1326-1398
def __call__(
    self,
    embeddings: jax.Array,  # [B, T, D]
    mask: jax.Array,  # [B, T]
    memory: Optional[Memory],
) -> TransformerOutput:
    fprop_dtype = embeddings.dtype
    _, seq_len, model_size = embeddings.shape
    padding_mask = mask.copy()
    mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]

    # Compute causal mask for autoregressive sequence modelling.
    causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len))).astype(
        fprop_dtype
    )  # [B=1, H=1, T, T]
    mask = mask * causal_mask  # [B, H=1, T, T]

    h = embeddings
    kv_memories = []

    def block(
        h, mask, padding_mask, memory,
        layer_index=None, widening_factor=None, name=None,
    ) -> DecoderOutput:
        return DecoderLayer(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            widening_factor=widening_factor or self.widening_factor,
            num_layers=self.num_layers,
            mesh=self.mesh,
            data_axis=self.data_axis,
            model_axis=self.model_axis,
            attn_output_multiplier=self.attn_output_multiplier,
            shard_activations=self.shard_activations,
            num_experts=self.num_experts,
            num_selected_experts=self.num_selected_experts,
            name=name,
            layer_index=layer_index,
        )(h, mask, padding_mask, memory)

    for i in range(self.num_layers):
        decoder_output = block(
            h, mask, padding_mask,
            memory.layers[i] if memory else None,
            layer_index=i,
            name=f"decoder_layer_{i}",
        )
        h, new_kv_memory = (
            decoder_output.embeddings,
            decoder_output.memory,
        )
        kv_memories.append(new_kv_memory)

    return TransformerOutput(
        embeddings=h,
        memory=Memory(layers=kv_memories),
    )
```

### 6.4.0 数据流再总结

`Transformer.__call__` 把整个主干串起来：

1. **构造 mask**：从 padding mask 和 causal mask 复合，得到 `[B, 1, T, T]` 的最终 mask
2. **for 循环 64 层**：每层调用 DecoderLayer，传入当前 hidden、mask、padding_mask、上一步 KV memory
3. **收集 KV memory**：每层输出新的 KVMemory，累积到 list
4. **返回**：TransformerOutput(embeddings, Memory(layers=kv_memories))

注意第 1380 行 `decoder_output.embeddings` 是 64 层逐层更新的 hidden，第 1393 行 `kv_memories` 是 64 个独立的 KVMemory。两者结构平行但耦合 - 上一层的 KV memory 在下一层的同位置 DecoderLayer 内被用，但 hidden 是顺序传递。

### 6.4.1 Python for-loop 而非 `hk.scan`

注意第 1380-1393 行用了 **Python for 循环**展开 64 层 - 不是 `hk.scan` 或 `jax.lax.scan`。

这意味着：

1. **每一层的参数都独立存放**：`decoder_layer_0/...`、`decoder_layer_1/...`，64 个不同的 prefix
2. **编译时 64 层全部 unroll** - XLA 看到的是一个超长的计算图
3. **优点**：可以为每层加独立约束、独立调试，partition rules 可以匹配 `decoder_layer_[0-9]+` 这种正则
4. **缺点**：编译时间长、占内存。Grok-1 实测 JIT compile 阶段会用几十 GB 内存

`hk.scan` 风格则会把 64 层折成一个 leading-64 维 - 参数 stack 在一起，编译只要看 1 层。Mixtral 用 PyTorch 写时是 for-loop，行为类似 Grok-1。

`partition_rules` 里特殊处理 `layer_stack`（`model.py:102-104`）暗示原本可能有 scan 版本，但当前代码用 for-loop。

### 6.4.2 causal mask 构造

```python
mask = mask[:, None, None, :]  # [B, 1, 1, T]   - padding mask
causal_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))  # [1, 1, T, T] - lower triangular
mask = mask * causal_mask  # [B, 1, T, T]
```

最终 mask 是 padding mask 与 causal mask 的逐元素乘积。位置 (i, j)：

$$
\text{mask}[i, j] = \mathbb{1}[\text{token}_j \neq \text{pad}] \cdot \mathbb{1}[j \le i]
$$

在 attention 计算里 mask = 0 的位置被设为 -1e30 然后 softmax 出 0。

注意 `seq_len` 是 prefill 时的整个 prompt 长度。在 decode 阶段 seq_len = 1，causal_mask 是个 `[1, 1, 1, 1]` 的全 1 矩阵 - 因为 decode 时只 attend 一个 query 到所有历史 K/V，causal mask 在 query 维度退化。

但等等 - decode 阶段 K/V 长度是 8192（缓存全长）而非 1。这个 mask 是 `[B, 1, 1, 1]`，怎么对应到 K/V 长度？

答案在 `MultiHeadAttention.__call__` 里（`model.py:832-838`）：

```python
new_step = kv_memory.step + sequence_length
memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
memory_mask = memory_mask[:, None, None, :]  # [B, H, T, T]
if mask is not None:
    mask = memory_mask * mask
else:
    mask = memory_mask
```

`memory_mask` 是 `[B, 1, 1, 8192]`，"位置 < 当前 step" 的为 1。然后和外面传入的 causal mask 相乘。所以 decode 时实际 mask 是 memory_mask（causal mask 退化无效）。

## 6.5 `LanguageModel`：加 embedding 与 logits

`model.py:1201-1289`：

```python
# model.py:1211-1279
def __call__(
    self,
    tokens: jax.Array,
    memory: Optional[Memory] = None,
    *,
    batch: Dict[str, jax.Array] = {},
    last_hid_only: bool = False,
    length: Optional[jax.Array] = None,
) -> LanguageModelOutput:
    del batch  # Unused.

    config = self.config

    input_mask = jnp.greater(tokens, config.pad_token)

    # Embed the input tokens and positions.
    in_out_embed = InOutEmbed(
        self.config.vocab_size,
        embed_dim=self.config.model_size,
        sharding=P(None, ("data", "model")),
    )
    input_embeddings = in_out_embed(tokens).astype(config.fprop_dtype)
    input_embeddings = with_sharding_constraint(
        input_embeddings, P("data", None, self.model.model_axis)
    )
    input_embeddings *= config.embedding_multiplier_scale

    model_output = self.model(
        input_embeddings, input_mask, memory=memory,
    )  # [B, T, D]
    embeddings, model_state = model_output.embeddings, model_output.memory
    if self.model.shard_activations:
        embeddings = with_sharding_constraint(
            embeddings, P("data", None, self.model.model_axis)
        )
    else:
        embeddings = with_sharding_constraint(embeddings, P("data", None))
    rank_logger.debug(f"Final embedding shape: {embeddings.shape}")
    embeddings = layer_norm(embeddings, self.model)
    assert embeddings.dtype == self.fprop_dtype

    if last_hid_only:
        last_step = jnp.maximum(jnp.sum(input_mask.astype(jnp.int32), axis=1) - 1, 0)
        last_hid = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(embeddings, last_step)
        return last_hid

    if length is not None:
        last_step = jnp.maximum(length.astype(jnp.int32) - 1, 0)
        embeddings = jax.vmap(lambda x, i: x[i], in_axes=0, out_axes=0)(embeddings, last_step)
        embeddings = jnp.expand_dims(embeddings, axis=1)

    # Decode the embeddings (here, we use tied weights).
    rank_logger.info(embeddings.shape)
    out = in_out_embed.decode(embeddings)
    rank_logger.info(out.shape)
    out *= config.output_multiplier_scale

    if self.model.shard_activations:
        out = with_sharding_constraint(out, P("data", None, self.model.model_axis))
    else:
        out = with_sharding_constraint(out, P("data", None))

    return LanguageModelOutput(
        logits=out,
        model_state=model_state,
    )
```

按顺序：

1. **input_mask = tokens > pad_token (=0)**：找出非 pad 位置
2. **InOutEmbed**：embedding lookup，结果 `[B, T, d]`，cast 到 bf16
3. **embedding 乘 78.38**（≈ $\sqrt{d}$）
4. **过 Transformer 主干**（64 层 DecoderLayer）
5. **最终 RMSNorm**：`embeddings = layer_norm(embeddings, self.model)` - 注意这是最后一次 norm，在 final logit 计算前
6. **`length` 参数**：如果给定，只取每个 sequence 在 `length-1` 位置的 hidden（vmap over batch）。这是 prefill 时取"最后一个 token"做 sampling 用
7. **`in_out_embed.decode`**：用 tied weights 做 logits 投影：`logits = embeddings @ embed_mat.T`
8. **乘 output_multiplier_scale**（≈ $1/\sqrt{3}$）

### 6.5.1 Embedding tying

`InOutEmbed`（`model.py:1110-1143`）：

```python
# model.py:1110-1143
class InOutEmbed(hk.Embed):
    """Module for embedding tokens in a low-dimensional space."""

    def __init__(
        self,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
        sharding: Optional[P] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            name=name,
        )
        self.sharding = sharding

    @property
    def embeddings(self):
        embed_mat = hk.get_parameter(
            "embeddings",
            [self.vocab_size, self.embed_dim],
            dtype=jnp.float32,
            init=hk.initializers.Constant(0),
        )
        if self.sharding:
            embed_mat = with_sharding_constraint(embed_mat, self.sharding)
        return embed_mat

    def decode(
        self,
        inputs: jax.Array,
    ) -> jax.Array:
        return jnp.dot(inputs, self.embeddings.T.astype(inputs.dtype))
```

- 输入：`tokens` -> `embed_mat[tokens]`（继承自 `hk.Embed.__call__`）
- 输出：`embeddings @ embed_mat.T`（`decode` 方法）

**同一个 `embed_mat` 既做输入 embedding 又做输出 logits 投影。** 这是 tied embedding，节省 0.8B 参数。

LLaMA-2 同样 tied，Mistral / Mixtral 也是。GPT-3 不 tied。Tied 还是 untied 的争论从 2017 开始就有，主流大模型 2023 之后基本都 tied。

### 6.5.2 `length` 参数：prefill 与 decode 都用

在 prefill 时，`length` = 真实 prompt 长度（不含 pad）。代码会从 `[B, T_pad, d]` 中只取出位置 `length-1` 的那个 hidden，得到 `[B, 1, d]`，再过 decode 出 `[B, 1, V]`。

这样在 prefill 阶段我们不需要计算 8192 个位置的 logits（浪费 8190 倍计算），只算最后一个。decode 阶段也用这个机制：传入 length=1，只取第 0 个位置。

## 6.6 KV cache 接口

`model.py:178-206`：

```python
# model.py:178-206
class KVMemory(NamedTuple):
    k: Optional[jax.Array]
    v: Optional[jax.Array]
    step: Optional[jax.Array]


def init_layer_memories(
    batch_size: int,
    sequence_len: int,
    num_kv_heads: int,
    key_size: int,
    num_layers: int,
    step: Optional[jax.Array] = None,
    dtype=jnp.bfloat16,
):
    return [
        KVMemory(
            k=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            v=jnp.zeros((batch_size, sequence_len, num_kv_heads, key_size), dtype=dtype),
            step=step,
        )
        for _ in range(num_layers)
    ]


class Memory(NamedTuple):
    layers: List[KVMemory]
```

- `KVMemory`：每层一个，包含 k、v、step
- `Memory`：64 个 KVMemory 的 list

每个 K/V 的 shape：

```
k: [B, T=8192, num_kv_heads=8, key_size=128] -> 8 * 8192 * 8 * 128 = 67M 元素 / batch / layer
v: 同上
```

bf16 dtype，每元素 2 bytes，per-layer per-batch：8M * 2 = 16M bytes / layer / batch / direction = **32 MB / layer / batch**。

64 层 = **2.05 GB / batch**。

这是 Grok-1 推理时除了模型权重外的主要显存开销。当 batch_size = 1，KV cache 就是 2 GB；batch_size = 8（mesh 8 卡，每卡 1 个 batch），共 16 GB KV cache。

GQA 在这里救命 - 如果是 MHA（48 KV heads），KV cache 会变成 12 GB / batch。

### 6.6.1 `step` 字段

每个 batch 元素有独立的 step（`jnp.zeros(batch_size, dtype=jnp.int32)`）。这意味着不同 batch 元素可以处于不同的生成阶段 - 是为 continuous batching / 异步 serve 准备的。

`runners.py:73-74` 用 `jax.lax.dynamic_update_index_in_dim` 把新 batch 元素插入 cache：

```python
return jax.tree_map(lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u[0], i, axis=0),
                    memory, slice)
```

### 6.6.2 KV cache 的 sharding

`model.py:476-486`：

```python
def get_memory_sharding(self):
    return Memory(
        layers=[
            KVMemory(
                k=P(self.data_axis, self.model_axis),
                v=P(self.data_axis, self.model_axis),
                step=P(self.data_axis),
            )
            for _ in range(self.num_layers)
        ],
    )
```

K/V partition 是 `(data, model)` - batch 沿 data 切，**第 1 维（seq_len）沿 model 切**。等等？seq_len 沿 model 切？

这有点反直觉。常规 KV cache sharding 是按 head 维度切（让每张卡持有部分 head）。Grok-1 按 seq_len 切意味着每张卡持有部分 token 位置的 K/V。8 卡，每卡持有 1024 个 token 位置的 K/V。

但 `model.py:826` 的 `update_into_shmap` partition spec 是 `P("data", None, "model")` - 第 0 维 batch 沿 data，第 1 维 seq_len 不切，第 2 维（head）沿 model。

这两处 partition spec 不一致！可能的解释：

- `get_memory_sharding` 给的 `(data, model)` 是 2D 的，对应 KV 的 4D shape 时只匹配前两维，后两维（head, key_size）默认按需切
- `with_sharding_constraint(..., P("data", None, "model"))` 是 3D 的，匹配前三维

JAX 的 PartitionSpec 是从前往后匹配，剩余维度 unconstrained。所以两处 spec **可以兼容** - `(data, model)` 等价于 `(data, model, None, None)`，而 `update_into_shmap` 用的 `(data, None, model)` 等价于 `(data, None, model, None)`。

两者关于 seq_len 维度的约束不一样 - 一个 model 切，一个不切。这可能是个 ckpt-vs-shmap 边界处理。XLA 编译器会自动插 reshard 让两端 spec 一致。

实际效果：Grok-1 的 KV cache 在加载/存储时按 (B, T) 切到 8 卡，在 attention 计算时 reshape 成 (B, head) 切。每次 step 有一次 reshard 通信。

### 6.6.3 KV cache 总显存账

按 `run.py:54` 的 `bs_per_device=0.125`，8 卡总 batch = 1。KV cache 总显存：

$$
S_{\text{kv}} = 64 \cdot 1 \cdot 8192 \cdot 8 \cdot 128 \cdot 2 \text{ (k+v)} \cdot 2 \text{ bytes}
$$
$$
= 64 \cdot 33554432 \approx 2.15 \text{ GB}
$$

如果 batch=8（每卡 1 个 batch），KV cache 涨到 17 GB - 仍可接受。

但如果想做高吞吐 serving（batch=64），KV cache 变成 137 GB - 超过单卡，需要切分到多卡。Grok-1 的 KV cache 按 (data, model) sharding，正好能切到 8 卡，每卡 ~17 GB。

实际生产 serving 用 paged attention 能节省 KV cache（每个 token 分页，未用页不分配显存），但 Grok-1 默认实现不支持这个。

## 6.7 `LanguageModelConfig.partition_rules`

`model.py:1193-1194`：

```python
def partition_rules(self):
    return LM_PARTITION_RULES + self.model.partition_rules()
```

合并语言模型层和 Transformer 层的 rule。`LM_PARTITION_RULES`（`model.py:162-174`）只有 3 条 - embedding、positional_embeddings（实际不用，因为 RoPE 不需要 positional embedding 参数）、和最终 rms_norm。

完整 rule 列表会在 `runners.py:202-209` 被用：

```python
# runners.py:202-209
sharding = jax.tree_util.tree_map_with_path(
    apply_rules(self.model.partition_rules()),
    shapes,
)
```

`apply_rules` 函数（`model.py:92-109`）对每个参数路径尝试匹配 rule，匹配上就返回对应 PartitionSpec。

## 6.8 整张表：所有 module 与参数名

| Module | 参数路径 | shape | partition |
| --- | --- | --- | --- |
| `language_model/in_out_embed` | `embeddings` | (131072, 6144) | (None, (data, model)) |
| `language_model/rms_norm` | `scale` | (6144,) | None |
| `language_model/transformer/decoder_layer_i/rms_norm` (i ∈ 0..63) | `scale` | (6144,) | None |
| `decoder_layer_i/rms_norm_1` | `scale` | (6144,) | None |
| `decoder_layer_i/rms_norm_2` | `scale` | (6144,) | None |
| `decoder_layer_i/rms_norm_3` | `scale` | (6144,) | None |
| `decoder_layer_i/multi_head_attention/query/w` | (6144, 6144) | (data, model) |
| `decoder_layer_i/multi_head_attention/key/w` | (6144, 1024) | (data, model) |
| `decoder_layer_i/multi_head_attention/value/w` | (6144, 1024) | (data, model) |
| `decoder_layer_i/multi_head_attention/linear/w` | (6144, 6144) | (model, data) |
| `decoder_layer_i/router/w` | (6144, 8) | (data,) |
| `decoder_layer_i/moe/linear_v/w` | (8, 6144, 32768) | (None, data, model) |
| `decoder_layer_i/moe/linear/w` | (8, 6144, 32768) | (None, data, model) |
| `decoder_layer_i/moe/linear_1/w` | (8, 32768, 6144) | (None, model, data) |

总共约 770 个参数 tensor（64 层 × 12 + 头尾若干）。

## 6.9 总结：model.py 的设计风格

1. **Haiku module 树扁平**：很多 `@hk.transparent` 让参数名缩短
2. **for-loop 展开 64 层**：编译大、但灵活
3. **sandwich norm**：每层 4 个 RMSNorm
4. **GeGLU MoE**：8 专家 × top-2
5. **GQA 48:8**：6 倍 KV 压缩
6. **8-bit 量化是一等公民**：Linear 内置反量化路径
7. **tied embedding**：节省 0.8B

下一章看 checkpoint.py，把这 770 个 tensor 怎么从磁盘 load 到 GPU 讲清楚。

## 延伸阅读

- [Cohere Command R Technical Report](https://docs.cohere.com/docs/command-r) - sandwich norm 的另一个工业实例
- [Improving Transformer Optimization Through Better Initialization](https://proceedings.mlr.press/v119/huang20f.html) - 多种 norm 布局对训练稳定性的讨论
- [DeepNet: Scaling Transformers to 1000 Layers](https://arxiv.org/abs/2203.00555) - 极深网络下的 norm 策略
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - tied embedding 的早期工作
