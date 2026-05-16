# 第 6 章 model.py 精读·下：Block 组装与切分

本章把 `model.py` 剩下的部分讲完：

- `MHABlock`：把 attention 封装成"输入 → 输出"形式的子层
- `DenseBlock`：单个 SwiGLU FFN（也是 MoE 的 layer_fn）
- `DecoderLayer`：sandwich norm + MoE 选择
- `LanguageModel` 与 `Transformer`：顶层组装
- `InOutEmbed`：tied embedding 实现
- causal mask
- KV cache 的接口

读完这一章，读者应当能够在头脑中完整还原一遍 `forward(tokens) -> logits` 的执行流程：从 token id 经由 embedding lookup 进入残差流，依次穿过 64 层 DecoderLayer，最后由 tied 的输出 embedding 投影回 logits。

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

这一层本质上只是一个**自注意力的封装**：通过 `side_input = inputs` 让 query、key、value 三个输入都指向同一个张量，可以理解为"self-attention block"的一种语法糖式写法。

`@hk.transparent` 装饰器的作用是让这个 module 的参数 scope **不进入** Haiku 的命名空间层级。也就是说，`MultiHeadAttention` 创建的参数名直接挂在父 module（DecoderLayer）下面，最终参数路径不会出现 `mha_block/multi_head_attention/query/w` 的形式，而是直接呈现为 `decoder_layer_0/multi_head_attention/query/w`。

这种设计的好处是让 ckpt 中的参数路径更简短、更易读，代价则是 `MHABlock` 无法作为独立的 module 完成 init/apply。

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

`num_q_heads`、`num_kv_heads`、`key_size` 这三个字段在 DenseBlock 内部**完全没有被使用**：它们出现在构造参数列表里，但 `__call__` 方法里没有任何地方读取它们。这看起来像是从一个更大的 module 复制过来时遗留的代码，不会影响功能正确性。

### 6.2.1 SwiGLU 还是 GeGLU？

回到第 5 章中的讨论：这里用的是 `jax.nn.gelu`，激活函数是 **GELU**，所以严格按命名习惯来说应当称为 **GeGLU**。但社区在口语和文献里都习惯把这类 gated FFN 统称为 SwiGLU，本书后文也会沿用 SwiGLU 作为泛称。

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

每个 sub-layer 实际使用了**两次 RMSNorm**。所有 RMSNorm 调用统一通过 `hk_rms_norm(x)` 完成，每一次调用都会创建一个新的 RMSNorm module 实例（按 Haiku 的默认命名规则，name 会自动递增）。

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

这种命名机制要求**调用顺序保持固定**：如果存在 if 分支让某些 RMSNorm 调用被跳过，自动递增出来的 name 序号就会与预期错位。Grok-1 的 DecoderLayer 不存在这种条件分支（`if num_experts > 1` 只影响 MoE 与 DenseBlock 的分支选择，不影响 RMSNorm 调用次数），因此这套命名机制在当前代码下是安全的。

不过这种"依靠隐式 name counter 来维护参数命名"的风格比较 fragile。Flax 在后续版本中引入显式 name 必填的设计，一部分原因正是为了规避这类隐式计数带来的陷阱。

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

64 层这个深度在 pre-norm 设置下未必一定会出现残差爆炸，所以 Grok-1 选择 sandwich norm 并非数值稳定性上的必需选项，更可能反映出 xAI 团队在经验上偏向更保守的设计取向。

### 6.3.3 `if self.num_experts > 1` 分支

```python
if self.num_experts > 1:
    h_dense = MoELayer(...)(layer_norm(h), padding_mask)
else:
    h_dense = base_dense_block(layer_norm(h))
```

由于 `num_experts = 8 > 1`，Grok-1 在实际推理时一定走 MoE 分支。但这个 if 分支同时保留了 dense 路径，意味着可以用完全相同的一份代码训练 dense baseline 模型，这是一种对研究使用比较友好的设计。

### 6.3.4 `with_sharding_constraint`

每一次 residual add 完成之后，都会显式调用一次 sharding constraint，把张量的 partition 约束到指定的目标形状。在 `shard_activations=True` 的情况下：

```python
sharding = P("data", None, "model")
```

即 batch 维沿 data 轴切分、sequence 维保持不切、hidden 维沿 model 轴切分。这是激活值在 Grok-1 中的"标准布局"。

XLA 编译器会利用这些 sharding 约束来决定具体的通信策略，例如 attention 输出之后的 hidden 张量是否需要在 model 轴上进行 all-gather。

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

`Transformer.__call__` 负责把整个 Transformer 主干串联起来：

1. **构造 mask**：把 padding mask 与 causal mask 做逐元素相乘，得到形状为 `[B, 1, T, T]` 的最终 mask
2. **for 循环跑 64 层**：每一层都调用一次 DecoderLayer，传入当前 hidden、合成 mask、padding mask、以及对应层的 KV memory
3. **收集 KV memory**：每一层输出新的 KVMemory，逐层 append 到 list 中
4. **返回**：把最终 hidden 与 64 层的 KV memory 一起打包为 TransformerOutput(embeddings, Memory(layers=kv_memories)) 返回

注意第 1380 行的 `decoder_output.embeddings` 表示 64 层之间逐层更新的 hidden，第 1393 行收集到的 `kv_memories` 则是 64 个相互独立的 KVMemory。两者在结构上是平行的，但实际计算中存在耦合：每一层在做 attention 时会用到该层自己的 KV memory，但 hidden 是顺序在层与层之间传递的。

### 6.4.1 Python for-loop 而非 `hk.scan`

注意第 1380-1393 行使用的是 **Python 原生 for 循环**来展开 64 层 DecoderLayer，并没有采用 `hk.scan` 或 `jax.lax.scan` 这种结构化的循环 primitive。

这种实现方式带来几个直接后果：

1. **每一层的参数都被独立存放**，参数路径分别是 `decoder_layer_0/...`、`decoder_layer_1/...`，共 64 个不同的前缀
2. **编译时 64 层全部被完全 unroll**，XLA 实际看到的是一张极长的、把全部 64 层串联展开的计算图
3. **优点**：可以为每一层添加独立的 sharding 约束、独立调试单层行为，partition rules 也可以直接用 `decoder_layer_[0-9]+` 这种正则统一匹配
4. **缺点**：编译时间长、内存占用大，Grok-1 实测在 JIT compile 阶段会消耗几十 GB 主机内存

如果改用 `hk.scan` 风格实现，则会把 64 层折叠成一个 leading-64 的维度，参数 stack 在一起，编译时实际只需要分析一层。Mixtral 在 PyTorch 实现中也是 for-loop 形式，整体行为与 Grok-1 类似。

`partition_rules` 中针对 `layer_stack` 做了专门处理（见 `model.py:102-104`），这暗示代码早期版本中可能存在过基于 scan 的实现，但当前开源版本最终使用的是 for-loop 实现。

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

在 attention 的具体计算中，mask = 0 对应的位置会被赋成 -1e30，softmax 之后这些位置的注意力权重会被压到 0。

注意 `seq_len` 指的是 prefill 阶段输入的整个 prompt 长度。在 decode 阶段 seq_len = 1，causal_mask 退化为 `[1, 1, 1, 1]` 的全 1 矩阵：decode 时只有一个 query 位置需要 attend 到所有历史 K/V，因此 causal mask 在 query 维度上失去了实际意义。

需要进一步澄清的是，decode 阶段实际的 K/V 序列长度是 8192（KV cache 的总长度），并不是 1。这里的 mask 形状是 `[B, 1, 1, 1]`，那它如何与长度为 8192 的 K/V 对应？

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

这里的 `memory_mask` 形状为 `[B, 1, 1, 8192]`，其语义是"位置序号小于当前 step"的位置取值为 1。它随后与外层传入的 causal mask 做逐元素相乘。因此在 decode 阶段，真正起作用的实际是 memory_mask，外层传入的 causal mask 在此时已经退化为全 1 失去筛选作用。

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

**同一个 `embed_mat` 既承担输入 token 的 embedding 查表，也用于输出 logits 的投影。** 这正是 tied embedding 的实现方式，对 Grok-1 这种 131072 词表、6144 隐藏维的模型来说可以节省大约 0.8B 参数。

LLaMA-2 同样采用 tied embedding，Mistral 与 Mixtral 也都是 tied。GPT-3 选择 untied。tied 与 untied 的优劣讨论从 2017 年起就持续存在，但 2023 年之后主流大模型在工程取舍上几乎都倒向 tied 这一边。

### 6.5.2 `length` 参数：prefill 与 decode 都用

在 prefill 阶段，`length` 表示真实的 prompt 长度（不含 pad）。代码会从形状为 `[B, T_pad, d]` 的 hidden 中按 `length-1` 索引取出每个 batch 元素对应的最后一个有效位置的 hidden，得到 `[B, 1, d]`，然后再过 decode 投影得到 `[B, 1, V]` 的 logits。

这样在 prefill 阶段就不必为 8192 个位置全部计算 logits（否则相对于实际需要的最后一个位置就是 8190 倍的浪费），只需要计算最后一个位置即可。decode 阶段沿用同样的机制：传入 length=1，只取序列的第 0 个位置作为 hidden。

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

这是 Grok-1 推理时除模型权重之外最主要的显存开销。当 batch_size = 1 时 KV cache 大小约为 2 GB；当 batch_size = 8（mesh 8 卡，每卡承担 1 个 batch）时合计大约 16 GB。

GQA 在这里起到了至关重要的显存压缩作用：如果改用 MHA（48 个 KV head），KV cache 会膨胀到大约 12 GB 每个 batch。

### 6.6.1 `step` 字段

每个 batch 元素拥有各自独立的 step 计数（初始化为 `jnp.zeros(batch_size, dtype=jnp.int32)`）。这种设计允许同一 batch 中的不同元素处于不同的生成阶段，正是为 continuous batching 与异步 serving 场景准备的接口。

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

K/V 的 partition 写成 `(data, model)`：batch 维沿 data 轴切分，**第 1 维（即 seq_len）沿 model 轴切分**。需要强调一下：seq_len 在这里是沿 model 轴切的。

这一点与常见做法有一定反差。常规的 KV cache sharding 方案是按 head 维度切（让每张卡持有部分 head 的 K/V）。Grok-1 按 seq_len 切意味着每张卡只持有 K/V 中部分 token 位置的数据。在 8 卡配置下，每张卡持有大约 1024 个 token 位置的 K/V。

但 `model.py:826` 的 `update_into_shmap` partition spec 是 `P("data", None, "model")` - 第 0 维 batch 沿 data，第 1 维 seq_len 不切，第 2 维（head）沿 model。

这两处给出的 partition spec 看起来并不一致。一种合理的解释如下：

- `get_memory_sharding` 中提供的 `(data, model)` 是 2D 形式，应用到 KV 这种 4D shape 时只匹配前两维，剩余的 head 与 key_size 维度按 JAX 默认行为推断
- `with_sharding_constraint(..., P("data", None, "model"))` 是 3D 形式，匹配前三维

JAX 的 PartitionSpec 在匹配维度时是从前往后逐个对齐，未指定的后续维度默认 unconstrained。基于这条规则，两处 spec **是可以兼容的**：`(data, model)` 实际等价于 `(data, model, None, None)`，而 `update_into_shmap` 中使用的 `(data, None, model)` 等价于 `(data, None, model, None)`。

两者对 seq_len 这一维的约束并不相同：一处是按 model 切，另一处是不切。这种不一致大概是 ckpt 加载边界与 shmap 边界之间的过渡处理。XLA 编译器在遇到不一致的 sharding 约束时，会自动在两端之间插入 reshard 通信，把布局对齐。

最终的实际效果是：Grok-1 的 KV cache 在加载与存储时按 (B, T) 切分到 8 卡；进入 attention 计算之前会被 reshape 并改成沿 (B, head) 维度切分。每一个 decode step 都会触发一次这种 reshard 通信。

### 6.6.3 KV cache 总显存账

按 `run.py:54` 的 `bs_per_device=0.125`，8 卡总 batch = 1。KV cache 总显存：

$$
S_{\text{kv}} = 64 \cdot 1 \cdot 8192 \cdot 8 \cdot 128 \cdot 2 \text{ (k+v)} \cdot 2 \text{ bytes}
$$
$$
= 64 \cdot 33554432 \approx 2.15 \text{ GB}
$$

如果 batch=8（每张卡负责 1 个 batch），KV cache 总量上升到大约 17 GB，对 8 卡 80 GB H100 来说仍然在可接受范围内。

但如果用于高吞吐 serving 场景（例如 batch=64），KV cache 总量会上升到约 137 GB，单张卡已经无法容纳，必须切分到多卡上。Grok-1 的 KV cache 按 (data, model) 进行 sharding，正好能切到 8 张卡，每张卡承担约 17 GB。

工业级 serving 一般会使用 paged attention 来进一步节省 KV cache 显存（按 token 分页存储，未使用的页不分配显存），但 Grok-1 默认的开源实现并未支持这种机制。

## 6.7 `LanguageModelConfig.partition_rules`

`model.py:1193-1194`：

```python
def partition_rules(self):
    return LM_PARTITION_RULES + self.model.partition_rules()
```

这一行将语言模型层与 Transformer 主干层的 partition rule 合并到一起。`LM_PARTITION_RULES`（位于 `model.py:162-174`）只包含 3 条规则：分别对应 embedding、positional_embeddings（实际不会被命中，因为 Grok-1 使用 RoPE，没有显式 positional embedding 参数）、以及最终输出前的 rms_norm。

完整的 partition rule 列表最终会在 `runners.py:202-209` 处被消费使用：

```python
# runners.py:202-209
sharding = jax.tree_util.tree_map_with_path(
    apply_rules(self.model.partition_rules()),
    shapes,
)
```

`apply_rules` 函数（位于 `model.py:92-109`）会针对每一个参数路径依次尝试匹配规则列表，匹配成功时返回对应的 PartitionSpec。

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

合计大约 770 个参数 tensor（其中 64 层 × 12 个主干参数，再加上头尾若干个额外参数）。

## 6.9 总结：model.py 的设计风格

1. **Haiku module 树整体扁平**：大量使用 `@hk.transparent` 让参数名缩短
2. **for-loop 显式展开 64 层**：编译产物大但灵活，便于逐层加约束与调试
3. **sandwich norm**：每一层包含 4 个独立的 RMSNorm
4. **GeGLU MoE**：8 个专家 + top-2 路由
5. **GQA 48:8**：6 倍 KV 压缩
6. **8-bit 量化作为一等公民**：Linear 内部内置了反量化路径
7. **tied embedding**：节省约 0.8B 参数

下一章看 checkpoint.py，把这 770 个 tensor 是如何从磁盘加载到 GPU 显存的过程讲清楚。

## 延伸阅读

- [Cohere Command R Technical Report](https://docs.cohere.com/docs/command-r) - sandwich norm 的另一个工业实例
- [Improving Transformer Optimization Through Better Initialization](https://proceedings.mlr.press/v119/huang20f.html) - 多种 norm 布局对训练稳定性的讨论
- [DeepNet: Scaling Transformers to 1000 Layers](https://arxiv.org/abs/2203.00555) - 极深网络下的 norm 策略
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - tied embedding 的早期工作
