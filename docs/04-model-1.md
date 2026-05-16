# 第 4 章 model.py 精读·上：RMSNorm、RoPE、GQA Attention

本章覆盖 `model.py` 中除 MoE 与 DecoderLayer 之外的"底层零件"：

- `Linear`（自定义的 `hk.Linear` 子类）
- `RMSNorm` 与 `hk_rms_norm`
- `rotate_half` 与 `RotaryEmbedding`
- `MultiHeadAttention`
- `make_attention_mask`

读完本章你应该能在脑子里把"一次 attention 调用"全程跑一遍：输入是 `[B, T, d]`，输出是 `[B, T, d]` + 更新后的 KV cache，中间经过哪些 reshape、哪些 RoPE 角度、哪些 partition 约束。

## 4.1 `Linear`：内置 8-bit 反量化

`model.py:525-584`。Grok-1 自己写了一个 `Linear`，继承 `hk.Linear`，主要为了：

1. 支持 `QuantizedWeight8bit` - 权重存 int8 + per-channel fp32 scale
2. 让 sharding 直接写在构造参数里

```python
# model.py:525-584
class Linear(hk.Linear):
    def __init__(
        self,
        output_size: int,
        with_bias: bool = True,
        sharding: Optional[P] = None,
        mesh: Any = None,
        name: Optional[str] = None,
        shard_axis: int = 0,
    ):
        super().__init__(
            output_size=output_size,
            with_bias=with_bias,
            name=name,
        )
        self.sharding = sharding
        self.mesh = mesh
        self.shard_axis = shard_axis

    def __call__(self, inputs: jax.Array) -> jax.Array:
        fprop_dtype = inputs.dtype
        ...
        w = hk.get_parameter(
            "w", [input_size, output_size], jnp.float32, init=hk.initializers.Constant(0)
        )

        if hasattr(w, "scales"):
            shape = inputs.shape
            inputs = jnp.reshape(inputs, (-1, shape[-1]))

            @functools.partial(
                shard_map,
                mesh=self.mesh,
                in_specs=(self.sharding, self.sharding),
                out_specs=self.sharding,
                check_rep=False,
            )
            def mul(w, s):
                return w.astype(s.dtype) * s

            w = mul(w.weight, w.scales)
        out = jnp.dot(inputs, w.astype(fprop_dtype))
        if self.with_bias:
            b = hk.get_parameter(
                "b", [self.output_size], jnp.float32, init=hk.initializers.Constant(0)
            )
            b = jnp.broadcast_to(b, out.shape)
            out = out + b.astype(fprop_dtype)

        return out
```

关键观察：

- **`hasattr(w, "scales")`** 是一个"运行时鸭子类型"检查。`hk.get_parameter` 在加载 ckpt 时会把参数替换成 `QuantizedWeight8bit` 实例（这个 dataclass 注册为 pytree node 在 `model.py:47-51`），它就有 `.scales` 字段。如果没量化，直接 dense matmul
- **反量化在 `shard_map` 内做** - 这样每个 device 只乘自己那一片的 scale，省一次 all-gather
- 反量化后的 `w` dtype 与 scales 一致（fp32），然后再 cast 到 fprop dtype 做 matmul

与 Llama2 实现的区别：Llama2 用 `nn.Linear` 直接、没有内置量化支持；想要 int8/int4 需要靠 `bitsandbytes` 或 GPTQ 后处理。Grok-1 是**训练时就为量化推理做好了一等公民支持**。

这种"训练时就为量化做好支持"的设计，在 2023 年还不算主流。当时 GPTQ、AWQ 这些训练后量化方法刚成熟，业界标准做法是**先训 bf16，再量化**。Grok-1 反其道而行 - 让 Linear 内置反量化路径，这暗示 xAI 在训练 / 推理边界做了**量化感知训练**（QAT）或者至少**量化感知推理**的设计。

ckpt 里的 `QuantizedWeight8bit` 大概率是这样产生的：训练 bf16 → 训练结束做 per-channel int8 量化（每个 output channel 一个 fp32 scale）→ 保存为 `(weight: int8, scales: fp32)` 二元组。这种"per-channel 静态量化"是工业界最稳的 int8 方案，质量损失通常 <0.5% PPL。

### 4.1.1 `QuantizedWeight8bit`

`model.py:37-51`：

```python
# model.py:37-51
@dataclass
class QuantizedWeight8bit:
    weight: jnp.array
    scales: jnp.array

    @property
    def shape(self):
        return self.weight.shape


tree_util.register_pytree_node(
    QuantizedWeight8bit,
    lambda qw: ([qw.weight, qw.scales], ()),
    lambda _, children: QuantizedWeight8bit(children[0], children[1]),
)
```

`weight` 是 int8（实际 dtype 由 ckpt 决定）、`scales` 是 fp32 或 bf16。`register_pytree_node` 把它变成 JAX 可以遍历的 pytree。这样 `jax.tree_map(...)` 等操作能自动展开。

`run.py:17` 重导出为 `QW8Bit`，但 `run.py` 没真的用到 8-bit - 默认是全精度 bf16 加载。

## 4.2 `RMSNorm`：fp32 中间计算

`model.py:587-624`：

```python
# model.py:587-624
class RMSNorm(hk.RMSNorm):

    def __init__(
        self,
        axis: Union[int, Sequence[int], slice],
        eps: float = 1e-5,
        name: Optional[str] = None,
        create_scale: bool = True,
        sharding: Optional[P] = None,
    ):
        super().__init__(axis, eps, create_scale=create_scale, name=name)
        self.sharding = sharding

    def __call__(self, inputs: jax.Array):
        fprop_dtype = inputs.dtype
        param_shape = (inputs.shape[-1],)
        if self.create_scale:
            scale = hk.get_parameter(
                "scale",
                param_shape,
                dtype=jnp.float32,
                init=hk.initializers.Constant(0),
            )
            if self.sharding:
                scale = with_sharding_constraint(scale, self.sharding)
            scale = jnp.broadcast_to(scale.astype(jnp.float32), inputs.shape)
        else:
            scale = 1.0
        inputs = inputs.astype(jnp.float32)
        scale = scale.astype(jnp.float32)
        mean_squared = jnp.mean(jnp.square(inputs), axis=[-1], keepdims=True)
        mean_squared = jnp.broadcast_to(mean_squared, inputs.shape)

        normed_inputs = inputs * jax.lax.rsqrt(mean_squared + self.eps)

        outputs = scale * normed_inputs

        return outputs.astype(fprop_dtype)
```

数学上：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

注意几件事：

1. **eps = 1e-5**（默认）- 与 Llama2 一致
2. **整个计算上 fp32**，最后才 cast 回 fprop dtype（bf16）。这是数值稳定的标配
3. **`scale` 默认 init 是 Constant(0)** - 仅做加载占位，真实值从 ckpt 读。注意 RMSNorm 的 scale 在训练时一般初始化为 1，如果真用 0 初始化训练，整个 layer 输出永远是 0。这印证了 init 仅是占位
4. **没有 mean 减法、没有 bias** - 与原始 LayerNorm 不同。这是 RMS 而非 mean-variance norm

`hk_rms_norm`（`model.py:489-496`）是个 wrapper：

```python
# model.py:489-496
def hk_rms_norm(
    x: jax.Array,
    fixed_scale=False,
    sharding=P(None),
) -> jax.Array:
    """Applies a unique LayerNorm to x with default settings."""
    ln = RMSNorm(axis=-1, create_scale=not fixed_scale, sharding=sharding)
    return ln(x)
```

`fixed_scale=False` 默认会创建 scale 参数。Grok-1 的所有 RMSNorm 都有可学习 scale。

### 4.2.0 为什么 RMSNorm 而非 LayerNorm

LayerNorm 计算均值再减、计算方差再除：

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

RMSNorm 省略了均值减法和 bias：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma
$$

理论分析（[RMSNorm paper](https://arxiv.org/abs/1910.07467)）发现：**LayerNorm 的"均值减"步骤对模型质量贡献很小**，删掉后只有微小回退（小于 0.1 PPL）但能省 ~15% 计算。在大模型时代这是免费午餐，所以 LLaMA 起所有主流 base 模型都用 RMSNorm。

bias 项 $\beta$ 同理 - 删掉对质量影响极小，但每个 norm 少 6144 个参数（× 64 层 × 4 个 norm = 1.5M 参数）。在 314B 里数字小，但是一个"减重"哲学的体现。

### 4.2.1 与 LLaMA-2 / Mistral 的对比

- **LLaMA-2** 用同样的 RMSNorm，scale 初始化为 1，eps = 1e-6
- **Mistral / Mixtral** 同 LLaMA-2
- **Grok-1** eps = 1e-5（默认），略大；scale 在 ckpt 里加载，init 仅占位
- **Grok-1 用了 4 个 RMSNorm 每层**（pre/post-attention + pre/post-FFN）- 比 LLaMA-2 多一倍

第 6 章会展开这个 sandwich norm。

## 4.3 RoPE：`rotate_half` 与 `RotaryEmbedding`

`model.py:627-691`。

```python
# model.py:627-633
def rotate_half(x: jax.Array) -> jax.Array:
    """Obtain the rotated counterpart of each feature"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)
```

注意 `jnp.split(x, 2, axis=-1)` 把最后一维**前半 / 后半**两块。这是 **"GPT-NeoX 风格"** 的 RoPE 排列，不是 **"interleaved"** 风格。

| 风格 | 拆分方式 | 代表实现 |
| --- | --- | --- |
| GPT-NeoX | `x = [x_first_half, x_second_half]` 然后 rotate | Grok-1, GPT-J, GPT-NeoX, LLaMA, Mistral, Mixtral |
| Interleaved | `x = [x_0, x_1, x_2, x_3, ...]` 偶数维度配对 | 原始 RoPE 论文, RoFormer |

**这两种风格的 RoPE 数值不等价** - ckpt 是用 NeoX 风格训练的，加载到 interleaved 实现会得到错误结果。这是社区迁移 ckpt 时最常见的坑之一。

### 4.3.1 RotaryEmbedding

```python
# model.py:635-691
class RotaryEmbedding(hk.Module):
    def __init__(
        self,
        dim: int,
        name: Optional[str] = None,
        base_exponent: int = 10000,
    ):
        super().__init__(name)
        self.dim = dim
        self.base_exponent = base_exponent
        assert self.dim % 2 == 0

    def __call__(
        self,
        x: jax.Array,
        seq_dim: int,
        offset: jax.Array,
        const_position: Optional[int] = None,
        t: Optional[jax.Array] = None,
    ) -> jax.Array:
        fprop_dtype = x.dtype
        exponents = jnp.arange(0, self.dim, 2, dtype=jnp.float32)
        inv_freq = jnp.asarray(
            1.0 / (self.base_exponent ** (exponents / self.dim)), dtype=jnp.float32
        )

        if jnp.shape(offset) == ():
            offset = jnp.expand_dims(offset, 0)

        if const_position:
            t = const_position * jnp.ones(
                (1, x.shape[seq_dim],), dtype=jnp.float32,
            )
        elif t is None:
            t = jnp.arange(x.shape[seq_dim], dtype=jnp.float32) + jnp.expand_dims(offset, -1)
        phase = jnp.einsum("bi,j->bij", t, inv_freq)
        phase = jnp.tile(phase, reps=(1, 2))[:, :, None, :]

        x = x * jnp.cos(phase) + rotate_half(x) * jnp.sin(phase)
        x = x.astype(fprop_dtype)

        return x
```

逐段：

**Line 665-668：频率表**

$$
\text{inv\_freq}[i] = \frac{1}{10000^{2i/d_h}}, \quad i \in [0, d_h/2)
$$

`exponents = [0, 2, 4, ..., d_h - 2]`，除以 `dim = d_h = 128`，得到 64 个频率。

**Line 670-684：每个位置的相位**

`offset` 是 KV cache 中已经缓存的 token 数。`t` 是从 0 开始的相对位置 + offset。在 prefill 阶段 offset = 0，t = [0, 1, ..., T-1]；在 decode 阶段 offset = cur_len, t = [cur_len]。

`phase` 形状是 `[B, T, d_h/2]`，然后 `jnp.tile(phase, reps=(1, 2))` 复制成 `[B, T, d_h]`（前后两半相同），再加上 head 维度变成 `[B, T, 1, d_h]`。

**Line 688：旋转应用**

$$
x' = x \cdot \cos(\phi) + \text{rotate\_half}(x) \cdot \sin(\phi)
$$

这等价于对每个 (dim_i, dim_{i+d/2}) 对做 2D 旋转。

### 4.3.1.1 GPT-NeoX 风格还是 interleaved，到底有什么实质区别

很多人不理解为什么"前后半"和"奇偶交错"会让 RoPE 数学等价但实现不可互换。我们来推导。

设 head_dim = 4，hidden vector $x = (x_0, x_1, x_2, x_3)$，频率 inv_freq = $(\omega_0, \omega_1)$，位置 $t$：

**Interleaved（原始 RoPE 论文）：**

$$
x'_0 = x_0 \cos(t\omega_0) - x_1 \sin(t\omega_0), \quad x'_1 = x_0 \sin(t\omega_0) + x_1 \cos(t\omega_0)
$$
$$
x'_2 = x_2 \cos(t\omega_1) - x_3 \sin(t\omega_1), \quad x'_3 = x_2 \sin(t\omega_1) + x_3 \cos(t\omega_1)
$$

把 (0, 1) 当做一个 2D 复数旋转、(2, 3) 当做另一个 2D 复数旋转。

**NeoX（Grok-1）：**

$$
x'_0 = x_0 \cos(t\omega_0) - x_2 \sin(t\omega_0), \quad x'_2 = x_0 \sin(t\omega_0) + x_2 \cos(t\omega_0)
$$

注意配对的不是 (0,1) 而是 (0, 2) - 即 (前半第 i 位, 后半第 i 位) 配对。

**这两种风格在 Q @ K 内积上是等价的吗？**

数学上是等价的 - 它们都实现"位置 t 的旋转矩阵作用于配对维度"。但**具体的 Q、K 向量值不同**，因为坐标排列不同。

后果：

- 一个用 NeoX 风格训练的 ckpt 里，Q 的 dim 0 对应"位置无关分量"，dim 2 对应它的"配对位置感知分量"
- 如果你拿这个 ckpt 用 interleaved 实现做 forward，会把 dim 0 和 dim 1 配对 - 完全错乱

这是个**坐标系问题**，不是数学问题。两种实现都"对"，但 ckpt 不能跨。

### 4.3.2 base = 10000

`base_exponent = 10000` 是 RoPE 的"标准"基数。这意味着：

- 最高频维度：周期约 $2\pi$ token
- 最低频维度（i = d_h/2 - 1）：周期约 $2\pi \cdot 10000^{(d_h - 2)/d_h} \approx 2\pi \cdot 8500$ token

在 `sequence_len = 8192` 的范围内，最低频维度只完成不到一个周期 - 远没到 RoPE 外推的混叠点。所以 Grok-1 用标准 base 10000 是合理的，没必要像 Llama-3.1 那样升到 500000。

### 4.3.3 与 Llama / Mistral 的对比

| 项 | Grok-1 | LLaMA-2 | Mistral 7B | Mixtral 8x7B |
| --- | --- | --- | --- | --- |
| 风格 | NeoX | NeoX | NeoX | NeoX |
| base | 10000 | 10000 | 10000 | 1000000 |
| 应用位置 | 在 KV cache update **之前** | 一致 | 一致 | 一致 |
| 应用对象 | Q + K（不对 V） | 一致 | 一致 | 一致 |

注意 Mixtral 8x7B 把 base 升到了 1000000，是为了支持 32k 上下文外推。Grok-1 只支持 8k，所以 10000 足够。

## 4.4 注意 RoPE 在 KV cache 更新前后的位置

`model.py:800-844` 是 attention 调用里的关键段：

```python
# model.py:800-844
rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
key_heads = rotate(key_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))
query_heads = rotate(query_heads, seq_dim=1, offset=(kv_memory.step if kv_memory else 0))

@functools.partial(jax.vmap)
def update_into(mem, start, update):
    return jax.lax.dynamic_update_slice_in_dim(mem, update, start, axis=0)

if kv_memory:
    if mesh is not None:
        @functools.partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("data", None, "model"),
                P("data"),
                P("data", None, "model"),
            ),
            out_specs=P("data", None, "model"),
            check_rep=False,
        )
        def update_into_shmap(mems, starts, updates):
            return update_into(mems, starts, updates)

        key_heads = update_into_shmap(kv_memory.k, kv_memory.step, key_heads)
        value_heads = update_into_shmap(kv_memory.v, kv_memory.step, value_heads)
    else:
        key_heads = update_into(kv_memory.k, kv_memory.step, key_heads)
        value_heads = update_into(kv_memory.v, kv_memory.step, value_heads)

    new_step = kv_memory.step + sequence_length
    memory_mask = jnp.arange(kv_memory.k.shape[1]) < new_step[:, None]
    memory_mask = memory_mask[:, None, None, :]  # [B, H, T, T]
    if mask is not None:
        mask = memory_mask * mask
    else:
        mask = memory_mask

    new_memory = KVMemory(
        k=key_heads,
        v=value_heads,
        step=new_step,
    )
```

执行顺序：

1. 投影 Q/K/V（在 `model.py:774-799`）
2. 对 K 和 Q **先应用 RoPE**（`model.py:801-803`）
3. 然后把新 K/V 写回 KV cache（`model.py:826-830`）
4. 构造 memory mask（已写入的位置才能被 attend）

注意 RoPE 应用在 KV cache update 之前 - 这意味着 cache 里**存的是已经过 RoPE 的 K**。这是和 LLaMA 一致的做法。

`update_into` 用 `jax.lax.dynamic_update_slice_in_dim` 把新 K/V 写到 `kv_memory.step` 指定的位置。`jax.vmap` 把它沿 batch 维 vmap，因为不同 batch 元素可能有不同的 step。

`memory_mask`：构造一个 `[T]` 的 0/1 mask，"位置 < 当前 step" 的为 1。然后 expand 成 `[B, 1, 1, T]`，再和 causal mask 相乘。

## 4.5 `MultiHeadAttention.__call__` 全程

`model.py:720-911` 是 attention 主体。我们已经看过 RoPE + cache 部分（4.4），现在看 attention 本身的计算。

### 4.5.0 GQA 的工程动机

GQA（Grouped Query Attention）是 MHA（Multi-Head Attention）和 MQA（Multi-Query Attention）的折中。

- **MHA**：每个 head 都有独立 Q/K/V，质量最好，但 KV cache 最大
- **MQA**：所有 head 共享同一组 K/V（即 num_kv_heads = 1），KV cache 最小，但质量下降明显
- **GQA**：把 Q head 分组，组内共享 K/V

对于 Grok-1，64 层 × 8192 token × KV cache：

- MHA（如果 num_kv_heads = 48）：64 × 8192 × 48 × 128 × 2 (k+v) × 2 bytes = 12 GB / batch
- GQA 48:8：64 × 8192 × 8 × 128 × 2 × 2 = **2 GB / batch**
- MQA：64 × 8192 × 1 × 128 × 2 × 2 = 256 MB / batch

Grok-1 选 8:1 GQA（48 Q : 8 KV），KV cache 比 MHA 节省 6 倍，比 MQA 大 8 倍。质量 - 内存的中间点。

LLaMA-2 70B 选 8:1 GQA（64 Q : 8 KV），Mixtral 8x7B 选 4:1（32 : 8）。Grok-1 介于二者之间。

### 4.5.1 Q/K/V 投影

```python
# model.py:773-799
assert self.num_q_heads % self.num_kv_heads == 0
query_heads = projection(
    query, self.key_size, self.num_q_heads, name="query",
    sharding=P("data", "model"), mesh=mesh,
)  # [B, T', H, Q=K]

key_heads = projection(
    key, self.key_size, self.num_kv_heads, name="key",
    sharding=P("data", "model"), mesh=mesh,
)  # [B, T, H, K]

value_heads = projection(
    value, self.value_size, self.num_kv_heads, name="value",
    sharding=P("data", "model"), mesh=mesh,
)  # [B, T, H, V]
```

`_linear_projection`（`model.py:893-911`）：

```python
# model.py:893-911
def _linear_projection(
    self, x, head_size, num_heads, sharding=None, name=None, mesh=None,
):
    y = Linear(
        num_heads * head_size,
        with_bias=False,
        name=name, sharding=sharding, mesh=mesh,
    )(x)
    *leading_dims, _ = x.shape
    return y.reshape((*leading_dims, num_heads, head_size))
```

Q/K/V projection 都是 `with_bias=False` - 注意这点 - 与 `MultiHeadAttention.__init__` 的 `with_bias=True` 默认不冲突，因为这里硬编码 False。

输出 shape：

- Q: `[B, T, 48, 128]`
- K: `[B, T, 8, 128]`
- V: `[B, T, 8, 128]`

### 4.5.2 GQA reshape 与 attention 计算

```python
# model.py:846-866
query_heads = with_sharding_constraint(query_heads, P(self.data_axis, None, "model", None))
key_heads = with_sharding_constraint(key_heads, P(self.data_axis, None, "model", None))
value_heads = with_sharding_constraint(value_heads, P(self.data_axis, None, "model", None))
b, t, h, d = query_heads.shape
_, _, kv_h, _ = key_heads.shape
assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"

query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))
query_heads = with_sharding_constraint(
    query_heads, P(self.data_axis, None, "model", None, None)
)

# Compute attention weights.
# Attention softmax is always carried out in fp32.
attn_logits = jnp.einsum("...thHd,...Thd->...hHtT", query_heads, key_heads).astype(
    jnp.float32
)
attn_logits *= self.attn_output_multiplier
max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)
```

Q 被 reshape 成 `[B, T, 8, 6, 128]` - 8 个 KV group，每 group 6 个 Q 头。

注意 einsum：`"...thHd,...Thd->...hHtT"`

- 输入 Q：`[..., t, h=kv_h, H=q_per_kv, d=128]`
- 输入 K：`[..., T, h=kv_h, d=128]`（无 H 维度，因为 K 在 group 内共享）
- 输出：`[..., h=kv_h, H=q_per_kv, t, T]`

也就是说 K 在 `q_per_kv` 上是 **broadcast**，不需要显式复制 K。这就是 GQA 的精髓 - **节省 6 倍的 K/V 内存与 IO**。

接着：

```python
attn_logits *= 1/sqrt(128)
attn_logits = 30 * tanh(attn_logits / 30)
```

这是第 3 章 3.3.2 说过的软裁剪。

### 4.5.3 mask 与 softmax

```python
# model.py:867-876
mask = mask[:, :, None, :, :]

if mask is not None:
    if mask.ndim != attn_logits.ndim:
        raise ValueError(...)
    attn_logits = jnp.where(mask, attn_logits, -1e30)
attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)  # [H, T', T]
```

mask 在传入时是 `[B, 1, T, T]`，加一个 `None` 维度变成 `[B, 1, 1, T, T]`，与 logit 的 `[B, kv_h, q_per_kv, T, T]` 广播。

`softmax` 是 fp32 计算（因为 logit 是 fp32），然后 cast 回 query dtype（bf16）。这是数值稳定的标配。

### 4.5.4 加权求和与最终投影

```python
# model.py:878-891
attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads)
attn = with_sharding_constraint(attn, P(self.data_axis, None, "model", None, None))
leading_dims = attn.shape[:2]
attn = jnp.reshape(attn, (*leading_dims, -1))  # [T', H*V]
attn = with_sharding_constraint(attn, P(self.data_axis, None, "model"))
final_projection = Linear(
    self.model_size,
    with_bias=False,
    sharding=P("model", "data"),
    mesh=mesh,
)
return MHAOutput(final_projection(attn), new_memory)
```

einsum `"...hHtT,...Thd->...thHd"` 同样让 V 在 H（q_per_kv）维 broadcast。

reshape 把 `[B, T, kv_h, q_per_kv, V]` 压成 `[B, T, kv_h * q_per_kv * V] = [B, T, 48 * 128] = [B, T, 6144]`。

最终 projection `(6144, 6144)` with no bias，输出 `[B, T, 6144]`。

## 4.6 与 LLaMA、Mistral、GPT-NeoX 的差异速查

| 项 | Grok-1 | LLaMA-2 70B | Mistral 7B / Mixtral | GPT-NeoX |
| --- | --- | --- | --- | --- |
| Q 头数 | 48 | 64 | 32 | 64 |
| KV 头数 | 8 | 8 | 8 | 64 (no GQA) |
| Q/KV 比 | 6 | 8 | 4 | 1 |
| head dim | 128 | 128 | 128 | 96 |
| RoPE 风格 | NeoX | NeoX | NeoX | NeoX |
| RoPE base | 10000 | 10000 | 10000 / 1000000 | 10000 |
| Attention logit 软裁剪 | **有（30·tanh）** | 无 | 无 | 无 |
| QKV bias | 否 | 否 | 否 | 是 |
| Output proj bias | 否 | 否 | 否 | 是 |
| Norm | RMSNorm，pre+post | RMSNorm，pre | RMSNorm，pre | LayerNorm，pre |

Grok-1 的差异点：

1. **GQA 比例 6:1** - 比 LLaMA-2 70B 的 8:1 更"密"。代价是 KV cache 更大一点（少 6 倍 vs 少 8 倍），收益是表达能力更强
2. **30·tanh 软裁剪** - 全本书唯一一处显式 logit 裁剪
3. **sandwich norm** - 见第 6 章

下一章进入本仓库最难的部分：MoE 路由的 shard_map 实现。

## 延伸阅读

- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - GQA 的原始论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE 原始论文
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm 原始论文
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) - 同代 GQA + MoE 的最直接对照
