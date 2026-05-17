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

!!! note "残差流（residual stream）"
    这是阅读现代 Transformer 源码时反复出现的关键概念。把每层的输入 hidden 张量 $h \in \mathbb{R}^{B \times T \times d}$ 想象成一条"信息高速公路"：每一个 sub-layer（attention 或 FFN）从这条流上**读**当前状态，计算出一个增量 $\Delta h$，然后再**写**回流上（即 $h \leftarrow h + \Delta h$）。

    这种"读-算-写"的结构是残差连接（residual connection）的实质：每个 sub-layer 只负责**修改**hidden，而不是**重新生成**hidden。它带来的好处是梯度在反向传播时有一条几乎无衰减的直通路径（identity path），训练得起 64 层、80 层、甚至 100+ 层的网络。Grok-1 的 sandwich norm 设计（6.3 节展开）本质上是在"读"和"写"两端各加一次 norm，把残差流的量级控制得更死。

    在 PyTorch 里，残差流就是 `for layer in self.layers: h = h + layer(self.norm(h))` 这一行中的 `h`；在 Grok-1 的 JAX 代码里，它对应 `Transformer.__call__` 中的局部变量 `h`，被 64 次 `h += ...` 反复更新。理解残差流是一根贯穿全模型的张量，是理解 Transformer 整体结构的关键。

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

!!! note "self-attention vs cross-attention"
    第 5 章的 `MultiHeadAttention.__call__` 签名是 `(query, key, value, mask, memory)` 四个张量分立的形式。这种通用接口同时支持两种模式：

    - **self-attention（自注意力）**：query 与 key/value 来自同一个序列，即 query == key == value，模型在处理"序列内部"的依赖关系。GPT 系列、LLaMA 系列、Grok-1 这类 decoder-only 模型只用 self-attention。
    - **cross-attention（交叉注意力）**：query 来自一个序列，key/value 来自另一个序列。典型场景是 encoder-decoder 架构里的 decoder 看 encoder 输出，或者 Flamingo 这种多模态模型里文本 token 看图像 token。

    `MHABlock` 是 Grok-1 对 self-attention 这种"最常用情形"的封装：把同一个 inputs 张量同时塞进 q、k、v 三个位置。这一层之所以单独存在而不直接调用 `MultiHeadAttention`，是为了让 `DecoderLayer` 的代码更直观：每层一句话 `MHABlock(layer_norm(h), mask, layer_memory)` 就完成自注意力的全部细节。

`@hk.transparent` 装饰器的作用是让这个 module 的参数 scope **不进入** Haiku 的命名空间层级。也就是说，`MultiHeadAttention` 创建的参数名直接挂在父 module（DecoderLayer）下面，最终参数路径不会出现 `mha_block/multi_head_attention/query/w` 的形式，而是直接呈现为 `decoder_layer_0/multi_head_attention/query/w`。

!!! note "`hk.transparent` 与 PyTorch 的 `nn.Sequential`"
    在 PyTorch 里，每个 `nn.Module` 会按照它在父 module 中被赋值的属性名（例如 `self.attn = MyAttention()`）形成参数命名前缀，`state_dict()` 输出形如 `layers.0.attn.query.weight` 的扁平字典。如果想"去掉中间一层命名"，PyTorch 没有原生的 transparent 机制，通常做法是直接把子 module 的参数 "拷贝指针" 到外层（或者干脆把子 module 的代码 inline 到父 module 里）。

    Haiku 的 `@hk.transparent` 提供了一种轻量做法：被装饰的 module 在参数命名时**透明**地穿透过去，让内部子 module 直接挂在父 module 的命名空间下。这种设计的实际收益是 ckpt 中的参数路径更短、ckpt 文件结构更清晰。对应到 Grok-1 的 ckpt（第 7 章详述），770 个参数 tensor 的命名几乎都遵循 `language_model/transformer/decoder_layer_i/<子模块>/<参数>` 这种四级路径，正是因为 `MHABlock`、`DenseBlock` 这些中间层都用了 `@hk.transparent`。

这种设计的好处是让 ckpt 中的参数路径更简短、更易读，代价则是 `MHABlock` 无法作为独立的 module 完成 init/apply。换言之，`@hk.transparent` 装饰之后的 module 只能作为别人的"嵌入式子模块"被使用，不能脱离父 module 单独存在。

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

数学上把它逐步拆开看：

记输入 $x \in \mathbb{R}^{B \times T \times d}$，$d = 6144$、$d_{ffn} = 32768$。三个权重矩阵：

- $W_v \in \mathbb{R}^{d \times d_{ffn}}$：value 分支的升维投影（代码中的 `linear_v`）
- $W_g \in \mathbb{R}^{d \times d_{ffn}}$：gate 分支的升维投影（代码中的 `linear`）
- $W_1 \in \mathbb{R}^{d_{ffn} \times d}$：降维投影（代码中的 `linear_1`）

完整计算：

$$
h_v = x W_v \in \mathbb{R}^{B \times T \times d_{ffn}}
$$

$$
h_g = \text{GELU}(x W_g) \in \mathbb{R}^{B \times T \times d_{ffn}}
$$

$$
\text{FFN}(x) = (h_g \odot h_v) W_1 \in \mathbb{R}^{B \times T \times d}
$$

其中 $\odot$ 表示按位相乘（Hadamard product）。可以简写为：

$$
\text{FFN}(x) = W_1 \cdot (\text{GELU}(W_g x) \odot W_v x)
$$

这里"gate"分支并不只是一个 0-1 的门控信号 - 它是经过 GELU 之后的连续浮点数，可以放大、压制、甚至反向 value 分支的某一维。这给 FFN 带来了比传统 `down(activation(up(x)))` 形式更强的表达力，是 SwiGLU/GeGLU 系列在多项基准上略胜一筹的根本原因。

注意 `Linear(...)`（中间那个，没传 name）会得到 `linear` 作为默认 name（来自 `hk.Linear` 的默认）。所以三个矩阵的参数名是：

- `linear_v`：up projection 1（value path）
- `linear`：up projection 2（gate path）
- `linear_1`：down projection

这正好匹配 partition rules 里 `linear`、`linear_v`、`linear_1` 三个名字。

!!! note "PyTorch 对照：LLaMA 的 `FeedForward`"
    LLaMA 在 PyTorch 里的 SwiGLU FFN 实现是这样的：

    ```python
    class FeedForward(nn.Module):
        def __init__(self, dim, hidden_dim):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # value

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))
    ```

    与 Grok-1 的 `DenseBlock` 几乎是一比一对应：`w1` 对应 `linear`（gate）、`w3` 对应 `linear_v`（value）、`w2` 对应 `linear_1`（down）。两个最显眼的差别：

    1. **激活函数**：LLaMA 用 SiLU（Swish），Grok-1 用 GELU。两者图形相似（都是 S 型平滑曲线），数值表现也接近，互换一般不会显著影响下游指标
    2. **partition / sharding**：PyTorch 实现里完全没有 `mesh`、`sharding=P("data", "model")` 这类参数。需要做张量并行时要靠外部包装（例如 `ColumnParallelLinear`、`RowParallelLinear`）。Grok-1 的每个 `Linear` 都把 sharding 信息直接写进构造参数里，编译时就让 XLA 知道这个矩阵是按列切还是按行切

    这种差别反映出 JAX 的"声明式并行"哲学 - 在写模型代码时就把所有数据布局信息标注好，编译器根据这些标注自动生成跨设备通信，不需要工程师手写 `all_reduce`。代价是模型代码里到处都是 sharding 参数，阅读起来稍微吃力。

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

把三种 norm 布局并排放在一起对比：

**Post-norm（原始 Transformer，2017 年 "Attention is All You Need" 用的方案）：**

```
y = LayerNorm(x + SubLayer(x))
```

子层先与残差相加，再 norm。残差流上的每一个状态都经过 norm 重新归一化。训练困难，因为深层时残差被反复 norm，原始 identity path 被 norm 反复缩放后梯度难以无衰减地穿过几十层。这是早期 Transformer 训练时需要"warmup learning rate"等技巧的根本原因之一。

**Pre-norm（GPT-2 起的主流，LLaMA、Mistral、Mixtral、DeepSeek、Qwen 等都是这种）：**

```
y = x + SubLayer(LayerNorm(x))
```

先把残差流 norm 一下再喂进子层，子层的输出直接加回残差流（不再 norm）。优点是 identity path 完全没有 norm 干扰，梯度可以零衰减地从最后一层一路传到第一层；缺点是残差流自己的量级会随层数增加单调累积（每层都做加法但从不缩放）。在 32-64 层这个深度还可控，到 80+ 层时残差量级可能膨胀到训练不稳定。

**Sandwich norm（Grok-1、Cohere Command R、早期 Gopher）：**

```
y = x + LayerNorm(SubLayer(LayerNorm(x)))
```

子层的输入端做一次 norm（与 pre-norm 一致），子层的输出端再做一次 norm（再加回残差流）。两端都被 norm 约束，残差流上每一次写入都是已经归一化过的"小步长"更新，量级不会单调累积。代价是每层多一次 RMSNorm 调用，参数也多一倍（每个 sub-layer 有 2 个 norm，每层 2 个 sub-layer，共 4 个 RMSNorm）。

!!! note "为什么 RMSNorm 而不是 LayerNorm"
    Grok-1 这里用的是 RMSNorm（第 4 章逐行讲过）而不是经典的 LayerNorm。两者的区别：

    - **LayerNorm**：减均值、除标准差、乘 scale、加 bias，公式是 $\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$
    - **RMSNorm**：只除 root-mean-square、乘 scale，公式是 $\text{RMS}(x) = \frac{x}{\sqrt{\text{mean}(x^2)}} \cdot \gamma$

    RMSNorm 去掉了"减均值"和"加 bias"两步，参数从 $2d$ 减到 $d$，计算从两次 reduce 减到一次。实测在大模型上效果与 LayerNorm 接近，但更快。LLaMA、Mistral、Mixtral、Grok-1 等现代大模型几乎都用 RMSNorm。

64 层这个深度在 pre-norm 设置下未必一定会出现残差爆炸，所以 Grok-1 选择 sandwich norm 并非数值稳定性上的必需选项，更可能反映出 xAI 团队在经验上偏向更保守的设计取向 - 多花一点 norm 的算力，换来训练时无需为残差爆炸调参的省心。

值得多说一句的是，"sandwich norm" 并非 xAI 首创。Cohere 在 Command R 系列也用了这种设计；更早的 DeepMind Gopher（2021）就尝试过类似布局；2022 年的 NormFormer 论文系统比较过几种 norm 布局对训练稳定性的影响，结论之一就是"在 sub-layer 输出端再加一次 norm"能够显著降低梯度方差。Grok-1 选用 sandwich norm 大概率受到了这条线索的启发。

!!! note "Haiku module 的 init 与 apply 心智模型"
    Grok-1 的 `DecoderLayer` 是一个标准的 `hk.Module`，它的 `__call__` 方法内部用 `Linear(...)` / `MultiHeadAttention(...)` / `hk_rms_norm(...)` 等子模块。理解 Haiku module 怎么工作的关键是分清两个阶段：

    - **init 阶段**：`hk.transform(forward).init(rng, x_example)` 调用 forward 函数时，Haiku 会拦截每一个 `hk.get_parameter` 调用，从 init 函数生成对应 shape 的初始参数，最终返回一个嵌套字典 `params`
    - **apply 阶段**：`forward_t.apply(params, rng, x)` 调用时，Haiku 把 params 字典里的参数注入到对应的 `hk.get_parameter` 调用位置，让 forward 函数能拿到训练好的权重

    这种"两阶段"设计与 PyTorch 的差别：PyTorch 的 `nn.Module` 在 `__init__` 阶段就显式创建 `nn.Parameter` 对象（要写出 shape），并把参数挂在 `self.xxx` 上；Haiku 是"惰性"的，参数在第一次调用 `__call__` 时根据输入 shape 自动推断。Grok-1 大部分 `Linear` 的 `output_size` 与 `input_size` 都依赖 runtime 张量 shape（例如 `Linear(ffn_size(model_size, w))`），正是利用了这种惰性。

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

!!! note "为什么 hidden 维要沿 model 轴切，sequence 维不切"
    几句直观的解释：

    - **batch 维沿 data 轴切**：data 并行的本意就是"同一个模型、不同的输入"，让每张卡处理一部分 batch。Grok-1 单机 8 卡时 data 维长度是 1，意味着 batch 不真的被切（只有一份 data shard），但保留了未来扩到多机时立即可用的接口。
    - **sequence 维不切**：sequence 之间存在注意力的全局依赖（每个位置都要看到所有历史位置），如果按 sequence 切分会让 attention 变得需要大量跨设备通信。Megatron-LM 后来发展出"sequence parallelism"专门处理这种情况，但代码复杂度显著上升。Grok-1 选择不在 sequence 维做并行，简化设计。
    - **hidden 维沿 model 轴切**：这就是经典的"张量并行"（tensor parallelism）- 把一个大矩阵的列分到多张卡上，每张卡只负责一部分 hidden 通道。配合 attention 与 FFN 内部 head 维度的 sharding（详见第 5 章），整个模型的权重和激活都能被自然地切到 8 张卡上。

    用一句话总结 Grok-1 的并行策略：**沿 model 轴做张量并行、沿 data 轴预留 data 并行接口、sequence 维不并行**。

XLA 编译器会利用这些 sharding 约束来决定具体的通信策略，例如 attention 输出之后的 hidden 张量是否需要在 model 轴上进行 all-gather。

!!! note "PyTorch 对照：手写 all_reduce vs JAX 声明式 sharding"
    在 PyTorch 张量并行实现里（Megatron-LM、DeepSpeed-Tensor），residual add 之后通常要显式调用一次 `torch.distributed.all_reduce(h)` 把每张卡上的部分激活求和。代码大致长这样：

    ```python
    h_attn = self.attn(self.norm1(h))
    h_attn = self.row_parallel_output(h_attn)  # 内部含 all_reduce
    h = h + h_attn
    ```

    每个 sub-layer 的输出投影（`RowParallelLinear`）内部封装了 all_reduce。工程师在阅读时需要意识到"这里有一次 NCCL 通信"。

    Grok-1 的等价代码是 `h = with_sharding_constraint(h, P("data", None, "model"))` - 不直接调用通信原语，而是告诉编译器"我希望这一步之后张量的布局是这样的"。XLA 会检查输入张量当前的布局与目标布局是否一致，如果不一致就自动插入需要的 collective op（all_reduce、all_gather、reduce_scatter 等）。

    这种"声明式"风格的好处是模型代码只描述数学结构，不掺杂通信细节；缺点是新人很难直觉地看出哪一步会触发跨设备通信，必须借助 `jax.jit(...).lower(...).compile().as_text()` 查看 XLA 编译后的 HLO 才能知道实际通信开销。

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

!!! note "`hk.scan` 与 `jax.lax.scan`"
    JAX 提供 `jax.lax.scan(f, init, xs)` 作为"函数式 for 循环"原语：把同一个函数 $f$ 作用在 $xs$ 的每一个元素上，前一次调用的结果作为下一次调用的初始状态。它的好处是**循环体只编译一次**，编译器把它视为一个"重复执行 N 次"的单一原语，编译时间和编译产物大小都不会随 N 增长。

    `hk.scan` 是 Haiku 在 `jax.lax.scan` 之上的包装，让你能在 scan 里调用 hk module，参数会自动 stack 起来（例如 64 层 attention 的 query 权重会合并成一个 `[64, ...]` shape 的张量）。

    PyTorch 没有完全对应的原语 - 它的 for 循环就是 Python for 循环，每次迭代都会重新展开（因为 PyTorch 是 eager mode + 动态图）。`torch.utils.checkpoint.checkpoint_sequential` 在某种意义上接近这个语义，但只用于 activation checkpointing，不会合并参数。

这种实现方式带来几个直接后果：

1. **每一层的参数都被独立存放**，参数路径分别是 `decoder_layer_0/...`、`decoder_layer_1/...`，共 64 个不同的前缀
2. **编译时 64 层全部被完全 unroll**，XLA 实际看到的是一张极长的、把全部 64 层串联展开的计算图
3. **优点**：可以为每一层添加独立的 sharding 约束、独立调试单层行为，partition rules 也可以直接用 `decoder_layer_[0-9]+` 这种正则统一匹配
4. **缺点**：编译时间长、内存占用大，Grok-1 实测在 JIT compile 阶段会消耗几十 GB 主机内存

如果改用 `hk.scan` 风格实现，则会把 64 层折叠成一个 leading-64 的维度，参数 stack 在一起，编译时实际只需要分析一层。Mixtral 在 PyTorch 实现中也是 for-loop 形式，整体行为与 Grok-1 类似。

`partition_rules` 中针对 `layer_stack` 做了专门处理（见 `model.py:102-104`），这暗示代码早期版本中可能存在过基于 scan 的实现，但当前开源版本最终使用的是 for-loop 实现。这种"代码里残留的旧痕迹"在工业级开源项目里很常见 - 是研究迭代过程中尝试过、又放弃的技术路径的化石。

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

!!! note "`embedding_multiplier_scale` 与 `output_multiplier_scale` 的来历"
    Grok-1 在 embedding 之后乘 $\sqrt{d}$、在 logits 之前乘 $1/\sqrt{3}$，这两个 magic number 出现得非常突兀。来历分别是：

    - **乘 $\sqrt{d} \approx 78.38$**：原始 Transformer 论文（Vaswani et al. 2017）就在 embedding 后乘 $\sqrt{d}$。原因是 token embedding 的初始化方差通常是 $1/d$（让每维平均能量约为 $1/d$），但残差流期望的能量是 $1$ 量级。乘 $\sqrt{d}$ 之后每维能量变成 $1$，与残差流后续的 norm 假设吻合。
    - **乘 $1/\sqrt{3} \approx 0.577$**：这是 logits 的"温度缩放因子"。tied embedding 时，输出 logits = hidden @ embed_mat.T，由于 embed_mat 的元素方差与 hidden 的方差不一定匹配，乘一个常数可以让 logits 的分布更接近"训练时希望的尺度"，避免 softmax 出来要么全是 0 要么全是 1。$1/\sqrt{3}$ 这个具体值是 xAI 训练过程中调出来的经验值。

    这两个 scale 都是**训练时和推理时必须保持一致的固定常数**，不能修改。如果你自己训练 Grok-1 风格模型，这两个值可能需要重新搜索；但用 Grok-1 ckpt 做推理时必须严格用代码里写死的这两个值。

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

!!! note "tied embedding 的几何解读"
    把 embedding 矩阵 $E \in \mathbb{R}^{V \times d}$ 想象成一个"字典"：每一行 $E_i$ 是第 $i$ 个 token 在 $d$ 维空间中的方向向量。

    - **输入侧**：给定 token id $t$，模型读取 $E_t$ 作为这个 token 在残差流上的初始表示
    - **输出侧（tied）**：给定 hidden $h$，模型计算 $h \cdot E^T$，得到一个长度为 $V$ 的向量，第 $i$ 个分量是"hidden 与第 $i$ 个 token 的方向向量的内积"。内积越大表示 hidden 越接近这个 token，对应的 logit 也越大

    几何上理解：tied embedding 强迫"读"和"写"用同一套坐标系。如果一个 token 的输入 embedding 是 $E_t$，那它的输出 logit 也由 $E_t$ 决定。这种"读写对称"的约束在小模型上节省参数（减少 overfitting），在大模型上也几乎不损失性能。

    untied 的做法是在输出端单独放一个 $E_{out} \in \mathbb{R}^{V \times d}$，让"读"和"写"用不同的字典。多出来的参数主要被用来"修正"输入字典在做 logits 时不够精准的部分。GPT-3 选 untied，可能是因为 175B 这个规模下多 0.8B 参数也不算什么，但额外的灵活性可能略有收益。

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

!!! note "KV cache 是什么、为什么必须有"
    自回归生成时模型一次产出一个 token。如果每个 step 都重新跑完整的 forward（把全部历史 token 重新 attention 一遍），算力消耗会是 $O(T^2)$ - 第 1000 个 token 需要做 1000 次 attention，第 2000 个 token 需要做 2000 次。

    KV cache 的思路：每一层的 K 和 V 一旦算出来，**永远不会再变**（因为 K/V 只取决于历史 token，不取决于未来 token）。所以把它们缓存起来，每个 step 只需要为新增的 1 个 token 计算 K/V 并 append 到 cache 上，然后用 cache 里的所有 K/V 与当前 query 做 attention。

    这样每个 step 的计算量从 $O(T)$ 降到 $O(1)$（与历史长度无关），整个生成的总计算量从 $O(T^2)$ 降到 $O(T)$。代价是要存 $T \times d_{kv} \times L$ 大小的 cache - 这就是 KV cache 显存压力的来源。

    Grok-1 的 KV cache 预分配为 `(B, 8192, 8, 128)` - 8192 是最大 sequence 长度，预留好整个空间避免推理时动态扩展。

### 6.6.1 `step` 字段

每个 batch 元素拥有各自独立的 step 计数（初始化为 `jnp.zeros(batch_size, dtype=jnp.int32)`）。这种设计允许同一 batch 中的不同元素处于不同的生成阶段，正是为 continuous batching 与异步 serving 场景准备的接口。

!!! note "continuous batching"
    传统的 batched 推理要求 batch 内所有 sequence 同步开始、同步结束。如果一条 sequence 在第 100 个 token 就生成完了（吐出 EOS），它得等其他 sequence 继续生成到第 1000 个才能整体收尾。这种"齐步走"导致 GPU 利用率低。

    continuous batching（vLLM 等推理框架的核心机制）允许每条 sequence 独立结束，结束后立即把这个 slot 让出来给新的请求。Grok-1 的 `step` 字段每个 batch 元素一个独立 int32 计数器，意味着代码层面已经为这种模式预留了接口，只是默认的 `runners.py` 没有真正用上。要在 Grok-1 上做高吞吐 serving，可以基于这个接口扩展。

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

!!! note "PyTorch 对照：HuggingFace `transformers` 的 `past_key_values`"
    HuggingFace 的 `transformers` 库里 KV cache 叫 `past_key_values`，是一个 `tuple[tuple[Tensor, Tensor], ...]` - 外层 tuple 长度 = num_layers，每层是 `(K, V)` 二元组。模型 forward 时把 `past_key_values` 当参数传进去，模型把它和新 token 的 K/V 拼接后做 attention，再返回新的 `past_key_values`。

    Grok-1 的 `Memory(layers=List[KVMemory])` 在结构上完全对应。差别在于：

    1. **Grok-1 预分配最大长度**：cache shape 一开始就是 `(B, 8192, ...)`，新 token 的 K/V 通过 `jax.lax.dynamic_update_slice_in_dim` 写到对应位置。这种"预分配 + in-place update"的风格在 JAX 里更高效，因为可以避免每个 step 都做张量 concat（jax 数组 immutable，concat 会分配新内存）
    2. **PyTorch 通常动态扩展**：每个 step 把新 K/V `torch.cat` 到老 K/V 上，cache 长度随生成长度增长。代价是反复分配显存，但优点是不需要预先知道最大长度
    3. **Grok-1 多一个 `step` 字段**：每个 batch 元素独立的 step 计数器，前面提到这是为 continuous batching 准备的

    工业级 PyTorch 推理框架（vLLM、TensorRT-LLM）后来也采用"预分配 + paged attention"，与 Grok-1 这种"预分配 + 整块 cache"在思路上是同一方向的演进。

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

## 6.8.1 端到端数据流复盘

现在所有的拼图都齐了，把 `forward(tokens) -> logits` 这一整条数据流再串一遍。假设输入是 `tokens.shape = (1, 128)`（batch=1，prompt 长度 128），目标是计算最后一个位置的 logits。

**Step 0：input mask**

```python
input_mask = jnp.greater(tokens, 0)  # [1, 128] - True 表示非 pad
```

**Step 1：embedding lookup**

```python
input_embeddings = in_out_embed(tokens)  # [1, 128, 6144]
input_embeddings *= 78.38  # 乘 sqrt(d)
```

这一步把每个 token id 映射成一个 6144 维向量。结果就是残差流的初始状态。

**Step 2：构造 mask**

```python
mask = input_mask[:, None, None, :]  # [1, 1, 1, 128] - padding mask
causal_mask = jnp.tril(jnp.ones((1, 1, 128, 128)))  # 下三角全 1
mask = mask * causal_mask  # [1, 1, 128, 128]
```

**Step 3：64 层 DecoderLayer**

```python
h = input_embeddings  # [1, 128, 6144]
for i in range(64):
    # attention sub-layer
    h_attn = MHA(RMSNorm(h), mask, kv_memory[i])
    h_attn = RMSNorm(h_attn)
    h = h + h_attn

    # FFN sub-layer（MoE 分支，因为 num_experts=8）
    h_dense = MoELayer(RMSNorm(h), router, 8_experts)
    h_dense = RMSNorm(h_dense)
    h = h + h_dense
```

每一层 hidden 都保持 `(1, 128, 6144)` 这个 shape 不变；变的只是张量的具体数值。残差流被读 4 次（每层两个 sub-layer，每个 sub-layer 一次 pre-norm 读、一次 add 写），共写回 2 次（每层两个 sub-layer 各一次 residual add）。

**Step 4：最终 RMSNorm**

```python
embeddings = layer_norm(h)  # [1, 128, 6144]
```

注意这一步不在残差流上 - 它是单向变换。

**Step 5：取最后位置**

```python
last_step = length - 1  # 假设 length=128, last_step=127
last_hid = embeddings[:, last_step:last_step+1, :]  # [1, 1, 6144]
```

**Step 6：投影回 logits**

```python
logits = last_hid @ embed_mat.T  # [1, 1, 131072]
logits *= 0.577  # 乘 1/sqrt(3)
```

得到一个 131072 维的向量，每一维对应词表中一个 token 的得分。后续 sampler（第 8 章）从这个分布中采样出下一个 token。

整个 forward 走过 64 × (1 次 attention + 1 次 MoE) = 64 次 attention + 64 次 MoE 计算。如果按"激活参数 86B"近似，每个 token 需要约 $2 \times 86 \times 10^9 = 172$ GFLOPs，128 个 token 的 prefill 总计算量约 22 TFLOPs - 一张 H100 fp16 算力 989 TFLOPs，理论上 prefill 只需要约 22 毫秒。实际由于带宽限制和算子启动开销，prefill 128 token 大约要几百毫秒。

## 6.9 总结：model.py 的设计风格

回过头来俯瞰整份 `model.py`（1398 行），可以提炼出几条贯穿全文件的设计取向：

1. **Haiku module 树整体扁平**：大量使用 `@hk.transparent` 让参数名缩短，ckpt 路径更易读
2. **for-loop 显式展开 64 层**：编译产物大但灵活，便于逐层加约束与调试
3. **sandwich norm**：每一层包含 4 个独立的 RMSNorm，残差量级被双重约束
4. **GeGLU MoE**：8 个专家 + top-2 路由，每个 token 实际只走 25% 的 FFN 参数
5. **GQA 48:8**：6 倍 KV 压缩，单 token KV cache 从 12 GB 降到 2 GB
6. **8-bit 量化作为一等公民**：Linear 内部内置了反量化路径，无需在外层做特殊处理
7. **tied embedding**：节省约 0.8B 参数
8. **声明式 sharding**：每个 Linear 自带 sharding 信息，让 XLA 编译器自动生成跨设备通信
9. **预分配 KV cache + 函数式更新**：用 `dynamic_update_slice_in_dim` 替代 concat，避免每个 step 都做内存分配

把 Grok-1 的这套设计与 LLaMA、Mixtral、DeepSeek-V2 横向对比：架构上没有什么"独门绝技"，每一条选择在 2024 年初都能在前沿模型中找到对应案例。Grok-1 真正特别的地方在于**把这些设计组合起来的规模**：314B 总参 + 86B 激活 + 64 层 + 8 专家。这种规模在 2024 年 3 月的开源界是独一份的。

到这里 `model.py` 的内容就讲完了。理解了这一章之后，你应当能完整回答开篇提到的几类问题中"数据流走过哪些张量操作"以及"为什么这样设计"两类。下一章看 checkpoint.py，把这 770 个 tensor 是如何从磁盘加载到 GPU 显存的过程讲清楚 - 这是把"读懂的模型"变成"跑起来的模型"必经的一道工序。

## 延伸阅读

- [Cohere Command R Technical Report](https://docs.cohere.com/docs/command-r) - sandwich norm 的另一个工业实例
- [Improving Transformer Optimization Through Better Initialization](https://proceedings.mlr.press/v119/huang20f.html) - 多种 norm 布局对训练稳定性的讨论
- [DeepNet: Scaling Transformers to 1000 Layers](https://arxiv.org/abs/2203.00555) - 极深网络下的 norm 策略
- [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) - tied embedding 的早期工作
