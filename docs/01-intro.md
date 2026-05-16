# 第 1 章 引言

## 1.1 2024 年 3 月：MoE 的开源黄金窗口

回看 2024 年第一季度，开源大模型的格局正好处于一个转折点。

- **稠密派**：Meta 的 LLaMA-2 在 2023 年 7 月发布，最大 70B，是工业界用得最多的开源 dense 模型；2 月 21 日，Google 又开源了 Gemma 7B / 2B，依然走稠密路线
- **MoE 派的开端**：2023 年 12 月 8 日，Mistral AI 通过一条没有任何说明文字的磁力链放出 Mixtral 8x7B（47B 总参，13B 激活），用 MoE 把 Llama-2 70B 在多项基准上拉下马
- **闭源的怪兽**：GPT-4、Claude 3、Gemini 1.5 都已经被广泛猜测是 MoE，但只是猜测

就在这个时点，xAI 在 2024 年 3 月 17 日凌晨用一条与 Mixtral 极像的磁力链放出了 Grok-1。

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e
```

随附 `xai-org/grok-1` 这个 GitHub 仓库，约 9GB 代码，没有训练脚本、没有数据处理代码、没有任何 README 之外的文档，只有：

- `model.py` - 1398 行 JAX/Haiku 模型实现
- `runners.py` - 605 行推理 runner
- `checkpoint.py` - 221 行 ckpt 加载
- `run.py` - 72 行入口示例
- `tokenizer.model` - 2.2MB SentencePiece 模型

Grok-1 的总参 314B、激活参 ~86B，远超过同期所有开源模型。这个尺寸到今天（2026 年）依然是开源 MoE 里的头部 - 后来 Mixtral 8x22B (141B / 39B)、DeepSeek-V2 (236B / 21B)、DeepSeek-V3 (671B / 37B)、Qwen2.5-MoE 系列陆续出现，但 Grok-1 仍然是单 MoE 块设计里的最重者。

把这两条新闻放在 2024 年 3 月的语境里看，就能感受到 Grok-1 开源时的"重量"：

- 当时全世界能拿出 8x80GB GPU 服务器跑推理的研究团队不超过两千个
- 当时全世界能拿出 64x80GB GPU 集群做 SFT 的团队不超过一百个
- 当时全世界完整训练过 100B+ MoE 的团队不超过十个

xAI 把训练成果一次性放出来，让前两类团队拿到一个"现成的研究对象"。但因为没有给出训练代码、数据、配方，第三类团队（潜在的下一代模型开发者）拿到的只是结论而不是过程。这种"半开半闭"的开源策略，是理解 Grok-1 商业意图的关键。本书后面（特别是第 11 章）会反复回到这点。

## 1.2 314B = 8 × ? + 共享：把账先算个粗略

`run.py:25-49` 给出的实参：

```python
# run.py:25-49
grok_1_model = LanguageModelConfig(
    vocab_size=128 * 1024,                  # L26
    pad_token=0,                            # L27
    eos_token=2,                            # L28
    sequence_len=8192,                      # L29
    embedding_init_scale=1.0,               # L30
    output_multiplier_scale=0.5773502691896257,   # L31 = 1/sqrt(3)
    embedding_multiplier_scale=78.38367176906169, # L32 = sqrt(emb_size)? 见第 3 章
    model=TransformerConfig(
        emb_size=48 * 128,                  # L34 = 6144
        widening_factor=8,                  # L35
        key_size=128,                       # L36
        num_q_heads=48,                     # L37
        num_kv_heads=8,                     # L38
        num_layers=64,                      # L39
        attn_output_multiplier=0.08838834764831845,  # L40 ~= 1/sqrt(128)
        shard_activations=True,             # L41
        num_experts=8,                      # L43
        num_selected_experts=2,             # L44
        data_axis="data",                   # L46
        model_axis="model",                 # L47
    ),
)
```

简化记为：

- $d = 6144$（hidden）
- $L = 64$ 层
- $E = 8$ 专家、$k = 2$ 激活
- FFN widening factor $w = 8$，对 SwiGLU 实际中间维度由 `ffn_size(d, w)` 计算（见第 3 章）
- $H_q = 48$ Q 头、$H_{kv} = 8$ KV 头、$d_h = 128$

每层参数主要分两块：

**Attention（GQA）：**

$$
P_{\text{attn}} = d \cdot (H_q d_h) + 2 \cdot d \cdot (H_{kv} d_h) + (H_q d_h) \cdot d
$$

代入：$6144 \cdot 6144 + 2 \cdot 6144 \cdot 1024 + 6144 \cdot 6144 \approx 88\,\text{M}$

**FFN（8 个 SwiGLU 专家）：**

`ffn_size(6144, 8) = int(8 \cdot 6144) \cdot 2 / 3 = 32768`，再向上取 8 的倍数仍是 32768。每个专家有三个矩阵 `linear_v`、`linear`、`linear_1`，单专家 $\approx 3 \cdot 6144 \cdot 32768 \approx 0.6\,\text{B}$，8 个专家约 $4.83\,\text{B}$，乘 64 层得 $\approx 309\,\text{B}$ - 这是 314B 的主体。

**Embedding：** $131072 \cdot 6144 = 0.81\,\text{B}$，输入输出权重共享。

**激活参数（per token）：** 每个 token 只走 attention + 2 个专家：

$$
P_{\text{active}} \approx L \cdot (P_{\text{attn}} + 2 \cdot P_{\text{expert}}) + P_{\text{emb}} \approx 64 \cdot (0.088 + 2 \cdot 0.604) + 0.81 \approx 84\,\text{B}
$$

与官方宣称的"约 86B 激活"吻合。

详细每个矩阵的 shape、partition 在第 3 章一项一项列。

这个粗略账目的关键，是要把"总参 314B"和"激活参 86B"在脑子里分清楚：

- **总参 314B** 决定了**显存需求** - 不管哪条 token 路径，所有参数都得在 HBM 里
- **激活参 86B** 决定了**计算需求** - 每个 token 实际做 2 × 86B 次浮点运算量级的工作

这两个量是**完全独立的**。如果你的目标是"低成本部署"，你关心激活参；如果你的目标是"硬件门槛"，你关心总参。MoE 的精髓就是把这两件事解耦 - 用同样的算力可以装更大的模型。

但 Grok-1 把"显存"这一面榨到了极限：314B 即便 8-bit 量化也要 314GB 显存，单机 8 卡 H100 80GB 总共 640GB，留给 KV cache 和激活的余地只有 100GB 左右。这是 Grok-1 推理硬件门槛 8 卡起跳的根本原因 - 与激活参的大小关系不大，与总参直接相关。

第 7 章会算清楚显存账。

### 1.2.1 这些数字怎么读

如果你之前没接触过 100B+ 模型，几个直观对比可以建立感觉：

- **314B 总参 vs 人类大脑突触数**：人脑约 100 万亿（10^14）个突触，314B = 3.14 × 10^11 - 是人脑的 0.3%。当然神经网络的参数和生物突触不是一回事，但这个数量级对比让人意识到 "314B 不是大到不可想象"
- **314B 总参 vs GPT-3**：GPT-3 是 175B dense，激活参 = 总参 = 175B。Grok-1 总参是 GPT-3 的 1.8 倍，但激活参 86B 只是 GPT-3 的一半
- **86B 激活 vs LLaMA 系列**：LLaMA-2 70B 激活参 70B，LLaMA-3 70B 同样 70B。Grok-1 激活参 86B 略大但同级
- **86B 激活的算力含义**：按"每 token 2P 次浮点运算"近似，Grok-1 每 token 需要约 170 GFLOPs - 一张 H100 fp16 算力 ~989 TFLOPs，理论上每秒能处理 5800 token

这个 5800 token/s 是上界，实际由于 HBM 带宽限制（每 token 要读取 86GB×2/8GPU = 21.5 GB），单卡 H100 实际能跑约 100 token/s（受带宽限制）；8 卡协同后受通信限制和 op launch 开销，社区报告 5-10 token/s decode。

数字间的落差告诉你：**Grok-1 的推理瓶颈不是计算，是内存带宽和通信**。第 8 章会更详细展开。

## 1.3 只放 base，不放 SFT 与 RLHF：这点要展开

`README.md` 强调：

> Grok-1 is the raw base model checkpoint from the Grok-1 pre-training phase, which concluded in October 2023.

xAI 没有放：

1. **指令微调（SFT）权重** - Grok-1 base 直接对话效果差，prompt 工程也救不回来
2. **RLHF/DPO** 后的版本
3. **训练代码 / 数据 / 词表统计**
4. **Tokenizer 训练语料**（只有训好的 `tokenizer.model`）

这意味着什么？

- **对研究者**：你拿到的是 314B base，可读、可分析、可观测，但不能直接用作产品
- **对工程师**：要做 SFT，需要至少 16 × H100（量化后）的硬件 + 高质量数据集 + 几周训练时间。这是直接复刻 LLaMA-Chat 的成本数量级，社区基本承担不起
- **对 xAI 本身**：保留了 instruction-tuning 这一商业关键能力 - 当时 grok.com 上跑的 Grok 是已经 SFT/RLHF 过的版本，开源 base 不会直接威胁产品

事实证明社区的反应也印证了这一判断 - 截至 2026 年，HuggingFace 上 Grok-1 base 的下载量远少于同尺寸的 Mixtral 8x22B-Instruct，几乎找不到一个被广泛使用的 Grok-1 SFT fork。这是第 11 章会展开讨论的话题。

这里再多说一点关于"base 模型"的工程意义。Base 模型是预训练阶段的产物，目标是**最大化对训练分布的 likelihood**。这个目标和"听人指令、礼貌对话、拒绝越界"是**正交的** - base 模型见到 "Hello, how are you?" 大概率会接一段 Reddit 评论而不是回答你。让 base 变成 chatbot，需要 SFT（用指令-回复对训练）+ DPO/PPO/RLHF（按人类偏好微调）。

xAI 开源 base 而不开源后续步骤，等价于把"原料"给你，但"配方和工艺"自己留着。这是商业意义上的最优策略 - 既能博得"开源"美名，又不影响产品差异化。OpenAI、Anthropic、Google 从不公开任何 base，xAI 至少公开了 base，已经"走得比闭源派远"。但相比 Meta 公开 LLaMA-Chat、Mistral 公开 Mixtral-Instruct，xAI 的开源又"走得没那么远"。

本书的态度是：把这件事讲清楚，不下道德判断。开源策略是商业决定，没有"应该不应该"。

## 1.4 仓库结构概览

```
grok-1/
├── README.md / README_zh.md       # 入门说明
├── LICENSE.txt                    # Apache 2.0
├── pyproject.toml                 # 只配置了 ruff
├── requirements.txt               # 4 行依赖
├── run.py                         # 72 行示例入口
├── checkpoint.py                  # 221 行权重加载
├── runners.py                     # 605 行推理引擎
├── model.py                       # 1398 行模型实现
├── tokenizer.model                # 2.2MB SentencePiece
└── checkpoints/                   # 用户自行下载 ckpt-0/* 放这里
```

依赖只有四个：

```
dm_haiku==0.0.12
jax[cuda12-pip]==0.4.25
numpy==1.26.4
sentencepiece==0.2.0
```

注意几件事：

- 没有 PyTorch、没有 TensorFlow、没有 transformers - 纯 JAX 栈
- `dm_haiku` 是 DeepMind 早期主推的 JAX 神经网络库，2023 年开始 DeepMind 自己也在迁向 `flax.nnx`，但 xAI 选择继续用 Haiku
- JAX 锁到 0.4.25，对 jaxlib/cuda12 的版本绑死，迁移到新 JAX 会有兼容性问题（实测 0.4.30+ 已无法直接跑）
- 整个项目没有 test、没有 mypy、没有 CI 配置 - 这是一份"研究级开源"，不是产品级代码

### 1.4.1 几条依赖背后的"小故事"

值得再花一段说说这份 `requirements.txt`：

```
dm_haiku==0.0.12
jax[cuda12-pip]==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
numpy==1.26.4
sentencepiece==0.2.0
```

短短四行，每一行都不是随意写的。`dm_haiku 0.0.12` 是 2023 年 11 月发布的版本，紧贴 Grok-1 预训练结束（2023 年 10 月）的时间点。`jax 0.4.25` 在 2024 年 2 月发布，正好赶在 3 月开源前。`numpy 1.26.4` 是 2024 年 2 月发布的最后一个 1.x 系列版本 - 之后 2.x 出现，对部分 dtype 行为做了不兼容修改。`sentencepiece 0.2.0` 是 2024 年 2 月发布的稳定版。

这意味着 xAI 在开源前做了一次"依赖冻结" - 把那一刻所有能用的最新稳定版组合起来，写死版本号。这是工业级开源的标准做法，但 2 年后想跑就难了：

- `jax 0.4.25` 不兼容 CUDA 12.4+（NVIDIA 后续 cuDNN 升级，需要 jax 0.4.30+）
- `dm_haiku 0.0.12` 在 jax 0.4.30+ 下有若干 deprecation warning
- `numpy 1.26.4` 与 jax 0.4.30+ 配合 OK，但与 numpy 2.x 之后某些库不兼容

读到这里的人如果真要跑 Grok-1，建议用 Docker 锁定一个 2024 年 3 月前后的 CUDA 12.2 镜像，再 pip install 这四行。

## 1.5 后续 Grok 系列走向闭源

| 版本 | 发布日期 | 是否开源 |
| --- | --- | --- |
| Grok-1 | 2024-03 | 是（base 权重 + 代码） |
| Grok-1.5 | 2024-04 | 否 |
| Grok-1.5V | 2024-04 | 否 |
| Grok-2 | 2024-08 | 否 |
| Grok-2 mini | 2024-08 | 否 |
| Grok-3 | 2025-02 | 否 |

也就是说，xAI 后续没有继续走开源路线。Grok-1 作为 xAI 唯一公开的模型，技术细节有"博物馆价值"。这本书写作的根本动机也在于此：等到 Grok 系列再开源（如果有的话），架构很可能已经面目全非（Grok-3 的 reasoning 模式据传是基于 RL self-play 的全新栈），现在留下的细致拆解会成为这一支 lineage 的最详尽记录。

### 1.5.1 一个对照案例：LLaMA 系列

把 xAI 和 Meta 的开源策略对照一下：

| 维度 | Meta LLaMA | xAI Grok |
| --- | --- | --- |
| 持续开源 | LLaMA、LLaMA-2、LLaMA-3、LLaMA-3.1、LLaMA-3.2、LLaMA-3.3、LLaMA-4 全部开源权重 | 只有 Grok-1，后续都闭源 |
| 同时多尺寸 | 每代都放 8B、70B、有时还放 405B | 只放 314B 一个尺寸 |
| 开源 SFT 版本 | LLaMA-2-Chat / LLaMA-3-Instruct 系列同步发布 | 只有 base，无 chat |
| 配套工具链 | 完整的 finetune scripts、HF transformers 原生支持、官方推理优化 | 只有最小推理示例 |
| 商业许可 | 自定义许可（>7 亿月活需申请） | Apache 2.0（完全开放） |
| 实际生态影响 | 数千个 fork、几十万次下载、衍生模型数百个 | 几乎没有衍生生态 |

数字本身就在讲故事：**Apache 2.0 比 LLaMA 协议更开放，但 Grok-1 的生态影响远小于 LLaMA**。开源许可只是一个法律外壳，真正决定生态的是**易用性**：模型尺寸、配套工具、文档质量、社区参与度。Grok-1 在这些维度上都比 LLaMA 弱很多。

这也回答了"什么是真开源"这个看似简单的问题。代码可见、权重可下、协议宽松，这些都是必要条件，但不充分。**真正的开源还需要"可用性"** - 让普通研究者能在合理硬件上跑、能扩展、能改造。Grok-1 拿到了前几个条件，没拿到最后这个。

## 1.6 读源码前需要的 JAX/Haiku 基础

Grok-1 代码对 JAX 的依赖集中在几个 API。如果你不熟，下面这段简介足以读懂全书。

### 1.6.1 函数式与 transform

JAX 模型是**无状态函数**。Haiku 用 `hk.transform` 把一个带 `hk.Module` 的函数变成 `(init_fn, apply_fn)` 两件套：

```python
def forward(x):
    return MyModel()(x)

forward_t = hk.transform(forward)
params = forward_t.init(rng, x_example)   # 拿参数
y = forward_t.apply(params, rng, x)        # 跑前向
```

Grok-1 在 `runners.py:395-398` 显式用了 `hk.without_apply_rng(hk.transform(...))`，因为推理时 RNG 单独传入 sampler，模型自身不需要 dropout 之类的随机性。

### 1.6.2 `pjit` 与 mesh

JAX 的张量并行通过 `jax.experimental.pjit.pjit` 完成：

```python
mesh = jax.sharding.Mesh(devices, ("data", "model"))
fn_p = pjit.pjit(fn, in_shardings=P("data", "model"), out_shardings=P("data"))
with mesh:
    out = fn_p(x)
```

`PartitionSpec("data", "model")` 意为"第 0 维沿 data 轴切、第 1 维沿 model 轴切"。Grok-1 使用 (1, 8) 的本地 mesh - 1 个 data shard、8 个 model shard，正好对应一台 8 卡 H100/A100 节点。

### 1.6.3 `shard_map`

比 `pjit` 更底层。`shard_map` 让你写"每个 shard 看到的局部 tensor"的 SPMD 程序：

```python
@partial(shard_map, mesh=mesh, in_specs=P("model", None), out_specs=P("model", None))
def f(local_x):
    return local_x * 2
```

Grok-1 在 MoE 层用 `shard_map`（`model.py:319-357`）做 expert 维度的切分，是全书最难读的一段。

### 1.6.4 `with_sharding_constraint`

中间张量的强制 partition：

```python
x = with_sharding_constraint(x, P("data", None, "model"))
```

模型代码里到处都是这种约束，可以理解为"提示 XLA 编译器在这一步把张量布成这种形状"。

### 1.6.5 几个语义陷阱

JAX 与 PyTorch 在心智模型上有几处常踩坑：

**陷阱 1：函数式更新。** JAX 数组是 immutable，所有"更新"操作返回新数组。`x[0] = 1` 这种写法在 JAX 里不能直接用，需要 `x = x.at[0].set(1)`。Grok-1 的 KV cache 更新就大量用 `jax.lax.dynamic_update_slice_in_dim` 这种"函数式更新"原语。

**陷阱 2：JIT 边界。** 在 `jit` 编译过的函数里，Python 控制流（if/for）会在 trace 阶段固化。所以 `if x > 0: ...` 这种依赖 runtime 值的判断不能写在 jit 里，需要换成 `jnp.where(x > 0, ..., ...)`。Grok-1 的 `hk_forward` 用 `jnp.where` 替换 if 就是这个原因。

**陷阱 3：PRNG 显式传参。** JAX 没有"全局随机种子"概念，每次需要随机数都要传一个 PRNGKey。`jax.random.split(rng)` 让你从一个 key 派生出多个独立 key。Grok-1 的 `hk_sample_step` 在每一步都做 `rngs, rngs_ = jax.vmap(jax.random.split, out_axes=1)(rngs)` - 沿 batch 维 vmap 后产出新 rng。

**陷阱 4：dtype 严格性。** JAX 对 dtype 转换比 PyTorch 严格。如果你把 fp32 张量传给一个期待 bf16 的函数，会得到错误而不是自动 cast。Grok-1 的 `cast_bfloat16`（`model.py:78-82`）是一个显式 cast helper - 这种 helper 在 PyTorch 里很少见。

到这里 JAX 基础就够了。下一章我们用一张图把 Grok-1 的数据流串起来。

## 延伸阅读

- xAI 官方博客 [Open Release of Grok-1](https://x.ai/blog/grok-os)
- Mistral [Mixtral of Experts (8x7B) Paper](https://arxiv.org/abs/2401.04088) - 同代 MoE 对照
- DeepMind [Haiku Documentation](https://dm-haiku.readthedocs.io/) - 0.0.12 版本的 API 没有大变化
- JAX [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) - `pjit` 与 mesh 的官方文档
