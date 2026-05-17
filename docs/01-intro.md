# 第 1 章 引言

## 1.0 这本书为谁而写

在进入 Grok-1 的源码之前，我想先把"读者画像"摆出来。这本书假设你是一个**普通的机器学习工程师**：你训练过 Transformer，调过 LLaMA 系列的微调，能看懂注意力机制的公式推导，但你不一定亲手写过一份 314B 规模模型的并行推理代码，也不一定熟悉 JAX 与 Haiku 这套 DeepMind 早年主推的函数式神经网络栈。这本书想做的，是把 Grok-1 这份 9GB 大小、四个核心文件、两千多行 Python 代码的开源实现，逐节逐行地拆开讲清楚，让你读完之后能够独立回答以下几类问题：

- 当我向 Grok-1 输入一段 prompt 时，从 token id 到最终的概率分布，数据具体走过了哪些张量操作？每一步张量的形状和分片方式是什么？
- 314B 总参数、86B 激活参数、64 层、8 专家每次激活 2 个 - 这些数字背后的工程取舍是什么？为什么不是 4 专家激活 1 个、也不是 16 专家激活 4 个？
- Grok-1 与同代的 Mixtral 8x7B、稍晚的 DeepSeek-V2、Qwen2.5-MoE 比较，架构差异在哪里？哪些设计是 xAI 的独创、哪些是抄了前人的路？
- 一段 JAX/Haiku 写的模型代码与 PyTorch 实现有什么关键不同？为什么 xAI 选择 JAX 而不是当时社区主流的 PyTorch？

如果你完全没接触过 JAX、或者从未读过 100B 量级模型的源码，本章和第 2 章会把必要的背景补齐；如果你已经熟悉 Mixtral / DeepSeek-V2 的实现思路，可以从第 4 章开始直接看核心。但无论从哪开始读，每一章都尽量做到自包含 - 出现的新概念会就地解释，引用的代码片段会标到具体行号，公式会逐步推导而不是直接给结论。

## 1.1 2024 年 3 月：MoE 的开源黄金窗口

回看 2024 年第一季度，开源大语言模型的格局正好处于一个明显的转折点。要理解 Grok-1 在这个时点放出来意味着什么，得先把当时的生态分成三派来看。

- **稠密派**：Meta 的 LLaMA-2 在 2023 年 7 月发布，提供 7B、13B、70B 三个尺寸；其中 70B 在 2023 年下半年到 2024 年初是工业界用得最多的开源稠密模型，HuggingFace 上衍生的微调版本上千个。2024 年 2 月 21 日，Google 又开源了 Gemma 7B / 2B 两个尺寸，依然走稠密路线，没有跟随 MoE 潮流。这意味着 2024 年初最有影响力的开源稠密线仍由 Meta 和 Google 把持，参数规模上限定格在 70B 一档。

!!! note "稠密模型（dense model）"
    "稠密"是相对 MoE 的另一极：每一层的前馈网络（FFN，Feed-Forward Network）只有一份权重，所有 token 都走同一组参数，没有"路由"也没有"专家"的概念。**激活参数等于总参数** - 每个 token 都会把整张网络从头到尾跑一遍。LLaMA、Gemma、Mistral 7B 这些都是稠密模型。

    在同等总参规模下，稠密模型的推理算力消耗远比 MoE 大（因为每个 token 都得用上全部参数），但工程上更简单：不用考虑专家负载均衡、不用专门切 expert 维度的并行，社区的训练/微调/推理工具链也更成熟。Grok-1 如果做成稠密就是 314B 全部激活，推理算力会涨到现在的约 3.6 倍 - 这就是 MoE 之所以重要的根源。

- **MoE 派的开端**：2023 年 12 月 8 日，法国创业公司 Mistral AI 在 X（推特）上贴出一条磁力链（magnet link），没有附 README、没有博客文章、没有任何说明文字，社区下载完才发现里面是 **Mixtral 8x7B** - 47B 总参、13B 激活、8 个专家每次激活 2 个的混合专家（Mixture of Experts，MoE）架构。几天之内陆续跑出的基准测试让人意外：Mixtral 在 MMLU（大规模多任务理解）、HellaSwag（常识推理）、ARC（科学推理）、GSM8K（小学数学题）这些公认的 base 模型评测上，几乎全面胜过 Llama-2 70B，激活参却只有它的不到五分之一。

    这是 MoE 第一次在公开模型里把"同等激活参数下更聪明、同等总参数下更省算力"这件事正面证明出来。它也给了 xAI 三个月后用同款"磁力链 + 沉默"方式放 Grok-1 一个直接的先例 - 既能体现"我们就把权重直接放出来给你"的硬核姿态，又不用承担任何配套文档或售后支持的责任。

!!! note "MoE（Mixture of Experts，混合专家）"
    MoE 是把稠密 FFN 换成"N 个并行的小 FFN（专家）+ 一个路由器（router 或 gate）"。每个 token 进来，router 先打分挑出 top-k 个专家，token 只走这 k 个专家的子网络，其他专家这一步不参与计算。

    最关键的数字有两个：
    
    - **总参 (total parameters)**：所有专家加起来的参数量。决定**显存占用** - 推理时所有专家权重都得驻留在显存里，因为 router 可能任意挑选。
    - **激活参 (activated parameters)**：单个 token 实际经过的参数量 = 共享部分 + k 个专家。决定**计算量** - 每生成一个 token 大约要做 2 × 激活参数次浮点运算。
    
    Grok-1 是 8 专家、每次激活 2 个，所以激活参 ≈ 共享部分 + 2/8 × FFN 部分 ≈ 86B，而总参是 314B。这是 MoE 把"显存"和"算力"两件事解耦的核心机制。

- **闭源前沿模型**：到 2024 年第一季度，OpenAI 的 GPT-4、Anthropic 的 Claude 3、Google 的 Gemini 1.5 都已经发布。业界根据它们的推理延迟、token 价格、长上下文能力等间接信号广泛猜测它们都采用了 MoE 架构，但三家厂商都没有正面承认任何架构细节。也就是说，MoE 在闭源前沿模型里的地位**只能从外部推断**，开源侧需要自己拿出一份完整的、能跑、能调、能改造的 MoE 实现，才能让这些猜测落到代码层面被验证。

就在这个"开源稠密强、闭源 MoE 强、开源 MoE 弱"的三角格局下，xAI 在 2024 年 3 月 17 日凌晨用一条与 Mixtral 极像的磁力链放出了 Grok-1：

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e
```

随附 `xai-org/grok-1` 这个 GitHub 仓库，约 9GB 代码，没有训练脚本、没有数据处理代码、没有任何 README 之外的文档，只有几个核心 Python 文件：

- `model.py` - 1398 行 JAX/Haiku 模型实现
- `runners.py` - 605 行推理 runner（运行器）
- `checkpoint.py` - 221 行权重加载逻辑
- `run.py` - 72 行入口示例脚本
- `tokenizer.model` - 2.2MB 的 SentencePiece 分词器模型文件

Grok-1 的总参 314B、激活参约 86B，远超过同期所有开源模型。这个尺寸到今天（2026 年）依然是开源 MoE 里的头部 - 后来 Mixtral 8x22B (141B / 39B)、DeepSeek-V2 (236B / 21B)、DeepSeek-V3 (671B / 37B)、Qwen2.5-MoE 系列陆续出现，但 Grok-1 仍然是**单 MoE 块设计里的最重者**（DeepSeek-V3 虽然总参更大但用了更细粒度的"细颗粒专家"+"共享专家"两类，架构不同）。

把这两条新闻（Mixtral 磁力链、Grok-1 磁力链）放在 2024 年 3 月的语境里看，就能感受到 Grok-1 开源时的"重量"：

- 当时全世界能拿出 8 张 80GB 显存的 GPU、组成一台单机服务器、把这种模型真正跑起来做推理的研究团队，估算不超过两千个。这是"读懂 Grok-1 推理代码、能在自己机器上验证"的硬件门槛。
- 当时全世界能拿出 64 张 80GB GPU 组成集群、对这种规模的 base 模型做指令微调（SFT，Supervised Fine-Tuning）的团队，估算不超过一百个。这是"把 Grok-1 改造成可用对话模型"的工程门槛。
- 当时全世界完整训练过 100B 参数以上 MoE 模型的团队，公开可查的估算不超过十个。这是"复现 Grok-1 训练过程、做下一代模型"的科研门槛。

xAI 把训练成果一次性放出来，让前两类团队拿到一个"现成的研究对象"。但因为没有给出训练代码、数据、配方，第三类团队（潜在的下一代模型开发者）拿到的只是结论而不是过程。这种"半开半闭"的开源策略，是理解 Grok-1 商业意图的关键。本书后面（特别是第 11 章）会反复回到这一点。

## 1.2 314B = 8 × ? + 共享：把账先算个粗略

`run.py:25-49` 给出的所有模型实参一次性列在一个 dataclass 配置里。先把它原样贴出来：

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

为了后面写公式方便，把这些字段重新命名成数学符号：

- $d = 6144$ - hidden 维度（隐藏层维度）
- $L = 64$ - Transformer 块的层数
- $E = 8$ - 专家总数
- $k = 2$ - 每个 token 激活的专家数
- $w = 8$ - FFN 的 widening factor（拓宽倍数），SwiGLU 实际中间维度由 `ffn_size(d, w)` 函数计算（第 3 章会逐行讲）
- $H_q = 48$ - 查询头（Query head）数
- $H_{kv} = 8$ - 键值头（Key/Value head）数
- $d_h = 128$ - 每个头的维度

!!! note "GQA（Grouped Query Attention，分组查询注意力）"
    Grok-1 的 Q 头有 48 个、KV 头只有 8 个。如果像传统多头注意力（MHA，Multi-Head Attention）那样 Q/K/V 都 48 头，KV cache 会非常大（cache 大小 ∝ 头数 × 维度 × 序列长度）。

    GQA 让多个 Q 头**共享**同一组 KV 头：48 / 8 = 6，意味着每 6 个 Q 头共享 1 组 KV。这样 KV cache 直接缩到原来的 1/6，推理时显存占用和 HBM（高带宽显存）读取压力都显著降低。代价是模型表达力稍微下降一点，但实测影响很小。
    
    LLaMA-2 70B 也是这种"48Q / 8KV"的搭配，Mistral / Mixtral / Qwen 等模型也几乎都用 GQA。属于 2023 年后大模型的事实标准。

!!! note "SwiGLU（FFN 的激活函数变体）"
    标准 FFN 是 `down(activation(up(x)))` - 一个升维线性层 + 激活函数（早期 ReLU，后来 GELU）+ 一个降维线性层。**GLU**（Gated Linear Unit）系列把它改成两分支结构：`down((gate_proj(x) * activation(value_proj(x))))` - 一个分支算 value、另一个分支算 gate（值在 0-1 附近），两路按位相乘再 down-project。SwiGLU 就是激活函数选 SiLU（也叫 Swish，公式 $x \cdot \sigma(x)$）的 GLU 变体。

    代价是 FFN 多了一组 up 矩阵，参数从 2 块变 3 块，所以业内的惯例是把中间维度 $d_{ffn}$ 乘 $2/3$ 来对齐总参数量。Grok-1 的 `widening_factor=8`，经过 `ffn_size` 函数里的 `* 2 // 3` 就得到 $d_{ffn} = 8 \cdot 6144 \cdot 2 / 3 = 32768$。每个专家的三个矩阵（`linear`、`linear_v`、`linear_1`）都设了 `with_bias=False`（不带偏置项），这也是现代大模型的常见做法 - 减少参数同时降低数值不稳定。

每层参数主要分两块：注意力（attention）和前馈（FFN）。先把每块的公式写出来，再代入数字。

**Attention（GQA）每层参数：**

注意力层有四个线性投影：Q、K、V 三组输入投影 + 一组输出投影。注意 K、V 因为 GQA 只有 $H_{kv}$ 组：

$$
P_{\text{attn}} = \underbrace{d \cdot (H_q d_h)}_{\text{Q 投影}} + \underbrace{2 \cdot d \cdot (H_{kv} d_h)}_{\text{K 和 V 投影}} + \underbrace{(H_q d_h) \cdot d}_{\text{输出投影}}
$$

代入数字：

$$
P_{\text{attn}} = 6144 \cdot 6144 + 2 \cdot 6144 \cdot 1024 + 6144 \cdot 6144 \approx 88\,\text{M}
$$

这里 $H_q d_h = 48 \cdot 128 = 6144$ 恰好等于 $d$，$H_{kv} d_h = 8 \cdot 128 = 1024$。整个注意力层每层约 0.088B 参数。

**FFN（8 个 SwiGLU 专家）每层参数：**

每个专家是一个 SwiGLU FFN，三个矩阵都是 $d \times d_{ffn}$：

$$
P_{\text{expert}} = 3 \cdot d \cdot d_{ffn} = 3 \cdot 6144 \cdot 32768 \approx 0.604\,\text{B}
$$

8 个专家：

$$
P_{\text{FFN per layer}} = 8 \cdot 0.604\,\text{B} \approx 4.83\,\text{B}
$$

乘 64 层：

$$
P_{\text{FFN total}} = 64 \cdot 4.83\,\text{B} \approx 309\,\text{B}
$$

这是 314B 总参的主体 - 可以看到，绝大部分参数都在专家里，注意力部分相对很小。

**Embedding：** 输入 token 嵌入矩阵是 $V \times d = 131072 \cdot 6144 \approx 0.81\,\text{B}$，输入和输出共享同一份权重（也叫 tied embedding），所以不重复计入。

**激活参数（per token）：** 每个 token 走完整的 attention，但 FFN 部分只走 2 个专家（而不是全部 8 个）：

$$
P_{\text{active}} \approx L \cdot (P_{\text{attn}} + k \cdot P_{\text{expert}}) + P_{\text{emb}}
$$

代入：

$$
P_{\text{active}} \approx 64 \cdot (0.088 + 2 \cdot 0.604) + 0.81 \approx 84\,\text{B}
$$

与官方宣称的"约 86B 激活参"基本吻合（差异来自 RMSNorm、router 等小项）。

详细到每个矩阵的 shape、partition spec 在第 3 章会一项一项列出来。

这个粗略账目的关键，是要把"总参 314B"和"激活参 86B"在脑子里分得清清楚楚：

- **总参 314B** 决定了**显存需求** - 不管哪条 token 路径，所有参数都得在 HBM（High Bandwidth Memory，高带宽显存）里待命，因为 router 任何时候都可能挑到任意专家。
- **激活参 86B** 决定了**计算需求** - 每个 token 实际做的浮点运算量级约为 $2 \cdot 86\text{B} = 172\,\text{GFLOPs}$（一个 FLOPs 是一次浮点乘加）。

这两个量是**完全独立的两个维度**。如果你的目标是"低成本部署"，你关心的是激活参（它决定算力）；如果你的目标是"硬件门槛"，你关心的是总参（它决定显存）。MoE 的精髓就是把这两件事解耦 - 用同样的算力可以装一个总参大得多的模型。

但 Grok-1 把"显存"这一面压到了极限：314B 参数即便经过 8-bit 量化也仍然需要约 314GB 显存来存放权重本身；单机 8 卡 H100 80GB 总显存是 640GB，扣掉权重后留给 KV cache 和中间激活的余地只剩 100GB 左右。这是 Grok-1 推理硬件门槛之所以**从 8 卡起跳**的根本原因 - 限制来自总参 314B，而不是激活参 86B；想用更少的卡跑起来，权重直接放不下。

!!! note "8-bit 量化（INT8 quantization）"
    训练时模型参数通常是 fp32（每个权重 4 字节）或 bf16（2 字节）。推理阶段的精度其实用不到那么高 - 把每个权重压到 int8（1 字节）甚至 int4（半字节），模型质量损失通常很小但显存占用直接减半甚至减到四分之一。常见做法是把一组 weight（比如一整行）记录一个 fp16 的 scale 系数 + 一组 int8 数值，计算时动态 dequantize 回浮点。

    Grok-1 的开源 ckpt 已经是 int8 量化的（详见第 7 章），单个 shard 文件里就标了 `WEIGHT_FP8` 之类的 dtype。所以"314B 参数"实际硬盘占用约 314GB。即便如此，8 卡 H100 也只剩 100GB 左右给 KV cache 和激活，推理时硬件资源压得很紧。

第 7 章会算清楚完整的显存账：权重 + KV cache + 中间激活 + 通信缓冲，最终落到每张卡多少 GB。

### 1.2.1 这些数字怎么读

如果你之前没接触过 100B+ 量级的模型，几个直观对比可以帮你建立感觉：

- **314B 总参 vs 人类大脑突触数**：人脑约 100 万亿（$10^{14}$）个突触，314B = $3.14 \times 10^{11}$ - 是人脑的 0.3%。当然神经网络的参数和生物突触不是同一回事，但这个数量级对比让人意识到 "314B 不是大到不可想象"。
- **314B 总参 vs GPT-3**：GPT-3 是 175B 稠密模型，激活参 = 总参 = 175B。Grok-1 总参是 GPT-3 的 1.8 倍，但激活参 86B 只是 GPT-3 的一半。
- **86B 激活 vs LLaMA 系列**：LLaMA-2 70B 激活参 70B，LLaMA-3 70B 同样 70B。Grok-1 激活参 86B 略大但同级。
- **86B 激活的算力含义**：按"每 token 约 2 × 激活参数次浮点运算"近似，Grok-1 每 token 需要约 170 GFLOPs。一张 H100 在 fp16 精度下算力约 989 TFLOPs，理论上每秒能处理 $989\text{T}/170\text{G} \approx 5800$ token。

这个 5800 token/s 是上界（理论峰值），实际由于 HBM 带宽限制 - 每生成一个 token 都要把那部分参数从 HBM 搬到 SM（Streaming Multiprocessor，流多处理器）寄存器，单 token 需要搬运约 $86\text{GB} \times 2/8\text{GPU} = 21.5\,\text{GB}$ - 单卡 H100 实际能跑约 100 token/s（被带宽卡死）。8 卡协同后受 NVLink 通信带宽和 op launch 开销（每个算子启动的固定成本）限制，社区报告 Grok-1 decode 阶段实际能跑到 5-10 token/s。

数字间的落差告诉你一个非常重要的事实：**Grok-1 的推理瓶颈不是计算，是内存带宽和通信**。这是几乎所有大模型推理的共性问题，第 8 章会更详细展开如何用 KV cache、speculative decoding、量化等手段缓解这个瓶颈。

## 1.3 只放 base，不放 SFT 与 RLHF：这点要展开

`README.md` 强调：

> Grok-1 is the raw base model checkpoint from the Grok-1 pre-training phase, which concluded in October 2023.

xAI 没有放出来的东西很重要，需要列清楚：

1. **指令微调（SFT，Supervised Fine-Tuning）权重** - Grok-1 base 在直接对话场景下的输出质量较差，即便配合精心设计的 prompt 工程也难以补救
2. **RLHF（基于人类反馈的强化学习）或 DPO（直接偏好优化）后的版本**
3. **训练代码 / 数据 / 词表统计**
4. **Tokenizer 的训练语料**（只有训好的 `tokenizer.model` 文件）

这意味着什么？

- **对研究者**：你拿到的是 314B base 模型，可读、可分析、可观测，但**不能直接用作产品** - 它会续写、会乱接话，不会"听人话"。
- **对工程师**：要做 SFT，需要至少 16 × H100（量化后的训练规模）的硬件 + 高质量的指令数据集 + 几周训练时间。这是直接复刻 LLaMA-Chat 的成本数量级，业余社区基本承担不起。
- **对 xAI 本身**：保留了 instruction-tuning 这一商业关键能力 - 当时 grok.com 上跑的 Grok 是已经经过 SFT/RLHF 处理的版本，开源 base 不会直接威胁产品。

事实证明社区的反应也印证了这一判断 - 截至 2026 年，HuggingFace 上 Grok-1 base 的下载量远少于同尺寸的 Mixtral 8x22B-Instruct，几乎找不到一个被广泛使用的 Grok-1 SFT fork。这是第 11 章会展开讨论的话题。

这里再多说一点关于"base 模型"的工程意义，因为这是普通 MLE 容易模糊的概念。

**Base 模型是预训练阶段的产物**，目标是**最大化对训练分布的 likelihood**（似然），用人话说就是"看到一堆文本，学会预测下一个 token"。这个目标和"听人指令、礼貌对话、拒绝越界"是**正交的**（互不相关）- base 模型见到 `Hello, how are you?` 大概率会接一段 Reddit 评论或者一段维基百科开头，而不是回答你。让 base 变成 chatbot，需要后续两个阶段：

1. **SFT（Supervised Fine-Tuning）**：用人工或半自动构造的"指令 - 回复"对（比如 OpenAssistant、Alpaca、ShareGPT 数据集）对 base 模型继续训练，让它学会"看到指令要回答指令"。
2. **偏好对齐**：在 SFT 之上，用 RLHF（Reinforcement Learning from Human Feedback）、DPO（Direct Preference Optimization）或更新的 KTO、ORPO 等方法，按人类对回复质量的偏好微调，让模型不只是"答"，还要"答得好、答得安全"。

xAI 开源 base 而不开源后续两个阶段，相当于把"原料"提供出来，但"配方和工艺"自己保留。这是商业意义上的最优策略 - 既能获得"开源"带来的公关效果和研究界好感，又不影响商业产品的差异化护城河。

把同期几家厂商的开源策略横向比一下就更清楚：

- OpenAI、Anthropic、Google **从不公开任何 base 模型**（除了 OpenAI 早期的 GPT-2）
- xAI **公开了 base，但没公开 SFT/RLHF**
- Meta 公开了 base，**也公开了 LLaMA-2-Chat / LLaMA-3-Instruct**
- Mistral 公开了 Mixtral-Base，**也公开了 Mixtral-Instruct**

xAI 走在 OpenAI/Anthropic/Google 之前，但走在 Meta/Mistral 之后。本书的态度是：把这件事讲清楚，不下道德判断。开源策略是商业决定，没有"应该不应该"。

## 1.4 仓库结构概览

整个 `xai-org/grok-1` 仓库摊开看：

```
grok-1/
├── README.md / README_zh.md       # 入门说明，中英文各一份
├── LICENSE.txt                    # Apache 2.0 协议
├── pyproject.toml                 # 只配置了 ruff（一个 Python linter）
├── requirements.txt               # 4 行依赖
├── run.py                         # 72 行示例入口
├── checkpoint.py                  # 221 行权重加载
├── runners.py                     # 605 行推理引擎
├── model.py                       # 1398 行模型实现
├── tokenizer.model                # 2.2MB SentencePiece 分词模型
└── checkpoints/                   # 用户自行下载 ckpt-0/* 放这里
```

依赖只有四行：

```
dm_haiku==0.0.12
jax[cuda12-pip]==0.4.25
numpy==1.26.4
sentencepiece==0.2.0
```

注意几件事：

- **没有 PyTorch、没有 TensorFlow、没有 HuggingFace transformers** - 纯 JAX 技术栈
- `dm_haiku` 是 DeepMind 早期主推的 JAX 神经网络库，2023 年开始 DeepMind 自己也在迁向更新的 `flax.nnx`，但 xAI 选择继续用 Haiku（很可能是创始团队从 DeepMind 带过来的习惯，参见第 11 章关于 xAI 创始人背景的讨论）
- JAX 锁到 0.4.25，对 jaxlib/cuda12 的版本绑死，迁移到新 JAX 会有兼容性问题（实测 0.4.30+ 已无法直接跑）
- 整个项目没有 test、没有 mypy 配置、没有 CI 配置 - 这是一份"研究级开源"，不是产品级代码

### 1.4.1 几条依赖背后的"小故事"

值得再花一段说说这份四行 `requirements.txt`：

```
dm_haiku==0.0.12
jax[cuda12-pip]==0.4.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
numpy==1.26.4
sentencepiece==0.2.0
```

短短四行，每一行都不是随意写的。`dm_haiku 0.0.12` 是 2023 年 11 月发布的版本，紧贴 Grok-1 预训练结束（2023 年 10 月）的时间点。`jax 0.4.25` 在 2024 年 2 月发布，正好赶在 3 月开源前。`numpy 1.26.4` 是 2024 年 2 月发布的最后一个 1.x 系列版本 - 之后 2.x 出现，对部分 dtype 行为做了不兼容修改。`sentencepiece 0.2.0` 是 2024 年 2 月发布的稳定版。

这意味着 xAI 在开源前做了一次"依赖冻结" - 把那一刻所有能用的最新稳定版组合起来，写死版本号。这是工业级开源的标准做法（保证未来某天有人 clone 下来还能复现作者当时的环境），但 2 年后想跑就难了：

- `jax 0.4.25` 不兼容 CUDA 12.4+（NVIDIA 后续 cuDNN 升级，需要 jax 0.4.30+）
- `dm_haiku 0.0.12` 在 jax 0.4.30+ 下有若干 deprecation warning
- `numpy 1.26.4` 与 jax 0.4.30+ 配合 OK，但与 numpy 2.x 之后某些库不兼容

如果你读到这里真的想跑 Grok-1，建议用 Docker 锁定一个 2024 年 3 月前后的 CUDA 12.2 镜像，再 `pip install` 这四行。

## 1.5 后续 Grok 系列走向闭源

| 版本 | 发布日期 | 是否开源 |
| --- | --- | --- |
| Grok-1 | 2024-03 | 是（base 权重 + 推理代码） |
| Grok-1.5 | 2024-04 | 否 |
| Grok-1.5V | 2024-04 | 否（首次具备视觉理解） |
| Grok-2 | 2024-08 | 否 |
| Grok-2 mini | 2024-08 | 否 |
| Grok-3 | 2025-02 | 否（首次加入 reasoning 模式） |

也就是说，xAI 后续没有继续走开源路线。Grok-1 作为 xAI 唯一公开的模型，技术细节有"博物馆价值"。这本书写作的根本动机也在于此：等到 Grok 系列再开源（如果有的话），架构很可能已经面目全非（Grok-3 的 reasoning 模式据传是基于 RL self-play 的全新栈），现在留下的细致拆解会成为这一支 lineage 的最详尽记录。

### 1.5.1 一个对照案例：LLaMA 系列

把 xAI 和 Meta 的开源策略对照一下：

| 维度 | Meta LLaMA | xAI Grok |
| --- | --- | --- |
| 持续开源 | LLaMA、LLaMA-2、LLaMA-3、LLaMA-3.1、LLaMA-3.2、LLaMA-3.3、LLaMA-4 全部开源权重 | 只有 Grok-1，后续都闭源 |
| 同时多尺寸 | 每代都放 8B、70B、有时还放 405B | 只放 314B 一个尺寸 |
| 开源 SFT 版本 | LLaMA-2-Chat / LLaMA-3-Instruct 系列同步发布 | 只有 base，没有 chat |
| 配套工具链 | 完整的微调脚本、HuggingFace transformers 原生支持、官方推理优化（torch-compile、vLLM 集成） | 只有最小推理示例 |
| 商业许可 | 自定义许可（>7 亿月活需申请） | Apache 2.0（完全开放） |
| 实际生态影响 | 数千个 fork、几十万次下载、衍生模型数百个 | 几乎没有衍生生态 |

数字本身就在讲故事：**Apache 2.0 比 LLaMA 协议更开放，但 Grok-1 的生态影响远小于 LLaMA**。开源许可只是一个法律外壳，真正决定生态的是**易用性**：模型尺寸、配套工具、文档质量、社区参与度。Grok-1 在这些维度上都比 LLaMA 弱很多。

这也回答了"什么是真开源"这个看似简单的问题。代码可见、权重可下、协议宽松，这些都是必要条件，但不充分。**真正的开源还需要"可用性"** - 让普通研究者能在合理硬件上跑、能扩展、能改造。Grok-1 拿到了前几个条件，没拿到最后这个。

## 1.6 读源码前需要的 JAX/Haiku 基础

Grok-1 代码对 JAX 的依赖集中在几个 API。如果你来自 PyTorch 世界，下面这段简介足以读懂全书。我会用"PyTorch 对照"的方式讲，因为大部分 MLE 都熟悉 PyTorch。

### 1.6.1 函数式与 transform

PyTorch 的模型是**有状态对象**：`model = Transformer(); y = model(x)`，参数存在 `model` 实例里。

JAX 的模型是**无状态函数**：参数是显式传进去的，不挂在任何对象上。这种风格的好处是天然适合 `jit`、`vmap`、`pmap` 这些函数变换；代价是写起来啰嗦一些。

Haiku 是 DeepMind 提供的"语法糖"，让你能用接近 PyTorch 的写法定义模型（用 `hk.Module`），然后用 `hk.transform` 把这个带状态的写法转成 `(init_fn, apply_fn)` 两件套：

```python
def forward(x):
    return MyModel()(x)  # 看起来像 PyTorch 调用

forward_t = hk.transform(forward)
params = forward_t.init(rng, x_example)   # 拿参数（类似 PyTorch 的 model.state_dict()）
y = forward_t.apply(params, rng, x)        # 跑前向（必须显式传 params）
```

Grok-1 在 `runners.py:395-398` 显式用了 `hk.without_apply_rng(hk.transform(...))`，因为推理时 RNG（随机数发生器）单独由 sampler 管理，模型自身不需要 dropout 之类的随机性，所以把 rng 参数从 apply 签名里去掉。

### 1.6.2 `pjit` 与 mesh

PyTorch 的张量并行通常通过 `torch.distributed` + `nn.parallel` + 手写 `all_reduce` 实现。JAX 提供了一套更高层的抽象：你声明每个张量应该怎么切，剩下的让 XLA 编译器自动生成通信代码。

```python
mesh = jax.sharding.Mesh(devices, ("data", "model"))
fn_p = pjit.pjit(fn, in_shardings=P("data", "model"), out_shardings=P("data"))
with mesh:
    out = fn_p(x)
```

**Mesh** 是把物理设备（比如 8 个 GPU）组织成一个 N 维网格的抽象。Grok-1 用 (1, 8) 的本地 mesh - 1 个 data shard、8 个 model shard，正好对应一台 8 卡 H100/A100 节点。

**PartitionSpec("data", "model")** 意为"输入张量的第 0 维沿 data 轴切、第 1 维沿 model 轴切"。如果某一维不切（完整复制到每个 device），用 `None` 占位。

### 1.6.3 `shard_map`

`pjit` 是"我告诉编译器张量怎么切，剩下你管"；`shard_map` 更底层，让你**直接写"每个 shard 看到的局部 tensor"的 SPMD（Single Program Multiple Data，单程序多数据）代码**：

```python
@partial(shard_map, mesh=mesh, in_specs=P("model", None), out_specs=P("model", None))
def f(local_x):
    # 这里的 local_x 已经是"这台设备看到的那一片"
    return local_x * 2
```

Grok-1 在 MoE 层用 `shard_map`（`model.py:319-357`）做 expert 维度的切分，因为 expert 路由有数据相关的稀疏行为（每个 token 走哪些专家是 runtime 才知道的），`pjit` 处理不好这种 dynamic 模式。这是全书最难读的一段，第 5 章会逐行讲。

### 1.6.4 `with_sharding_constraint`

中间张量的强制 partition：

```python
x = with_sharding_constraint(x, P("data", None, "model"))
```

模型代码里到处都是这种约束，可以理解为"提示 XLA 编译器在这一步把张量重新布成这种形状"。编译器会插入必要的 collective 通信（如 all-gather、reduce-scatter）来满足约束。

### 1.6.5 几个语义陷阱

JAX 与 PyTorch 在心智模型上有几处常见的工程陷阱，把它们提前点出来能省你不少调试时间：

**陷阱 1：函数式更新。** JAX 数组是 **immutable（不可变）**，所有"更新"操作返回新数组。`x[0] = 1` 这种写法在 JAX 里**不能直接用**，需要 `x = x.at[0].set(1)`。Grok-1 的 KV cache 更新就大量用 `jax.lax.dynamic_update_slice_in_dim` 这种"函数式更新"原语 - 每次更新都"返回一个新 cache"，编译器再优化成 in-place。

**陷阱 2：JIT 边界。** 在 `jit` 编译过的函数里，Python 控制流（if/for）会在 trace（追踪）阶段固化。所以 `if x > 0: ...` 这种**依赖 runtime 值**的判断不能写在 jit 里，需要换成 `jnp.where(x > 0, ..., ...)`（一个张量级的 if-else，两个分支都会被 trace）。Grok-1 的 `hk_forward` 用 `jnp.where` 替换 if 就是这个原因。

**陷阱 3：PRNG 显式传参。** JAX **没有"全局随机种子"概念**，每次需要随机数都要传一个 PRNGKey（伪随机数键）。`jax.random.split(rng)` 让你从一个 key 派生出多个独立 key。Grok-1 的 `hk_sample_step` 在每一步都做 `rngs, rngs_ = jax.vmap(jax.random.split, out_axes=1)(rngs)` - 沿 batch 维做 vmap（向量化映射）后产出新 rng。

**陷阱 4：dtype 严格性。** JAX 对 dtype 转换比 PyTorch 严格。如果你把 fp32 张量传给一个期待 bf16 的函数，会得到错误而不是自动 cast。Grok-1 的 `cast_bfloat16`（`model.py:78-82`）是一个显式 cast helper - 这种"显式 cast 帮助函数"在 PyTorch 里很少见，但在 JAX 代码里是必备工具。

到这里 JAX 基础就够了。下一章我们用一张图把 Grok-1 的整个数据流串起来。

## 延伸阅读

- xAI 官方博客 [Open Release of Grok-1](https://x.ai/blog/grok-os) - 当时发布时的官方声明
- Mistral [Mixtral of Experts (8x7B) Paper](https://arxiv.org/abs/2401.04088) - 同代 MoE 对照，写得非常清楚
- DeepMind [Haiku Documentation](https://dm-haiku.readthedocs.io/) - 0.0.12 版本的 API 没有大变化，新版文档基本可用
- JAX [Distributed arrays and automatic parallelization](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html) - `pjit` 与 mesh 的官方教程
- Ainslie 等 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) - GQA 原论文
- Shazeer [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU 系列的提出
