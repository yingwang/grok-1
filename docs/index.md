# Grok-1 中文精读

!!! note "MoE（Mixture of Experts）"
    标准 transformer 里，每个 token 在 attention 之后会进入 FFN（前馈网络），同一层所有 token 共用同一个 FFN。MoE 把这一层换成一组 FFN，每个叫一个"专家"。token 进来时，一个小网络（路由器）给所有专家打分，挑分数最高的几个，token 只在这几个里完成 FFN 计算，剩下的不参与。

    好处很明显：模型总量可以做得很大，算力却不用跟着同比膨胀。Grok-1 一共 8 个专家，每个 token 只激活其中 2 个，所以参数虽然有 314B，单个 token 实际用上的只有 86B 左右。

!!! note "JAX / Haiku"
    JAX 是 Google 出的数值计算库，可以理解成"带自动微分和 XLA 编译的 numpy"。和 PyTorch 最大的差别是它走纯函数式那一套：tensor 不可变、模型不持有参数、随机数要显式传 key、控制流要用 `jnp.where` 不能用 Python 的 `if`。好处是 jit + 多卡并行（pjit / shard_map）写起来很自然，代价是心智模型和 PyTorch 几乎不通用。

    Haiku 是 DeepMind 在 JAX 上做的薄薄一层神经网络框架，作用是把"带 `hk.Module` 的函数"转成 `(init_fn, apply_fn)` 两件套，让你用近似 PyTorch 的写法描述模型，但参数仍然是外部传入的 pytree。Grok-1 就是纯 JAX + Haiku 0.0.12 写的，没有任何 PyTorch 代码 - 这也是它至今没有像 LLaMA 那样长出生态的原因之一。

## 这本书是什么

2024 年 3 月 17 日，xAI 用一条磁力链放出了 Grok-1 的全部权重和推理代码：314B 参数、8 选 2 的 MoE、Apache 2.0 协议。这是当时已开源的最大模型，也是 xAI 至今唯一一次开源核心 base model。一年多之后，Grok-1.5、Grok-2、Grok-3 都没有再走开源路线。也就是说，这 9GB 的代码和约 300GB 的权重，是公众能合法、完整地拆解 xAI 训练栈的唯一窗口。

!!! note "magnet link（磁力链）"
    BitTorrent 协议里的一种"去中心化的下载入口"，本质是一串包含文件哈希的 URI，丢给 BT 客户端后，客户端会去 DHT 网络里找到正在做种的 peer，然后从这些 peer 那里把文件拉下来。优点是发布方不用挂 HTTP 服务器、不会被单点限流。xAI 当时给的就是一条 magnet 链接，约 300GB 权重，社区在第一天用 BT 协同分发完。

本书逐行精读这 9GB 代码。重点是：把 `run.py`、`checkpoint.py`、`runners.py`、`model.py` 四个文件里每一处不寻常的设计选择讲清楚 - RoPE 用哪种归一化、attention logit 为什么要过 `tanh` 软裁剪、MoE 为什么用最朴素的 `einsum` 而不是写 kernel、KV cache 怎么和 JAX 的 `shard_map` 协作、`hk.transparent_lift` 怎么在 Haiku 里偷偷复制 8 份 expert 参数。

## 读者画像

预设你已经：

- 写过至少一份 Transformer 的从零实现，对 Q/K/V 投影、softmax、residual 不需要再讲
- 知道 MoE 的大致概念（Mixtral、Switch Transformer 一类）
- 接触过 JAX 或者 Haiku，至少看过 `jit`、`vmap`、`pjit` 的文档；如果没用过，第 1 章给一个最小回顾

不预设你已经：

- 看完 Mixtral 8x7B 的源码
- 知道 GQA 和 MQA 的内核差异
- 熟悉 SPMD 编程下 partition_spec 的写法

如果你的目标只是"跑一下 Grok 看看效果"，本书的密度对你来说偏高，建议先看 `README_zh.md`。

## 11 章的脉络

| 章 | 一句话钩子 |
| --- | --- |
| 1 引言 | Grok-1 落地时的 MoE 生态，与"只放 base、不放 SFT/RLHF"这件事的含义 |
| 2 总体架构 | 一张数据流图把 314B 的 64 层 8 专家穿起来 |
| 3 配置与超参 | 把 `run.py` 里 14 个数字逐个翻译成 314B / 86B 的算术 |
| 4 model.py 上 | RMSNorm、RoPE、GQA - Grok 用了 48 Q vs 8 KV 的少数派配置 |
| 5 model.py 中 | MoE 路由：纯 softmax + top-2，无 aux loss，无 capacity drop，与 Mixtral 仔细对照 |
| 6 model.py 下 | DecoderLayer 装配，竟然用了"双 RMSNorm 包夹"的少见写法 |
| 7 checkpoint.py | 770 个 pickle 分片 + `/dev/shm` + 多 host 协作的硬盘 IO 设计 |
| 8 runners.py | prefill / decode 双轨、KV cache 形状、`pjit` + mesh 把 314B 切到 8 卡 |
| 9 Tokenizer | SentencePiece 131072 词表与 LLaMA-2 的对比 |
| 10 与其他 MoE 对比 | Grok-1 vs Mixtral / DeepSeek-V2 / Qwen-MoE 的路由/容量/粒度 |
| 11 工程教训 | 为什么社区没有出现 Grok-1 的高质量微调版本 |

## 必读 vs 选读

- **必读**：第 2、3、5 章。这三章把"Grok-1 到底是什么"讲完了
- **如果你想动手**：再读第 7、8 章，知道为什么单卡跑不起来、知道 mesh 怎么切
- **如果你写 paper**：第 10 章给出与同代 MoE 的横向对照
- **选读**：第 11 章是观点性内容

## 法律提示

- 代码与权重均以 Apache 2.0 开源，允许商用
- 但 **权重只能从原 magnet 链接或 xai-org HuggingFace 仓库下载**，不要从第三方镜像取
- 本书所有代码片段直接引自 `xai-org/grok-1` 仓库（commit 来自 2024-03），保留版权头中的 X.AI Corp 标识
- 本书文字内容遵循 CC-BY 4.0，可自由转载署名

## 写作约定

- 代码引用形如 `model.py:225-248`，行号与上述仓库一致
- 数字精确到源码：`vocab_size=128 * 1024 = 131072`、`emb_size=48 * 128 = 6144`、`num_layers=64`、`num_experts=8`、`num_selected_experts=2`、`sequence_len=8192`、`key_size=128`、`num_q_heads=48`、`num_kv_heads=8`
- 不确定的事实（训练数据规模、训练成本）一律明确标注"未公开"
- 不写"——"，统一改 `-`

下一章正式开始。
