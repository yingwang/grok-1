# Grok-1

[English](./README.md) | 简体中文

本仓库包含用于加载和运行 **Grok-1** 开源权重模型的 JAX 示例代码。

请确保下载模型检查点，并将 `ckpt-0` 目录放置于 `checkpoints` 目录下 —— 详见 [下载权重](#下载权重) 一节。

随后运行：

```shell
pip install -r requirements.txt
python run.py
```

即可测试代码。

该脚本会加载检查点，并在一段测试输入上从模型进行采样。

由于该模型规模庞大（**3140 亿参数**），使用示例代码测试模型需要一台具备足够 GPU 显存的机器。本仓库中 MoE 层的实现并未做效率优化，选择这一实现方式是为了避免依赖自定义算子，从而便于验证模型实现的正确性。

---

## 一、项目简介

**Grok-1** 是由 [xAI](https://x.ai/) 于 2024 年 3 月发布的大型语言模型，并以 Apache 2.0 协议开源。它是 xAI 在 2023 年 10 月完成预训练的基础模型，**未经任何特定任务（例如对话）的微调**，即“裸模型”状态。这使得 Grok-1 成为研究界一个非常重要的开放基础模型样本。

主要亮点：

- **3140 亿参数**，是发布时全球最大体量的开源大模型之一；
- 采用 **Mixture-of-Experts（MoE，混合专家）** 架构，每个 token 仅激活 2 个专家（约 86B 参数被激活），在保证容量的同时降低推理计算量；
- 完全开源权重，采用 **Apache 2.0** 许可证，允许商用；
- 使用 **JAX + Rust** 自研训练栈在大规模 GPU 集群上从零训练。

---

## 二、模型规格

| 项目 | 参数 |
| --- | --- |
| **参数量** | 3140 亿（314B） |
| **架构** | 8 个专家的 Mixture of Experts (MoE) |
| **每 token 激活的专家数** | 2 |
| **激活参数量** | 约 860 亿（86B） |
| **层数（Transformer Layers）** | 64 |
| **注意力头数** | Query：48；Key / Value：8（采用 GQA 分组查询注意力） |
| **嵌入维度（Embedding Size）** | 6,144 |
| **位置编码** | 旋转位置编码（Rotary Position Embedding, RoPE） |
| **分词器（Tokenizer）** | SentencePiece，词表大小 131,072 |
| **最大上下文长度** | 8,192 tokens |
| **附加特性** | 支持激活分片（activation sharding）与 8-bit 量化 |

> 备注：由于 MoE 每层只激活 2 个专家，**前向推理时实际参与计算的参数约为 86B**，远低于总参数量 314B。这也是 MoE 架构在节省算力方面的核心优势。

---

## 三、架构详解

### 1. Transformer 主干

Grok-1 遵循当前主流的 **Decoder-Only Transformer** 设计：

- **64 层** 堆叠的 Transformer Block；
- 每层包含 **多头自注意力（Multi-Head Self-Attention）** 与 **MoE 前馈网络（MoE FFN）**；
- 使用 **Pre-LayerNorm（RMSNorm）** 提升训练稳定性；
- 通过 **RoPE** 编码相对位置信息，便于推广到更长序列。

### 2. Grouped-Query Attention（GQA）

- 48 个 Query 头，但仅 8 个 Key / Value 头；
- 多个 Query 头共享同一组 KV，显著降低 **KV Cache** 显存占用与解码时的内存带宽压力；
- 在保留多头表达能力的同时，使长上下文推理更高效。

### 3. Mixture of Experts（MoE）

- 每个 FFN 层包含 **8 个专家网络**；
- 由门控网络（Router）根据 token 表征 **动态选择 2 个专家**（Top-2 Gating）参与计算；
- 既保留了大模型的容量优势，又把每个 token 的有效计算量限制在远小于稠密 314B 的规模；
- 仓库中的 MoE 实现以**可读性与正确性优先**，未做内核级优化，便于研究者验证与魔改。

### 4. 量化与并行

- 支持 **8-bit 权重量化**，方便在显存受限的多卡环境中加载；
- 支持 **激活分片**（activation sharding），可在多 GPU 之间划分激活值，缓解显存压力；
- 示例代码使用 JAX 的 `pjit` / mesh 机制做张量并行。

---

## 四、硬件需求

由于参数量高达 314B：

- **完整精度加载**（bf16/fp16）至少需要约 **600+ GB GPU 显存**；
- **8-bit 量化** 后大约需要 **300+ GB GPU 显存**；
- 因此运行示例代码通常需要 **多卡服务器**，例如 8×H100 / 8×A100 (80GB)、或类似规模的集群；
- 仅做模型结构学习、阅读源码则无需任何 GPU。

仓库中的 MoE 实现 **未做性能优化**，吞吐量不代表 Grok-1 真实部署性能，仅用于学术验证。

---

## 五、目录结构

```
grok-1/
├── README.md           # 英文说明
├── README_zh.md        # 中文说明（本文档）
├── LICENSE.txt         # Apache 2.0 许可证
├── CODE_OF_CONDUCT.md  # 行为准则
├── checkpoint.py       # 检查点加载逻辑
├── model.py            # 模型定义（Transformer / MoE / Attention 等）
├── runners.py          # 推理 runner，封装并行与采样
├── run.py              # 入口脚本，加载模型并进行一次采样
├── pyproject.toml      # 项目配置
├── requirements.txt    # Python 依赖
├── tokenizer.model     # SentencePiece 分词器
└── checkpoints/        # 放置 ckpt-0 模型权重的目录
```

---

## 六、快速开始

### 1. 安装依赖

建议使用 Python 3.10+，并提前配置好与本地 CUDA 版本匹配的 JAX。

```shell
pip install -r requirements.txt
```

### 2. 下载权重

#### 方式一：使用磁力链接（torrent）

```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e&tr=https%3A%2F%2Facademictorrents.com%2Fannounce.php&tr=udp%3A%2F%2Ftracker.coppersurfer.tk%3A6969&tr=udp%3A%2F%2Ftracker.opentrackr.org%3A1337%2Fannounce
```

#### 方式二：通过 HuggingFace Hub 🤗

```shell
git clone https://github.com/xai-org/grok-1.git && cd grok-1
pip install huggingface_hub[hf_transfer]
huggingface-cli download xai-org/grok-1 --repo-type model --include ckpt-0/* --local-dir checkpoints --local-dir-use-symlinks False
```

下载完成后请确认目录结构如下：

```
checkpoints/
└── ckpt-0/
    └── ...（权重分片文件）
```

### 3. 运行示例

```shell
python run.py
```

脚本会：

1. 加载分词器与检查点；
2. 在一段预置的英文 prompt 上前向推理；
3. 通过采样方式打印模型续写结果。

如需修改输入，可直接编辑 `run.py` 中的 prompt 字段。

---

## 七、常见问题

**Q1：Grok-1 可以直接用于对话吗？**
A：不能直接获得理想的对话体验。Grok-1 是 **未经过 SFT / RLHF 的基础模型**，更适合作为下游微调（指令微调、对话微调、领域微调）的起点，而非开箱即用的 ChatBot。

**Q2：是否可以商用？**
A：可以。代码与权重均采用 **Apache 2.0 协议**，允许商业使用，但请仔细阅读 `LICENSE.txt`。

**Q3：示例代码推理很慢，是 Grok-1 本身就慢吗？**
A：不是。本仓库中的 MoE 实现以可读性为优先，并未做算子融合或专用内核优化。生产部署可参考 vLLM、SGLang、TensorRT-LLM 等推理框架的 MoE 优化方案。

**Q4：和 Grok-1.5 / Grok-2 有什么关系？**
A：Grok-1 是 xAI 公开开源的 **第一代** 基础模型。后续版本（Grok-1.5、Grok-2 等）目前并未开源权重，本仓库只涉及 Grok-1。

---

## 八、许可证

本仓库中的源代码以及随版本发布的 Grok-1 模型权重，均采用 **Apache 2.0 License**。许可证仅适用于本仓库中的源文件以及 Grok-1 的模型权重。详细条款请参阅 [`LICENSE.txt`](./LICENSE.txt)。

---

## 九、参考链接

- 官方介绍博客：<https://x.ai/blog/grok-os>
- HuggingFace 模型主页：<https://huggingface.co/xai-org/grok-1>
- 原始仓库：<https://github.com/xai-org/grok-1>
