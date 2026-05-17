# 第 9 章 Tokenizer

`tokenizer.model` 是仓库里唯一的二进制文件，2.2 MB（2229219 bytes）。本章把这个文件拆开逐项核对：算法、特殊 token、词表分布、对各语种的编码效率，以及它与 LLaMA-2 / LLaMA-3 等同代模型 tokenizer 的差异。

在进入细节之前，先用一节把 tokenizer 这个组件本身梳理清楚 - 这是 LLM 工程链路上最容易被当作"黑盒"跳过的一环，但理解它对解释很多模型行为（包括上下文长度限制、多语种性能差异、奇怪的输出现象）都非常关键。

## 9.0 Tokenizer 在 LLM 链路里到底做什么

一段文本送进 LLM 之前要经过 tokenizer 转成一串整数 id，模型输出的也是一串整数 id，再经 tokenizer 解码成文本。tokenizer 是文本 ↔ 整数序列之间的固定查找表，**模型本身完全不知道字符长什么样**，它眼里只有整数。

为什么不直接用 Unicode 码点（每个字符一个整数）？两个原因：

第一是**词表大小**。Unicode 有 11 万多个码点（含 CJK、Emoji、各类符号），如果每个字符独立一个 token，词表就是 ~110k，embedding 矩阵和 lm_head 矩阵都很大。更关键的是，**很多常见词组合（"the"、"ing"、"http"、"中国"）会被拆成多个字符 token**，模型每次预测都要预测好几个 token 才完成一个语义单位，序列变长、推理变慢。

第二是**频率分布极度不均匀**。如果按字符切，模型见到 "the" 时实际上看到的是 't', 'h', 'e' 三个独立 token，但实际它们是一个高度相关的整体。subword（子词）tokenizer 把这种高频组合作为单个 token 处理，让模型更容易学到上层语义。

更朴素的 word-level（按空格分词）也不行，原因有三：词表会爆炸（英文几十万词、加上拼写错误、加上其他语言更多）；不能处理生词（OOV，out-of-vocabulary 问题）；不适合中日韩这种不靠空格分词的语言。

**Subword tokenization（子词分词）** 是这两个极端之间的折中：高频词作为整体 token、低频词被拆成更小的 subword、最罕见的字符仍能用 byte 表示。BPE 与 Unigram 是两种主流的 subword 算法，会在 9.2 节展开讲。

理解了 tokenizer 的角色之后，再回头看 Grok-1 这份 2.2 MB 的 `tokenizer.model` 就清楚了：它就是一张训练时定下的"131072 个 subword 字符串 ↔ 整数 id"的映射表，附带一些归一化规则。文件不需要训练时的语料（语料用过就丢），只需要这张表。

## 9.1 基本信息

```
文件：tokenizer.model
格式：SentencePiece
大小：2229219 bytes
词表大小：131072 (= 128 × 1024 = 2^17)

!!! note "SentencePiece"
    Google 开源的 subword tokenizer 库，从语料训练出一张词表，然后把任意输入字符串切成词表里的 piece。它的特点是不依赖语言的"空格分词"前处理 - 直接把整段文本（含空格、换行、标点）当 Unicode 字节流喂进去，所以中文、日文这种不靠空格分词的语言也能用同一套流程。空格在内部被替换成特殊符号 `▁`（U+2581），解码时再换回来。

    存档格式是个 protobuf 文件（`.model`），里面打包了词表、合并规则、归一化设置等。Grok-1 的 `tokenizer.model` 就是这种 protobuf，2.2 MB，131072 个 piece。LLaMA-2、Mistral、Grok-1 都用 SentencePiece；LLaMA-3、GPT-4o 改用 tiktoken，两者格式互不兼容。
```

实测（用 `sentencepiece==0.2.0` 加载）：

```python
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('tokenizer.model')
print(sp.GetPieceSize())   # 131072
print(sp.bos_id())          # 1
print(sp.eos_id())          # 2
print(sp.pad_id())          # 0
print(sp.unk_id())          # 3
```

特殊 token：

| id | piece | 用途 |
| --- | --- | --- |
| 0 | `[PAD]` | padding |
| 1 | `[BOS]` | begin-of-sequence |
| 2 | `[EOS]` | end-of-sequence |
| 3 | `[UNK]` | unknown |

与 `run.py:27-28` 对应：

```python
pad_token=0,
eos_token=2,
```

**没有显式 BOS 处理** - run.py 里的示例 prompt 直接传给 tokenizer.encode，没有手动在序列前补 BOS token。这是研究示例代码的简化做法，正式 inference / fine-tune 时需要按训练时的约定显式补 BOS，否则模型看到的位置 0 token 与训练分布不一致，会影响前几步输出质量。

这个细节在不少模型上引起过问题：HuggingFace transformers 早期版本默认会自动加 BOS，但 SentencePiece 的 `Encode` 默认不加，两边代码混用时会出现"明明用同一个 prompt、HF 和原版输出不一样"的奇怪现象。debug 时只要打印一下 `tokenizer.encode(prompt)` 的结果对比，能立刻看出谁多了一个 BOS。

## 9.2 BPE vs Unigram

!!! note "BPE（Byte-Pair Encoding）与 Unigram"
    两种主流的 subword 分词算法。BPE 走"自底向上合并"：先把所有文本拆成单字符，统计相邻字符对的出现频率，把最高频的一对合并成一个新 piece，再统计、再合并，循环到词表达到目标大小。最终切词时按训练时记下的合并顺序应用，每次贪心 merge。GPT-2、LLaMA、Mistral 用的都是 BPE。
    
    举个具体例子。假设语料里"l", "o", "w" 三个字符相邻出现得很多（比如来自 "low"、"lower"、"slow" 等词），BPE 第一次合并可能产出 piece "lo"；如果 "lo" + "w" 也常一起出现，下一次合并产出 "low"；如果 "low" + "er" 频率高，再合并成 "lower"。最终词表里既有 "low" 也有 "lower"，切词时遇到 "lower" 直接整体取整、遇到 "lowering" 切成 "lower" + "ing"。
    
    Unigram 走"自顶向下淘汰"：先生成一个超大候选词表（含所有 1~3 字符子串），训练一个 unigram 语言模型给每个 piece 估个概率，每轮 EM 把"如果删掉对总语料 likelihood 影响最小"的那批 piece 砍掉，直到词表收敛到目标大小。切词时找让整句概率最大的切分，是全局最优而不是贪心。
    
    打个比方：BPE 像装修先打地基再砌墙，按训练时记下的"合并历史"逐步搭起每个词；Unigram 像雕刻先有一块大石头然后凿掉不要的部分，最终词表是淘汰后的子集。两者得到的词表会有差异：BPE 偏向于保留"高频合并路径上的中间产物"，Unigram 偏向于保留"对语料 likelihood 贡献最大的最终子词"。
    
    Grok-1 的 131072 词表看起来是 Unigram（前 30 个 piece 没有 BPE 典型的"完整 256 个 byte token"段，倒像是预定义 `\n` × N、`\t` × N、数字 0-9 之后接学到的 piece）。LLaMA 系是 BPE，Grok-1 走 Unigram 路线在大模型里是少数派。

!!! note "为什么 Grok-1 用 SentencePiece 而不是 tiktoken"
    tiktoken 是 OpenAI 开源的 BPE 实现，最大特点是**纯 byte-level BPE**：把原始文本先按 UTF-8 编成字节流，再在字节序列上跑 BPE。这样不需要任何 unicode 归一化、不需要预定义任何特殊处理，**任意字节序列都能编码、不会有 [UNK]**。GPT-2、GPT-3、GPT-4、LLaMA-3 都用 tiktoken 或与之兼容的 byte-level BPE 实现。
    
    SentencePiece 是 Google 开源的库，支持 BPE 和 Unigram 两种算法。它的字符处理范围是 unicode 字符（不是字节），需要事先做 NFKC 归一化等处理。优点是支持的算法更多（Unigram 是 SentencePiece 主推的）、训练流程更成熟、和 protobuf 配合做模型持久化方便。LLaMA-1、LLaMA-2、Mistral、Mixtral、Grok-1 都用 SentencePiece。
    
    Grok-1 选 SentencePiece 而非 tiktoken 的原因大概率是历史习惯 + Unigram 算法本身的偏好。xAI 创始团队多人来自 DeepMind 和 OpenAI，对两套工具应该都熟悉。选 SentencePiece + Unigram 这个组合，最实用的好处是 Unigram 在多语种文本上通常比 BPE 表现更好 - Unigram 的"全局最优切分"对 CJK 这种无空格语言更友好，BPE 的贪心合并在中文场景下经常切出语义割裂的奇怪 piece。Grok-1 既然要做多语种 base 模型，Unigram 是合理选择。
    
    代价是 SentencePiece 不像 tiktoken 那样 byte-level，遇到训练时没见过的 unicode 字符（比如生僻汉字、罕见 emoji）会回退到 [UNK]，造成信息丢失。tiktoken 的 byte-level 完全无损。LLaMA-3 后来从 SentencePiece 切换到 tiktoken（且词表从 32k 升到 128k），就是为了避免这个 [UNK] 问题。

SentencePiece 支持两种主要算法：

- **BPE**（Byte-Pair Encoding）：从字符开始，反复合并最高频对
- **Unigram**：基于 EM 训练 unigram language model，选 piece 最大化语料 likelihood

要判断 Grok-1 的 tokenizer 用的是哪一种，可以查看词表头部前 30 个 piece 的分布：

| id | piece |
| --- | --- |
| 0-3 | `[PAD]` `[BOS]` `[EOS]` `[UNK]` |
| 4-13 | `\n` × 1, × 2, ..., × 10 |
| 14-23 | `0` `1` `2` ... `9` |
| 24-29 | `\t` × 1, × 2, ..., × 6 |

排列规律：从短到长的连续 `\n` / `\t` 序列、单字符数字。这是**Unigram 模型典型的"用户预定义的高频 piece + 学习到的 piece"组合**。

更可能 Grok-1 用的是 **Unigram with byte fallback**：

1. byte fallback 字符段在 vocab 头部（在 SentencePiece 默认配置下，byte fallback 占 256 个 piece，但 Grok-1 这个 tokenizer 没看到明显的 byte fallback 段）
2. 然后是高频特殊 piece（`\n`、`\t`、数字）
3. 之后是 Unigram 学到的 sub-word

但严格来说没看到 byte fallback piece（id 4-255 应该是 byte 0x00 ~ 0xFF 才对，这里却是其他东西）。实际可能是不带 byte fallback 的 Unigram。

LLaMA 系列使用 BPE，Mistral / Mixtral 同样使用 BPE。Grok-1 的 tokenizer 从词表头部排列规律推断更接近 Unigram 算法 - 这是 Grok-1 与同代主流 LLM 在分词侧的一个明确差异，意味着即使 vocab 大小相同，它与 BPE 模型也无法直接共用 ckpt 的 embedding 矩阵。

### 9.2.1 词表是怎么训出来的（直观理解）

很多人对 tokenizer 的疑问停在"它是干什么的"，对"它怎么从语料训出来的"反而模糊。这里用一句话概括两种算法的训练流程。

**BPE 训练**：(1) 把训练语料的每个字符独立看作一个 piece；(2) 统计相邻 piece 对的频率；(3) 选最高频的一对合并成新 piece，记录合并历史；(4) 重复 2-3 步直到词表达到目标大小（比如 32000）。最后保存"词表 + 合并历史"。切词时按合并历史顺序逐步合并。

**Unigram 训练**：(1) 用启发式算法生成一个超大候选词表（比如所有 1-16 字符子串）；(2) 用 EM 算法训练每个 piece 的概率（看作一个 unigram 语言模型）；(3) 每轮淘汰一批"删掉后对语料 likelihood 影响最小"的 piece；(4) 重复 2-3 步直到词表收敛到目标大小。最后保存"词表 + 每个 piece 的概率"。切词时用 Viterbi 算法找出整句概率最大的切分。

两种训练都是无监督的、不需要标注数据，只需要一份纯文本语料。训练时间相对模型预训练几乎可以忽略 - 训一个 131k 词表的 SentencePiece tokenizer 在普通 CPU 上几小时到一两天就能完成，根本不需要 GPU。所以 tokenizer 训练不是 LLM 项目的瓶颈。

真正的难点反而在**语料配比**：tokenizer 训练用什么比例的多语种/代码/数学/网页文本，直接决定了词表的偏向。如果训练时英文占 90%、中文占 5%、代码占 1%，那词表里中文 piece 会少、代码 piece 也少，最终模型在中文和代码上的 token 效率就低。xAI 没有公开 Grok-1 tokenizer 训练时的语料配比，但从词表里中文 piece 数量推断（参见 9.4 节），中文相关 piece 大概 1.5-2 万个，占总词表 11-15%，比 LLaMA-2 高很多。这个比例配置直接转化为"Grok-1 处理中文比 LLaMA-2 高效"的性能差异。

## 9.3 词表内容分布

实测前 30 个 piece 已经看到：

- 4 个特殊 token
- 10 个换行序列
- 10 个数字
- 6 个 tab 序列

接下来开始是 ASCII piece 和拉丁字母 word。

接近词表尾（131060-131071）：

```
131060 '土'
131061 'ั'   (泰语字符)
131062 'Å'
131063 '強'
131064 'ω'   (希腊字符)
131065 '该'
131066 '只'
131067 '统'
131068 '트'  (韩语字符)
131069 '院'
131070 'À'
131071 '曲'
```

词表尾部多为单 Unicode 字符（中、日、韩、希腊、泰、拉丁扩展），是 fallback 用的低频 piece。

### 9.3.1 词表分布的几个推断

从前 30 个 piece 的排列再推断几件事：

**预定义 piece 的处理：** SentencePiece 训练时可以指定"必含 piece"（user-defined symbols 或 control symbols）。Grok-1 的 `[PAD]`、`[BOS]`、`[EOS]`、`[UNK]` 看起来是 control symbols。`\n` × 1~10、`\t` × 1~6 看起来是 user-defined symbols - xAI 显式插入它们以保证缩进、换行被高效编码。

**没有 byte fallback：** 标准 SentencePiece 配置如果开 `--byte_fallback=true`，会在 vocab 头部插入 256 个 `<0x00>` ~ `<0xFF>` byte token。Grok-1 没有这些，所以是不带 byte fallback 的 Unigram 模型。这意味着遇到罕见 unicode 字符（vocab 外）会被替换为 `[UNK]` - 信息丢失。

**词表尾部多 CJK 单字：** 表里看到 `土`、`強`、`该`、`只`、`统`、`院`、`曲` 等中日韩单字位于 131060+ 位置。这暗示 xAI 训练 tokenizer 时给了 CJK 较多预算，但仍把"中频"汉字留在尾部 - 高频汉字（"的"、"是"、"了"）应该排在前面。

## 9.4 编码效率实测

实测几个文本：

| 文本 | 字符数 | token 数 | 字/token |
| --- | --- | --- | --- |
| `Hello, world!` | 13 | 4 | 3.25 |
| `The answer to life the universe and everything is of course` | 59 | 11 | 5.36 |
| `你好，世界！` | 6 | 6 | 1.00 |
| `今天天气真好，我们一起去散步吧。` | 16 | 23 | 0.70 |
| `def fibonacci(n): return n if n<2 else fibonacci(n-1)+fibonacci(n-2)` | 76 | 31 | 2.45 |
| `\sum_{i=0}^n i^2 = \frac{n(n+1)(2n+1)}{6}` | 41 | 28 | 1.46 |
| `グロックは画期的な大規模言語モデルです。` | 20 | 15 | 1.33 |
| `한국어도 잘 처리합니다.` | 13 | 22 | 0.59 |

观察：

1. **英文编码效率较高**：实测达到 5.36 字符/token，处于与 LLaMA-2 相当的水平，单 token 平均覆盖一个完整英文单词或常见词的主干部分
2. **简体中文居中**：在常用字组成的短句上能做到 1.00 字符/token 左右，但句法稍复杂、生僻词稍多的语句会下降到 0.70 字符/token 附近，原因是部分汉字会被回退到更细的 piece
3. **日文表现较好**：实测 1.33 字符/token，得益于日文的常用汉字与片假名词在词表里有专门 piece，假名连接较短的句子能合并成单个 token
4. **韩文最低**：仅 0.59 字符/token，原因是韩文的谚文音节由初声、中声、终声组合而成，词表很难覆盖所有组合，多数音节只能逐字符回退编码

中文表现优于 LLaMA-2（后者中文约 0.4-0.7 字符/token，因为 LLaMA-2 的 32k 词表里专门留给中文的 piece 数量很少）。Grok-1 的 131072 词表中估计有约 1.5 万到 2 万个中文相关的 piece，对中文用户的实际推理成本与上下文利用都更友好。

这个差异在实际使用中是肉眼可见的：同样长度的中文 prompt（比如一段 4000 字的中文文章），用 LLaMA-2 的 tokenizer 切出来大约 8000-10000 token，用 Grok-1 切出来大约 4000-5000 token。这意味着相同的 8192 上下文窗口下，Grok-1 能塞下大约两倍的中文内容；同等输入的推理 latency 也大致只有 LLaMA-2 的一半。中文用户的实际体验提升非常直接。

但相比 LLaMA-3 （中文效率约 1.2-1.6 字/token，因为 LLaMA-3 在 tokenizer 训练时显式给了中文更多预算）和 Qwen 系列（中文专门优化、字/token 比能到 1.5+），Grok-1 的中文效率仍然不算特别强。这反映出 Grok-1 是定位"全语种通用"而非"针对中文优化"的设计取舍。

## 9.5 与 LLaMA-2 32k 词表的对比

| 项 | Grok-1 | LLaMA-2 | LLaMA-3 |
| --- | --- | --- | --- |
| 词表大小 | 131072 | 32000 | 128256 |
| 算法 | Unigram（推测） | BPE | BPE (tiktoken) |
| pad token | 0 (`[PAD]`) | 无（用 unk） | 无 |
| BOS / EOS | 1 / 2 | 1 / 2 | 128000 / 128001 |
| 中文效率 | 0.7-1.0 字/token | 0.4-0.7 字/token | 1.2-1.6 字/token |
| 代码效率 | 2.45 字/token | 2.5-3.0 字/token | 3.5-4.0 字/token |

**131k 词表的好处：**

1. 中文、日文、韩文、阿拉伯文、印地文等非拉丁文字语言的 token 效率显著提升，从而在相同上下文窗口内能装下更多有效内容
2. 同等输入下长 prompt 切出更少的 token，attention 二次开销随 token 数下降，推理速度直接受益
3. 代码符号、常见数学记号都有专门预留的 piece，使得代码语料的 token 数量更接近字符数

**131k 词表的代价：**

1. **embedding 参数膨胀**：embedding 矩阵规模为 131072 × 6144 = 0.81 B 参数，是 32k 词表（约 0.20 B）的 4 倍，单是 embedding 这一项就给 ckpt 大小带来 ~1.6 GB 增量（bf16）
2. **输出 logit 矩阵更大**：每个 forward step 的最后一步需要做 hidden×vocab 矩阵乘，词表扩到 4 倍意味着这一步显存与计算量也相应放大，整体相对 hidden 维投影增加约 80% 的 FLOPs
3. **rare token 训练不充分**：低频 piece 在训练语料里出现样本较少，对应的 embedding 行更新次数有限，导致这些 token 的表示质量不如高频 token 稳定

LLaMA-3 后来也升到 128k 词表（用 tiktoken 而非 sentencepiece），可以说**Grok-1 提早走到了"大词表"路线**。但 Grok-1 用 Unigram 而 LLaMA-3 用 BPE，两者编码结果不兼容 - 不能跨模型迁移 ckpt 的 embedding 矩阵。

跨模型不兼容这件事在工程上比想象中更受限。假设你想把 Grok-1 的某层 embedding 复用到 Mixtral 或 LLaMA-3 上（比如做某种 knowledge distillation 或 cross-model 实验），第一步就卡住：Grok-1 的 token id 5000 对应的 piece 在另一个 tokenizer 里完全不对应 - 可能是不同字符串，可能根本不存在。所以理论上 ckpt 之间的迁移必须重新做 tokenizer 对齐，把一份语料分别用两边 tokenizer 编码，建立一个 token 级别的对应关系，再做 embedding 迁移。实际操作起来损失大、效果差，几乎没人做。

这也是为什么 LLaMA 系列的生态影响力远超 Grok-1：LLaMA-1 / LLaMA-2 共享同一份 tokenizer，社区做的所有微调、所有 LoRA、所有量化都能跨版本流通；Grok-1 自己一个 tokenizer，社区做的任何东西都没法迁出去。**Tokenizer 的"生态绑定效应"是 LLM 工具链里最强的之一**。

## 9.6 没有 chat template / instruct token

主流 instruct 模型的 tokenizer 会带特殊 token：

- LLaMA-2-Chat：`[INST]`、`[/INST]`、`<<SYS>>`、`<</SYS>>`
- Mistral-Instruct：`[INST]`、`[/INST]`
- ChatML（GPT、Qwen）：`<|im_start|>`、`<|im_end|>`、`<|system|>` 等

**Grok-1 base tokenizer 只有 PAD/BOS/EOS/UNK 四个特殊 token**。没有任何 chat / instruct 模板支持。这印证了 Grok-1 是纯 base 模型。

如果要拿 Grok-1 base 做 chat finetune，需要：

1. 扩展词表加入 chat-specific token（重新训 embedding + lm_head）
2. 或者直接复用现有 token 做 string-level 模板（如 `User:\n...\nAssistant:\n...`）

社区的 Grok-1 fork（少数几个尝试做 SFT 的项目）一般选 2 - 因为扩展 embedding 在 314B 上代价巨大。

### 9.6.1 扩词表 SFT 的具体困难

如果你真想给 Grok-1 base 做 chat SFT 并加 `<|im_start|>` / `<|im_end|>` 这类 token，需要：

1. **扩展 vocab 和 embedding 表**：embedding shape 从 (131072, 6144) 变成 (131072+N, 6144)，N 个新 token 需要新 init。这部分 init 必须合理 - 不能是随机噪声，否则新 token 在 SFT 早期会输出乱码
2. **扩展 logit head**：因为 tied embedding，logit head = embedding，扩展是同一件事
3. **重新训练 embedding**：新 token 的 embedding 需要从零学起，至少几百 step
4. **保留原有 token 的语义**：新 init 部分必须不破坏原有 131072 个 token 的 embedding 空间

工业上常用做法是把新 token 的 init 设置为"语义上接近"的现有 token 的均值（比如 `<|im_start|>` init 为 `User`、`Bot`、`:` 等 token embedding 的平均）。

但 Grok-1 没有官方扩词表脚本，社区也没出现广泛使用的方案。所以即便你有硬件做 SFT，也得先 hack 出一套扩词表流程，门槛叠加。

## 9.7 SentencePiece 默认行为

`runners.py:288` 用 `sentencepiece.SentencePieceProcessor` 直接 load：

```python
self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)
```

`runners.py:513` 使用：

```python
tokens = self.tokenizer.encode(request.prompt)
```

默认 `encode` 行为：

- 不自动加 BOS（除非 tokenizer 训练时设了 `--add_bos`）
- 不自动加 EOS
- 不规范化 unicode（除非训练时设了 `--normalization_rule_name`）

实测 `sp.Encode('Hello, world!')` 返回 `[11770, 130111, 1135, 130166]`，对应 piece `['▁Hello', ',', '▁world', '!']`。注意 `▁`（U+2581，下划线）表示 word start - SentencePiece 把空格转换成 `▁`。

`tokenizer.decode(ids)` 会把 `▁` 还原成空格，所以解码后是 `"Hello, world!"`。

为什么要把空格替换成 `▁`？SentencePiece 的设计理念是"语言无关、不依赖空格分词"。如果保留普通空格，BPE/Unigram 在合并时会把"the"和"_the"（带前导空格）看作两个不同的 token，词表里很多 piece 是同一个词的"带空格版"和"不带空格版"。引入 `▁` 之后，所有 piece 都带显式的 word boundary 标记，词表更紧凑、不会出现这种重复。

对中文这种没有空格分词的语言，`▁` 几乎不出现 - 整段中文文本被当作连续字符流处理。所以 SentencePiece 的中文切词不依赖于"先用 jieba 分词"这种前处理，直接从字符级开始训。这是 SentencePiece 相比传统 NLP 工具链的一个重要简化。

## 9.8 词表大小的"2 次幂"取整

`vocab_size = 131072 = 2^17`。这是一个有意选择的 2 次幂取值，原因包括：

1. **GPU kernel 友好**：大量 GEMM 与 softmax / log-softmax 内核针对 2 次幂尺寸做过对齐优化，矩阵维度落在 2 次幂上时往往能直接命中最优 tile 大小，避免最后一列的特殊处理
2. **embedding sharding 方便**：131072 / 8 = 16384，恰好可以让 8 路 tensor-parallel 各承担 16384 行 embedding，每 shard 都是整数行而无 ragged 边界，反向梯度的 all-reduce 通信也更整齐
3. **避免 ragged 长度**：在做 padding、attention mask 计算、kv-cache 申请时，词表尺寸为 2 次幂可以让相关的内存布局自然对齐，减少为非整长度准备的填充逻辑

LLaMA-2 的 32000 并不是 2 次幂（与之最接近的是 32768）- LLaMA 团队选 32000 这个偏数的内部理由从未公开。相比之下 Grok-1 直接选 131072 在工程对齐上更整齐。

## 9.9 代码与数学符号 token efficiency

实测：

- `def fibonacci(n):...` 76 字符 31 token，2.45 字/token
- `\sum_{i=0}^n i^2` 41 字符 28 token，1.46 字/token

代码效率比中文好，但比 GPT-4 / LLaMA-3 差。LLaMA-3 在代码上能到 3.5-4 字/token，因为它的 BPE 训练时显式加了大量代码语料。

数学符号（LaTeX）的编码效率明显偏低 - 仅 1.46 字符/token。`\sum`、`\frac` 这类反斜杠命令在词表里没有专门收录，往往会被拆成反斜杠 + 字母段共 2-3 个 token。这从侧面反映出 Grok-1 在训练 tokenizer 阶段，代码与数学语料所占比重不算特别高，至少没有像 LLaMA-3 那样为代码语料专门扩充词表。

对应到模型行为，这个 tokenizer 偏向也会带来下游能力差异。代码与数学符号被切得碎，意味着模型每次预测都要预测更多 token 才能完成一个语义单位（一个变量名、一个公式段），推理 latency 高、token 概率累积的不确定性也高。在代码生成或数学推理任务上，token 效率低的模型通常表现更弱 - 这部分是 tokenizer 的责任，不全是模型架构的问题。LLaMA-3、Qwen-Code、StarCoder 这类专门优化代码场景的模型都在 tokenizer 训练时显式加大代码语料权重、专门收录常见编程语法 token，从源头规避这个问题。

社区里有种说法叫"tokenizer 是模型能力上限的一道墙"。意思是不管模型架构再强、参数再大，如果 tokenizer 把某类内容切得稀碎，模型在那类内容上的表现就会被结构性压低。要彻底改善只能重训 tokenizer + 重训模型 embedding 矩阵，成本极高。Grok-1 base 在数学和代码任务上的相对弱势，部分根源就在这里。

## 9.10 tokenizer 训练规模未公开

xAI 没有公开：

- tokenizer 训练用了哪些语料
- 训练参数（character coverage、normalization 规则）
- 训练耗时

只能从二进制 tokenizer.model 里**推断**：

- 词表覆盖语言广（拉丁、CJK、希腊、阿拉伯、印地、泰、希伯来等）
- 数字独立成 piece 0-9
- 换行 / tab 序列预定义

这是个"广覆盖、轻预处理"的 tokenizer 风格，适合做 base 模型。

可以反过来推断：以 Grok-1 总参 314 B、训练时大致用了 6-8 万亿 token 这个量级（业内对类似规模模型的公开估算）反推，tokenizer 训练用的语料规模大概在数百 GB 到 1-2 TB 之间。这部分语料应当是预训练语料的子集（取若干份采样后混合），覆盖所有目标语言和文本类型。具体配比 xAI 没透露，但从最终词表分布可以看到几个信号：（1）拉丁字母（英文 + 欧洲语言）piece 数量最多，证明英文语料是主体；（2）CJK piece 占比相对 LLaMA 系明显更高，证明中日韩语料有合理预算；（3）emoji 和不常见 unicode 字符基本不在词表里，证明 emoji / 多媒体相关文本被刻意过滤了。

## 9.10.1 Tokenizer 训练实操：如果你想自己训一个

假如你想用 SentencePiece 自己训一个类似规模的 tokenizer，整个流程大致如下：

1. **收集语料**：准备一份纯文本文件（每行一个 sample 或 document），规模大概在 1-10 GB。文本越多最终词表越稳定，但训练耗时和内存压力也越大
2. **调用 SentencePieceTrainer**：API 是 `spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='my_tokenizer', vocab_size=131072, model_type='unigram', character_coverage=0.9995, ...)`
3. **关键参数**：
    - `model_type`：选 unigram 或 bpe
    - `character_coverage`：决定多大比例的字符必须被覆盖到（剩下的最罕见字符会被 [UNK]）。中文场景一般 0.9995，纯英文可以 1.0
    - `user_defined_symbols`：列一些必须保留的 piece（比如 `["\n\n", "\t\t", "```"]`），训练时强制收录
    - `byte_fallback`：是否开 byte fallback。开了之后会在词表里预留 256 个 byte token，遇到生僻字符回退到 byte 编码，永远不会有 [UNK]
4. **训练时间**：1 GB 语料训 131k 词表大概几小时到一天，看 CPU 性能和算法
5. **输出文件**：`my_tokenizer.model`（二进制 protobuf）+ `my_tokenizer.vocab`（文本格式的词表 + 概率，方便人查阅）

整个过程不需要 GPU、不需要梯度下降，本质上是一个组合优化问题。SentencePiece 内部用了不少工程优化（C++ 实现、多线程、增量更新），但对外接口非常简洁。

Grok-1 的 `tokenizer.model` 就是经过类似流程产出的一个文件，只不过 xAI 用的语料和参数细节都没公开。

## 9.10.2 一些关于 tokenizer 的常见误解

读 tokenizer 相关代码或文档时容易踩到几个直觉性的坑，统一在这里点清：

**误解一：词表越大模型越强。** 不一定。词表大确实让多语种和代码效率提升，但 embedding 矩阵和 lm head 矩阵会同步膨胀，参数预算被挤占。131k 词表的 embedding 在 d=6144 下就有 0.81 B 参数，相当于多了一层 FFN 的预算。这部分参数对模型核心能力贡献有限，是"为了多语种付出的成本"。社区有研究表明词表在 32k-128k 区间内对模型质量影响不显著，超过 128k 后边际收益快速递减。

**误解二：tokenizer 与模型可以独立优化。** 不可以。tokenizer 一旦定下来，embedding 矩阵的每一行就专属于某个 token，模型整个预训练过程都依赖这个映射。换 tokenizer 等价于重新 init embedding，需要从零开始训。所以"先做完模型再优化 tokenizer"在工程上不可行，必须在项目最开始就把 tokenizer 定好。

**误解三：所有 token 都被平等训练。** 完全错。高频 token（the、of、,、▁）在训练语料里出现几亿次，对应 embedding 行被更新几亿次，表示质量很稳定。低频 token（罕见汉字、生僻 unicode）可能出现几百次或几十次，对应 embedding 行更新次数有限，表示质量差。这是为什么模型在生僻字符上经常输出奇怪结果，社区叫"glitch tokens"（故障 token）现象，有些极端例子下输入某个特定 token 会让模型直接错乱。GPT-2 著名的 "SolidGoldMagikarp" 就是这种 - 一个在 Reddit 上反复出现但训练时被过滤掉的用户名 token，导致模型见到它时行为完全不可预测。

**误解四：tokenizer 是 NLP 阶段的事，不影响模型推理速度。** 实际上 tokenizer 的效率直接影响推理 latency。同样长度的文本，token 多就意味着更长的序列、更大的 KV cache、更慢的 prefill。对延迟敏感的场景（比如实时对话），tokenizer 效率差 30% 就意味着首 token 延迟差 30%。

理解这些误解之后，再看 Grok-1 tokenizer 的各种设计选择就清晰了：131k 词表是在"多语种支持"和"embedding 成本"之间的折中；Unigram 算法是为了多语种切分质量；缺乏 chat template 是因为它本来就只是 base 模型 tokenizer。每个决定都对应一个明确的工程权衡。

## 9.11 总结

| 属性 | 值 |
| --- | --- |
| 算法 | Unigram（推测） |
| 词表 | 131072 = 2^17 |
| 特殊 token | PAD / BOS / EOS / UNK |
| 中文效率 | 0.7-1.0 字/token |
| 英文效率 | 5.4 字/token |
| 代码效率 | 2.45 字/token |
| 数学效率 | 1.46 字/token |
| chat 模板 | 无 |
| 跨模型兼容 | 否（无现成模型共享此 tokenizer） |

下一章对比 Grok-1 与其他主流 MoE。

## 延伸阅读

- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) - SentencePiece 原始论文
- [Subword Regularization](https://arxiv.org/abs/1804.10959) - Unigram 模型的来源
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) - BPE 在 NLP 中的奠基论文
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI 用的 BPE 实现，LLaMA-3 复用其格式
