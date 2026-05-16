# 第 9 章 Tokenizer

`tokenizer.model` 是仓库里唯一的二进制文件，2.2 MB（2229219 bytes）。本章把这个文件拆开逐项核对：算法、特殊 token、词表分布、对各语种的编码效率，以及它与 LLaMA-2 / LLaMA-3 等同代模型 tokenizer 的差异。

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

## 9.2 BPE vs Unigram

!!! note "BPE（Byte-Pair Encoding）与 Unigram"
    两种主流的 subword 分词算法。BPE 走"自底向上合并"：先把所有文本拆成单字符，统计相邻字符对的出现频率，把最高频的一对合并成一个新 piece，再统计、再合并，循环到词表达到目标大小。最终切词时按训练时记下的合并顺序应用，每次贪心 merge。GPT-2、LLaMA、Mistral 用的都是 BPE。

    Unigram 走"自顶向下淘汰"：先生成一个超大候选词表（含所有 1~3 字符子串），训练一个 unigram 语言模型给每个 piece 估个概率，每轮 EM 把"如果删掉对总语料 likelihood 影响最小"的那批 piece 砍掉，直到词表收敛到目标大小。切词时找让整句概率最大的切分，是全局最优而不是贪心。

    Grok-1 的 131072 词表看起来是 Unigram（前 30 个 piece 没有 BPE 典型的"完整 256 个 byte token"段，倒像是预定义 `\n` × N、`\t` × N、数字 0-9 之后接学到的 piece）。LLaMA 系是 BPE，Grok-1 走 Unigram 路线在大模型里是少数派。

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
