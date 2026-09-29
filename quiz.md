## P3.a What Unicode character does chr(0) return?

直接 :r !python3 -c 'print(repr(chr(0)))'

```
'\x00'
```

### 标准答案

`chr(0)` 返回 Unicode 码点 `U+0000` 的 NUL（null）控制字符；在 Python 中，它是一个长度为 1 的 `str`（其 `repr` 为 `'\x00'`）。

## P3.b How does this character’s string representation (**repr**()) differ from its printed representation?

`__repr__` 是机器友好，print 是人类可读。具体来讲好像 print(a) 输出的是 a 的 `__str__` 函数。不太确定。

### 标准答案

`repr(chr(0))` 将该字符显示为带引号的转义形式 `'\x00'`，而 `print(chr(0))` 写入的是实际的 NUL 字符（以及 `print` 默认添加的换行符）；由于 NUL 没有可见字形，终端上看起来只有一个空行。

## P3.c What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations

应该是不可读的字符吧，显示上应该是空。实验下看看。以下 shell 输出的粘贴：

```
❯ python3
Python 3.14.5 (main, May 10 2026, 10:21:34) [Clang 21.0.0 (clang-2100.0.123.102)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
>>> exit

chezmoi on  main took 39s
```

### 标准答案

NUL 出现在文本中时仍是一个真实字符并占据一个位置，例如 `s = "this is a test" + chr(0) + "string"` 满足 `len(s) == 21`、`s[14] == chr(0)`，且 UTF-8 编码中包含 `0x00`；它通常不可见，所以打印时两侧文本看似直接相连，但 Python 不会把它当作字符串结束符（某些 C 或操作系统接口可能拒绝或特殊处理它）。

## P4.a What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings

因为 UTF-8 标准规定是最少可以只用 8 bit 也就是 1 个 byte 来表示一个 char 的。相应的 UTF-16 和 UTF-32 是分别最少要 2、4 个 byte 才行。显然 UTF-8 更省存储。

### 标准答案

UTF-8 与 ASCII 兼容，ASCII 字符只需 1 byte，而 UTF-16 和 UTF-32 分别至少需要 2 和 4 bytes，并会为 ASCII 文本引入大量 `0x00`，因此在通常包含大量 ASCII 的语料上，UTF-8 会产生更短、更适合 byte-level tokenizer 的序列；同时 UTF-8 没有 UTF-16/32 的大小端问题。UTF-8 并非对所有文本都更省空间，例如 `"汉"` 在 UTF-8 中占 3 bytes、在 UTF-16 中占 2 bytes。

## P4.b Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results

example:

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

有种 not even wrong 的感觉。既然输入已经是 bytes 类型，直接 decode 就好了吧。

### 标准答案

该函数遍历 `bytes` 时得到的是单个 byte 的整数值，并将每个 byte 隔离后解码；这只对单 byte 的 ASCII 字符有效，因为一个非 ASCII 字符的 UTF-8 编码必须由连续的 2 到 4 bytes 一起解码。例如 `b'\xc3\xa9'` 整体解码应得到 `"é"`，但该函数首先尝试单独解码 `b'\xc3'`，会抛出 `UnicodeDecodeError: unexpected end of data`。

## P4.c Give a two-byte sequence that does not decode to any Unicode character(s)

思路是直接双层 for loop 拼两个 byte 找个 decode 报错的就行了吧。

e.g.

```python
def solution():
    for i in range(256):
        for j in range(256):
            seq = chr(i) + chr(j)
            try:
                seq.decode()
            except:
                print(f"Found seq [chr({i}), chr({j})]")
                break
```

### 评价

穷举全部两字节组合、尝试用 UTF-8 解码的思路是可行的，但当前实现不正确。`chr(i) + chr(j)` 得到的是 `str` 而不是 `bytes`，所以调用 `.decode()` 会抛出 `AttributeError`；裸 `except:` 又把这个编程错误误判成了解码失败。此外，`break` 只会退出内层循环。可以改为构造 `bytes([i, j])`、只捕获 `UnicodeDecodeError`，并在找到结果后直接 `return`：

```python
def solution() -> bytes | None:
    for i in range(256):
        for j in range(256):
            seq = bytes([i, j])
            try:
                seq.decode("utf-8")
            except UnicodeDecodeError:
                return seq
    return None
```

### 标准答案

`b'\xc0\x80'` 是一个无法解码的两字节序列：合法的双字节 UTF-8 首 byte 必须在 `0xC2` 到 `0xDF` 之间，而 `0xC0 0x80` 是被 UTF-8 禁止的 `U+0000` 过长编码（overlong encoding），因此解码时会抛出 `UnicodeDecodeError`。

---

## P9.a （讲义 p.9 · `train_bpe_tinystories` (a)）How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?

### 回答

训练用了 178 秒（2 分 58 秒，其中纯训练 177.9s），峰值进程树 RSS 4.69 GiB；词表 10,000（9,743 个 merge，含 `<|endoftext|>`）。最长 token 是 15 字节的 `Ġaccomplishment` / `Ġdisappointment` / `Ġresponsibility`，即 “ accomplishment” 这类带前导空格的完整长单词——这是合理的：它们在 TinyStories 里分别出现 1516 / 614 / 558 次，是高频词单元，而且词表里没有任何 ≥16 字节的 token，说明 10K 容量基本都花在常见词上。

细节（`artifacts/p9_tinystories/worklog.json`）：

- 语料 `data/TinyStoriesV2-GPT4-train.txt` 共 2,227,753,162 B（2.07 GiB）；`elapsed_seconds` 177.99 = training 177.90 + serialization 0.02 + analysis 0.001。
- 内存 `peak_total_rss_gib` 4.689，测量方式是每 0.25s 采样 parent + descendants 的 RSS 求和；满足讲义 ≤30 min / ≤30 GB RAM。
- 最长 token（`analysis.longest_token`，3 个并列 15B）：id 7160 `Ġaccomplishment`、id 9143 `Ġdisappointment`、id 9379 `Ġresponsibility`；次长的一批也全是整词（`Ġuncomfortable`、`Ġcompassionate`、`Ġunderstanding` 等 14B）。
- 语料计数（`grep -oE ' (accomplishment|disappointment|responsibility)' data/TinyStoriesV2-GPT4-train.txt | sort | uniq -c`）：1516 / 614 / 558。
- 这次运行是在 macOS（8 CPU、Python 3.13.5）上做的，worklog 里的路径是 `/Users/liangyuanwei/...`；要跟 p10 的 Linux 数字对比内存时要记得这点。

## P9.b （讲义 p.10 · `train_bpe_tinystories` (b)）Profile your code. What part of the tokenizer training process takes the most time?

### 回答

主进程的 CPU 几乎都花在串行的 merge 循环上：每一步都用 `max()` 扫一遍全部活跃 pair 选最优（`builtins.max` 自身 39.2s），它的 key lambda 又占 15.4s（3.69 亿次调用），两者合计约 55s。预分词（worker 进程）是墙钟上的大头，但 cProfile 看不到 worker，主进程只把它记成 116.4s 的 `ProcessPoolExecutor` 收尾等待。

细节（`artifacts/p9_tinystories/train.cprof.txt`，主进程 177.887s）：

- `{built-in method builtins.max}`：tottime 39.2s / cumtime 54.6s / 12,641 次调用 —— 对应 `max(pair_counts.items(), key=...)`（该次训练时在 297 行，现在是 `cs336_basics/p9_bpe_tokenizer_training.py:352`）。
- `p9_bpe_tokenizer_training.py:297(<lambda>)`：15.4s / 369,218,707 次 —— 就是上面 max 的 key 函数；9,743 步 merge 平均每步扫约 3.8 万个 pair。
- `apply_merge` 4.1s（9,743 次），增量缓存更新 `_add_word_to_cache` cum 2.0s、`_remove_word_from_cache` cum 1.4s。
- `ProcessPoolExecutor.__exit__` / `shutdown` / `join` cum 116.4s（占墙钟约 2/3）是在等预分词 worker；文件头也注明 “cProfile covers the main process only”，要看 worker 内部得用 `py-spy --subprocesses`。
- 附带一点：内存采样函数 `_sample`（psutil 每 0.25s 拉一次 descendants）本身就花掉 cum 7.7s，约 4% 墙钟。
- 结论：并行预分词已经把主要工作挪出主进程，剩下能优化的就是 merge 阶段（选 pair 的 O(活跃 pair) 扫描 + 缓存更新）。

## P10.a （讲义 p.10 · `train_bpe_expts_owt` (a)）Train a byte-level BPE tokenizer on OpenWebText (vocab 32,000) and serialize it. What is the longest token in the vocabulary? Does it make sense?

### 回答

最长 token 是 64 字节，共 2 个：64 个连续的 `-`（id 25836）和 16 遍重复的 “ÃÂ”（id 25822，字节 `C3 83 C3 82` ×16）。这很合理：OWT 里 ≥64 个连续 `-` 的片段有 6,491 个（grep 计数），“ÃÂ” 连排 ≥16 的片段有 4,679 个，都是高频重复内容；64 字节的 token 不是随机长串，而是网页里的排版分隔线和双重编码 mojibake（脏数据）。

细节（`artifacts/p10_openwebtext/worklog.json` + `train.log`）：

- 元数据：语料 11.10 GiB，178 chunks（4 workers，64 MiB 目标，最大对齐 chunk 63.9 MiB）；vocab 32,000 / merges 31,743；`elapsed_seconds` 7732.34（2h08m52s），峰值 RSS 8.094 GiB，满足 ≤12 h / ≤100 GB。
- 阶段耗时（`train.log` 时间戳）：预分词 ≈920s（12:01:33 → 12:16:53），pair cache ≈37s（6,601,892 个 unique pre-token），merge 循环 ≈6,773s（约占 88%）。
- 64B token 的“家族”（都来自 `vocab.json`）：32 字节的 `-`×32（id 10900）、`_`×32（15947）、`=`×32（25146）、`.`×32（28585）、“ÃÂ”×8（16885），48 字节的 em-dash×16（31274）。
- 语料侧证据：`LC_ALL=C grep -oaP '(-){64,}' data/owt_train.txt | wc -l` = 6491（≥32 个的是 13743）；`grep -oaP '(\xc3\x83\xc3\x82)' | wc -l` = 75587 个 “ÃÂ” 单元，其中 ≥16 连排 4679 个（同为 grep 计数）；em-dash×16 有 4724 个。

## P10.b （讲义 p.10 · `train_bpe_expts_owt` (b)）Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.

### 回答

两者用的是同一套训练流程（同一 GPT-2 预分词正则、同一 `<|endoftext|>` 特殊 token），差异只来自语料和词表上限：TinyStories 的 10K 词表基本被常见英文词填满，最长 token 才 15 字节且全是完整单词，碰到非 ASCII 的 merge 只占 0.2%；OWT 的 32K 词表在中心分布上几乎一样（平均字节长 6.34 vs 5.79、中位数都是 6），但尾部多出 47 个 ≥16 字节、9 个 ≥32 字节的 token，而且这些长 token 全是分隔线/mojibake，碰到非 ASCII 的 merge 升到 1.4%。也就是说：语料越杂，多出来的词表容量主要花在“噪声片段”上，而不是更长更细的英文词。

细节（由 `vocab.json` / `merges.txt` 反解成 bytes 后统计）：

| | TinyStories (10K) | OpenWebText (32K) |
| --- | --- | --- |
| 语料大小 | 2.07 GiB | 11.10 GiB |
| merges 数 | 9,743 | 31,743 |
| token 平均 / 中位字节长 | 5.79 / 6 | 6.34 / 6 |
| p99 / 最大字节长 | 12 / 15 | 13 / 64 |
| ≥16B / ≥32B token 数 | 0 / 0 | 47 / 9 |
| 含非 ASCII 字节的 token | 148 | 559 |
| 触碰非 ASCII 的 merge | 20（0.2%） | 431（1.4%） |
| 最长 token | `Ġaccomplishment` 等 3 个英文长词 | `-`×64；“ÃÂ”×16 |

## P12.a （讲义 p.12 · `tokenizer_experiments` (a)）Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained tokenizers, encode them. What is each tokenizer's compression ratio (bytes/token)?

### 回答

TinyStories 样本（10 篇、7,565 bytes）用 10K tokenizer 是 **4.161 bytes/token**（1,818 tokens）；OpenWebText 样本（10 篇、31,617 bytes）用 32K tokenizer 是 **4.704 bytes/token**（6,722 tokens）。

细节（`artifacts/p12_tokenizer_experiments/worklog.json`，由 `uv run python -m cs336_basics.p12_tokenizer_experiments --steps sample` 生成）：

- 压缩率 = `source_bytes`（样本的 UTF-8 字节数）/ `token_count`。
- 两个 tokenizer：TS `vocab 10,000 / merges 9,743`，OWT `vocab 32,000 / merges 31,743`；special token 都是 `<|endoftext|>`。
- 采样按 `<|endoftext|>` 切文档（分隔符保留在样本里，它本身是 1 个 token），两边都用各自的 tokenizer 编码自己的语料。

## P12.b （讲义 p.12 · `tokenizer_experiments` (b)）What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer?

### 回答

压缩率从 4.704 掉到 **3.199 bytes/token**（6,722 → 9,883 tokens，多出约 47%）。主因不是文本分布本身，而是 **tokenizer 与语料不匹配**：10K 词表只覆盖 TinyStories 的简单词汇，OWT 里的长词、专名、代码/HTML 片段和 mojibake 大量没有对应 token，只能拆成更多小片段甚至单字节。special token 这里不构成问题：两个 tokenizer 训练时用的是同一个 `<|endoftext|>`，不存在 special 集合不一致。

细节：

- 同一份 OWT 样本：OWT tokenizer 6,722 tokens（4.704 B/token），TS tokenizer 9,883 tokens（3.199 B/token）。
- 这与 P10.b 的观察互为印证：语料越杂，词表尾部越多花在分隔线/mojibake 上；反过来小词表处理杂语料就更吃亏。

## P12.c （讲义 p.12 · `tokenizer_experiments` (c)）Estimate the throughput of your tokenizer (bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

### 回答

当前纯 Python 实现的吞吐约 **0.65 MiB/s（TS tokenizer）** 和 **0.58 MiB/s（OWT tokenizer）**；按 825 GiB 外推，Pile 需要约 **362 / 403 小时**（约 15–17 天）。

细节（`worklog.json` 的 `throughput`，样本是各语料开头约 8 MiB 的完整行）：

| | TinyStories (10K) | OpenWebText (32K) |
| --- | --- | --- |
| source_bytes | 8,388,727 | 8,388,703 |
| token_count | 2,037,859 | 1,910,074 |
| elapsed_seconds | 12.34 | 13.74 |
| bytes_per_second | 679,835 | 610,470 |
| Pile（825 GiB） | ≈ 362 h | ≈ 403 h |

- 外推公式：`825 * 1024**3 / bytes_per_second / 3600`（把 825GB 视为 825 GiB；按 10^9 算会小约 7%）。
- 两个 tokenizer 吞吐接近，OWT 稍慢是因为 31,743 merges 的 rank 表比 9,743 大，`_encode_pretoken` 每轮要多比较。
- 瓶颈在纯 Python 的逐 pretoken merge（regex 预分词本身很快）；符合讲义对“Pile 规模会非常慢”的预期。

## P12.d （讲义 p.12–13 · `tokenizer_experiments` (d)）Encode the training/development sets into uint16 NumPy arrays. Why is uint16 an appropriate choice?

### 回答

因为两个词表（10,000 / 32,000）都小于 `2**16 = 65,536`，token ID 又都是非负数，`uint16` 是能无损表示所有 ID 的最小整数类型：比 `uint32` 省一半空间，比 `int64` 省 3/4。实现上 ID 先收进 `array("H")`，超过 65,535 会直接 `OverflowError`，相当于顺带做了范围校验。

细节：

- ID 范围：TS 0–9,999，OWT 0–31,999；`uint16` 上界 65,535，余量充足。
- 产物示例：`TinyStoriesV2-GPT4-valid.txt`（22 MB UTF-8）→ 5,465,883 tokens → `tinystories_valid_ids.npy`，dtype `uint16`、10.4 MiB（换成 `uint32` 则是 20.8 MiB）。
- 写盘用流式 `.npy`（先写占位 header，结束时回填真实 shape），所以 12 GB 的 OWT 不会把语料和 ID 数组同时留在内存里。
- “不会取负索引”这点也对：ID 只作为 embedding 的正索引使用；无符号类型正合适。
- 生成全套数组：`uv run python -m cs336_basics.p12_tokenizer_experiments --steps encode-datasets`（四个语料合计约 6–7 小时，输出到 `artifacts/p12_tokenizer_experiments/`）。
