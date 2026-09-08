# FAQ

<!-- pi-faq-record: source=manual -->
## `bytes`、`str`、regex 与 `BinaryIO` 的关系

### 问题

在重写 `find_chunk_boundaries()` 时，为什么文件按二进制读取、special token 也转换成 `bytes` 后，仍然需要对 token 做 regex escape？`bytes` 和 `str` 在 regex 中有什么区别？`BinaryIO` 又表示什么？

### 交互记录

- `find_chunk_boundaries()` 读取的是二进制文件，因此 `file.read()` 返回 `bytes`，special tokens 也应该使用 `bytes`，这样 offset 才对应原始文件中的 byte offset。
- `bytes.find()` 是字面量搜索，不解析正则语法，所以传入 `b"<|endoftext|>"` 时不需要 escape。
- `regex.search()` 即使接收的是 `bytes` pattern，仍然会把 `|`、`.`、`*`、`+`、`(`、`)` 等解释为正则元字符。`bytes` 只表示数据的表示形式，并不会关闭 regex 解析。
- `find_chunk_boundaries()` 中如果使用 regex，应当分别 escape 每个 token，再使用未 escape 的 `b"|"` 连接 alternatives：

  ```python
  pattern = b"|".join(
      regex.escape(token)
      for token in split_special_tokens
  )
  ```

- `str` pattern 必须配合 `str` subject，`bytes` pattern 必须配合 `bytes` subject。不能把两种类型混用。
- `prepare_docs()` 使用文本模式读取文件，得到 `str`；`find_chunk_boundaries()` 使用二进制模式读取文件，得到 `bytes`。这是两个函数不同的接口层次。

### 最终结论

#### 1. `bytes` 和 `str` 是数据类型，不是解释模式

可以从两个独立维度理解：

| 数据类型 | API | 行为 |
|---|---|---|
| `str` | `.find()` | 字面量文本搜索 |
| `str` | `regex.search()` | 正则搜索 |
| `bytes` | `.find()` | 字面量字节搜索 |
| `bytes` | `regex.search()` | 正则搜索 |

是否需要 escape 由“是否使用 regex”决定，而不是由 `bytes` 或 `str` 决定。

#### 2. 为什么这个误解很常见

`bytes` 常被称为 raw data、binary data，很多 API（如 `file.write()`、socket API、`.find()`）会把它原样处理，因此容易产生“bytes 不会被解释”的直觉。

更准确的记忆方式是：

> `bytes` 决定数据长什么样；regex 决定数据如何被解释。

`b` 前缀只表示 Python 字面量是 `bytes`；它不等于“关闭正则语法”。`r` 前缀也只影响 Python 源码层面的反斜杠处理，不会关闭 regex 解析。

#### 3. `BinaryIO` 的含义

```python
def find_chunk_boundaries(file: BinaryIO, ...):
```

`BinaryIO` 是类型注解，表示一个提供二进制 I/O 接口的文件流，通常支持：

```python
file.read(...)   # 返回 bytes
file.seek(...)
file.tell()
```

典型对象包括：

```python
with open(path, "rb") as f:
    ...
```

以及测试中的：

```python
from io import BytesIO
f = BytesIO(b"some data")
```

`BinaryIO` 不是一段 `bytes` 数据本身：

- `bytes` 是已经在内存中的字节序列
- `BinaryIO` 是可以移动游标、持续读取的二进制流

类型注解不会自动转换或运行时强制检查对象；它主要帮助读者、IDE 和类型检查器理解函数契约。

#### 4. 当前 `find_chunk_boundaries()` 的实现重点

先单独把 boundary finder 做对，不要同时处理进程池。目标是：

- 返回以 `0` 和文件大小为端点的递增 offset 列表
- 每个中间 boundary 位于某个 special token 的起始位置
- special tokens 作为字面量匹配，而不是未经 escape 的 regex
- 空 token 列表和空 token 单独处理

如果不需要 regex，也可以对每个 token 使用 `bytes.find()`，再取最早的非负位置；这种方式天然是字面量搜索，不需要 escape。

<!-- pi-faq-record: source=session-mine -->
## p9/p10 tokenizer 实验：序列化、worklog 与 MVP 下采样

### 问题

讲义要求把 TinyStories / OpenWebText 的 tokenizer 训练结果序列化到磁盘，但没有指定一个新的专有格式。如何选择格式，并如何记录耗时、内存和 profiling 信息？

### 结论

- 讲义真正固定的是后续 `Tokenizer.from_files(vocab_filepath, merges_filepath, ...)` 的输入契约：`vocab` 是 `dict[int, bytes]`，`merges` 是有序的 `list[tuple[bytes, bytes]]`。
- 可以采用 GPT-2 byte-level BPE 的可读文本格式：`vocab.json` 保存 `byte-to-unicode string -> token_id`，`merges.txt` 每行保存一个编码后的 pair，按生成顺序排列。
- 本仓库的 loader 会把每一行按两个字段解析，因此 `merges.txt` 不写 `#version: 0.2` header；格式选择应服从实际 loader，而不是只看外部库的惯例。
- `worklog.json` 与 tokenizer 格式分离，记录输入路径 / 大小 / 可选 SHA-256、配置、git / Python 环境、各阶段耗时、parent + worker RSS 峰值、最长 token，以及 profile 文件路径。
- `cProfile` 主要覆盖主进程；训练使用 `ProcessPoolExecutor` 时，主进程 profile 可能把 worker 等待时间显示为进程池 shutdown。要观察完整进程树，应使用 `py-spy --subprocesses`。

### MVP 下采样

为了快速验证，可以按已有 `<|endoftext|>` delimiter 流式复制前 N 个完整文档到 subset，再调用同一个 `train(...)`。默认不启用下采样；启用时不截断文档中间、不合成 delimiter、不注入额外 special token，并把 subset 路径和复制数量写进 worklog。

这类 MVP 的目标是保持训练假设不变，只减少样本规模。下采样参数应该改变输入数据量，而不是改变 tokenizer 的 special-token 定义、训练代码路径或 merge 规则。

<!-- pi-faq-record: source=session-mine -->
## 增量 cache 优化：先定位受影响对象，再维护派生状态

### 问题

每轮 BPE merge 都重新构建 `pair_counter`，观察到 merge 只影响包含 selected pair 的 word 后，如何把这个性质变成不容易错的第一版？

### 结论

先区分 source of truth 和 derived cache：

```text
token_seq_counter: 当前 Word -> 频次       # source of truth
pair_counts[pair]: 全局 pair occurrence 数  # derived cache
pair_to_words[pair]: 包含 pair 的 Word 集合 # derived reverse index
```

维护的核心不变量是：

```text
pair_counts == 从当前 token_seq_counter 全量重建的统计
pair_to_words[p] == 当前包含 p 的所有 word
```

因此一次 merge 不需要手动推导左右邻居，而可以写成：

```text
cache' = cache - contribution(old_word) + contribution(new_word)
```

实现上先复制 `pair_to_words[max_pair]`，删除所有 old words 的 token / pair count / reverse-index membership，再聚合 new words，最后统一加入它们的贡献。`pair_counts` 要按 occurrence 计数；`pair_to_words` 只表示 membership，所以使用 `set`。当前阶段明确保留 `max(pair_counts.items(), ...)`，不引入 heap；先验证增量状态的正确性和收益。

### 为什么不要一开始做 occurrence-level delta

有以下结构时：

```text
(A, B) -> AB
```

虽然只改变局部邻居，但要正确维护位置级 delta，必须处理开头 / 结尾、重复 occurrence、`(A, A)`、重叠匹配以及 left-to-right 规则。第一版先使用 `pair -> affected words`，对 affected word 完整重算 pair，已经能消除无关 word 的全量扫描；只有 profiling 证明仍不够快时，才引入 occurrence position 或 linked list。

### 推荐调试流程

1. 保留 naive reference。
2. 写出 source of truth、cache 和 invariant。
3. 用小例子和边界用例验证 `merge_word`。
4. 用 differential test 逐轮比较 `vocab` / `merges`。
5. 用随机测试把 cache 与全量重建结果比较。
6. 正确性稳定后再降低更新粒度。

核心判据：如果一个函数同时计算新值、判断边界、修改多个 cache，通常应该拆成纯计算函数和对称的 remove / add helper。
