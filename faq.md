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
