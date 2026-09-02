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
