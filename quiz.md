## P3.a What Unicode character does chr(0) return?

直接 :r !python3 -c 'print(repr(chr(0)))'

```
'\x00'
```

## P3.b How does this character’s string representation (**repr**()) differ from its printed representation?

`__repr__` 是机器友好，print 是人类可读。具体来讲好像 print(a) 输出的是 a 的 `__str__` 函数。不太确定。

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

## P4.a What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings

因为 UTF-8 标准规定是最少可以只用 8 bit 也就是 1 个 byte 来表示一个 char 的。相应的 UTF-16 和 UTF-32 是分别最少要 2、4 个 byte 才行。显然 UTF-8 更省存储。

## P4.b Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results

example:

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

有种 not even wrong 的感觉。既然输入已经是 bytes 类型，直接 decode 就好了吧。

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
