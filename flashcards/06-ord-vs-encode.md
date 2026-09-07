# 卡 06 · `[迁移]` `bytes([ord(ch)])` 只对 Latin-1 有效

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘
- **标记**：`[迁移]` —— 玩具语料全 ASCII，码点恒 < 128，与 `encode("utf-8")` 逐字节拆等价

## 背景

BPE 在**字节**层面工作：初始 vocab 是 256 个单字节 token，每个 word 要拆成
`tuple[bytes, ...]`。把 str 拆成单字节序列时写了 `tuple(bytes([ord(ch)]) for ch in w)`。

## 正面 —— 三个表达式，哪些成功、结果是什么？

```python
bytes([ord('a')])
bytes([ord('é')])
bytes([ord('中')])
```

## 答案 —— `b'a'` / `b'\xe9'` / **`ValueError: bytes must be in range(0, 256)`**

`ord` 给的是 Unicode 码点，`bytes([...])` 要的是 0-255 的字节值。两者只在
码点 < 256（即 Latin-1）时**数值上巧合相等**。

`'é'`（U+00E9）能过，但它的 UTF-8 编码其实是 2 字节 `b'\xc3\xa9'`
—— 所以 `b'\xe9'` 这个「成功」比失败更危险，它悄悄产出了错的字节，且不会有任何报错。

正确写法 —— 先 encode，再逐字节拆：

```python
tuple(bytes([b]) for b in w.encode("utf-8"))
```

## 自测

```sh
python3 -c "print(bytes([ord('é')]), tuple(bytes([b]) for b in 'é'.encode()))"
python3 -c "print(bytes([ord('中')]))"
```

## 手写要点

「str → bytes」只有一条合法通路：`encode`。**看到 `ord` 和 `bytes` 出现在同一个表达式里就该警觉**
—— 这是在用码点冒充字节。

## 相关卡

- [卡 11](11-bytes-int-zero-fill.md) —— 同一行代码的另一个坑：`bytes(b)` 与 `bytes([b])` 差一对方括号，语义完全不同
- [卡 18](18-input-contract-edge-cases.md) —— `open()` 不写 `encoding=` 的同类问题（隐式依赖环境）
