# 卡 29 · `[陷阱]` decode 必须先把所有 token 的 bytes 拼起来再解码

- **来源**：p11 `decode` 的实现与验证
- **标记**：`[陷阱]` —— byte-level tokenizer 的 token 边界不是字符边界；逐 token 解码看着自然但会错

## 背景

byte-level BPE 的 token 是任意字节片段：**一个 UTF-8 字符可以跨多个 token**。
所以「先逐个 token 解码成字符再拼接」和「先拼字节再解码」不是一回事。

## 正面 —— 预测两种实现的输出

```python
# vocab 里 0:b"\xe2", 1:b"\x82", 2:b"\xac"（合起来是 "€"）
decode_per_token([0, 1, 2])     # 逐 token decode 再拼
decode_join([0, 1, 2])          # 先 join bytes 再 decode
```

## 答案

- 逐 token：`b"\xe2"`、`b"\x82"`、`b"\xac"` 各自都不是合法 UTF-8 → 报错或 `"���"`
- join：`b"\xe2\x82\xac"` → `"€"`

## 修法

```python
return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
```

- `errors="replace"` 是讲义要求：任意 ID 序列不保证合法 UTF-8，非法字节要换成 U+FFFD；
- 非法单字节（如 `b"\xff"`）在 join 之后也会正常替换成 `"\ufffd"`。

## 自测

```sh
uv run python - <<'PY'
from cs336_basics.p11_tokenizer import Tokenizer

tok = Tokenizer({0: b"\xe2", 1: b"\x82", 2: b"\xac", 3: b"\xff", 4: b"a"}, [], None)
assert tok.decode([0, 1, 2]) == "€"
assert tok.decode([3, 4, 3]) == "\ufffda\ufffd"
assert tok.decode([]) == ""
print("ok")
PY
```

## 手写要点

1. 字节级表示的 token 边界**不是**字符边界；编解码要成对思考：
   encode 先整体 UTF-8 → bytes，decode 先整体拼 bytes → UTF-8。
2. `errors="replace"` 不是可选项：模型可以输出任意 ID 序列。
3. 任何「单位转换」先问：单位是 token、字节还是字符？三者不能混。

## 相关卡

- [卡 10](10-mixed-token-representation.md) —— 表示层不统一
- [卡 11](11-bytes-int-zero-fill.md) —— bytes 构造的坑
