# 卡 11 · `[必错]` `bytes(b)` 与 `bytes([b])`：差一对方括号，语义完全不同

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 复盘（修 [卡 10](10-mixed-token-representation.md) 时踩的坑）
- **标记**：`[必错]` —— 但**不报错、不崩、测试跑完**，只是所有 token 内容全错

## 背景

BPE 要把一个 word 的 UTF-8 字节串拆成「每个字节一个 `bytes` 对象」的序列。
修 [卡 10](10-mixed-token-representation.md) 时，把 `tuple(b for b in token)` 改成了
`tuple(bytes(b) for b in token)` —— 少了一对方括号。

## 正面 —— 预测两个输出

```python
print(repr(bytes(104)), len(bytes(104)))
print(repr(bytes([104])))
```

## 答案

- `bytes(104)` → **`b'\x00' * 104`，长度 104**。`bytes(int)` 的语义是
  「造一个**长度为该整数**的全零字节串」。
- `bytes([104])` → `b'h'`。这才是「把这个整数当成一个字节」。

## 为什么这个 bug 特别阴

`int → b'\x00' * int` 恰好是**单射**（每个字节值 ↔ 唯一长度），所以：

- pair 计数、合并流程、循环控制**全部照样跑通**
- 没有异常，没有崩溃，`ruff` / `ty` 全 pass
- 只是所有 token 的**内容**全错

实测报错信号是 pytest 的 5000 行 bytes diff，第一条 merge 显示为
`(b'\x00'*32, b'\x00'*116)` —— 32 是 `' '`、116 是 `'t'`，其实就是参考答案的
`(b' ', b't')`，只是内容被编码成了「零字节的长度」。

顺带一个更深的坑：`b'\x00'*32 + b'\x00'*116 == b'\x00'*148`，
恰好等于字节值 148 的单字节 token —— **合并产物会和别的 token 撞车**。

## 自测

```sh
python3 -c "print(repr(bytes(104))[:16], len(bytes(104)), repr(bytes([104])))"
```

## 手写要点

这种「能跑但全错」的 API 误用，代价最高。**改完先在 REPL 里对一个具体值确认一次**
（`repr(bytes([104]))`），比跑一遍 pytest 快 100 倍、信号清晰 100 倍。

附带一条流程教训：这个方括号在诊断建议里是有的，手抄时掉了。
**关键的字符级片段，复制粘贴比手打可靠。**

## 相关卡

- [卡 06](06-ord-vs-encode.md) —— 同一行代码的另一个坑（`ord` 冒充字节）
- [卡 10](10-mixed-token-representation.md) —— 本卡的上游 bug
- [卡 14](14-pure-function-deserves-asserts.md) —— 三行断言本可以当场逮住它
