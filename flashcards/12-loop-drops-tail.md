# 卡 12 · `[必错]` 边扫边跳步的循环：`while i < len(seq) - 1` 会丢掉最后一个元素

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 的 `build_new_seq` 复盘
- **标记**：`[必错]` —— 静默丢数据，不报错

## 背景

把一个 word 序列里所有 `pair` 出现处合并掉。因为要看 `seq[i]` 和 `seq[i+1]` 两个元素，
循环条件顺手写成了 `while i < len(seq) - 1`：

```python
def build_new_seq(seq, pair):
    new_seq = []
    i = 0
    while i < len(seq) - 1:            # ← 病灶
        if (seq[i], seq[i + 1]) == pair:
            new_seq.append(pair[0] + pair[1])
            i += 2
        else:
            new_seq.append(seq[i])
            i += 1
    return tuple(new_seq)
```

## 正面 —— 预测两个输出

```python
build_new_seq((b'h', b'e', b'l', b'l', b'o'), (b'h', b'e'))
build_new_seq((b'h', b'e', b'l', b'l', b'o'), (b'l', b'o'))
```

## 答案

- `(b'he', b'l', b'l')` —— **`b'o'` 丢了**
- `(b'h', b'e', b'l', b'lo')` —— 正确

## 为什么

「边扫边跳步」的循环，退出时 `i` 可能落在**两个**位置：`len(seq)` 或 `len(seq) - 1`。
后者意味着还剩最后一个元素没被 append，而循环条件已经不成立了。

第二个例子正确，是因为合并跳了 2 步，`i` 从 3 直接到 5，越过了 `len-1`。
**同一段代码时对时错**，取决于合并位置的对齐 —— 这是最难查的形态。

## 正确写法

```python
while i < len(seq):                                        # 走到底
    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:  # 配对判断自己管边界
        new_seq.append(pair[0] + pair[1])
        i += 2
    else:
        new_seq.append(seq[i])
        i += 1
```

另一种等价写法是保留 `while i < len(seq) - 1`，**在循环外**补
`if i == len(seq) - 1: new_seq.append(seq[-1])`
—— 但**两种不能混用**，见 [卡 13](13-two-fixes-stacked.md)。

## 自测

```sh
python3 -c "
def f(seq, pair):
    new, i = [], 0
    while i < len(seq) - 1:
        if (seq[i], seq[i+1]) == pair: new.append(pair[0]+pair[1]); i += 2
        else: new.append(seq[i]); i += 1
    return tuple(new)
print(f((b'h',b'e',b'l',b'l',b'o'), (b'h',b'e')))
print(f((b'h',b'e',b'l',b'l',b'o'), (b'l',b'o')))
"
```

## 手写要点

写「窗口 + 变步长」的循环时，**先把「退出时 i 可能落在哪些值上」列出来**。
默认选择应该是 `while i < len(seq)` + 循环内部做 `i + 1 < len(seq)` 的边界判断
—— 这个形状天然不会漏尾。

## 相关卡

- [卡 13](13-two-fixes-stacked.md) —— 修本卡时把两个互斥方案叠加，又引入重复 append
- [卡 14](14-pure-function-deserves-asserts.md) —— 三行断言本可以当场逮住它
