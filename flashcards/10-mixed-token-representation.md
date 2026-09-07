# 卡 10 · `[必错]` 表示层不统一：同一个位置上时而 int 时而 tuple

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py`（真语料 BPE 训练）复盘
- **标记**：`[必错]` —— `TypeError`，但**第 1 轮能跑通、第 2 轮才炸**

## 背景

BPE 训练把每个 word 表示成「当前切分」的序列，每轮把出现最多的相邻 pair 合并成一个新 token。
关键不变量是：**序列里每个元素都是同一种类型的「一个 token」**。

原代码里：

```python
# pretokenize：起点
token_seq = tuple(b for b in token)        # 元素是 int（0-255）

# merge_word（当时名为 build_new_seq）：合并产物
new_seq.append(pair)                       # 元素是 tuple[int, int]  ← 类型变了

# train：消费
vocab[len(vocab)] = bytes([max_pair[0], max_pair[1]])
```

## 正面 —— 预测两件事

1. 当时的 `build_new_seq((104,101,108,108,111), (101,108))` 返回什么？
2. 主循环第几轮会抛 `TypeError: 'tuple' object cannot be interpreted as an integer`？

## 答案

1. `(104, (101, 108), 108)` —— 合并产物是个**嵌套 tuple**，和它左右的 `int` 兄弟不是同一类东西。
2. **第 2 轮**。第 1 轮时序列里全是 `int`，`max_pair` 是 `(int, int)`，`bytes([...])` 正常；
   第 2 轮 `max_pair` 里开始出现 tuple，`bytes([tuple, ...])` 才炸。

## 三个连带损伤（比 TypeError 更值钱）

1. `merges` 收集的是嵌套 int-tuple，不是签名要求的 `list[tuple[bytes, bytes]]`。
2. **tie-break 语义被悄悄改变**：规格要求按 **bytes 字典序**取并列中较大者，
   而 `max()` 比较 int 只在单字节时**碰巧**等价 —— 见 [卡 04](04-tiebreak-three-stages.md)。
3. `init_vocab` 里用的是 `bytes([i])`（正确），和序列里的 `int` 表示不一致，
   于是测试里 `set(vocab.values())` 那条断言也会跟着挂。

## 正确做法 —— 全程统一用 `bytes` 对象

```python
token_seq = tuple(bytes([b]) for b in token)   # 起点：单字节 bytes
new_seq.append(pair[0] + pair[1])              # 合并：bytes 拼接，仍是 bytes
vocab[len(vocab)] = max_pair[0] + max_pair[1]  # 消费：同一种类型
```

## 自测

```sh
python3 -c "
seq=(104,101,108,108,111)
merged=[seq[0], (seq[1],seq[2]), seq[3], seq[4]]
print(merged)
print([type(x).__name__ for x in merged])   # 同一个 list 里两种类型 → 病灶
"
```

## 手写要点

**先钉死不变量，再写实现。** 一句话写下来贴在函数上方：
「这个序列的元素永远是 `bytes`，从 pretokenize 到 vocab 都不变。」

诊断口诀：如果一个 bug 是「第 1 轮对、第 2 轮炸」，**几乎一定是「输出类型 ≠ 输入类型」**
—— 迭代把自己的输出又喂回了输入。检查方法：`print([type(x).__name__ for x in seq])`。

## 相关卡

- [卡 07](07-vocab-vs-merges.md) —— 同一族问题的入门版（一个 list 里混三种类型）
- [卡 11](11-bytes-int-zero-fill.md) —— 修本卡时踩的下一个坑
- [卡 16](16-annotation-without-enforcement.md) —— 本卡的注解**当时就写对了**，但没人对账
