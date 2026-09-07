# 卡 14 · `[流程]` 纯函数的边界逻辑值得三行断言 —— 别让端到端测试当调试器

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 复盘的**元教训**（最值钱的一条）
- **标记**：`[流程]` —— 关于工作方法，不是某个 API 的坑

## 背景

`tests/test_train_bpe.py` 是端到端测试：读 130KB 语料 → 训练 500 词表 →
把 `merges` 和参考实现逐条比对。一旦不一致，pytest 吐出**被截断的 5000 行 bytes diff**。

一次修 BPE 训练的过程中，连续三轮都靠这个端到端测试当唯一信号：

| 轮次 | 实际 bug | 端到端给的信号 |
|---|---|---|
| 1 | 序列元素 int/tuple 混用（[卡 10](10-mixed-token-representation.md)） | `TypeError`，指向第 90 行（离病灶 50 行远） |
| 2 | `bytes(b)` 零填充（[卡 11](11-bytes-int-zero-fill.md)） | 5000 行 `b'\x00\x00...'` diff |
| 3 | 两个修法叠加导致重复 append（[卡 13](13-two-fixes-stacked.md)） | 5000 行 diff（换了内容） |

## 正面

`merge_word(word, pair)`（修复前名为 `build_new_seq`）是个**纯函数**：输入输出都是几个字节的 tuple，无 IO、无全局状态。
写出三行断言，使其能同时覆盖上表中的三个 bug。

## 答案

```python
assert merge_word((b'h',b'e',b'l',b'l',b'o'), (b'h',b'e')) == (b'he',b'l',b'l',b'o')
assert merge_word((b'h',b'e',b'l',b'l',b'o'), (b'l',b'o')) == (b'h',b'e',b'l',b'lo')
assert merge_word((b'a',b'b'),                (b'a',b'b')) == (b'ab',)
```

三条断言的分工（**这是本卡的重点**）：

- 第 1 条：pair 在**开头**、且合并后**右侧还有元素** → 抓丢尾（[卡 12](12-loop-drops-tail.md)）和重复 append（[卡 13](13-two-fixes-stacked.md)）
- 第 2 条：pair 在**结尾**、合并**跳步越过边界** → 正是那个「碰巧正确」的对照组，
  没有它就看不出 bug 是「时对时错」
- 第 3 条：整个序列**恰好就是一个 pair**（最短输入）→ 抓 `i` 初值/终值的极端情况

顺带地，只要断言写的是 `b'he'` 而不是 `(b'h', b'e')`，第 1 条同时钉死了
「合并产物必须是拼接后的 `bytes`」这个不变量（[卡 10](10-mixed-token-representation.md)）。

## 手写要点

**在函数里挑出「纯 + 边界密集」的那一个，给它自己的小测试。** 判据：

- 输入输出都是小的、可打印的值
- 有 `while` / 索引运算 / 变步长
- 被主循环调用成千上万次（错一次，信号会被放大成天书）

`merge_word` 三个条件全中，是全文件最该有单元断言的函数。

对照：改完先在 REPL 里验一个具体值（[卡 11](11-bytes-int-zero-fill.md)），
成本几秒；靠端到端测试反推，成本是三轮往返。

## 相关卡

- [卡 12](12-loop-drops-tail.md)、[卡 13](13-two-fixes-stacked.md)、[卡 11](11-bytes-int-zero-fill.md) —— 本卡要抓的三个 bug
- [卡 20](20-handwrite-train-bpe.md) —— 手写 `train_bpe` 时的完整自查清单
