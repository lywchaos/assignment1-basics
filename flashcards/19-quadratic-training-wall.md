# 卡 19 · `[迁移]` O(vocab_size × 语料) 的复杂度墙：每轮全量重算 pair 计数

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 的 `train` 主循环 review
- **标记**：`[迁移]` —— `test_train_bpe_speed` 实测 1.0s（限 1.5s）通过，
  但离作业要求的真实数据集差 3 个数量级

## 背景

主循环每轮做两件事，**两件都是全表扫描**：

```python
for _ in range(len_init_vocab, vocab_size):
    pair_counter = {}
    for k, v in token_seq_counter.items():      # ① 从零重算全语料 pair 计数
        for p in zip(k, k[1:]):
            pair_counter[p] = ...
    max_pair = ...
    token_seq_counter = apply_merge(token_seq_counter, max_pair)   # ② 重扫全部 word
```

## 正面 —— 三个数字

1. `corpus.en`（130KB）/ `vocab_size=500` 实测多久？profile 里时间分布如何？
2. `tinystories_sample_5M.txt`（5MB）/ `vocab_size=1000` 实测多久？
3. 按此外推，TinyStories 全量（2.1GB）/ `vocab_size=10000` 大约多久？

## 答案（实测）

1. **约 1.0s**（`cProfile` 下 2.46s，limit 1.5s，**余量只有 1.5 倍**）。分布：

   ```
     ncalls  tottime  cumtime  filename:lineno(function)
          1    1.073    2.464  train        ← ① 每轮重算 pair_counter
        243    0.659    0.823  apply_merge  ← ② 每轮重扫 word 表
    5688002    0.611    0.611  dict.setdefault
   ```

   两个热点各占一半，**都是「每轮 O(语料)」**。

2. **7.0s**。
3. 数据量 ×420、轮数 ×13 → **数十小时量级**，不可接受。

复杂度是 **O(vocab_size × 唯一 pre-token 总长度)**。每轮的 merge 通常只影响
**极少数** word，但代码把所有 word 都重新数了一遍 —— 这就是全部的浪费。

## 正确做法（作业「优化」部分的核心）

1. **增量维护 `pair_counts`**，不再每轮从零重算
2. 建**倒排索引** `pair -> {出现该 pair 的 word id}`，每次 merge 只 touch 受影响的 word
3. 每个 word 用「位置索引 / 链表」表示邻接关系，合并时只改局部，
   同步对**周边 pair** 做 `-1 / +1` 修正（这是最容易写错的一步：
   合并 `(A,B)` 会毁掉 `(左邻, A)` 和 `(B, 右邻)`，同时生成 `(左邻, AB)` 和 `(AB, 右邻)`）
4. 选 max 用堆 + 惰性删除；或者接受 `max()` 的 O(P)，因为 P 随合并单调变小

## 手写要点

判据：**「每轮从零重算一个只有局部变化的全局统计量」是典型的算法级浪费。**
识别信号：外层循环里有一个 `xxx = {}` 紧跟着一个全量遍历。

规模墙的通用教训：**性能测试用的 fixture 规模，往往只够验证「不是灾难性的慢」。**
本例 `test_train_bpe_speed` 的 1.5s 上限在 130KB 语料上，
对 O(V×N) 和 O(N + V·局部) 两种实现**几乎没有区分能力** —— 前者只慢 1.5 倍，照样通过。

## 相关卡

- [卡 17](17-read-whole-file-scale-wall.md) —— 同一份代码的空间规模墙
- [卡 15](15-setdefault-as-counter.md) —— profile 第三行那个常数级开销（值得修，但不是主因）
