# 卡 27 · `[流程]` 复用前先分「策略」与「原语」：整函数往往搬不动

- **来源**：p11 `encode` 实现时讨论「p9 的 split / pretokenize / merge 能不能复用」
- **标记**：`[流程]`

## 背景

写 `encode` 时的直觉是「split 和 pretokenize 可以复用 p9，merge 好像没法复用」。逐一看之后：

| p9 的函数 | 能直接复用吗 | 原因 |
|---|---|---|
| `_split_special_tokens(text, specials)` | 不能 | 训练语义是**丢弃** special（不参与 merge 统计）；编码要保留它们并映射成 ID |
| `pretokenize(doc)` | 不能 | 返回**聚合计数** `Counter`（训练要频次）；编码要的是有序的 pretoken 序列 |
| `merge_word(word, pair)` | **能** | 它只是「合并某个 pair 的全部不重叠出现」，是原语 |
| 「选哪个 pair」 | 不能 | 训练按全局 pair 频次；编码按 merges 的创建顺序（rank） |

结论：**可复用的粒度是原语，不是流程。**

## 修法

把 p9 里隐藏的原语抽成公共函数：

- `split_special_tokens`（保留 delimiter）—— 训练侧 `_split_special_tokens` 变成「调用它再过滤」；
- `iter_pretokens` / `to_word`（产出序列而非计数）—— `pretokenize` 变成「从它计数」；
- `merge_word` 本来就是公共原语，编码侧只补 policy：每轮取 rank 最小的相邻 pair。

训练行为逐项对拍未变，`encode` 全部复用这几个原语。

## 手写要点

1. 「能不能复用」别按整函数回答。先问：这个函数里**哪部分是策略**（随场景变：方向、顺序、聚合方式），
   **哪部分是原语**（不变：merge 一个 pair 的全部出现、按 bytes 拆分）。
2. 同名概念在不同场景下**返回形态可能不同**：drop vs keep、计数 vs 序列、对象 vs 迭代器。
   形态不对就不能直接复用，但可以把共同部分抽出来让两边都调用。
3. 重构的正确性判据：原场景输出逐项不变（本例用等价性对拍验证）。

## 相关卡

- [卡 21](21-type-driven-domain-aliases.md) —— 领域命名与函数边界
- [卡 22](22-incremental-cache-convergence.md) —— 工程化拆解
