# 卡 22 · `[流程]` 从发现局部性到可维护的增量缓存

- **来源**：`p9_bpe_tokenizer_training.py` 增量 cache 复盘
- **标记**：`[流程]` —— 从性能性质收敛到数据结构和状态转换的方法
- **相关**：[卡 19](19-quadratic-training-wall.md)、[卡 20](20-handwrite-train-bpe.md)、[卡 21](21-type-driven-domain-aliases.md)

## 题面

朴素 BPE 每轮都会遍历所有 word 重建 `pair_counter`。观察到一次 merge 只影响包含 selected pair 的 word 后，如何设计一个既能加速又不容易错的第一版？

不要一开始回答「如何只修改左右邻居」，先回答：

1. 哪个对象是真实状态？
2. 哪些结构是 derived cache？
3. 未来的更新操作需要快速回答什么查询？
4. 一次 merge 的状态转换是什么？

## 答案

### 1. 先写 source of truth

```text
token_seq_counter: 当前 Word -> 频次
```

其它结构都是从它推导出来的缓存：

```text
pair_counts[pair]      = 当前所有 pair occurrence 的加权总数
pair_to_words[pair]    = 当前包含 pair 的 Word 集合
```

缓存不变量是：

```text
pair_counts == 从当前 token_seq_counter 全量重建的 pair 统计
pair_to_words[p] == 当前包含 p 的所有 word
```

### 2. 按未来查询设计反向索引

未来需要回答的是：

```text
给定 max_pair，哪些 word 需要更新？
```

所以索引应是：

```python
pair_to_words: dict[Pair, set[Word]]
```

而不是：

```text
bytes -> pairs
word -> pairs
pair -> list[Word]  # 同一个 word 可能重复出现
```

`pair_counts` 和 `pair_to_words` 的语义不同：

- 一个 word 内 pair 出现多次，`pair_counts` 要重复计数；
- 反向索引只表示 word 是否包含 pair，所以用 `set`。

### 3. 把 merge 写成对称的状态转换

```text
cache' = cache - contribution(old_word) + contribution(new_word)
```

因此先写两个对称 helper：

```python
_remove_word_from_cache(old_word, count, ...)
_add_word_to_cache(new_word, count, ...)
```

一次 merge 的稳定流程是：

```text
1. 复制 affected_words
2. 删除所有 old words 的 token / pair count / 反向索引
3. 用 merge_word 计算 new words，并聚合相同 new word 的频次
4. 统一加入所有 new words 的 token / pair count / 反向索引
```

先删除、后加入很重要。这样可以正确处理多个 old words 生成同一个 new word，或 new word 与未受影响 word 冲突的情况。

### 4. 优化粒度逐级降低

```text
全量扫描所有 word
    -> 只扫描 affected words
    -> 只扫描 affected word 的相关 pair
    -> 只更新 affected occurrence 的邻居
```

第一版选择第二层已经有很大收益：用 `pair -> affected words` 定位范围，但对每个 affected word 完整重算 pair。当前阶段保留 `max(pair_counts.items(), ...)`，明确不引入 heap；只有 profiling 证明 affected-word 版本仍然不够时，才引入 occurrence position、linked list 等更复杂结构。

## 本次暴露的能力缺口

不是没有发现局部性，而是：

- 过早从「affected words」跳到「affected pair occurrences」；
- 没先写 cache invariant；
- 把纯计算、索引维护和多个 dict 的 mutation 混在一个函数里；
- 没保留 naive 实现作为 differential oracle；
- 通过不断增加边界分支来补救结构问题。

改进目标不是「以后绝不发散」，而是让发散尽快收敛：每提出一个优化想法，先写它需要支持的查询、状态不变量和最小数据结构。

## 工作流程

1. 保留一个能证明正确的 naive reference。
2. 用 profile 确认热点，而不是凭直觉优化。
3. 写 source of truth、derived cache 和不变量。
4. 明确一次操作影响的对象集合。
5. 先实现最粗但可证明正确的增量版本。
6. 用小例子、边界用例、随机 invariant check 和 differential test 验证。
7. 只有验证通过且仍有性能问题时，才降低更新粒度。

## 自测

看到下面的代码时，应该先停下来重做设计，而不是继续加 `if`：

```python
if at_start and at_end:
    ...
elif at_start:
    ...
elif at_end:
    ...
```

回答：

1. 这里是否有两个职责混在一起？
2. 是否能改写成「删除旧对象完整贡献 + 加入新对象完整贡献」？
3. 是否有一个 source of truth 可以从中重建缓存？
4. 反向索引回答的是未来真正要问的问题吗？
5. 是否有一个 naive oracle 可以逐轮对照？

## 一句话

> 先问「我需要快速回答什么查询」，再设计索引；先问「哪个状态是真实的」，再设计缓存；先选能证明正确的优化粒度，再追求极致速度。
