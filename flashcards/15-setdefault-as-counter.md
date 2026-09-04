# 卡 15 · `[风格]` `dict.setdefault` 当计数器：语义绕 + 实测占 25% 运行时间

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py`（第 20、27、81 行）review
- **标记**：`[风格]` —— 结果正确，但可读性和性能都有代价

## 背景

文件里三处计数逻辑（pretoken 计数、counter 合并、pair 计数）都写成同一个形状：

```python
ret[k] = ret.setdefault(k, 0) + v
```

## 正面 —— 两个问题

1. `d.setdefault(k, 0)` 的语义是什么？用它来读一个可能不存在的 key，问题在哪？
2. `corpus.en` / `vocab_size=500` 下 `cProfile` 显示 `setdefault` 被调用了多少次、占多少时间？

## 答案

1. `setdefault` 的语义是「**如果 key 不存在就插入默认值**，然后返回该 key 的值」
   —— 它是个**写操作**。这里只想读，却用了个会写的方法，而且下一行马上又写一次同一个 key。
   想读就用 `d.get(k, 0)`；想累加就让容器自己管默认值。
2. 实测：**5,688,002 次调用，0.611s / 总 2.464s ≈ 25%**。

```
   ncalls  tottime  cumtime  filename:lineno(function)
        1    1.073    2.464  p9_bpe_tokenizer_training.py:66(train)
      243    0.659    0.823  p9_bpe_tokenizer_training.py:53(apply_merge)
  5688002    0.611    0.611  {method 'setdefault' of 'dict' objects}
```

## 正确写法

```python
from collections import Counter, defaultdict

pair_counter = Counter()          # 或 defaultdict(int)
for pair in zip(k, k[1:]):
    pair_counter[pair] += v       # 一次 __getitem__ + 一次 __setitem__，无多余方法调用
```

`Counter` 还自带 `+` / `update` / `most_common`，`merge_counter` 那个手写函数直接可以删掉
（换成 `sum(counters, Counter())` 或循环 `total += c`）。

> 注意：`Counter.update(dict)` 有它自己的坑，见 [卡 03](03-dict-comprehension-collapses-keys.md)。

## 同一处 review 的另外三条

1. **取 max 的规则应该显式表达。** 现在是「求最大值 → 筛并列 → 取字典序最大」三步 + 两个中间 list：

   ```python
   max_pair = max(pair_counter, key=lambda p: (pair_counter[p], p))
   ```

   一行把「频次优先，并列取 bytes 字典序更大者」这条**最容易错的规则**变成肉眼可读
   —— 见 [卡 04](04-tiebreak-three-stages.md)、[卡 10](10-mixed-token-representation.md)。

2. **白建一个 list 只为做一次 `in`。** `apply_merge` 第 58 行
   `pairs = list(zip(k, k[1:]))` 然后 `if max_pair in pairs` ——
   `if max_pair in zip(k, k[1:])` 就够：生成器、命中即短路、零 list 分配
   （但**只能用一次**，见 [卡 02](02-iterator-consumed-by-in.md)）。

3. `pretokenize(doc, pat)` 的 `pat` 参数所有调用点都传全局 `PAT`，要么给默认值要么去掉。

## 手写要点

热循环里的每个方法调用都要付 CPython 的函数调用开销。**「用一个写方法来读」不只是语义别扭，
它还真的会出现在 profile 的前三行。**

判据：计数/累加场景，第一反应应该是 `Counter` / `defaultdict(int)`，
而不是手工 `get` / `setdefault` 组合。

## 相关卡

- [卡 03](03-dict-comprehension-collapses-keys.md) —— 同一处计数逻辑的另一种写坏方式
- [卡 19](19-quadratic-training-wall.md) —— profile 里排第一的 `train` 自身耗时是更大的问题
