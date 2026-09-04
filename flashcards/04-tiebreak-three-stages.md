# 卡 04 · `[必错]` tie-break 的三段式阶梯：不知道 → 以为做了 → 真做了

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘
- **标记**：`[必错]` —— **本次复盘唯一真正改变了玩具语料输出**的 bug

## 背景

BPE 规格书规定：每轮选频次最高的 pair；**若多个 pair 频次并列，取字典序（bytes 比较）更大的那个**。
在 `low lower widest newest` 语料上，第一轮 `(b'e', b's')` 和 `(b's', b't')` 都出现 9 次
—— 第 1 个 merge 选哪个，直接决定后面 7 步全对还是全错。

## 正面 —— 下面三种写法分别返回什么？

```python
# v1
return counter.most_common(1)[0][0]

# v2
commons = counter.most_common(1)
return max(p for p, _ in commons)

# v3
max_count = max(counter.values())
return max(p for p, c in counter.items() if c == max_count)
```

## 答案 —— BPE 要字典序更大的 `(b's', b't')`。v1 ❌、v2 ❌、v3 ✅

| 版本 | 病灶 | 返回 |
|---|---|---|
| v1 | 不知道要 tie-break。`most_common` 是稳定排序，同 count 按**插入顺序** —— 等于让语料遍历顺序决定答案 | `(b'e', b's')` |
| v2 | **以为**自己 tie-break 了。`most_common(1)` 只返回 1 个元素，在单元素序列上取 `max` 是空操作 | `(b'e', b's')` |
| v3 | 先求最高频次 → 筛出全部并列者 → 字典序取大 | `(b's', b't')` |

**v2 才是最该记的那一格。** 代码里**出现了 `max`**，review 时眼睛会直接滑过去；
静态检查也全 pass。病根一句话：**截断早于筛选**。

## 判据

看到 `max` / `min` / `sorted[0]`，先问：

> **这个 max 的候选集里到底有几个元素？是谁把它变成这么多的？**

## 自测

```sh
python3 -c "from collections import Counter; c=Counter({(b'e',b's'):9,(b's',b't'):9,(b'l',b'o'):7}); print('v1', c.most_common(1)[0][0]); print('v2', max(p for p,_ in c.most_common(1))); m=max(c.values()); print('v3', max(p for p,n in c.items() if n==m))"
```

## 手写要点

规格书里写了平票规则，就必须把规则**显式编码进 key 或筛选条件**，且**筛选必须早于截断**。

更好的写法是把整条规则压进一个 key，让「频次优先、bytes 字典序 tie-break」肉眼可读：

```python
max_pair = max(pair_counter, key=lambda p: (pair_counter[p], p))
```

## 相关卡

- [卡 09](09-truncate-before-filter.md) —— 「截断早于筛选」的通用反模式，跨四个领域的同型 bug
- [卡 10](10-mixed-token-representation.md) —— 表示层从 `bytes` 退化成 `int` 时，
  这条 tie-break 规则会被**悄悄改变语义**（int 比较 vs bytes 字典序）
