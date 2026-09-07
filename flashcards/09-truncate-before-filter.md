# 卡 09 · `[通用]` 通用反模式：截断早于筛选

- **来源**：由 [卡 04](04-tiebreak-three-stages.md)（BPE tie-break）抽象而来
- **标记**：`[通用]` —— 跨领域的同型 bug，不限于 BPE

## 正面 —— 下面四段代码出自四个不同领域，错误是同一个。是什么？

```python
# 1. Counter top-k
max(p for p, _ in counter.most_common(1))

# 2. 排序 + 切片
best = [x for x in sorted(items, key=score)[:1] if x.valid]

# 3. SQL
SELECT * FROM (SELECT * FROM t LIMIT 10) WHERE flag = 1 ORDER BY score DESC

# 4. 检索 / 推荐
hits = index.search(q, top_k=1)
return [h for h in hits if h.score > threshold]
```

## 答案 —— 四个都把**截断放在了筛选/排序之前**

于是后面那个看似严谨的 `max` / `if valid` / `ORDER BY` / 阈值过滤，作用在一个已经只剩 1 条的集合上
—— **退化成空操作**。

这类 bug 的特征：

- 代码里**看得见**正确的语义（`max`、`ORDER BY`、阈值），review 时眼睛直接滑过去
- 静态检查、类型检查全 pass（类型完全正确，错的是**集合基数**）
- 小数据上往往碰巧对（只有一个候选时，两种写法等价）

## 判据

每看到一个聚合/选取操作，把手指放在它的输入上问：

> **这个集合里现在有几个元素？上一行是不是已经把它删小了？**

口诀：**先算全集 → 再排序/筛选 → 最后才截断。** 截断永远是流水线的最后一道工序。

## 手写要点

`[:1]` / `LIMIT` / `top_k=1` / `most_common(1)` 这些写法出现时，
先确认它后面没有任何还想对「多个候选」做事的代码。

## 相关卡

- [卡 04](04-tiebreak-three-stages.md) —— 本卡的具体来源（BPE 第 1 个 merge 选错）
