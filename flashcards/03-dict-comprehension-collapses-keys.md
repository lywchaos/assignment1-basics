# 卡 03 · `[迁移]` dict 推导式喂给 Counter，会把重复 key 折叠掉

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘
- **标记**：`[迁移]` —— 在玩具语料上**不触发**（实测全 12 轮、两条轨迹下，
  `low / lower / widest / newest` 及其所有中间态**从未**出现 token 内重复相邻 pair，
  故与正确写法等价）；换到真语料才错

## 背景

BPE 每轮要统计全语料的相邻 pair 频次：对每个 word（`tuple[bytes, ...]`）及其出现次数
`count`，把它的每个相邻 pair 都加上 `count`。写法是
`pair_counter.update({pair: count for pair in zip(t, t[1:])})`。

## 正面 —— token 是 `(b'a', b'a', b'a')`、词频 5，预测 pair 计数

```python
from collections import Counter
t = (b'a', b'a', b'a')
c = Counter()
c.update({pair: 5 for pair in zip(t, t[1:])})
print(c)
```

## 答案 —— `Counter({(b'a', b'a'): 5})`，**正确答案是 10**

`zip` 产出了两个相同的 `(b'a', b'a')`，但 dict 推导式后写的 key 覆盖前一个，两次变一次。
`Counter.update(dict)` 本身是「加法」没错，错在**传进去之前就已经丢了信息**。

正确写法 —— 让累加发生在 Counter 里，而不是在 dict 构造里：

```python
for pair in zip(token, token[1:]):
    pair_counter[pair] += count
```

## 阴险之处

`low lower widest newest` 这个语料里没有任何词包含重复相邻 pair，所以玩具例子上
**结果完全正确**，换到真语料（`aaa`、`---`、`...`、`\n\n`）才开始错。

## 自测

```sh
python3 -c "from collections import Counter; t=(b'a',b'a',b'a'); c=Counter(); c.update({p:5 for p in zip(t,t[1:])}); print(c)"
```

## 手写要点

推导式的 key 有可能重复时，就不能用推导式聚合。判据：**key 是不是「位置的函数」而非「值的函数」。**

## 相关卡

- [卡 15](15-setdefault-as-counter.md) —— 同一处计数逻辑的另一种写坏方式（`setdefault`）
