# 卡 02 · `[陷阱]` `in` 消耗迭代器：同一个 zip 不能用两次

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘
- **标记**：`[陷阱]` —— 不是原始 bug，是**修 [卡 01](01-zip-in-container-membership.md) 时容易新引入**的错误

## 背景

[卡 01](01-zip-in-container-membership.md) 的病灶是 `max_pair in [zip(w, w[1:])]`（多了一层 list）。
自然的修法是把 zip 存进变量再用：`pairs = zip(w, w[1:])`，然后在函数里多处引用 `pairs`。
这一步就踩进本卡。

## 正面 —— 预测输出

```python
t = (b'a', b'b', b'c')
pairs = zip(t, t[1:])
print((b'a', b'b') in pairs)
print((b'b', b'c') in pairs)
print((b'a', b'b') in pairs)
```

## 答案 —— `True` / `True` / `False`

`in` 对迭代器是**边消耗边比较**：第 1 次匹配到 `(a,b)` 就停在那儿；第 2 次从剩下的
`(b,c)` 继续找，命中；第 3 次迭代器已耗尽，返回 `False`。

要复用就物化成 `set` / `list`，或者每次现场重建 `zip`。

## 自测

```sh
python3 -c "t=(b'a',b'b',b'c'); p=zip(t,t[1:]); print((b'a',b'b') in p, (b'b',b'c') in p, (b'a',b'b') in p)"
```

## 手写要点

`zip` / `map` / `filter` / 生成器都是一次性的。**一个变量名如果要被读两次以上，它就不该是迭代器。**

反过来，只读一次的场合，生成器优于 list —— 见
[卡 15](15-setdefault-as-counter.md) 里 `max_pair in zip(k, k[1:])` 短路省分配的用法。

## 相关卡

- [卡 01](01-zip-in-container-membership.md) —— 本卡的上游 bug
- [卡 13](13-two-fixes-stacked.md) —— 另一个「修 bug 时引入新 bug」的例子
