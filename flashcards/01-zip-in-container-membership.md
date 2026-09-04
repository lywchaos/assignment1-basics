# 卡 01 · `[必错]` zip 对象塞进容器后，`in` 判定恒为 False

- **来源**：`cs336_basics/p7_bpe_example.py`（toy BPE 训练）复盘
- **标记**：`[必错]` —— 在玩具语料上就产生错误输出

## 背景

BPE 训练的主循环里，每轮要判断「某个 word 的当前切分里，是否含有本轮要合并的 pair」。
word 表示成 `tuple[bytes, ...]`（例如 `(b'l', b'o', b'w')`），相邻 pair 用
`zip(w, w[1:])` 生成。判断写成了 `if max_pair in [zip(w, w[1:])]`。

## 正面 —— 预测两个输出

```python
t = (b'a', b'b')
print((b'a', b'b') in [zip(t, t[1:])])
print((b'a', b'b') in zip(t, t[1:]))
```

## 答案 —— `False` / `True`

`[zip(...)]` 是「装了一个 zip 对象的 list」，长度为 1，元素类型是 `zip`。拿 tuple 去比一个 zip
对象，永远不相等。去掉方括号后 `in` 走迭代器协议，逐个比较元素，才是想要的语义。

## 真实后果

BPE 主循环里合并分支一次都没进过，`token_counter` 从头到尾没变，每轮都选出同一个
`(b'e', b's')`，vocab 被塞进 7 个重复项。**程序不报错、不崩、跑得很欢** —— `ruff check` 和
`ty check` 也全 pass。

## 自测

```sh
python3 -c "t=(b'a',b'b'); print((b'a',b'b') in [zip(t,t[1:])], (b'a',b'b') in zip(t,t[1:]))"
```

## 手写要点

写完 `in` 判定，问自己：右边那个东西的**元素**是什么类型？和左边同类吗？
凡是「包了一层容器」的迭代器表达式，都要停下来数一遍层数。

## 相关卡

- [卡 02](02-iterator-consumed-by-in.md) —— 本卡的修法如果写成「先存 `zip` 再复用」，会换成另一个更隐蔽的 bug
