# 卡 28 · `[必错]` encode 阶段 merge 必须按创建顺序：外层扫 pair 会选错对象

- **来源**：p11 `_encode_pretoken` 的实现评审（第一版思路：「构造 list[pair] → 双循环判断 pair == merge → 命中就改 word」）
- **标记**：`[必错]` —— 训练按频率，编码按 rank；选择键换了，但「选哪个」必须显式写对

## 背景

训练每轮选**全局频率最高**的 pair；编码没有频率，只有 merges 的**创建顺序**：
每轮要选「当前相邻 pair 里 rank 最小」的那个。两种「双循环」写法长得像，结果不同：

- 正确：外层遍历 `merges`（rank 顺序），内层找它是否出现在当前 word；
- 错误：外层遍历 word 的相邻 pair，碰到能合的（最靠左）就合。

## 正面 —— 预测两个输出

```python
merges = [(b"b", b"c"), (b"a", b"b")]   # (b,c) 先创建，rank 0
word   = (b"a", b"b", b"c")
```

## 答案

- rank 最小优先：`(b"a", b"bc")` —— 先合 rank 0 的 `(b,c)`
- 外层先扫 pair：`(b"ab", b"c")` —— 先合了最靠左的 `(a,b)`，**错**

## 为什么

合并 rank=i 的 pair 只会产生「包含新 token」的 pair，而它必然 rank>i
（新 pair 只能在那个 token 产生之后被创建）。所以「按 merges 列表顺序扫一遍」等价于
rank 优先级；但「按 word 位置顺序找第一个可合的」不是——位置顺序与 rank 顺序无关。

## 实现与性能

```python
# __init__ 建一次
self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

# 每轮取 rank 最小的相邻 pair，用 merge_word 合并它的所有不重叠出现
pair = min((p for p in zip(word, word[1:]) if p in self.merge_ranks),
           key=self.merge_ranks.__getitem__, default=None)
```

- `word` 是 tuple，不能原地改；`merge_word(word, pair)` 返回新 tuple（这也是它可复用的一面，见[卡 27](27-reuse-policy-vs-primitive.md)）；
- 实测同一个 13 字节 pretoken（GPT-2 全部 5 万条 merges）：rank 查表 **44µs** vs 外层扫 50k merges **15.3ms**（约 349×）；
- 不要把每个 pretoken 丢回 `self.encode()`（会重复 special 扫描 + PAT），直接用已定稿的 match 走 `_encode_pretoken`。

## 自测

```sh
uv run python - <<'PY'
from cs336_basics.p11_tokenizer import Tokenizer

vocab = {0: b"a", 1: b"b", 2: b"c", 3: b"ab", 4: b"bc", 5: b"abc"}
toy = Tokenizer(vocab, [(b"b", b"c"), (b"a", b"b")], None)
print(toy._encode_pretoken((b"a", b"b", b"c")))   # [0, 4] = a + bc
PY
```

## 手写要点

1. 「先做哪个」是算法语义（选择键），不能让它变成循环顺序的副产品；写之前先问：这一步的选择键是什么？
2. 选择键有 O(1) 查表就别每轮全表扫描；merges 有 5 万条，差三个数量级。
3. 同一份 merges 被训练和编码两处消费：训练用频次、编码用 rank；改动要两边都想一遍。

## 相关卡

- [卡 04](04-tiebreak-three-stages.md) —— 选择键的另一面：平票怎么办
- [卡 27](27-reuse-policy-vs-primitive.md) —— 策略（选哪个）与原语（怎么合）分离
