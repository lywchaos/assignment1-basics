# 卡 16 · `[流程]` 注解写对了但不被检查 —— 不变量没有守卫

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` review
- **标记**：`[流程]` —— 解释了 [卡 10](10-mixed-token-representation.md) 那个 bug**为什么能藏住**

## 背景

合并 word 的 helper（当时名为 `build_new_seq`，现名 `merge_word`）签名从一开始就写对了：

```python
def build_new_seq(seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
```

但实现里 `seq` 的元素其实是 `int`，合并产物是 `tuple[int, int]`
（[卡 10](10-mixed-token-representation.md)）。**注解和实现完全矛盾，`ty check` 却全 pass。**

## 正面

为什么类型检查器没能发现这个矛盾？

## 答案

因为上游的注解太松，`bytes` 这个约束在整条链路上**没有任何一处被真正断言**：

```python
def pretokenize(doc: str, pat: str) -> dict[bytes, int]:   # ← 错：实际是 dict[tuple[bytes, ...], int]
def merge_counter(counters: list[dict]) -> dict:            # ← 太松：裸 dict，零信息量
```

`train` 从 `merge_counter` 拿到的是裸 `dict`，检查器只知道「key 是某种东西」，
传给合并 helper 时无从比对。**一条链上只要有一环是 `dict` / `Any`，
下游所有精确注解都失去守卫作用。**

## 正确做法 —— 给不变量起名字

```python
Word = tuple[bytes, ...]        # 一个 pre-token 的当前切分；元素永远是 bytes
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]

def pretokenize(doc: str, pat: str = PAT) -> WordCounts: ...
def merge_counter(counters: list[WordCounts]) -> WordCounts: ...
def merge_word(word: Word, pair: Pair) -> Word: ...
def train(...) -> tuple[dict[int, bytes], list[Pair]]: ...
```

顺带：`merges = []` 没有注解，写成 `merges: list[Pair] = []` 才能让返回类型被真正检查。

## 手写要点

三条：

1. **注解和实现打架时，注解通常是对的那个。** 它记录的是写代码时的意图；
   实现是手滑的地方。发现矛盾，别改注解去迁就实现。
2. **类型别名是最便宜的文档。** `Word` 比 `tuple[bytes, ...]` 更能提示「这是一个概念，
   它的元素类型是不变量」。
3. **`ty check` / `mypy` 全 pass 不代表类型是对的** —— 只代表没有**可证明的**矛盾。
   注解越松，检查器越沉默。裸 `dict` / `list` / `Any` 是检查覆盖率的黑洞。

## 相关卡

- [卡 10](10-mixed-token-representation.md) —— 被这个漏洞放过去的真 bug
- [卡 07](07-vocab-vs-merges.md) —— 「类型写不出来 == 设计还没想清楚」
- [卡 14](14-pure-function-deserves-asserts.md) —— 静态检查覆盖不到的地方，用运行时断言补
- [卡 21](21-type-driven-domain-aliases.md) —— 类型别名如何把「数据形状」提升成「领域词汇」
