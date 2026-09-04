# 卡 21 · `[流程]` 类型驱动编程：先给领域概念命名，再写函数

- **来源**：重构 `cs336_basics/p9_bpe_tokenizer_training.py` 时，从基本类型注解提升为 `Word` / `Pair` / `WordCounts`
- **标记**：`[流程]` —— 关注如何用类型表达设计意图，而不只是让 typechecker 通过

## 背景

BPE 训练里有三个核心概念：

- 一个 pre-token 的当前切分，例如 `(b"l", b"o", b"w")`
- 一次候选合并，例如 `(b"l", b"ow")`
- pre-token 到语料频次的映射

它们当然可以直接写成基本容器类型，但这样只能看见**数据形状**，看不见**领域角色**。

## 正面 —— 比较两版签名

```python
# v1：只使用基本类型

def apply_merge(
    counts: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]: ...


# v2：先建立领域词汇
Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]
WordCounts = dict[Word, int]


def apply_merge(counts: WordCounts, pair: Pair) -> WordCounts: ...
```

问题：

1. 对 typechecker 而言，v1 和 v2 的约束能力有区别吗？
2. 对读代码和继续实现的人而言，区别是什么？

## 答案

1. **基本没有区别。** 普通类型别名是 structural alias；`Pair` 会展开成
   `tuple[bytes, bytes]`，不会产生一种新的运行时类型。
2. **意图密度有明显区别。** v1 只告诉你「这是某种 tuple 和 dict」；v2 告诉你
   「这个参数是一个 BPE pair」「返回值仍是 word frequency table」。函数签名本身已经在描述算法。

类型别名的核心价值不是少打几个字符，而是给领域概念建立一套**受检查的词汇表**：

```python
Word = tuple[bytes, ...]       # 不变量：一个 word 的每个当前 token 都是 bytes
Pair = tuple[bytes, bytes]     # 不变量：一次 merge 恰好有左右两个 token
WordCounts = dict[Word, int]   # 不变量：key 是 Word，value 是语料频次
```

有了这三个词，再设计数据流就变成：

```text
str document
  -> WordCounts
  -> choose Pair
  -> WordCounts
  -> (vocab, list[Pair])
```

这就是「基于类型编程」最实用的版本：**先写出领域名词和函数边界，再让函数体去满足它们。**

## 为什么它能帮助本次 BPE 实现

原始 bug 的根因是表示层不统一：pre-token 序列起初含 `int`，合并后又塞入嵌套 tuple。
如果整条调用链从一开始都使用 `WordCounts -> Pair -> WordCounts`，就更容易在写实现时追问：

> `merge_word` 返回的还是 `Word` 吗？它的每个元素仍然都是 `bytes` 吗？

相比之下，裸 `dict` / `tuple` 容易让人只关注容器操作，而忘了容器在领域里代表什么。
这和 [卡 16](16-annotation-without-enforcement.md) 是一体两面：类型别名负责表达不变量，
精确的端到端注解负责让 typechecker 真正检查这个不变量。

## 重要边界 —— 类型别名增加意图，不增加 nominal 隔离

预测下面代码能否通过静态类型检查：

```python
Pair = tuple[bytes, bytes]
ByteRange = tuple[bytes, bytes]


def merge(pair: Pair) -> None: ...


byte_range: ByteRange = (b"a", b"z")
merge(byte_range)
```

**答案：能通过。** `Pair` 和 `ByteRange` 都展开成同一个
`tuple[bytes, bytes]`，typechecker 认为它们兼容。

选择工具时按需要的强度递进：

| 需求 | 工具 | 示例 |
|---|---|---|
| 给重复出现的结构命名、提高可读性 | 类型别名 | `Pair = tuple[bytes, bytes]` |
| 区分底层同为 `int` 的 ID | `NewType` | `TokenId = NewType("TokenId", int)`、`WordId = NewType("WordId", int)` |
| 字段有不同角色，需要按名字访问 | `NamedTuple` / `dataclass` | `Merge(left: bytes, right: bytes)` |
| 构造时必须验证复杂不变量 | 封装类 + 校验 | 禁止空 token、限制状态转换 |

本例用普通别名正合适：pair 的左右位置已经由 tuple 顺序表达，算法又依赖 tuple/bytes 的字典序，
没有必要为了「看起来类型更强」引入包装对象。

## 自测

把上面的 `Pair` / `ByteRange` 例子交给 `ty` 或 `mypy`：它会通过。
然后把两个 ID 都写成裸 `int`，再改成两个 `NewType`，观察后者如何阻止参数传反：

```python
from typing import NewType

TokenId = NewType("TokenId", int)
WordId = NewType("WordId", int)


def lookup(token_id: TokenId) -> bytes: ...


word_id = WordId(3)
lookup(word_id)  # typechecker 应报错
```

## 手写要点

看到以下信号时，先停下来定义领域类型：

1. 同一个较长的容器类型在多个签名里重复出现。
2. 两个参数底层形状相同，但业务角色不同。
3. 一个函数的返回值会直接成为另一个函数的输入，需要维持跨函数不变量。
4. 读签名时只能说出「这是 dict/tuple」，说不出「它在算法里是什么」。

命名要按**领域角色**而不是实现细节：`Word` / `Pair` / `WordCounts` 是好名字；
`BytesTuple` / `TupleDict` 只是把语法又念了一遍，没有增加意图信息。

## 相关卡

- [卡 07](07-vocab-vs-merges.md) —— 两个容器职责不同，先写清元素类型
- [卡 10](10-mixed-token-representation.md) —— 类型不变量被破坏后，第 2 轮才爆炸
- [卡 16](16-annotation-without-enforcement.md) —— 裸 `dict` 让精确注解失去检查能力
- [卡 14](14-pure-function-deserves-asserts.md) —— 类型只能保证形状，边界行为仍需小测试
