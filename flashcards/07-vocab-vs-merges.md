# 卡 07 · `[必错]` vocab 存「合并后的 token」，merges 存「pair」—— 两个容器职责不同

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘
- **标记**：`[必错]` —— 把 pair 塞进 vocab 是真 bug；`merges` 完全没被记录，
  等于训练结果里最关键的一半丢了

## 背景

BPE 训练的返回值有两个容器，作业的 `tests/adapters.py` 里签名是硬要求：

```python
def run_train_bpe(...) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

原代码只维护了一个 `vocab` list，且 `vocab.append(max_pair)`。

## 正面

选出 `max_pair = (b'e', b'st')` 之后，vocab 里该追加什么？merges 里该追加什么？两者的类型分别是什么？

## 答案

- **`vocab`**：追加**合并后的新 token**，`max_pair[0] + max_pair[1]` → `b'est'`，
  类型 `dict[int, bytes]`（id → token bytes）。
- **`merges`**：追加**pair 本身** `(b'e', b'st')`，类型 `list[tuple[bytes, bytes]]`，
  且**顺序有意义** —— encode 时要按同样顺序重放这些合并。

原代码 `vocab.append(max_pair)` 把 pair 塞进了 vocab，于是一个 list 里混了三种类型：
`str`（`"<|endoftext|>"`，还该是 `bytes`）、`bytes`（256 个字节）、`tuple[bytes, bytes]`（pair）。

## 手写要点

**两个容器职责不同就绝不合并。** 写之前先把每个容器的元素类型写成注解 ——
**类型写不出来 == 设计还没想清楚。**

## 相关卡

- [卡 10](10-mixed-token-representation.md) —— 「一个容器里混两种元素类型」的升级版：
  同一个位置上时而是 `int` 时而是 `tuple`，第 2 轮才炸
- [卡 16](16-annotation-without-enforcement.md) —— 注解写对了但没人对账，等于没写
