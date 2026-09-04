# 卡 05 · `[迁移]` 「按目标数量循环」隐含假设资源够用：空 Counter 上取最大值会炸

- **来源**：`cs336_basics/p7_bpe_example.py` 复盘；`cs336_basics/p9_bpe_tokenizer_training.py` 重构前也有此问题，现已用空计数器 `break` 修复
- **标记**：`[迁移]` —— 玩具语料上实测要合并 12 次、`len(vocab)` 到 **269** 才耗尽 pair，
  而 example 写死 264（7 次合并），侥幸没触发

## 背景

BPE 主循环写成 `for _ in range(len(init_vocab), vocab_size)` 或 `while len(vocab) < vocab_size`
—— 这类「按目标数量循环」的写法，隐含假设「语料里的 pair 一直够用」。
当所有 word 都已被合并成单个 token 时，没有任何相邻 pair 可数，`pair_counter` 为空。

## 正面 —— 预测两个输出

```python
from collections import Counter
print(Counter().most_common(1)[0][0])   # A
print(max(Counter().values()))          # B
```

## 答案

- A —— `IndexError: list index out of range`。`most_common(1)` 返回的是 list，空 Counter 上就是 `[]`，`[0]` 直接炸。
- B —— `ValueError: max() iterable argument is empty`。

注意这两个是**同一个 bug 的两种面孔**：从 `most_common` 改写成 `max(...)` 之后，
错误类型会从 `IndexError` 变成 `ValueError`，很容易被误认为是新问题。

实测 `p9_bpe_tokenizer_training.py`：拿 `"hello hello world"` 当语料、`vocab_size=400`，
就会在第 84 行抛 `ValueError: max() iterable argument is empty`。

主循环开头必须有终止条件：

```python
if not pair_counter:
    break
```

## 自测

```sh
python3 -c "from collections import Counter; print(Counter().most_common(1)[0][0])"
python3 -c "from collections import Counter; print(max(Counter().values()))"
```

## 手写要点

写循环时同时写两个出口：**目标达成**，和**资源耗尽**。
只写前者 = 把「输入一定足够」当成了不需要检查的前提。

## 相关卡

- [卡 18](18-input-contract-edge-cases.md) —— 同一个函数的其它输入契约漏洞（空 special_tokens、encoding、vocab_size 过小）
