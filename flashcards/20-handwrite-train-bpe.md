# 卡 20 · `[手写]` 综合手写卡 —— `train_bpe`（真语料版）

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py`，对应 `tests/test_train_bpe.py`
- **标记**：`[手写]` —— 把 [卡 10](10-mixed-token-representation.md)–[卡 19](19-quadratic-training-wall.md) 串起来的验收题
- **前置**：先过 [卡 08](08-handwrite-toy-bpe.md)（玩具语料版）

## 题面

不看参考实现，手写：

```python
def train(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```

要点：按 special token 切文档 → GPT-2 正则 pre-tokenize → 统计 pre-token 频次 →
迭代合并最高频 pair 直到 `len(vocab) == vocab_size`。

## 自查清单

★ = 会导致 `tests/test_train_bpe.py` 失败，必须过；其余为规模/健壮性项。

**表示层**

1. ★ 序列元素**全程**是 `bytes`（不是 `int`、不是嵌套 tuple）？合并产物用 `a + b` 拼接？—— [卡 10](10-mixed-token-representation.md)
2. ★ 单字节用 `bytes([b])` 而**不是** `bytes(b)`？（后者是零填充）—— [卡 11](11-bytes-int-zero-fill.md)
3. ★ str → bytes 走 `encode("utf-8")`，没有 `ord`？—— [卡 06](06-ord-vs-encode.md)
4. ★ `vocab` 存合并后的 token，`merges` 存 pair 且有序？—— [卡 07](07-vocab-vs-merges.md)
5. 给 `Word = tuple[bytes, ...]` / `Pair` 起了类型别名，`merges: list[Pair] = []` 有注解？—— [卡 16](16-annotation-without-enforcement.md)

**合并逻辑（最容易错的地方）**

6. ★ 扫描循环用 `while i < len(seq)`，最后一个元素**不会丢**？—— [卡 12](12-loop-drops-tail.md)
7. ★ 而且**没有重复 append**（没把两种修法叠加）？—— [卡 13](13-two-fixes-stacked.md)
8. ★ 写了三行断言验 `merge_word`，**在跑 pytest 之前**？—— [卡 14](14-pure-function-deserves-asserts.md)

**统计与选择**

9. ★ pair 计数在 `Counter` / `defaultdict` 里累加，不是 dict 推导式？—— [卡 03](03-dict-comprehension-collapses-keys.md)
10. ★ tie-break：频次优先 + **bytes 字典序**取大，且筛选早于截断？—— [卡 04](04-tiebreak-three-stages.md)、[卡 09](09-truncate-before-filter.md)
11. `zip` 结果没有被复用第二次？—— [卡 02](02-iterator-consumed-by-in.md)

**契约与规模**

12. `pair_counter` 为空时 `break`？—— [卡 05](05-loop-assumes-resource-suffices.md)
13. `special_tokens=[]`、`vocab_size < 257`、special token 前缀重叠都处理了？`open(..., encoding="utf-8")`？—— [卡 18](18-input-contract-edge-cases.md)
14. 分块 + `multiprocessing` pre-tokenize，没有 `f.read()` 整个文件？—— [卡 17](17-read-whole-file-scale-wall.md)
15. pair 计数是**增量更新** + 倒排索引，不是每轮全量重算？—— [卡 19](19-quadratic-training-wall.md)

## 验收 oracle

```sh
just test-bpe            # 或 uv run pytest tests/test_train_bpe.py
```

三条测试各自的作用（**知道每条在测什么，比知道它们全绿更重要**）：

| 测试 | 真正在验什么 | 盲区 |
|---|---|---|
| `test_train_bpe` | `merges` 与参考实现**逐条相等** —— 表示层、tie-break、合并逻辑全在这条里 | 报错是 5000 行 bytes diff，定位能力极差（[卡 14](14-pure-function-deserves-asserts.md)） |
| `test_train_bpe_speed` | 130KB 语料 1.5s 内 | O(V×N) 实现也只用 1.0s，**基本区分不出复杂度**（[卡 19](19-quadratic-training-wall.md)） |
| `test_train_bpe_special_tokens` | 5MB 语料上 special token 不被合并进其它 token | 只喂了 1 个 special token（[卡 18](18-input-contract-edge-cases.md)） |

**逐条清单里 1、2、6、7 四项，任一错都表现为「5000 行 diff」** —— 所以第 8 项
（先写三行断言）是整张清单里回报最高的一条。

## 相关卡

- [卡 08](08-handwrite-toy-bpe.md) —— 玩具语料版，先过那张
- [卡 00](00-meta-lessons.md) —— 两次复盘的元教训汇总
