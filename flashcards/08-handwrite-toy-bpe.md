# 卡 08 · `[手写]` 综合手写卡 —— toy BPE 训练（古法编程）

- **来源**：`cs336_basics/p7_bpe_example.py`，对应 handout §BPE example
- **标记**：`[手写]` —— 不是单点 bug，是把 [卡 01](01-zip-in-container-membership.md)–[卡 07](07-vocab-vs-merges.md) 串起来的验收题

## 题面

不看任何参考，从空文件手写 toy BPE 训练：

```python
CORPUS = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
# 初始 vocab: b"<|endoftext|>" + 256 个单字节, vocab_size = 264 (即 7 次合并)
```

要求返回 `(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])`。

## 自查清单

写完后逐条打勾。★ = 本语料上就会错，必须过；其余为迁移项。

1. ★ 合并分支真的进去过吗？（打一行 print 或断言 `token_counter` 每轮都在变）—— [卡 01](01-zip-in-container-membership.md)
2. ★ 有没有把 `zip` 结果复用第二次？—— [卡 02](02-iterator-consumed-by-in.md)
3. pair 计数是在 Counter 里累加，而非 dict 推导式里？—— [卡 03](03-dict-comprehension-collapses-keys.md)
4. ★ tie-break 写了字典序优先，**且筛选早于截断**？—— [卡 04](04-tiebreak-three-stages.md)
5. `pair_counter` 为空时有 break？—— [卡 05](05-loop-assumes-resource-suffices.md)
6. str → bytes 走的是 `encode("utf-8")`？—— [卡 06](06-ord-vs-encode.md)
7. ★ vocab 存 bytes（不是 pair）；merges 存 pair 且有序返回？—— [卡 07](07-vocab-vs-merges.md)

## 验收 oracle

与 handout 例子一致，7 次合并依次为：

```
1. (b's', b't')      -> b'st'
2. (b'e', b'st')     -> b'est'
3. (b'o', b'w')      -> b'ow'
4. (b'l', b'ow')     -> b'low'
5. (b'w', b'est')    -> b'west'
6. (b'n', b'e')      -> b'ne'
7. (b'ne', b'west')  -> b'newest'
```

最终 pretoken 状态：

```
{(b'low',): 5, (b'low', b'e', b'r'): 2, (b'w', b'i', b'd', b'est'): 3, (b'newest',): 6}
```

**注意第 1 步就是分水岭**：如果你的第一个 merge 是 `(b'e', b's')`，说明
[卡 04](04-tiebreak-three-stages.md) 没过。

## 相关卡

- [卡 20](20-handwrite-train-bpe.md) —— 真语料版的手写卡（`train_bpe`，含 pytest oracle）
