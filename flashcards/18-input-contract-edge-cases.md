# 卡 18 · `[迁移]` 输入契约的四个漏洞：空列表 / 重叠前缀 / 隐式 encoding / 参数过小

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` review，四条**实测**结果
- **标记**：`[迁移]` —— 测试只喂 `special_tokens=["<|endoftext|>"]` + 合理 `vocab_size`，全部躲过

## 背景

`train(input_path, vocab_size, special_tokens)` 的三个参数都没有任何校验。
`prepare_docs` 用 `regex.split("|".join(escaped), corpus)` 按 special token 切文档。

## 正面 —— 预测四个场景的行为

```python
# 语料 = "hello hello world"
prepare_docs(p, [])                                # A
train(p, 400, ["<|endoftext|>"])                   # B
len(train(p, 10, ["<|endoftext|>"])[0])            # C
prepare_docs(q, ["<|eot|>", "<|eot|><|eot|>"])     # D  (语料 = "a<|eot|><|eot|>b")
```

## 答案（实测）

| | 场景 | 实际结果 |
|---|---|---|
| A | `special_tokens=[]` | `"\|".join([])` == `""` → **空 pattern 把每个字符都切开**：`['', 'h', 'e', 'l', 'l', 'o', ' ', 'h', ...]` ❌ |
| B | `vocab_size` 大于语料能支撑的合并数 | `ValueError: max() iterable argument is empty` ❌（详见 [卡 05](05-loop-assumes-resource-suffices.md)） |
| C | `vocab_size=10`，小于初始 vocab 的 257 | **静默返回 257 项**，比请求的多 —— `range(257, 10)` 是空的，循环一次不跑 ⚠️ |
| D | 重叠 special token，短的排在前 | `['a', '', 'b']` —— 这次结果碰巧无害，但 **regex 交替是从左到右首次匹配**，短的先命中 ⚠️ |

## 四条修法

1. **A**：`if not special_tokens: return [corpus]`（或直接 `docs = [corpus]`）
2. **B**：主循环开头 `if not pair_counter: break`
3. **C**：`if vocab_size < len(vocab): raise ValueError(...)` —— 静默违反契约比报错更坏
4. **D**：**按长度降序排**再拼 pattern，让最长匹配优先：

   ```python
   parts = sorted(special_tokens, key=len, reverse=True)
   pattern = "|".join(regex.escape(s) for s in parts)
   ```

   D 现在无害，但**后面实现 `Tokenizer.encode` 时一定会咬人**：
   作业要求 special token 不可被拆分，若 `<|eot|>` 排在 `<|eot|><|eot|>` 前面，
   长 token 就永远匹配不到。

## 附加：隐式 encoding

`open(input_path)` 没写 `encoding=`，用的是 locale 的 `getpreferredencoding(False)`。
macOS 上是 UTF-8 所以一直没暴露，但换机器/换 locale 会读出乱码或抛
`UnicodeDecodeError`。BPE 全程按 UTF-8 字节工作，**必须显式** `encoding="utf-8"`。

## 自测

```sh
python3 -c "
import regex
print(regex.split('|'.join([]), 'hello')[:6])                    # A
print(regex.split('|'.join(['<\|eot\|>','<\|eot\|><\|eot\|>']), 'a<|eot|><|eot|>b'))  # D
print(list(range(257, 10)))                                       # C：空循环
"
python3 -c "import locale; print(locale.getpreferredencoding(False))"
```

## 手写要点

三条判据：

1. **`"sep".join(可能为空的列表)` 得到空串**，而空串作为 regex / 分隔符 / 前缀，
   语义通常是「匹配一切」而不是「什么都不做」。所有 `join` 前先问：这个列表可能为空吗？
2. **regex 交替 `a|b` 是从左到右首次匹配，不是最长匹配。** 候选有前缀关系时必须
   按长度降序排。
3. **静默违反契约比抛异常更坏。** 「请求 10，返回 257」不会有人发现，
   直到下游按 `vocab_size` 建 embedding 矩阵时形状不匹配。

## 相关卡

- [卡 05](05-loop-assumes-resource-suffices.md) —— 场景 B 的完整卡
- [卡 06](06-ord-vs-encode.md) / [卡 17](17-read-whole-file-scale-wall.md) —— 编码边界的其它坑
