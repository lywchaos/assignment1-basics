# 卡 24 · `[迁移]` 空白 run 跨 chunk：按换行拆不保证与整段等价

- **来源**：p11 `encode_iterable` 的边界规则讨论与实测；5M fixture 与 3.8KB fixture 的对比
- **标记**：`[迁移]` —— 3.8KB fixture 恰好不触发，换真实数据/更大文件才显形

## 背景

PAT 前 4 个分支都从单字符/可选空格开始且不含 `\s`，容易得出「按换行切分安全」的结论。
但第 5 个分支带 lookahead：

```python
\s+(?!\S)
```

它让「空白 run 怎么分段」变成**全局信息**：run 后面跟非空白还是文本末尾，决定它被切成几段。

## 正面 —— 预测 findall / token ID 的差异

```python
s1 = "a \nb"      # 空格 + 换行 + 字母
s2 = "a\n\n\nb"   # 三个换行
```

分别比较整段 `findall(PAT, s)` 与按行 findall 再拼接。

## 答案

```
"a \nb"     whole=['a',' ','\n','b']      linewise=['a',' \n','b']        （ID 恰好相同）
"a\n\n\nb"  whole=['a','\n\n','\n','b']   linewise=['a','\n','\n','\n','b']（ID 不同！）
```

- `"a\n\n\nb"`：整段 `[64, 628, 198, 65]`，按行 `[64, 198, 198, 198, 65]`
  —— GPT-2 里 `'\n\n'` 是单个 token（628）；
- 5M fixture（tiktoken）：整段 1,289,382 tokens vs 按行 1,289,994，首个分歧就是 `'\n\n'` vs `'\n'+'\n'`；
- 3.8KB 的 `tinystories_sample.txt` 却 **923 == 923**：max 空白 run 只有 2、没有行尾空格。

空格和换行还不一样：`"a  b"` 里最后一个空格会被 ` ?\p{L}+` 吸进 `" b"`，换行不会被 ` ?` 吸
（` ?` 只匹配字面空格）。所以「哪些空白 token 受影响」要按字符分开看。

## 修法（流式 encoder 的两条 hold 规则）

```
hold = max(len(t) for t in special_tokens) - 1
① 只定稿 start < len(buffer) - hold 的 special（覆盖"可能被更长 special 延长"和"半个 special"）
② 对剩余 regular 文本，只 PAT buffer[:len(buffer)-hold]，
   保留最后 2 个 match（卡 23），其余 emit
③ EOF：yield from self.encode(buffer)
```

常数内存实测：5M 文件上 max buffer = 4 chars、max pretoken = 23 bytes；连跑 3 轮
RSS 增量 `1244KB → 32KB → 0KB`。注意这不是绝对 O(1)：上界是
`max(单个 chunk, 单个 pretoken, max_special_len)`；病态输入（一整个 100MB 的字母 run，
或 iterable 一次 yield 100MB）下 hold 会很大。

### 两个容易漏的 special 情况

- **完整 special 不在「最末位置」也可能被延长**：specials = `{<|eot|>, <|eot|>ab}`，
  chunk1 = `"x<|eot|>a"`。`<|eot|>` 完整且后面还有 `a`，但 chunk2 = `"b"` 时整段应匹配
  `<|eot|>ab`。定稿判据不是「是否在末尾」，而是「match 的 start 之后至少还有
  `max_special_len` 个字符」。
- **chunk 边界把 special 切两半**：`"x<|eot"` 里没有完整 match，整段会被当普通文本；
  必须保留末尾 `max_special_len - 1` 个字符。

### 附带坑：不要对任意前缀调 encode

把「已定稿前缀」丢回 `self.encode(prefix)` 重跑看着省事，但 prefix 末尾会被当成文本末尾，
`\s+(?!\S)` 与 ` ?` 的附着会变：

```
"a\n\n\n\nb"  整段                              [64, 628, 198, 198, 65]
              encode("a\n\n\n\n") + encode("b")  [64, 628, 628, 65]
```

定稿时应该直接 emit 已经算出的 match（或只在「以完整 special 结尾」的边界调 encode）。

## 自测

```sh
uv run python - <<'PY'
from pathlib import Path
from cs336_basics.p11_tokenizer import Tokenizer

tok = Tokenizer.from_files(
    "artifacts/p9_tinystories/vocab.json",
    "artifacts/p9_tinystories/merges.txt",
    ["<|endoftext|>"],
)
text = Path("tests/fixtures/tinystories_sample_5M.txt").read_text(encoding="utf-8")
whole = tok.encode(text)
linewise = [i for line in text.splitlines(keepends=True) for i in tok.encode(line)]
print(len(whole), len(linewise), whole == linewise)   # 应输出 False
PY
```

## 手写要点

1. 判断 regex 能否流式，先找**带 lookahead / 跨字符前缀**的分支；它们定义了非局部边界。
2. 空白 run 的分段取决于「后面还有没有非空白」——chunk 末尾的空白必须 hold 到下一段，
   不能当作文本末尾处理。
3. 「按行/按块拆」是**实现细节**，不是正确性保证；唯一判据是与整段 `encode` 逐 ID 相等。

## 相关卡

- [卡 23](23-streaming-hold-multi-char-alternative.md) —— contraction 截断
- [卡 17](17-read-whole-file-scale-wall.md) / [卡 25](25-test-green-is-not-testing.md) —— 规模与测试有效性
