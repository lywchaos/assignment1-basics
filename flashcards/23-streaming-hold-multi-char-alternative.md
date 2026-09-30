# 卡 23 · `[迁移]` 流式切分切进多字符 alternative：只 hold 最后一个 match 不够

- **来源**：`cs336_basics/p11_tokenizer.py` 的 `encode_iterable`；在 22MB 的 `TinyStoriesV2-GPT4-valid.txt` 上对拍 `encode_iterable` vs `encode` 时发现（差 14/5,465,883 个 token）
- **标记**：`[迁移]` —— fixture（3.8KB）和既有测试全绿，真实规模才咬人

## 背景

流式实现的做法是：每读一个 chunk，把 buffer 里**最后一个** PAT match 留到下一轮（其余立刻定稿），
依据是「只有最后一个 match 可能被后续文本改变」。这个假设对**单字符起始 + 最大 run** 的分支成立，
但 PAT 里有一个多字符 alternative：

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
              ^^^^^^^^^^^^^^^^^^ 3 字符的 contraction
```

## 正面 —— 预测两段输出（tiny tokenizer 把 `'re` 合成一个 token）

```python
# vocab 含全部 256 字节，merges 含 (b"'", b"r"), (b"'r", b"e")
tok.encode("a'reb")                            # 整段
list(tok.encode_iterable(["a'r", "eb"]))       # 切在 "'r" 和 "e" 之间
```

## 答案

- 整段：`['a', "'re", 'b']`
- 流式（旧实现）：`['a', "'", 're', 'b']` —— **`"'"` 被提前定稿，contraction 拆开了**

切在 `"a'r"` 时，slice 的分词是 `a | ' | r`：`'` 无法完成 `'re`，落到 punct 分支单独成 match，
`r` 才是最后一个 match；旧规则只 hold `r`，于是 `'` 被 emit。

受影响的只有 **2 字符后缀**的三种 contraction：

| 后缀 | cut 在 `'` 与第 2 个字符之间（如 `"a'l" | "lb"`） | 原因 |
|---|---|---|
| `s` `t` `d` `m` | 不触发 | 截断处 `'` 就是最后一个 match，已经被 hold |
| `ll` `ve` `re` | 触发 | `'` 是倒数第二个 match |

## 修法

step 2 保留**最后 2 个** match（不是 1 个）。为什么 2 是紧的上界：contraction 是 PAT 里唯一的
多字符 alternative，被截断后最多拆成 `'` + 残留字母两个 match；其余分支（字母/数字/标点 run、
空白 run）截断后的残缺部分都落在最后一个 match 上。**若 PAT 改了，这个「2」必须重新推导。**

## 为什么既有测试没抓到

1. 官方 `test_encode_iterable_tinystories_matches_tiktoken` 只用 3.8KB 样本对拍；
   5M 那个测试只测内存、不比较值；
2. 之前 fuzz 的盲区：带 special 时 `hold=12`，短文本整段被 carry，step 2 的截断几乎不执行；
3. **分词不同不一定改变 ID**（`'` + `re` 可能恰好和 `'re` BPE 出相同结果），随机 fuzz 命中率低。
   真实 22MB 数据里组合足够多，做一次 stream-vs-whole 的逐 ID 对拍就暴露了。

## 自测

```sh
uv run python - <<'PY'
import itertools
from cs336_basics.p11_tokenizer import Tokenizer

vocab = {i: bytes([i]) for i in range(256)}
for offset, merged in enumerate([b"'r", b"'re", b"'l", b"'ll", b"'v", b"'ve"]):
    vocab[256 + offset] = merged
merges = [(b"'", b"r"), (b"'r", b"e"), (b"'", b"l"), (b"'l", b"l"), (b"'", b"v"), (b"'v", b"e")]
tok = Tokenizer(vocab, merges, None)
for length in range(5):
    for chars in itertools.product("a'lrve", repeat=length):
        text = "".join(chars)
        want = tok.encode(text)
        for cut in range(len(text) + 1):
            assert list(tok.encode_iterable([text[:cut], text[cut:]])) == want, (text, cut)
print("ok")
PY
```

## 手写要点

1. 「只有最后一个 match 会变」是**需要按文法逐条验证的假设**，不是常识：多字符 alternative
   被截断会把一个 match 拆成两个。
2. hold 的边界由 regex 分支结构决定：先列出「哪些分支的 match 可能跨过截断点」，取最坏拆分 match 数。
3. 判断流式是否等价必须做**字节/ID 级对拍**；「没报错、结果看着合理」不算证据。

## 相关卡

- [卡 24](24-whitespace-runs-cross-chunks.md) —— 另一条非局部来源：空白 run 的 lookahead
- [卡 25](25-test-green-is-not-testing.md) —— 为什么官方测试和第一版 fuzz 都漏了它
