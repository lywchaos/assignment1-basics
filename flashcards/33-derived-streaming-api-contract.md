# 卡 33 · `[流程]` 派生流式 API：签名之外的等价性、隐式依赖与副作用

- **来源**：p11 `encode_iterable` 读码讨论：为什么假边界仍需要 `safe_len`、为什么要保留两个 PAT match，以及实现前该写清哪些契约
- **标记**：`[流程]` —— 从已有批量 API 派生增量 API 时，先写契约和依赖，再写缓冲逻辑

## 正面 —— 只有两个签名，够写实现了吗？

```python
def encode(self, text: str) -> list[int]: ...
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...
```

动手实现前，试着回答：

1. 怎样才算和 `encode` **等价**？chunk 边界能不能充当分词边界？
2. 什么时候读取输入、吐出 ID、处理剩余尾巴？调用者中途停止迭代会怎样？
3. 哪些 `encode` 的内部规则必须保持一致？保留两个 match 是接口保证，还是对 `PAT` 的假设？
4. `str` 字符、UTF-8 字节、PAT match、BPE token ID 是不是同一个计量单位？
5. 哪些对象会被修改？如果迭代时 tokenizer 配置改变，还能保证等价吗？
6. “流式”是否意味着固定内存、每个 chunk 都马上有输出？

再预测一个具体例子：specials 是 `"<|sep|>"` 和 `"<|end|>"`，chunks 是
`["<|sep|>X<|end|", ">"]`。输出已确认的 `<|sep|>` 后，对剩余 `buffer` **整个**跑 PAT，
只保留最后两个 match，能否保证正确？

## 答案 —— 类型是形状，契约还包括语义和时间

**对外的 oracle**：对任意有限的 `str` 块序列，完整消费后所得的 **ID 序列**必须逐项等于
`encode("".join(chunks))`。拼接不额外加分隔符；包括空块、跨块 special、跨块 pre-token。
只比较 `decode` 后的字符串会漏掉不同的 token 切分。

| 层面 | 这里应明确的约定 |
|---|---|
| 输入与结果 | 输入是按原顺序拼接的 `str`，不是独立文档；产出按顺序排列的 `int`，不是“每块一组 ID”。 |
| 生命周期 | 返回 generator，**迭代时**才推进输入；可以多块无输出，EOF 才定稿尾巴。若中途停止，剩余输入未必已读取；输入迭代抛错也不会自动 flush。 |
| 可变状态 / 副作用 | `encode_iterable` 自身不修改 tokenizer 配置，但会消费传入的迭代器。迭代期间要求 tokenizer 配置保持不变：`special_re` / `hold` 预先计算，`self.encode` 又会读取当前配置。 |
| 资源 | 不应为流式处理先拼接全文；但超大 chunk 或超长未完成的 pre-token 会使缓冲增长，不能承诺严格 O(1) 内存或每块即时产出。 |

**向下依赖的具体规则**（不能从 `Iterable[str] -> Iterator[int]` 推出来）：

| 被依赖的规则 | 为什么会影响流式切点 |
|---|---|
| `encode` 对 special 的识别与优先级 | special 是可以单独结算前缀的真边界；匹配重叠 token 时，流式版本必须与 `encode` 的最长优先规则一致。最长长度 `L` 导出字符级 `hold = L - 1`。 |
| `PAT` 的分支、贪心与预读 | 普通文本不能随意截出前缀重新 `encode`。当前 `PAT` 在假边界最多需要保留最后两个 **match**；`"You'r"` 暂时匹配为 `"You", "'", "r"`，补上 `e` 后变为 `"You", "'re"`。**改 `PAT` 就得重新论证这个 2。** |
| BPE 只在一个 pre-token 内合并 | 已确定的 match 可以交给 `_encode_pretoken`；尚未确定的 match 不能提前产出 ID。两个 match 不等于两个字符，更不等于两个 token ID。 |
| 表示单位 | `safe_len` 和 match 位置使用 Python `str` 的字符索引；`to_word` 再转 UTF-8 字节；BPE 的输出才是 token ID。不同单位不能混算。 |

**为什么“真边界处理完，假边界直接扫整个 buffer 并留两个”仍不行？**
第一块中的 `<|sep|>` 可以确认，但剩余 `buffer = "X<|end|"` 仍有半个 special。
`PAT` 会得到 `['X', '<|', 'end', '|']`；只保留末尾两个，就会提前输出 `'<|'`，
以后无法把它还原成完整的 `<|end|>`。所以假边界扫描也需要 `safe_len` 来排除
**未来可能成为 special 的字符**；留两个 match 只解决 **PAT 自身**的未定稿问题。

还有一个不同层次的副作用：`Tokenizer.__init__` 在添加缺失的 special 时会**原地修改传入的 `vocab`**
（`cs336_basics/p11_tokenizer.py:21-29`）。这属于构造器的契约，不应误写成 `encode_iterable` 的运行时副作用。

## 写在哪里？

- **函数 docstring**：写调用者需要知道的等价性、输入前提、惰性消费、EOF、可变状态和资源限制。
  例如：“完整消费的 ID 序列与对拼接全文调用 `encode` 相同；chunk 不是分词边界；
  迭代期间 tokenizer 配置不变；输出可能延迟，内存不保证严格常数。”
- **常数附近的注释**：写实现依赖的推导，例如“最长 special 导出 `L-1` 个字符”和
  “当前 `PAT` 的后缀至多需留两个 match”；别让魔法数只有一句“保守一点”。
- **可执行测试**：用 `encode` 当 oracle，随机切块并逐 ID 对拍；定向覆盖半个 special、
  较短 special 可能延长、`"You'r" | "e"`、空白 run 和 EOF。docstring 不能代替反例与测试。

## 自测

遮住上面的答案，先猜三行输出，再运行：

```sh
uv run python - <<'PY'
import regex
from cs336_basics.p9_bpe_tokenizer_training import PAT
from cs336_basics.p11_tokenizer import Tokenizer

chunks = ["<|sep|>X<|end|", ">"]
tok = Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<|sep|>", "<|end|>"])
print(regex.findall(PAT, "X<|end|")[:-2])
print((regex.findall(PAT, "You'r"), regex.findall(PAT, "You're")))
print(list(tok.encode_iterable(chunks)) == tok.encode("".join(chunks)))
PY
```

预期依次是 `['X', '<|']`、`(['You', "'", 'r'], ['You', "'re"])`、`True`。
第一行是**删掉 `safe_len` 但保留两个 match 会提前输出什么**，不是现有实现的输出。

## 手写要点

1. 派生增量 API 时，先写完整输入的 **oracle 等式**，再问哪部分结果已不受未来输入影响。
2. 签名没写出的东西也属于契约：消费时机、EOF、输入所有权、可变配置、异常和资源上界。
3. 依次列出底层规则的隐式依赖；每个 `2`、`L-1` 都应能指回某条具体规则。
4. **docstring 记承诺，邻近注释记证明，测试记可执行判据。** 三者互补，不能互相替代。

## 相关卡

- [卡 32](32-true-vs-fake-boundary.md) —— 真边界、假边界与两套 hold 的推导
- [卡 23](23-streaming-hold-multi-char-alternative.md) —— 为什么只保留一个 PAT match 会错
- [卡 24](24-whitespace-runs-cross-chunks.md) —— 为什么不能在假边界重新 `encode` 前缀
- [卡 25](25-test-green-is-not-testing.md) —— 逐 ID 对拍与让测试真正触及危险路径
