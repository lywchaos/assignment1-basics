# 卡 17 · `[迁移]` `f.read()` 整个语料入内存 —— 测试 fixture 能过，真数据集 OOM

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 的 `prepare_docs`（第 8-10 行）review
- **标记**：`[迁移]` —— fixture 最大 5MB，全部通过；作业后续要求的数据集会直接炸

## 背景

```python
def prepare_docs(input_path, special_tokens) -> list[str]:
    with open(input_path) as f:
        corpus = f.read()                                   # ← 整个文件
    docs = regex.split("|".join(...), corpus)               # ← 同量级的中间 list
    return docs
```

`tests/fixtures/` 里最大的是 `tinystories_sample_5M.txt`（5MB），毫无压力。

## 正面

作业后面要在 TinyStories（**2.1GB**）和 OpenWebText（**11GB**）上训练 BPE。
这段代码的峰值内存大约是文件大小的几倍？为什么？

## 答案

至少 **3–4 倍**，量级上 11GB 语料需要 40GB+：

1. `f.read()` 得到一个 str —— Python 的 str 对 ASCII 文本约 1 字节/字符，但**整个文件都在内存里**
2. `regex.split` 的结果 list 里，所有 doc 加起来又是**一份完整拷贝**
3. `pretokenize` 再为每个 pre-token 建 `tuple[bytes, ...]`
   —— 每个字节变成一个独立的 `bytes` 对象（单字节 `bytes` 有 CPython 缓存，
   但 tuple 本身每个槽位 8 字节指针，**膨胀 8 倍以上**）

## 正确做法（作业明确要求的）

1. 用 `find_chunk_boundaries`（见 `cs336_basics/pretokenization_example.py`）
   把文件按 special token 边界切成 N 块，**只记录 byte offset**，不读内容
2. `multiprocessing.Pool` 每个 worker 只 `seek` + 读自己那一块，返回该块的 `WordCounts`
3. 主进程把各 worker 的 counter 合并 —— 这正是 `merge_counter` 存在的意义
4. 全程只在内存里保留**聚合后的 pre-token 计数**（词表级，远小于语料），而非语料本身

关键洞察：**BPE 训练只需要 pre-token 的频次表，不需要原文。**
一旦聚合完成，原始语料就可以丢掉 —— 内存需求从 O(语料) 降到 O(唯一 pre-token 数)。

## 附带的小问题（同一段代码）

`open(input_path)` 没写 `encoding=` —— 依赖 locale 的 `getpreferredencoding()`。
macOS 上恰好是 UTF-8 所以没暴露，但这是**平台相关**的。见
[卡 18](18-input-contract-edge-cases.md)。

## 手写要点

判据：**「一次读进来」的写法，在 fixture 上永远是对的。** 所以它不会被测试抓到 ——
只会在你真正跑作业要求的数据集时炸。

写 IO 时问一句：**这个函数会被喂多大的输入？测试 fixture 的规模和生产规模差几个数量级？**
本例是 5MB vs 11GB，差 2000 倍。

## 相关卡

- [卡 19](19-quadratic-training-wall.md) —— 同一份代码的另一面规模墙（时间而非空间）
- [卡 18](18-input-contract-edge-cases.md) —— 同一个函数的输入契约漏洞
