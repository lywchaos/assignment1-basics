# 卡 30 · `[通用]` 比较两个数字前先隔离变量；内存测「状态」不测「噪声」

- **来源**：p12 (b) 的直觉答案 + p11 常数内存的测量过程
- **标记**：`[通用]`

## 误区一：语料相似 ⇒ 压缩率相近

直觉答案：「OWT 和 TinyStories 都是纯英文，分布差别不至于离谱，压缩率应该差不多」。

实测：同一份 OWT 样本，OWT tokenizer **4.704 bytes/token**，换成 TS tokenizer **3.199**（差 32%）。
主因不是文本分布，而是**tokenizer 词表**（32K vs 10K：OWT 的长词/代码/HTML/mojibake 在 10K 词表里
大量没有对应 token）。special token 两边一致，根本不是变量。

判据：比较两个数之前先列出**所有不同的自变量**（tokenizer、语料规模/噪声、special 集合、采样方式），
再看哪个量级能解释差异。

## 误区二：跑完 RSS 涨了 ⇒ 内存不常数

第一次流式跑 5M 文件，RSS 涨了 ~1.2MB，看着像累积。实际是 CPython 分配器保留 arena：

```
round 1: RSS delta = 1244 KB
round 2: RSS delta =   32 KB
round 3: RSS delta =    0 KB
```

证明「算法状态有界」要直接量**算法持有的量**（实测 max buffer = 4 chars、max pretoken = 23 bytes），
而不是进程 RSS 或 tracemalloc 峰值（百万级短命对象的分配元数据会把峰值放大）。
真实数据点：`RLIMIT_AS = RSS + 1MB` 下，连 200KB 的 `"x" * n` 都会 MemoryError ——
`RLIMIT_AS` 量的是地址空间，不是 RSS。

## 手写要点

1. 任何「比较」结论，先写自变量清单；「都差不多」不是论据。
2. 任何「资源」结论，先分清「算法持有」与「分配器保留」；多轮测量看增量是否归零。
3. 结论要归因到**可干预的主变量**：压缩率差 → 换 tokenizer；RSS 涨 → 换测量方法。

## 相关卡

- [卡 25](25-test-green-is-not-testing.md) —— 限制类测试先放 canary
- [卡 26](26-long-job-observability.md) —— 长任务的可度量性
