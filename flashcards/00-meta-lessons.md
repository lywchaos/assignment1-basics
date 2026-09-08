# 卡 00 · 元教训汇总

两次复盘（`p7_bpe_example.py` → 卡 01-09，`p9_bpe_tokenizer_training.py` → 卡 10-22）
的横向总结。这张不是「预测输出」型的卡，是复习时最后读的一张。

## 零号事实

这些 bug 里，**没有一个**能被 `ruff check` 或 `ty check` 抓到 —— 所有出错版本静态检查全 pass。
其中大半属于**程序正常退出、输出看起来合理**的静默错误
（卡 [01](01-zip-in-container-membership.md)、[03](03-dict-comprehension-collapses-keys.md)、
[04](04-tiebreak-three-stages.md)、[06](06-ord-vs-encode.md)、
[11](11-bytes-int-zero-fill.md)、[12](12-loop-drops-tail.md)、[13](13-two-fixes-stacked.md)）。

推论：**静态检查是地板，不是天花板。** 它的沉默不构成任何正确性证据 ——
尤其当注解本身太松时（[卡 16](16-annotation-without-enforcement.md)）。

## 一、七条可迁移判据

| 纪律 | 判据 | 相关卡 |
|---|---|---|
| **迭代器纪律** | `zip`/`map`/生成器：数容器层数，且只读一次。要读两次以上就别用迭代器 | [01](01-zip-in-container-membership.md) [02](02-iterator-consumed-by-in.md) |
| **聚合纪律** | 任何计数/累加，问「key 会重复吗」；任何 max，问「平票怎么办」 | [03](03-dict-comprehension-collapses-keys.md) [04](04-tiebreak-three-stages.md) [15](15-setdefault-as-counter.md) |
| **顺序纪律** | 先算全集 → 再排序/筛选 → **最后**截断。看到 `max`，先问它的候选集有几个元素 | [04](04-tiebreak-three-stages.md) [09](09-truncate-before-filter.md) |
| **表示层纪律** | 一个序列的元素类型是**不变量**，从产生到消费不许变；先给领域概念命名，再让函数边界维持它 | [06](06-ord-vs-encode.md) [07](07-vocab-vs-merges.md) [10](10-mixed-token-representation.md) [11](11-bytes-int-zero-fill.md) [21](21-type-driven-domain-aliases.md) |
| **边界纪律** | 变步长循环先列出「退出时 i 可能落在哪些值」；循环要有两个出口（目标达成 + 资源耗尽） | [05](05-loop-assumes-resource-suffices.md) [12](12-loop-drops-tail.md) [13](13-two-fixes-stacked.md) |
| **规模纪律** | 问「测试 fixture 和生产输入差几个数量级」。一次性 read、每轮全量重算，在 fixture 上永远是对的 | [17](17-read-whole-file-scale-wall.md) [18](18-input-contract-edge-cases.md) [19](19-quadratic-training-wall.md) |
| **增量纪律** | 先区分 source of truth 与 derived cache；按未来查询建立反向索引；先做 affected-object 级更新，再考虑 occurrence-level delta | [19](19-quadratic-training-wall.md) [22](22-incremental-cache-convergence.md) |

## 二、最大的一条：修 bug 时最容易造出「更难发现的 bug」

两次复盘各贡献了一种形态：

1. **「不知道要做 X」→「以为自己做了 X」**（[卡 04](04-tiebreak-three-stages.md) 的 v1 → v2）
   代码里已经**出现了 X 的字面痕迹**（`max`），review 时眼睛直接滑过去。
2. **「二选一的修法两个都用上」**（[卡 13](13-two-fixes-stacked.md)）
   同一个不变量有了两个守卫，于是双重生效。

对策：每次修完必须跑一个**能区分各个版本的 oracle**，而不是看「不报错、输出看着像那么回事」。
p7 的 oracle 是「第 1 步 merge 到底是 `(s,t)` 还是 `(e,s)`」；
p9 的 oracle 是 `merge_word`（修复前名为 `build_new_seq`）的[三行断言](14-pure-function-deserves-asserts.md)。

判据补充：**同一个不变量不应该有两个守卫。** 如果有，其中一个必然多余，
而多余的守卫往往不是无害，而是双重生效。

## 三、诊断口诀（按症状查病）

| 症状 | 大概率病因 | 卡 |
|---|---|---|
| 第 1 轮对、第 2 轮炸 | **输出类型 ≠ 输入类型** —— 迭代把自己的输出喂回了输入。查 `[type(x).__name__ for x in seq]` | [10](10-mixed-token-representation.md) |
| 同一段代码时对时错 | 变步长循环**跳过了某个边界值**，边界分支时灵时不灵 | [12](12-loop-drops-tail.md) [13](13-two-fixes-stacked.md) |
| 不报错但结果全错，值看着有规律 | API 语义误用，且误用恰好是**单射**（如 `bytes(104)` → 104 个 `\x00`） | [11](11-bytes-int-zero-fill.md) |
| 小数据全对、大数据才错 | 聚合处的 key 冲突 / 平票 / 编码假设，玩具语料碰巧不触发 | [03](03-dict-comprehension-collapses-keys.md) [04](04-tiebreak-three-stages.md) [06](06-ord-vs-encode.md) |
| 测试全绿但心里没底 | 测试的**盲区**：fixture 规模、参数只覆盖 happy path | [17](17-read-whole-file-scale-wall.md) [18](18-input-contract-edge-cases.md) [19](19-quadratic-training-wall.md) |

## 四、工作方法层面（p9 复盘的核心收获）

1. **端到端测试不是调试器。** p9 连续三轮拿 `pytest tests/test_train_bpe.py` 当唯一信号，
   每次得到 5000 行被截断的 bytes diff。而 `merge_word`（当时名为 `build_new_seq`）是纯函数，
   三行断言就能同时覆盖那三个 bug（[卡 14](14-pure-function-deserves-asserts.md)）。
   **挑「纯 + 边界密集 + 被调用上万次」的那个函数，给它自己的小测试。**
2. **改完先在 REPL 验一个具体值。** `repr(bytes([104]))` 花两秒，
   比跑一遍 pytest 快 100 倍、信号清晰 100 倍（[卡 11](11-bytes-int-zero-fill.md)）。
3. **关键的字符级片段，复制粘贴优于手打。** `bytes([b])` 的方括号就是手抄时掉的。
4. **注解和实现打架时，注解通常是对的那个** —— 它记录意图，实现是手滑处
   （[卡 16](16-annotation-without-enforcement.md)）。
5. **优化先做可证明正确的中间版本。** 保留 naive oracle，写 cache invariant，先定位 affected objects，再降低到 occurrence-level update；不要把多个未验证的优化同时带进实现（[卡 22](22-incremental-cache-convergence.md)）。
