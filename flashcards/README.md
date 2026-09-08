# Flashcards

CS336 assignment1-basics 的错题本。**每张卡一个文件、自包含** —— 单独抽出来也能看懂，
不必按顺序读。

## 用法

遮住「答案」，先在脑内**预测输出**（不许先跑），再用「自测」一行命令验证。
vim 里可以 `:r !<自测命令>` 直接把结果读进来对照。

复习一轮的建议路径：`[必错]` → `[陷阱]` → `[流程]` → `[迁移]` → 手写卡 →
[卡 00 元教训](00-meta-lessons.md)。

## 标记含义

| 标记 | 含义 |
|---|---|
| `[必错]` | 在原始语料/规模上就产生错误输出，必须过 |
| `[陷阱]` | 不是原始 bug，是**修上一个 bug 时容易新引入**的错误 |
| `[迁移]` | 原始玩具语料/fixture 上**实测不触发**，换真语料或大规模才咬人 |
| `[风格]` | 不影响正确性，影响可读性 / 性能 / 可维护性 |
| `[流程]` | 关于工作方法，不是某个 API 的坑 |
| `[通用]` | 跨领域的反模式，已脱离 BPE 语境 |
| `[手写]` | 综合验收题，含自查清单和 oracle |

## 索引

来源 `p7` = `cs336_basics/p7_bpe_example.py`（toy BPE），
`p9` = `cs336_basics/p9_bpe_tokenizer_training.py`（真语料 BPE 训练）。

| 卡 | 主题 | 标记 | 来源 |
|---|---|---|---|
| [00](00-meta-lessons.md) | 元教训汇总：六条纪律 + 诊断口诀 | — | 两次复盘 |
| [01](01-zip-in-container-membership.md) | zip 对象塞进容器后，`in` 判定恒为 False | `[必错]` | p7 |
| [02](02-iterator-consumed-by-in.md) | `in` 消耗迭代器：同一个 zip 不能用两次 | `[陷阱]` | p7 |
| [03](03-dict-comprehension-collapses-keys.md) | dict 推导式喂给 Counter，重复 key 被折叠 | `[迁移]` | p7 |
| [04](04-tiebreak-three-stages.md) | tie-break 三段式：不知道 → 以为做了 → 真做了 | `[必错]` | p7 |
| [05](05-loop-assumes-resource-suffices.md) | 「按目标数量循环」隐含假设资源够用 | `[迁移]` | p7 / p9 |
| [06](06-ord-vs-encode.md) | `bytes([ord(ch)])` 只对 Latin-1 有效 | `[迁移]` | p7 |
| [07](07-vocab-vs-merges.md) | vocab 存 token，merges 存 pair —— 职责不同 | `[必错]` | p7 |
| [08](08-handwrite-toy-bpe.md) | **手写卡**：toy BPE 训练（7 次合并 oracle） | `[手写]` | p7 |
| [09](09-truncate-before-filter.md) | 通用反模式：截断早于筛选（跨四个领域） | `[通用]` | 抽象 |
| [10](10-mixed-token-representation.md) | 表示层不统一：同一位置时而 int 时而 tuple | `[必错]` | p9 |
| [11](11-bytes-int-zero-fill.md) | `bytes(b)` vs `bytes([b])`：差一对方括号 | `[必错]` | p9 |
| [12](12-loop-drops-tail.md) | `while i < len(seq) - 1` 丢掉最后一个元素 | `[必错]` | p9 |
| [13](13-two-fixes-stacked.md) | 修 bug 时把两个互斥方案叠加 | `[陷阱]` | p9 |
| [14](14-pure-function-deserves-asserts.md) | 纯函数值得三行断言，别让端到端测试当调试器 | `[流程]` | p9 |
| [15](15-setdefault-as-counter.md) | `setdefault` 当计数器：语义绕 + 占 25% 运行时间 | `[风格]` | p9 |
| [16](16-annotation-without-enforcement.md) | 注解写对了但不被检查 —— 不变量没有守卫 | `[流程]` | p9 |
| [17](17-read-whole-file-scale-wall.md) | `f.read()` 整个语料：fixture 能过，11GB OOM | `[迁移]` | p9 |
| [18](18-input-contract-edge-cases.md) | 输入契约四漏洞：空列表 / 重叠前缀 / encoding / 参数过小 | `[迁移]` | p9 |
| [19](19-quadratic-training-wall.md) | O(vocab_size × 语料)：每轮全量重算 pair 计数 | `[迁移]` | p9 |
| [20](20-handwrite-train-bpe.md) | **手写卡**：`train_bpe` 真语料版（15 条清单） | `[手写]` | p9 |
| [21](21-type-driven-domain-aliases.md) | 类型驱动编程：先给领域概念命名，再写函数 | `[流程]` | p9 |
| [22](22-incremental-cache-convergence.md) | 从发现局部性到可维护的增量缓存 | `[流程]` | p9 |

## 按纪律分组

- **迭代器**：[01](01-zip-in-container-membership.md) [02](02-iterator-consumed-by-in.md)
- **聚合 / 选择**：[03](03-dict-comprehension-collapses-keys.md) [04](04-tiebreak-three-stages.md) [09](09-truncate-before-filter.md) [15](15-setdefault-as-counter.md)
- **表示层 / 类型**：[06](06-ord-vs-encode.md) [07](07-vocab-vs-merges.md) [10](10-mixed-token-representation.md) [11](11-bytes-int-zero-fill.md) [16](16-annotation-without-enforcement.md) [21](21-type-driven-domain-aliases.md)
- **循环边界**：[05](05-loop-assumes-resource-suffices.md) [12](12-loop-drops-tail.md) [13](13-two-fixes-stacked.md)
- **规模 / 契约**：[17](17-read-whole-file-scale-wall.md) [18](18-input-contract-edge-cases.md) [19](19-quadratic-training-wall.md)
- **工作方法**：[14](14-pure-function-deserves-asserts.md) [16](16-annotation-without-enforcement.md) [21](21-type-driven-domain-aliases.md) [22](22-incremental-cache-convergence.md) [00](00-meta-lessons.md)
