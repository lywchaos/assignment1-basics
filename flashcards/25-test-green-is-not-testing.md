# 卡 25 · `[流程]` 测试是绿的 ≠ 测试在测

- **来源**：p11 流式实现对拍排查；`tests/test_tokenizer.py` 的内存测试与 fixture 覆盖
- **标记**：`[流程]`

## 背景 —— 三个真实例子

1. **资源限制测试是空转**：`tests/test_tokenizer.py` 里

   ```python
   @memory_limit(int(1e6))
   def _encode_iterable(tokenizer, iterable):
       yield from tokenizer.encode_iterable(iterable)
   ```

   `_encode_iterable` 是 generator function：装饰器调用它只拿到 generator 对象，函数体没执行，
   `finally` 立刻把 `RLIMIT_AS` 还原。真正迭代时限制早没了。
   就算限制生效，测试循环里的 `ids.append`（5M → 约 130 万个 int）也会先超标。

2. **对拍 fixture 太小**：`test_encode_iterable_tinystories_matches_tiktoken` 只比 3.8KB 样本；
   5M 的测试只要求「不崩」，不比较值。contraction bug（[卡 23](23-streaming-hold-multi-char-alternative.md)）
   就这样躲过了整套官方测试。

3. **自己的 fuzz 有盲区**：带 special 时 `hold=12`，短文本整段被 carry，被怀疑的 step 2
   几乎不执行；随机切分要正好落在 `'` + 后缀第 2 字符上才触发。

## 五条纪律

| 纪律 | 做法 |
|---|---|
| **限制类测试先放 canary** | 对内存/时长/文件数上限，先写一个**必然超限**的对照（如 generator 里分配 50MB），确认限制真的会触发；别让测试静默空转 |
| **覆盖 ≠ 正确** | 「不报错」和「比较值」是两种测试。关键路径要有**真实数据 + 逐元素对拍**：22MB 的 stream-vs-whole 才暴露了 14 token 差异 |
| **fuzz 先证明分支被走到** | 参数组合（hold 大小、chunk 长度）有没有让被测分支真的执行？可用「故意植入 bug 能否被抓住」验证 |
| **限制的单位要看清** | `RLIMIT_AS` 量的是**地址空间**不是 RSS：`RLIMIT_AS = RSS + 1MB` 下 200KB 的分配就会 MemoryError。所以「1MB 窗口累积」在该限制下必挂——测试没 enforce 才让它活着 |
| **测试要能抓住旧 bug** | 修 bug 后从 git 取出旧版本跑同一组用例，**必须失败**（负向对照）；否则用例没有杀伤力。本例：`git show <merge>:cs336_basics/p11_tokenizer.py` 加载旧实现跑穷举，旧代码在 `'ll/'ve/'re` 的特定切点必然报错 |

## 自测

```sh
uv run python - <<'PY'
import os, resource, psutil

def memory_limit(max_mem):          # 与 tests/test_tokenizer.py 同款
    def decorator(f):
        def wrapper(*args, **kwargs):
            p = psutil.Process(os.getpid())
            prev = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (p.memory_info().rss + max_mem, -1))
            try:
                return f(*args, **kwargs)
            finally:
                resource.setrlimit(resource.RLIMIT_AS, prev)
        return wrapper
    return decorator

@memory_limit(int(1e6))
def generator_fn():
    big = bytearray(50_000_000)
    yield len(big)

print("generator 分配 50MB:", next(generator_fn()))   # 不报错：限制在迭代前已被恢复
PY
```

## 手写要点

1. 绿色的测试先问三件事：**它比较了什么值？输入有多大？被测分支真的执行了吗？**
2. 资源限制类测试必须验证限制「在测时是活的」——否则它测的是零。
3. 端到端对拍要用**真实规模**的数据；fixture 通过只是地板。

## 相关卡

- [卡 23](23-streaming-hold-multi-char-alternative.md) / [卡 24](24-whitespace-runs-cross-chunks.md) —— 被漏掉的 bug
- [卡 14](14-pure-function-deserves-asserts.md) —— 端到端测试不是调试器
- [卡 17](17-read-whole-file-scale-wall.md) —— 规模差异
