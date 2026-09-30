# 卡 31 · `[必错]` 包内模块的 import 要用完整包路径

- **来源**：p11 初版 `from p9_bpe_tokenizer_training import PAT`（跑 `pytest` / `python -m cs336_basics...` 必炸）
- **标记**：`[必错]` —— 报错很响（ModuleNotFoundError），但写的时候很像「同目录文件互相 import」

## 背景

`cs336_basics/` 是一个包。在 `cs336_basics/p11_tokenizer.py` 里写
`from p9_bpe_tokenizer_training import PAT`，只有 `sys.path` 恰好包含 `cs336_basics/` 时才能找到；
从仓库根目录运行（`uv run pytest`、`python -m cs336_basics.p11_tokenizer`）时，顶层名字是
`cs336_basics`，找不到裸模块名。

## 正面 —— 预测两条 import 的结果

```python
# 在 cs336_basics/p11_tokenizer.py 里
from p9_bpe_tokenizer_training import PAT
from cs336_basics.p9_bpe_tokenizer_training import PAT
```

## 答案

- 第一条：`ModuleNotFoundError: No module named 'p9_bpe_tokenizer_training'`
- 第二条：正常

## 修法

包内互相引用统一「从包顶层写全路径」：

```python
from cs336_basics.p9_bpe_tokenizer_training import (
    PAT,
    Word,
    iter_pretokens,
    merge_word,
    split_special_tokens,
    to_word,
)
```

「在 `cs336_basics/` 目录里直接跑脚本能 import 成功」只是假象——`sys.path[0]` 恰好是那个目录；
换 `pytest` 或 `-m` 就炸。

## 自测

```sh
uv run python -c "from cs336_basics.p11_tokenizer import Tokenizer; print('import ok')"
# 把文件里的 import 临时改回裸模块名，再跑一次就会看到 ModuleNotFoundError
```

## 手写要点

1. 判断 import 写法是否正确，看**运行入口**：`python -m 包.模块`、`pytest`、被别的包 import，
   三种情况下 `sys.path[0]` 不同。
2. 包内引用一律写完整包路径（或相对导入）；不要把「我本地能跑」当依赖。
3. 报 `ModuleNotFoundError` 时先看 `sys.path` 和 `__package__`，别猜。

## 相关卡

- [卡 26](26-long-job-observability.md) —— `-m` 要模块名而不是路径
- [卡 17](17-read-whole-file-scale-wall.md) —— 另一类「本地能跑」的假象
