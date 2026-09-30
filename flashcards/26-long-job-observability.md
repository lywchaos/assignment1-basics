# 卡 26 · `[流程]` 长任务脚本的可用性：进度、flush、可识别产物

- **来源**：p12 全量编码（~6.5h）的运行体验；`-m 路径` 报错被日志吞掉、重定向后 stdout 块缓冲、无进度输出、中断产物的识别
- **标记**：`[流程]`

## 背景

`--steps encode-datasets` 单个数据集最长 ~5.5h，最初脚本只在**每个数据集结束后**打印一行。
重定向到 `run.log` 后，Python 在非 tty 下用块缓冲，连这一行也可能几小时不落盘。实际踩到的坑：

- `uv run python -m cs336_basics/p12_tokenizer_experiments.py ...` → `-m` 要**模块名**（点分隔、
  不带 `.py`）；写路径直接 `ModuleNotFoundError`，错误被 `> run.log` 吞掉，终端看起来「没输出」；
- 中断后的 `.npy` 无法判断是完成还是半成品——除非记得 header 是占位 `shape=(0,)`；
- 长任务没有进度，只能 `ls`/`ps` 猜。

## 修法（写长任务脚本的清单）

1. **进度行**：按固定粒度（这里是每 1M tokens 的 chunk）检查时钟，默认每 30s 打一条：
   `(d) owt_train:  37.4% (4.1 GiB/11.1 GiB), 943,718,144 tokens, 0.61 MiB/s, ETA 3h21m`；
2. **`print(..., flush=True)`**：重定向到文件也实时落盘，不依赖调用者记得加 `-u`；
3. **开始/结束各一条**：开始行含输入大小与输出路径；结束行含耗时、吞吐、产物路径；
4. **产物自描述**：流式 `.npy` 先写占位 header、结束时回填真实 shape —— 「header shape=(0,)
   且文件非空」= 半成品，运行中不要 load；
5. **命令写法可复制**：docs 里写 `-m 包.模块`；路径写法（`python path/to/file.py`）能跑但依赖
   editable install，不适合当规范。
6. **worklog 按 `--steps` 覆盖写**：想保留 (a)–(c) 的记录就一次命令跑全三步；或运行前备份
   `worklog.json`（单独跑 `--steps encode-datasets` 会把旧记录整份覆盖）。

## 附：tee 同时看终端和写日志

```sh
uv run python -m cs336_basics.p12_tokenizer_experiments \
  --steps sample throughput encode-datasets 2>&1 | tee -a artifacts/p12_tokenizer_experiments/run.log
```

- `2>&1` 必须在 `|` 前面；bash 简写 `|&`；
- `tee` 默认覆盖，`-a` 追加；管道退出码是 `tee` 的，用 `set -o pipefail` 或 `${PIPESTATUS[0]}` 才
  能拿到 python 的失败；
- 6h 的任务建议放 `tmux` 里前台跑这条命令：既实时可见，又不怕断连。

## 自测

```sh
# 用 5s 间隔跑一遍 21.5MB 的 valid，观察进度行
uv run python - <<'PY'
from pathlib import Path
from cs336_basics.p12_tokenizer_experiments import encode_dataset, load_tokenizer

tok = load_tokenizer(Path("artifacts/p9_tinystories"))
encode_dataset(tok, Path("data/TinyStoriesV2-GPT4-valid.txt"), Path("/tmp/valid.npy"), progress_interval=5.0)
PY
```

## 手写要点

启动超过 1 小时的任务前，先回答四个问题：
**它在哪（进度）？还剩多久（ETA）？结果对不对（对拍）？断了怎么恢复（可识别产物）？**

## 相关卡

- [卡 25](25-test-green-is-not-testing.md) —— 真实数据的端到端对拍
- [卡 22](22-incremental-cache-convergence.md) —— 长任务/实验的工程化
