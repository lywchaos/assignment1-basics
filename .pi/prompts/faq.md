---
description: 查看 PDF FAQ 问答与归档命令的使用方法
argument-hint: "[用法]"
---
这是当前项目的 PDF FAQ 工作流。

可用命令：

- `/faq-a -p <页码> <问题>`：使用 `pdftotext` 读取 `cs336_assignment1_basics.pdf` 的指定页，并根据该页内容回答问题。例如：`/faq-a -p 3 我的问题是xxx`。
- `/faq-d`：在当前会话中找到最近一次 FAQ-A 问答，将相关问题、用户交互和最终结果追加到项目根目录的 `faq.md`；文件不存在时会创建。

约定：页码从 1 开始；FAQ-D 只归档当前会话中最近且相关的一轮 FAQ-A，不会把无关对话写入文件。

如果刚刚新增或修改了这些 prompt，请执行 `/reload` 让 pi 重新发现它们。
