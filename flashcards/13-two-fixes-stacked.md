# 卡 13 · `[陷阱]` 修 bug 时把两个互斥方案叠加

- **来源**：`cs336_basics/p9_bpe_tokenizer_training.py` 的 `build_new_seq` 复盘
- **标记**：`[陷阱]` —— 修 [卡 12](12-loop-drops-tail.md) 时新引入的 bug

## 背景

[卡 12](12-loop-drops-tail.md) 的病灶是 `while i < len(seq) - 1` 丢掉最后一个元素。
有两个**互斥**的修法：

- **方案 A**：循环条件改成 `while i < len(seq)`，配对判断内部加 `i + 1 < len(seq)` 边界
- **方案 B**：循环条件不动，**在循环外**补 `if i == len(seq) - 1: new_seq.append(seq[-1])`

实际改出来的是 A + B 同时用上，而且 B 那段还写在了**循环体内部**。

## 正面 —— 预测两个输出

```python
def build_new_seq(seq, pair):
    new_seq, i = [], 0
    while i < len(seq):                                          # 方案 A
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            new_seq.append(pair[0] + pair[1]); i += 2
        else:
            new_seq.append(seq[i]); i += 1
        if i == len(seq) - 1:                                    # 方案 B，还在循环体内
            new_seq.append(seq[-1])
    return tuple(new_seq)

build_new_seq((b'a', b'b', b'c'), (b'a', b'b'))
build_new_seq((b'h', b'e', b'l', b'l', b'o'), (b'l', b'o'))
```

## 答案

- `(b'ab', b'c', b'c')` —— **`b'c'` 重复了**
- `(b'h', b'e', b'l', b'lo')` —— 正确

## 为什么

循环条件已经是 `i < len(seq)`，`seq[i]` 自己就能吃掉最后一个元素，**不需要再补尾巴**。
而 B 写在循环体内：`i` 一走到 `len(seq)-1` 就先补一次 `seq[-1]`，然后循环再转一圈，
把 `seq[i]`（正是同一个 `seq[-1]`）又 append 一次。

第二个例子没重复，是因为合并跳了 2 步，`i` 从 3 直接到 5，没落在 `len-1` 上，`if` 没触发。
**又一次「时对时错」** —— 和 [卡 12](12-loop-drops-tail.md) 同一个成因（跳步越过了边界值）。

## 自测

```sh
python3 -c "
def f(seq, pair):
    new, i = [], 0
    while i < len(seq):
        if i < len(seq)-1 and (seq[i],seq[i+1])==pair: new.append(pair[0]+pair[1]); i += 2
        else: new.append(seq[i]); i += 1
        if i == len(seq)-1: new.append(seq[-1])
    return tuple(new)
print(f((b'a',b'b',b'c'), (b'a',b'b')))
print(f((b'h',b'e',b'l',b'l',b'o'), (b'l',b'o')))
"
```

## 手写要点

三条：

1. 拿到「二选一」的修复建议时，**选一个，然后把另一个从脑子里删掉**。
2. 改完**完整重读一遍改动区域**。看到「两处修改都在负责同一件事」（这里是「保证最后一个元素被输出」），
   立刻警觉。
3. 判据：**同一个不变量不应该有两个守卫**。如果有，其中一个必然是多余的，
   而多余的守卫往往不是「无害」，而是「双重生效」。

## 相关卡

- [卡 12](12-loop-drops-tail.md) —— 本卡的上游 bug
- [卡 02](02-iterator-consumed-by-in.md) —— 另一个「修 bug 时引入新 bug」的例子
- [卡 04](04-tiebreak-three-stages.md) —— 元教训同源：「不知道要做 X」修成「以为自己做了 X」
