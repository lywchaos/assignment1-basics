# Flashcards

来源：`cs336_basics/p7_bpe_example.py` 的一次复盘。7 张 bug 卡 + 1 张综合手写卡。

用法：遮住「答案」，先在脑内**预测输出**（不许先跑），再用「自测」一行命令验证。
vim 里可以 `:r !<自测命令>` 直接把结果读进来对照。

---

## 卡 1 · zip 对象塞进容器后，`in` 判定恒为 False

**正面** — 预测两个输出：

```python
t = (b'a', b'b')
print((b'a', b'b') in [zip(t, t[1:])])
print((b'a', b'b') in zip(t, t[1:]))
```

**答案** — `False` / `True`。

`[zip(...)]` 是「装了一个 zip 对象的 list」，长度为 1，元素类型是 `zip`。拿 tuple 去比一个 zip
对象，永远不相等。去掉方括号后 `in` 走迭代器协议，逐个比较元素，才是想要的语义。

真实后果：BPE 主循环里合并分支一次都没进过，`token_counter` 从头到尾没变，每轮都选出同一个
`(b'e', b's')`，vocab 被塞进 7 个重复项。**程序不报错、不崩、跑得很欢** —— ruff 和 ty 也全 pass。

**自测**

```sh
python3 -c "t=(b'a',b'b'); print((b'a',b'b') in [zip(t,t[1:])], (b'a',b'b') in zip(t,t[1:]))"
```

**手写要点** — 写完 `in` 判定，问自己：右边那个东西的**元素**是什么类型？和左边同类吗？
凡是「包了一层容器」的迭代器表达式，都要停下来数一遍层数。

---

## 卡 2 · `in` 消耗迭代器：同一个 zip 不能用两次

**正面** — 预测输出：

```python
t = (b'a', b'b', b'c')
pairs = zip(t, t[1:])
print((b'a', b'b') in pairs)
print((b'b', b'c') in pairs)
print((b'a', b'b') in pairs)
```

**答案** — `True` / `True` / `False`。

`in` 对迭代器是**边消耗边比较**：第 1 次匹配到 `(a,b)` 就停在那儿；第 2 次从剩下的
`(b,c)` 继续找，命中；第 3 次迭代器已耗尽，`False`。

所以卡 1 的修法若写成 `pairs = zip(...)` 再多处复用 `pairs`，会换成另一个更隐蔽的 bug。
要复用就物化成 `set` / `list`，或者每次现场重建 `zip`。

**自测**

```sh
python3 -c "t=(b'a',b'b',b'c'); p=zip(t,t[1:]); print((b'a',b'b') in p, (b'b',b'c') in p, (b'a',b'b') in p)"
```

**手写要点** — `zip` / `map` / `filter` / 生成器都是一次性的。一个变量名如果要被读两次以上，
它就不该是迭代器。

---

## 卡 3 · dict 推导式喂给 Counter，会把重复 key 折叠掉

**正面** — token 是 `(b'a', b'a', b'a')`、词频 5，预测 pair 计数：

```python
c = Counter()
c.update({pair: 5 for pair in zip(t, t[1:])})
print(c)
```

**答案** — `Counter({(b'a', b'a'): 5})`，**正确答案是 10**。

`zip` 产出了两个相同的 `(b'a', b'a')`，但 dict 推导式后写的 key 覆盖前一个，两次变一次。
`Counter.update(dict)` 本身是「加法」没错，错在**传进去之前就已经丢了信息**。

正确写法 —— 让累加发生在 Counter 里，而不是在 dict 构造里：

```python
for pair in zip(token, token[1:]):
    pair_counter[pair] += count
```

阴险之处：`low lower widest newest` 这个语料里没有任何词包含重复相邻 pair，所以玩具例子上
**结果完全正确**，换到真语料（`aaa`、`---`、`...`、`\n\n`）才开始错。

**自测**

```sh
python3 -c "from collections import Counter; t=(b'a',b'a',b'a'); c=Counter(); c.update({p:5 for p in zip(t,t[1:])}); print(c)"
```

**手写要点** — 推导式的 key 有可能重复时，就不能用推导式聚合。判据：key 是不是「位置的函数」
而非「值的函数」。

---

## 卡 4 · `most_common(1)` 的 tie-break 是插入顺序，不是字典序

**正面** — `(b'e', b's')` 和 `(b's', b't')` 都出现 9 次。`most_common(1)` 给哪个？
BPE 要的是哪个？

**答案** — `most_common` 给 `(b'e', b's')`（先插入的）；BPE 规定取**字典序更大**的
`(b's', b't')`。

`Counter.most_common` 底层是稳定排序，只按 count 排，同 count 保持插入顺序 —— 也就是
「碰巧由语料的遍历顺序决定」。这是 assignment 明确要求的 tie-break 规则，也是最容易
「跑通了但答案不对」的一处。

正确写法：

```python
max_pair = max(pair_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
```

`bytes` 之间可以直接比大小（按字节逐位比），所以复合 key `(count, pair)` 天然可用。

**自测**

```sh
python3 -c "from collections import Counter; c=Counter({(b'e',b's'):9,(b's',b't'):9}); print(c.most_common(1)[0][0], max(c, key=lambda p:(c[p],p)))"
```

**手写要点** — 任何取 max/argmax 的地方，先问「平票怎么办」。如果规格书里写了平票规则，
就不能用默认排序，必须把规则显式编码进 key。

---

## 卡 5 · 空 Counter 上取 `most_common(1)[0]` 会 IndexError

**正面** — 预测输出：

```python
print(Counter().most_common(1)[0][0])
```

**答案** — `IndexError: list index out of range`。`most_common(1)` 返回的是 list，
空 Counter 上就是 `[]`，`[0]` 直接炸。

BPE 里的触发条件：所有 word 都已被合并成单个 token，此时没有任何相邻 pair 可数。
`vocab_size=264`（只需 7 次合并）侥幸没触发，稍微调大就崩。

主循环开头必须有终止条件：

```python
if not pair_counter:
    break
```

**自测**

```sh
python3 -c "from collections import Counter; print(Counter().most_common(1)[0][0])"
```

**手写要点** — `while len(vocab) < vocab_size` 这类「按目标数量循环」的写法，隐含假设
「资源一定够」。写循环时同时写两个出口：目标达成，和**资源耗尽**。

---

## 卡 6 · `bytes([ord(ch)])` 只对 Latin-1 有效

**正面** — 三个表达式，哪些成功、结果是什么？

```python
bytes([ord('a')])
bytes([ord('é')])
bytes([ord('中')])
```

**答案** — `b'a'` / `b'\xe9'` / **`ValueError: bytes must be in range(0, 256)`**。

`ord` 给的是 Unicode 码点，`bytes([...])` 要的是 0-255 的字节值。两者只在
码点 < 256（即 Latin-1）时**数值上巧合相等**。`'é'`（U+00E9）能过，但它的 UTF-8 编码
其实是 2 字节 `b'\xc3\xa9'` —— 所以 `b'\xe9'` 这个「成功」比失败更危险，它悄悄产出了错的字节。

正确写法 —— 先 encode，再逐字节拆：

```python
tuple(bytes([b]) for b in w.encode("utf-8"))
```

**自测**

```sh
python3 -c "print(bytes([ord('é')]), tuple(bytes([b]) for b in 'é'.encode()))"
python3 -c "print(bytes([ord('中')]))"
```

**手写要点** — 「str → bytes」只有一条合法通路：`encode`。看到 `ord` 和 `bytes` 出现在
同一个表达式里就该警觉：这是在用码点冒充字节。

---

## 卡 7 · vocab 存「合并后的 token」，merges 存「pair」

**正面** — 选出 `max_pair = (b'e', b'st')` 之后，vocab 里该追加什么？merges 里该追加什么？
两者的类型分别是什么？

**答案**

- `vocab`：追加**合并后的新 token**，`max_pair[0] + max_pair[1]` → `b'est'`，
  类型 `dict[int, bytes]`（id → token bytes）。
- `merges`：追加**pair 本身** `(b'e', b'st')`，类型 `list[tuple[bytes, bytes]]`，
  且**顺序有意义** —— encode 时要按同样顺序重放这些合并。

原代码 `vocab.append(max_pair)` 把 pair 塞进了 vocab，于是一个 list 里混了三种类型：
`str`（`"<|endoftext|>"`，还该是 `bytes`）、`bytes`（256 个字节）、`tuple[bytes, bytes]`（pair）。
merges 则完全没有被记录 —— 训练结果里最关键的那一半丢了。

对照 `tests/adapters.py` 的签名，这是硬要求：

```python
def run_train_bpe(...) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
```

**手写要点** — 两个容器职责不同就绝不合并。写之前先把每个容器的元素类型写成注解，
类型写不出来 == 设计还没想清楚。

---

## 卡 8 · 综合手写卡（古法编程）

**题面** — 不看任何参考，从空文件手写 toy BPE 训练，语料与词表：

```python
CORPUS = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
# 初始 vocab: b"<|endoftext|>" + 256 个单字节, vocab_size = 264 (即 7 次合并)
```

要求返回 `(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]])`。

**自查清单**（写完后逐条打勾，对应卡 1-7）

1. 合并分支真的进去过吗？（打一行 print 或断言 `token_counter` 每轮都在变）
2. 有没有把 `zip` 结果复用第二次？
3. pair 计数是在 Counter 里累加，而非 dict 推导式里？
4. tie-break 显式写了字典序更大优先？
5. `pair_counter` 为空时有 break？
6. str → bytes 走的是 `encode("utf-8")`？
7. vocab 存 bytes、merges 存 pair，两者都返回了？

**验收 oracle**（与 handout p7 例子一致，7 次合并依次为）

```
1. (b's', b't')      -> b'st'
2. (b'e', b'st')     -> b'est'
3. (b'o', b'w')      -> b'ow'
4. (b'l', b'ow')     -> b'low'
5. (b'w', b'est')    -> b'west'
6. (b'n', b'e')      -> b'ne'
7. (b'ne', b'west')  -> b'newest'
```

最终 pretoken 状态：

```
{(b'low',): 5, (b'low', b'e', b'r'): 2, (b'w', b'i', b'd', b'est'): 3, (b'newest',): 6}
```

**注意第 1 步就是分水岭**：如果你的第一个 merge 是 `(b'e', b's')`，说明卡 4 没过。

---

## 元教训

这 7 个 bug 里，**没有一个**能被 `ruff check` 或 `ty check` 抓到（实测两者全 pass），
而且有 4 个（卡 1、3、4、6）属于**程序正常退出、输出看起来合理**的静默错误。

可迁移的三条判据：

- **迭代器纪律**：`zip`/`map`/生成器 —— 数容器层数，且只读一次。
- **聚合纪律**：任何计数/累加，问「key 会重复吗」；任何 max，问「平票怎么办」。
- **边界类型纪律**：str/bytes 边界只走 encode/decode；容器元素类型先写注解再写代码。
