## P3.a What Unicode character does chr(0) return?

直接 :r !python3 -c 'print(repr(chr(0)))'

```
'\x00'
```

## P3.b How does this character’s string representation (**repr**()) differ from its printed representation?

`__repr__` 是机器友好，print 是人类可读。具体来讲好像 print(a) 输出的是 a 的 `__str__` 函数。不太确定。

## P3.c What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations

应该是不可读的字符吧，显示上应该是空。实验下看看。以下 shell 输出的粘贴：

```
❯ python3
Python 3.14.5 (main, May 10 2026, 10:21:34) [Clang 21.0.0 (clang-2100.0.123.102)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
>>> exit

chezmoi on  main took 39s
```
