---
layout:     post
title:      "TensorFlow Notes 2 | 【TensorFlow深度学习框架教程二】"
subtitle:   "Python一小时入门导学"
date:       2018-03-31
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - Python
---

#### 概述

本导学基于Python2.7, 3.0以上只是部分语法有更新大同小异相信聪明的你查查就懂啦~

Python是著名的“龟叔”Guido van Rossum（吉多·范罗苏姆，荷兰人）在1989年圣诞节期间，为了打发无聊的圣诞节而编写的一个编程语言。

Python就为我们提供了非常完善的基础代码库，覆盖了网络、文件、GUI、数据库、文本等大量内容，被形象地称作“内置电池（batteries included）”。用Python开发，许多功能不必从零编写，直接使用现成的即可。除了内置的库外，Python还有大量的第三方库，也就是别人开发的，供你直接使用的东西。当然，如果你开发的代码通过很好的封装，也可以作为第三方库给别人使用。

许多大型网站就是用Python开发的，例如YouTube、Instagram，还有国内的豆瓣。

龟叔给Python的定位是“优雅”、“明确”、“简单”，所以Python程序看上去总是简单易懂，Python的哲学就是简单优雅，尽量写容易看明白的代码，尽量写少的代码，1行代码能实现的功能，决不写5行代码。如果一个资深程序员向你炫耀他写的晦涩难懂、动不动就几万行的代码，你可以尽情地嘲笑他。

*weak point*

- 运行速度慢，Python是解释型语言，你的代码在执行时会一行一行地翻译成CPU能理解的机器码，这个翻译过程非常耗时，所以很慢。而C程序是运行前直接编译成CPU能执行的机器码，所以非常快。

eg: C/0.001s  Python/0.1  慢了100倍，但由于网络更慢等待1s，用户feeling? 
就像拥堵的三环---法拉利恩佐和taxi赛跑一样

- 代码不能加密，发布程序就是发布源代码

开源运动、自由开放。所谓大公司不开放代码的原因是写的太烂了，一旦开源，也就没人敢用、、再说了大家那么忙也木有时间破解。

##### 交互式环境

```python
>>> print("hello world!")
hello world!
>>> 300 + 200
500
```

Notes: 
- 字符串使用单引号或双引号括起来，但不能混用单引号和双引号
- 缺憾---没有把代码保存下来，每输入一行就执行一行

如果写一个test1.py文件，内容如下：

```python
300 + 200
```

然后执行:

```python
python test1.py
```

发现什么都没有输出，想要输出结果，必须自己打印出来，

```python
#!/usr/bin/python  
# 告诉操作系统执行这个脚本的时候，调用/usr/bin下的python解释器
#coding:utf-8

sum = 300 + 200
print sum
```

再执行就可以看到结果：

```python
python test1.py
500
```

##### 输入和输出

- 输出

print()函数也可以接受多个字符串，用逗号“,”隔开，就可以连成一串输出：

```python
>>> print('Hello', 'I am', 'Python')
Hello, I am Pyhton
```

- 输入

Python提供了一个input()，可以让用户输入字符串，并存放到一个变量里，比如输入你的名字

```python
>>> name = input("Please input your name:")
python
```

当你输入name = input()并按下回车后，Python交互式命令行就在等待你的输入了。这时，你可以输入任意字符，然后按回车后完成输入。输入完成后，不会有任何提示，Python交互式命令行又回到>>>状态了。那我们刚才输入的内容到哪去了？答案是存放到name变量里了。可以直接输入name查看变量内容：

```python
>>> name
python
```

##### Python基础吧啦吧啦

- Python程序是大小写敏感的，如果写错了大小写，程序会报错。
- Python中有两种除法

```python
>>> 10.0 / 3
3.3333333333333335
```

```python
>>> 9 / 3
3
```

/除法计算结果是浮点数，还有一种除法//，称为地板除，两个整数的除法仍然是整数：

```python
>>> 10 // 3
3
```

##### 数据类型

- list

一种有序集合，可以随时添加和删除其中的元素。

```python
>>> animals = ['dog', 'cat', 'owl']  
>>> animals
['dog', 'cat', 'owl']
>>> len(animals)   # len()函数获取list元素的个数
3
>>> animals[0]     # 索引访问list中的元素，记得索引是从0开始
'dog'
>>> animals[2]
'owl'
>>> animals[3]    # 索引越界
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
IndexError: list index out of range
>>> animals[-1]   # 使用-1做索引，直接获取最后一个元素，类推-2，-3
'owl'
```

list的添加、删除和替换

```python
>>> animals.append('ant')    # 末尾添加元素
>>> animals
['dog', 'cat', 'owl', 'ant']
>>> animals.insert(1, 'bird')  # 插入到指定位置元素
>>> animals
['dog', 'bird', 'cat', 'owl', 'ant']
>>> animals.pop(2)    # 删除指定位置元素，注意返回的是被删除元素值
'cat'
>>> animals
['dog', 'bird', 'owl', 'ant']
>>> animals[1] = 'snake'    # 替换指定位置的元素，直接赋值给对应的索引位置
>>> animals
['dog', 'snake', 'owl', 'ant']
>>> L = ['apple', 8, True]    # list中的元素可以是不同的数据类型
>>> S = ['python', 'java', L, 'php']  # list中的元素也可以是另一个list
>>> S[2][1]      # S可以看做是一个二维数组
8
```

- tuple

元组，一旦初始化就不能修改，没有append()，insert()这样的方法，但可以正常使用索引获取元素。

```python
>>> animals=('dog', 'cat', 'owl')  # 初始化后这个tuple就不能变了
>>> animals
('dog', 'cat', 'owl') 
>>> t=(1,)     # 只有1个元素的tuple定义时必须加一个逗号,，来消除歧义
>>> t         # 这是因为括号()既可以表示tuple，又可以表示数学公式中的小括
              #号，这就产生了歧义，因此，Python规定，这种情况下，按小括号
              #进行计算，计算结果自然是1
(1,) 
```

- dict

dict的支持，dict全称dictionary，在其他语言中也称为map，使用键-值（key-value）存储，具有极快的查找速度。

```python
>>> d = {'dog': 12, 'cat': 15, 'owl': 18}
>> d['cat']
15
```

为什么dict查找速度这么快？因为dict的实现原理和查字典是一样的。假设字典包含了1万个汉字，我们要查某一个字，一个办法是把字典从第一页往后翻，直到找到我们想要的字为止，这种方法就是在list中查找元素的方法，list越大，查找越慢。

第二种方法是先在字典的索引表里（比如部首表）查这个字对应的页码，然后直接翻到该页，找到这个字。无论找哪个字，这种查找速度都非常快，不会随着字典大小的增加而变慢。

dict就是第二种实现方式，给定一个名字，比如'cat'，dict在内部就可以直接计算出cat对应的存放成绩的“页码”，也就是15这个数字存放的内存地址，直接取出来，所以速度非常快。

```python
>>> d['ant'] = 20 #把数据放入dict的方法，除了初始化时指定外，还可以通过key放入
>>> d
{'dog': 12, 'cat': 15, 'owl': 18, 'ant': 20}
#dict内部存放的顺序和key放入的顺序是没有关系的
>>> 'owl' in d   # 判断key是否存在
True
>>> d.get('bird')  # 返回None的时候交互式命令行不显示结果
>>> d.pop('cat')   # 删除元素
15
>>> d
{'dog': 12, 'owl': 18, 'ant': 20}
```

和list比较，dict有以下几个特点：

- 查找和插入的速度极快，不会随着key的增加而变慢；
- 需要占用大量的内存，内存浪费多。

而list相反：

- 查找和插入的时间随着元素的增加而增加；
- 占用空间小，浪费内存很少。

所以，dict是用空间来换取时间的一种方法。dict可以用在需要高速查找的很多地方，在Python代码中几乎无处不在，正确使用dict非常重要，需要牢记的第一条就是dict的key必须是不可变对象。

##### 条件判断和循环
- Python的语法比较简单，采用缩进方式，也就是当语句以冒号:结尾时，缩进的语句视为代码块。
- 好处是强迫你写出格式化的代码，但没有规定缩进是几个空格还是Tab。按照约定俗成的管理，应该始终坚持使用4个空格的缩进。
- 缩进的坏处就是“复制－粘贴”功能失效了（ps:cv工程师，写程序也是讲究一个感觉，刚开始不建议使用IDE，需要一个字母一个字母的敲出来，还有虽然很多人推荐你用Anaconda但是那个会剥夺你很多学习体验），这是最坑爹的地方。当你重构代码时，粘贴过去的代码必须重新检查缩进是否正确。此外，IDE很难像格式化Java代码那样格式化Python代码。

###### - 条件判断

```python
#!/usr/bin/python
#coding:utf-8

d = {'dog': 12, 'owl': 18, 'ant': 20}
if 'owl' in d:
    print d['owl']
else:
    print("no owl")
```

###### - 循环
- Python的循环有两种，一种是for...in循环

```python
#!/usr/bin/python
#coding:utf-8

animals = ['dog', 'cat', 'owl']
for x in animals:
    print(x)  # （x）,表示不换行
```

所以for x in ...循环就是把每个元素代入变量x，然后执行缩进块的语句。
- 第二种循环是while循环，只要条件满足，就不断循环，条件不满足时退出循环

```python
#!/usr/bin/python
#coding:utf-8

animals = ['dog', 'cat', 'owl']
l = len(animals)
while l > 0:
    print animals[l-1]
    l -= 1
```

##### 函数
- 内置函数

Python内置了很多有用的函数，我们可以直接调用。要调用一个函数，需要知道函数的名称和参数，比如求绝对值的函数abs，只有一个参数。也可以在交互式命令行通过help(abs)查看abs函数的帮助信息。

```python
>>> abs(10)  # 保证传入参数的个数和类型正确
10
>>> abs(-10)
10
>>> int('123')  # 类型转换
123
>>> int(23.45)
23
>>> str(123)
'123'
>>> a = abs # 函数名其实就是指向一个函数对象的引用，完全可以把函数名赋给一
            #个变量，相当于给这个函数起了一个“别名”
>>> a(-1)
1
```

- 自定义函数

在Python中，定义一个函数要使用def语句，依次写出函数名、括号、括号中的参数和冒号:，然后，在缩进块中编写函数体，函数的返回值用return语句返回。

```python
#!/usr/bin/python
#coding:utf-8

def get_animal(index):
    animals = ['dog', 'cat', 'owl']
    if (index < 0 or index >= len(animals)):  # 切记检查参数的合法性
        return -1
    animal = animals[index]
    return animal

if __name__=='__main__':
    index = input("please input the index:")
    animal = get_animals(index)
    print animal
```

当我们在命令行运行该模块文件时，Python解释器把一个特殊变量__name__置为__main__，而如果在其他地方导入该模块时，if判断将失败，因此，这种if测试可以让一个模块通过命令行运行时执行一些额外的代码，最常见的就是运行测试。

在Python交互环境中定义函数时，注意Python会出现...的提示。函数定义结束后需要按两次回车重新回到>>>提示符下。

##### 模块
为了编写可维护的代码，我们把很多函数分组，分别放到不同的文件里，这样，每个文件包含的代码就相对较少，很多编程语言都采用这种组织代码的方式。在Python中，一个.py文件就称之为一个模块（Module）。

使用模块有什么好处？

- 最大的好处是大大提高了代码的可维护性
- 编写代码不必从零开始。当一个模块编写完毕，就可以被其他地方引用。我们在编写程序的时候，也经常引用其他模块，包括Python内置的模块和来自第三方的模块。
- 使用模块还可以避免函数名和变量名冲突。相同名字的函数和变量完全可以分别存在不同的模块中，因此，我们自己在编写模块时，不必考虑名字会与其他模块冲突。
但是也要注意，尽量不要与内置函数名字冲突。

ok那你也许会说如果不同的人编写的模块名相同了怎么办？为了避免模块名冲突，Python又引入了按目录来组织模块的方法，称为包（Package）。请注意，每一个包目录下面都会有一个__init__.py的文件，这个文件是必须存在的，否则，Python就把这个目录当成普通目录，而不是一个包__init__.py可以是空文件。这在大型工程中经常使用到。

###### - 内置模块
Python本身就内置了很多非常有用的模块，只要安装完毕，这些模块就可以立刻使用。

```python
#!/usr/bin/python
#coding:utf-8

import time          # 导入time模块
'test model'
__author__='witt'

def get_animal(index):
    start = time.time()
    animals = ['dog', 'cat', 'owl']
    if (index < 0 or index >= len(animals)):  # 切记检查参数的合法性
        return -1
    animal = animals[index]
    t = time.time() - start
    print t
    return animal

if __name__=='__main__':
    index = input("please input the index:")
    animal = get_animals(index)
    print animal
    
```

###### - 第三方模块
在Python中，安装第三方模块，是通过包管理工具pip（Pip代表Pip Installs Packages，是Python的官方认可的包管理器；conda是一个与语言无关的跨平台环境管理器）完成的。

注意：Mac或Linux上有可能并存Python 3.x和Python 2.x，因此对应的pip命令是pip3

第三方库——Python Imaging Library，这是Python下非常强大的处理图像的工具库。不过，PIL目前只支持到Python 2.7，并且有年头没有更新了，因此，基于PIL的Pillow项目开发非常活跃，并且支持最新的Python 3。

```python
>>> apt-get install python-pip python-dev
>>> pip install Pillow
>>> from PIL import Image
>>> im = Image.open('test.png')
>>> print(im.size)
```

Numpy是Python的一个科学计算的库，提供了矩阵运算的功能，其实，list已经提供了类似于矩阵的表示形式，不过numpy为我们提供了更多的函数。

```pyhton
>>> sudo apt-get install python-numpy
>>> import numpy as np
>>> print np.array([1, 2, 3, 4], dtype=np.int32)  # 以list为参数产生一维数组
[1 2 3 4]
>>> print np.array([[1, 2],      # 以list为参数产生二维维数组  
                    [3, 4]])
[[1, 2]
 [3, 4]]
>>> print np.zeros((2, 2))  # 构造特定矩阵 np.ones
[[0. 0.]
[0. 0.]]
>>> a = np.zeros((2, 2))
>>> print a.ndim  # 获取数组维数
2
>>> print a.shape  # 获取数组每一维的大小
(2, 2)
# 数据索引，切片，赋值
>>> a = np.array([[1, 2, 3], [5, 6, 7]])
>>> print a[1, 2]
7
>>> print a[1,:]
[5 6 7]
>>> print a[1, 1:2]
[6]
>>> a[1,:]=[7, 8, 9]
>>> print a 
[[1 2 3]
 [7 8 9]]
```

其他常用第三方库还有Pandas、matplotlib等。

##### 面向对象编程
面向对象编程——Object Oriented Programming，简称OOP，是一种程序设计思想---从自然界而来，举例animal说明类class（抽象概念，类是抽象的模版）和实例instance（一个个具体的对象，但各自的数据可能不同），so面向对象的设计思想就是抽象出class，根据class创建instance。

**面向过程**的程序设计把计算机程序视为一系列的命令集合，即一组函数的顺序执行。为了简化程序设计，面向过程把函数继续切分为子函数，即把大块函数通过切割成小块函数来降低系统的复杂度。

而**面向对象**的程序设计把计算机程序视为一组对象的集合，而每个对象都可以接收其他对象发过来的消息，并处理这些消息，计算机程序的执行就是一系列消息在各个对象之间传递。

采用面向对象的程序设计思想，我们首选思考的不是程序的执行流程，而是animal这种数据类型应该被视为一个对象，这个对象拥有name和age这两个属性。

***
###### -数据封装
方法就是与实例绑定的函数，和普通函数不同，方法可以直接访问实例的数据；

通过在实例上调用方法，我们就直接操作了对象内部的数据，但无需知道方法内部的实现细节。

```python
#!/usr/bin/python
#coding:utf-8

'animal.py'

class Animal():
    
    def __init__(self, name, age): 
    # self指向创建的实例本身，在其内部就可以把各属性绑定到self
        self.name = name    # 成员属性、变量
        self.age = age  # 实例化后通过ani.name访问，ani指向一个Animal()的实例
            #既然ani实例本身就有这些数据，就没必要从外面函数访问，可以在类内部定义
_____________________________________________________________________________
    def get_animal(self):   
    # 成员方法  本质就是数据和逻辑的封装，不用知道内部实现的细节
        print(self.name, self.age)
    
    def get_dict(self):
        self.get_animal()    # self在成员方法之间调用时使用
        d={}
        d[self.name] = self.age
        return d

if __name__=='__main__':
    ani = Animal('dog', 15)
    print ani.name, ani.age
```

和普通的函数相比，在类中定义的函数只有一点不同，就是第一个参数永远是实例变量self，并且，调用时，不用传递该参数。除此之外，类的方法和普通函数没有什么区别。

###### -引入模块

```python
#!/usr/bin/python
#coding:utf-8

import animal   # 引入模块

ani = animal.Animal('dog', 15)  # 实例化一个对象，且绑定上属性
ani.get_animal()               # 访问成员函数
dict = ani.get_dict()
print dict
```

##### 文件读写
读写文件是最常见的IO操作。读写文件前，我们先要知道，在磁盘上读写文件的功能都是由操作系统提供的，现代操作系统不允许普通程序直接操作磁盘，所以读写文件就是请求操作系统打开一个对象（通常是文件描述符），然后通过操作系统提供的接口从这个文件对象中读取数据，或者把数据写入这个文件对象。

- 写文件

写文件调用open()函数时，传入标识符'w'或者'wb'表示写文本文件或写二进制文件：

```python
>>> f = open('test.py', 'w')
>>> f.write('hello world')
>>> f.close()

#文件使用完毕后必须关闭，因为文件对象会占用操作系统的资源，并且操作系统同一时间能打开的文件数量也是有限的
```

当我们写文件时，操作系统往往不会立刻把数据写入磁盘，而是放到内存缓存起来，空闲的时候再慢慢写入。只有调用close()方法时，操作系统才保证把没有写入的数据全部写入磁盘。忘记调用close()的后果是数据可能只写了一部分到磁盘，剩下的丢失了。

由于文件读写时都有可能产生IOError，一旦出错，后面的f.close()就不会调用。所以，为了保证无论是否出错都能正确地关闭文件，我们可以使用try ... finally来实现：

```python
try:
    f = open('test.py', 'w')
    f.write('hello world')
    print('done')
finally:
    if f:
        f.close()
```

但是每次都这么写实在太繁琐，所以，Python引入了with语句来自动帮我们调用close()方法：

```python
with open('test.py', 'w') as f:
    f.write('hello world')
    print('done')
```

- 读文件

```python
>>> f = open('test.py', 'r') # r是读取文本文件，读取二进制文件（图片、视频）使用'rb'
>>> f.read()
"Hello, world"
>>> f.close() 
```

```python
with open('test.py', 'r') as f:
    print(f.read())
```

##### 操作文件和目录
如果我们要操作文件、目录，可以在命令行下面输入操作系统提供的各种命令来完成。比如dir、cp等命令。

如果要在Python程序中执行这些目录和文件的操作怎么办？其实操作系统提供的命令只是简单地调用了操作系统提供的接口函数，Python内置的os模块也可以直接调用操作系统提供的接口函数。

操作文件和目录的函数一部分放在os模块中，一部分放在os.path模块中，这一点要注意一下。

```python
>>> import os
>>> os.name  # 操作系统类型
'posix'      # linux/unix/mac os 如果是nt，就是Windows
>>> os.path.abspath('.')  # 查看当前目录的绝对路径
'/users/jack'
>>> os.path.join('/users/jack', 'testdir') 
# 表示出新建目录的路径，不要直接拼接字符串，这样可以正确处理不同操作系统的路径分隔符（Linux返回/,windows使用\）
'users/jack/testdir'
>>> os.mkdir('users/jack/testdir')  # 创建目录
>>> os.rmdir('users/jack/testdir')  # 删除目录
>>> os.path.split('users/jack/testdir/file.txt')  # 拆分路径
>>> os.rname('file.txt', 'test.py')  # 重命名文件
>>> os.remove('test.py')  # 删掉文件
```

```python
 import os
 
a = os.path.abspath('.')    # 列出指定目录下的文件
for filename in os.listdir(a):
    print filename

```

***
That's all.Thx~
感谢：本篇导学深受[廖雪峰Python教程](https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000)影响
如果想通过实战来加深理解，可以参考这些[Python入门的小DEMO](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/2.python_syntax)