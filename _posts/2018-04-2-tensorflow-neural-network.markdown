---
layout:     post
title:      "TensorFlow Notes 3 | 【TensorFlow深度学习框架教程三】"
subtitle:   "初识TensorFlow和神经网络"
date:       2018-4-2
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - Neural Network
---

## 初识TensorFlow

### TensorFlow的前世今生

TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。
TensorFlow可被用于语音识别或图像识别等多项机器学习和深度学习领域，对2011年开发的深度学习基础架构DistBelief进行了各方面的改进，它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow将完全开源，任何人都可以用。
在过去的一年（2017年）中，TensorFlow成功成为Github年度报告盘点的赢家——因为Github Fork最多的项目就是它，这也是唯一上榜的机器学习项目（机器学习包含深度学习）。近两年，机器学习热度疯涨，在GitHub上无论是Star数量还是Fork数都很高，这也说明很多开发者都开始向这个方向转型，可能已经在准备将机器学习作为今后的主要研究方向之一。

### TensorFlow的组成

Tensorflow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。而基于Tensorflow的NN（Neural Network，神经网络）就是用张量表示数据，用计算图搭建NN，用会话执行计算图，优化线上的权重（参数），得到模型。

##### ①张量（Tensor）：

- Tensor：张量，即多维数组（列表,处理类型）
- 阶是张量的维数。零阶张量——>标量；1阶张量——>向量；n阶张量——>n维数组
- 张量中并没与保存具体的数字，而是保存的这些数字的计算过程
- 张量包括三种属性：
	- a. 名字（唯一的标识符，同时给出张量如何计算得来的）
	- b. 维度（get_shape函数可以获得维度信息）
	- c. 类型（张量运算必须保持类型一致）
- 张量的使用主要分为两类：
	- a. 对中间结果的引用
	- b. 计算图构造完成后，可以用来获得结果（通过会话）。在会话中可以使用Tensor.eval()函数获得张量的取值

##### ②计算图：

- Flow：计算模型（计算图）
- 用节点搭建NN（节点构成计算图）
- 只搭建不运算（Y = X * W）

#### ③会话（Session）：

- 执行计算图中的节点运算，拥有并管理TensorFlow此程序运行时的所有资源，计算完成后，需要关闭会话。
- 会话的形式有两种：
	- 需要明确调用会话生成函数和关闭会话函数。这种模式的缺点是：当程序因为异常而退出时，关闭会话函数可能就不会被执行，从而导致资源泄露。具体形式如下：
	
	```python
	sess=tf.Session()
	sess.run(…)
	sess.close()
	```

	- 将所有的计算放在with的内部。这种模式的优点：当上下文管理器退出时，会自动将所有资源释放。这样既解决了因为异常退出时资源释放的问题，同时也解决了忘记调用Session.close函数而产生的资源泄露。具体的形式如下：
	
	```python
	sess=tf.Session()
	sess.run(…)
	sess.close()
	```

下面给出一个例子，可以跑一跑试试看：

```python
import tensorflow as tf  
  
a=tf.constant([[1.0,2.0,3],[3.0,4.0,3]],name='a',dtype=tf.float32)  
b=tf.constant([[3,4,3],[5,6,3]],name='b',dtype=tf.float32)  
  
result=a+b  
  
with tf.Session() as sess:  
    print(result.eval())  

#sess=tf.Session()  
#print(result.eval())  
#sess.close()  
#通过a.graph可以查看张量所属的计算图。因为没有特意指定，所以这个计算图应该等于当前默认的计算图，所以结果为True  
#print(a.graph is tf.get_default_graph())  
  
#print(result)  
  
#sess=tf.Session()  
#print(result.eval(session=sess))  
  
#sess=tf.Session()  
#c=sess.run(result)  
#print(c)  
#sess.close()  
  
#with tf.Session() as sess:  
#    c=sess.run(result)  
#    print(c)  
  
#sess=tf.Session()  
#with sess.as_default():  
#    print(result.eval())  
  
#config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)  
#sess1=tf.InteractiveSession(config=config)  
#sess2=tf.Session(config=config)  
#c=sess2.run(result)  
#print(c)  
  
#sess=tf.InteractiveSession()  
#print(result.eval())  
```

#### ④参数：

- 即线上的权重w，用变量表示，常随机生成
- 常用函数：`tf.truncated_normal()`、`tf.random_uniform`、`tf.zeros`、`tf.ones`、`tf.fill()`、`tf.constant()`等