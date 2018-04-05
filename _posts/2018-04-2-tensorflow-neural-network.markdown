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
    - 前向传播
    - 反向传播
---

## 初识TensorFlow

### TensorFlow的前世今生

TensorFlow是谷歌基于DistBelief进行研发的第二代人工智能学习系统，其命名来源于本身的运行原理。Tensor（张量）意味着N维数组，Flow（流）意味着基于数据流图的计算，TensorFlow为张量从流图的一端流动到另一端计算过程。TensorFlow是将复杂的数据结构传输至人工智能神经网中进行分析和处理过程的系统。  
TensorFlow可被用于语音识别或图像识别等多项机器学习和深度学习领域，对2011年开发的深度学习基础架构DistBelief进行了各方面的改进，它可在小到一部智能手机、大到数千台数据中心服务器的各种设备上运行。TensorFlow将完全开源，任何人都可以用。  
在过去的一年（2017年）中，TensorFlow成功成为Github年度报告盘点的赢家——因为Github Fork最多的项目就是它，这也是唯一上榜的机器学习项目（机器学习包含深度学习）。近两年，机器学习热度疯涨，在GitHub上无论是Star数量还是Fork数都很高，这也说明很多开发者都开始向这个方向转型，可能已经在准备将机器学习作为今后的主要研究方向之一。

### TensorFlow的安装

关于TensorFlow的安装，网上也有很多教程了，对于大家来说应该没有什么难度，这里只讲讲CPU版本的安装  
[TensorFlow只支持Nvidia计算能力（compute capability）大于3.0的GPU。如果要支持GPU，那么还需要安装Nvidia的Cuda Tookit（版本大于等于7.0）和cuDNN（版本大于等于v2）]  
在Ubuntu/Linux 64-bit, Python 2.7环境， CPU Only环境下安装简单来说有三步：  

- 1.安装pip

```
$ sudo apt-get install python-pip python-dev
```

- 2.找到合适的安装包URL

```
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp27-none-linux_x86_x64.whl
```

- 3.通过pip安装TensorFlow

```
$ sudo pip install --upgrade $TF_BINARY_URL
```

通过以上简单3步，TensorFlow环境就安装完成了（可以根据自己机子的需求修改相应指令参数）。  
此时可以编译运行[验证环境小程序](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/1.tf_test_env)，测试TensorFlow环境是否已经成功安装。

### TensorFlow的组成

Tensorflow中的每一个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。  
而基于Tensorflow的NN（Neural Network，神经网络）就是用张量表示数据，用计算图搭建NN，用会话执行计算图，优化线上的权重（参数），得到模型。

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

##### ③会话（Session）：

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
在交互式环境下，通过设置默认会话获取张量更加方便。Tensorflow提供了一种在交互式环境下构建默认会话的函数tf.InteractiveSession()，这个函数会自动将生成的会话注册为默认会话。
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

##### ④参数：

- 即线上的权重w，用变量表示，常随机生成
- 常用函数：`tf.truncated_normal()`、`tf.random_uniform`、`tf.zeros`、`tf.ones`、`tf.fill()`、`tf.constant()`等，在下面的神经网络前向传播中会具体介绍。

---

## 神经网络（Neural Network）

### 神经网络解决分类问题的4个步骤：

#### ①提取问题中实体的特征向量作为神经网络的输入（构建输入）

#### ②定义神经网络的结构，并定义如何从神经网络的*输入到输出*（神经网络前向传播过程，先搭计算图，再用会话执行）：

##### 前向传播算法：

神经元是构建神经网络的最小单元，每个神经元也可以称为节点。神经网络的前向传播算法可以通过`tf.matmul`函数实现。

###### 神经网络的参数和TensorFlow变量：

TensorFlow中变量的作用就是保存和更新神经网络中的参数，变量通过`tf.Variable`函数实现，变量的初始化可以通过三种方式实现：

- 可以通过随机数生成函数，来对变量初始化，例如`tf.random_normal`(正态分布)、`tf.truncated_normal`(正太分布，但如果随机出来的值偏离平均值超过两个标准差，那么这个数将会被重新随机)、`tf.random_uniform`(平均分布)、`tf.random_gamma`(Gamma分布)
- 也可以通过常数来初始化一个变量。例如`tf.zeros`(产生全零的数组)、`tf.ones`(产生全1的数组)、`tf.fill`(产生一个全部为给定数字的数组)、`tf.constant`(产生一个给定值的常量)`。
- 也支持通过其他变量的初始值来初始化新的变量。例如：
	- w2=tf.Variable(weights.initialized_value())
	- w3=tf.Variable(weights.initialized_value()*2.0)
		- 以上代码w2的初始值与weights相同，w3的初始值则是weights初始值的两倍。

虽然在变量定义时给出了变量初始化的方法，但是这个方法并没有真正运行。在会话中需要将其运行。在会话中运行初始化主要有两种方式：

- 1.sess.run(w1.initializer)  
sess.run(w2.initializer)  
这种方式的缺点：当变量的数目增多时，或者变量之间存在依赖关系式，单个调用的方案就比较麻烦。

- 2.可以通过tf.global_variables_initializer函数实现所有变量的初始化（旧版TensorFlow中使用tf.initialize_all_variables()。但是这个函数已经被弃用，由tf.global_variables_initializer()代替）。  
init_op=tf.global_variables_initializer()  
sess.run(init_op)  
优点：不需要对变量一个一个初始化，同时这种方式会自动处理变量之间的依赖关系。
可以使用w1.assign(w2)函数，来对w1的值进行更新，需要注意的是要保证w1和w2张量中的数据类型是一致的。  

	举个例子如下：
	
	```python
	import tensorflow as tf  
	  
	#声明w1和w2两个变量，还通过seed参数设定了随机种子，这样可以保证每次运行结果是一样的  
	w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))  
	w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))  
	  
	#设定x（输入）为常数。注意这里x是一个1*2的矩阵  
	x=tf.constant([[0.7,0.9]])  
	  
	a=tf.matmul(x,w1)#实现x和w1相乘  
	y=tf.matmul(a,w2)  
	  
	sess=tf.Session()  
	  
	#初始化方法1  
	#sess.run(w1.initializer)#初始化w1  
	#sess.run(w2.initializer)#初始化w2  
	  
	#初始化方法2  
	init_op=tf.global_variables_initializer()  
	sess.run(init_op)  
	  
	print(sess.run(y))  
	  
	sess.close()  
	```

#### ③大量特征数据喂给NN（NN反向传播，训练模型），迭代优化参数（通过训练数据来调整神经网络中参数的取值）：

在神经网络优化算法中，最常用的方法是反向传播算法，在TensorFlow中通过`tf.train.AdamOptimizer()`实现，例如`train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)`。

TensorFlow提供了placeholder机制用于提供输入数据。placeholder相当于定义了一个位置，这个位置中的数据在程序运行时在指定。在placeholder定义时，这个位置上上的数据类型是需要指定的，同时placeholder的类型也是不可以改变的。Placeholder中数据的维度可以根据数据推导出，所以不一定要给出。

在会话中执行时，需要提供一个feed_dict来指定x的取值。Feed_dict是一个字典，字典中需要给出每个用到的placeholder的取值。

##### 反向传播：

- （1）训练NN模型 反向传输：在所有参数上用梯度下降，使NN模型在训练数据上的损失函数最小。
- （2）损失函数（Loss）：预测的y和已知答案y_的差距
	- 交叉熵（cross_entropy）:H(p,q) = -Σp(x)logq(x)
	- 均方误差：MSE(y_,y) = Σ(y - y_)^2 / n
	- 自定义
- （3）学习率Learning_rate：每次参数更新的幅度

#### ④使用训练好的神经网络来预测未知的数据（测试数据）。

---

简单总结一下TensorFlow中神经网络前传以及反传更新参数的过程：
- ①初始化参数，训练次数 = 0；
- ②选一小撮（batch_size）训练数据；
- ③前向传播得到预测值；
- ④反向传播更新参数；
- ⑤以上不停循环直到次数到或者目标到（目标一般是损失函数的要求）

下面给出了一个完整的程序来训练神经网络解决二分类问题，从这段程序中可以总结出训练神经网络的通用过程，可分为以下4个步骤：
- ①抽取实体特征作为输入喂给神经网络
- ②定义神经网络的结构和前向传播算法的结果
- ③定义损失函数以及选择反向传播算法优化的算法
- ④生成会话并且在训练数据集上反复运行反向传播优化算法。

```python
#coding:utf-8
#1.导入模块，生成模拟数据集。
import tensorflow as tf
#通过Numpy工具包模拟数据集
from numpy.random import RandomState
BATCH_SIZE = 8	#训练数据batch的大小 

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
X = rdm.rand(128,2)
#定义规则来给出样本的标签。所有x1+x2<1的样例都认为是正样本而其他的为负样本。1表示正样本；0表示负样本  
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]


#2. 定义神经网络的常量,参数，输入和输出节点,定义前向传播过程。
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))


x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_= tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#3. 定义损失函数及反向传播算法。
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))) 
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
#train_step = tf.train.MomentumOptimizer(0.001,0.9).minimize(cross_entropy)


#4. 创建一个会话来运行TensorFlow程序。反复运行反向传播
with tf.Session() as sess:
	#初始化参数
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 输出目前（未经训练）的参数取值。
    print "w1:\n", sess.run(w1)
    print "w2:\n", sess.run(w2)
    print "\n"
    
    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
		#每次选取batch_size个样本进行训练
        start = (i*BATCH_SIZE) % 128
        end = (i*BATCH_SIZE) % 128 + BATCH_SIZE
		#通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
			#每隔一段时间计算在所有数据集上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
    
    # 输出训练后的参数取值。
    print "\n"
    print "w1:\n", sess.run(w1)
    print "w2:\n", sess.run(w2)
```
```
#输出结果的部分打印信息
"""
w1: [[-0.81131822  1.48459876  0.06532937]
 [-2.44270396  0.0992484   0.59122431]]
w2: [[-0.81131822]
 [ 1.48459876]
 [ 0.06532937]]


After 0 training step(s), cross entropy on all data is 0.0674925
After 1000 training step(s), cross entropy on all data is 0.0163385
After 2000 training step(s), cross entropy on all data is 0.00907547
After 3000 training step(s), cross entropy on all data is 0.00714436
After 4000 training step(s), cross entropy on all data is 0.00578471


w1: [[-1.9618274   2.58235407  1.68203783]
 [-3.4681716   1.06982327  2.11788988]]
w2: [[-1.8247149 ]
 [ 2.68546653]
 [ 1.41819501]]
"""
```

如果想通过实战来加深理解，可以参考这些[初识TensorFlow和神经网络的小DEMO](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/3.tensorflow_neural_network)