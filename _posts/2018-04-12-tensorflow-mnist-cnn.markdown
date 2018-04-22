---
layout:     post
title:      "TensorFlow Notes 7 | 【TensorFlow深度学习框架教程七】"
subtitle:   "深入MNIST与初识卷积神经网络"
date:       2018-04-12
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - MNIST
    - CNN
---

## 深入MNIST
经过前几篇教程的学习，我们对于MNIST和TensorFlow都有了一定的了解。因此本教程在前面几篇文章的基础上，进一步深入MNIST，并且通过一个深度卷积神经网络在MNIST数据集上的表现来引出CNN——初识卷积神经网络。

TensorFlow是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。

在本教程中，我们将复习一下构建一个TensorFlow模型的基本步骤，以及MNIST的一些基本使用方法（假设已经看过本系列教程的前几篇）。并将通过这些步骤为MNIST构建一个深度卷积神经网络。

##### 导入MNIST数据集
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("保存数据集的路径/MNIST_data", one_hot=True)
#载入MNIST数据集，如果指定路径"保存数据集的路径/MNIST_data"下没有已经下载好的数据集，那么TensorFlow会自动从Yann LeCun的官网下载数据集
```
这在之前的教程中已经介绍了，还不太了解的可以再去看看。  
这里，mnist是一个轻量级的类。它以Numpy数组的形式存储着训练、校验和测试数据集。同时提供了一个函数，用于在迭代中获得minibatch，后面我们将会用到。

##### 运行TensorFlow的InteractiveSession
Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。

这里，我们使用更加方便的`InteractiveSession`类。通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。如果你没有使用`InteractiveSession`，那么你需要在启动session之前构建整个计算图，然后启动该计算图。
```Python
import tensorflow as tf
sess = tf.InteractiveSession()
```
###### 计算图
为了在Python中进行高效的数值计算，我们通常会使用像NumPy一类的库，将一些诸如矩阵乘法的耗时操作在Python环境的外部来计算，这些计算通常会通过其它语言并用更为高效的代码来实现。

但遗憾的是，每一个操作切换回Python环境时仍需要不小的开销。如果你想在GPU或者分布式环境中计算时，这一开销更加可怖，这一开销主要可能是用来进行数据迁移。

TensorFlow也是在Python外部完成其主要工作，但是进行了改进以避免这种开销。其并没有采用在Python外部独立运行某个耗时操作的方式，而是先让我们描述一个交互操作图，然后完全将其运行在Python外部。这与Theano或Torch的做法类似。

因此Python代码的目的是用来构建这个可以在外部运行的计算图，以及安排计算图的哪一部分应该被运行。

##### 构建Softmax回归模型
我们先建立一个拥有一个线性层的softmax回归模型。之后，我们再将其扩展为一个拥有多层卷积网络的softmax回归模型。

###### 占位符
我们通过为输入图像和目标输出类别创建节点，来开始构建计算图。
```Python
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
```
这里的`x`和`y`并不是特定的值，相反，他们都只是一个`占位符`，可以在TensorFlow运行某一计算时根据该占位符输入具体的值。

输入图片`x`是一个2维的浮点数张量。这里，分配给它的`shape`为`[None, 784]`，其中784是一张展平的MNIST图片的维度。`None`表示其值大小不定，在这里作为第一个维度值，用以指代batch的大小，意即`x`的数量不定。输出类别值`y_`也是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别。

虽然`placeholder`的`shape`参数是可选的，但有了它，TensorFlow能够自动捕捉因数据维度不一致导致的错误。

###### 变量
我们现在为模型定义权重`W`和偏置`b`。可以将它们当作额外的输入量，但是TensorFlow有一个更好的处理方式：`变量`。一个`变量`代表着TensorFlow计算图中的一个值，能够在计算过程中使用，甚至进行修改。在机器学习的应用过程中，模型参数一般用`Variable`来表示。
```Python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
我们在调用`tf.Variable`的时候传入初始值。在这个例子里，我们把`W`和`b`都初始化为零向量。`W`是一个784x10的矩阵（因为我们有784个特征和10个输出值）。`b`是一个10维的向量（因为我们有10个分类）。

"Before `Variables` can be used within a session, they must be initialized using that session. This step takes the initial values (in this case tensors full of zeros) that have already been specified, and assigns them to each `Variable`. This can be done for all `Variables` at once."

`变量`需要通过seesion初始化后，才能在session中使用。这一初始化步骤为，为初始值指定具体值（本例当中是全为零），并将其分配给每个`变量`,可以一次性为所有`变量`完成此操作。
```Python
sess.run(tf.initialize_all_variables())
```

###### 类别预测与损失函数
现在我们可以实现我们的回归模型了。这只需要一行！我们把向量化后的图片`x`和权重矩阵`W`相乘，加上偏置`b`，然后计算每个分类的softmax概率值。
```Python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
可以很容易的为训练过程指定最小化误差用的损失函数，我们的损失函数是目标类别和预测类别之间的交叉熵。
```Python
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```
注意，`tf.reduce_sum`minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的。