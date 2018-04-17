---
layout:     post
title:      "TensorFlow Notes 6 | 【TensorFlow深度学习框架教程六】"
subtitle:   "Softmax回归"
date:       2018-04-08
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - Softmax
    - Regression
---

## Softmax回归介绍
从上一个关于MNIST的教程中，我们得知了MNIST的每一张图片都表示一个数字，从0到9。我们希望得到给定图片代表每个数字的概率。比如说，我们的模型可能推测一张包含8的图片，代表数字8的概率是80%，但是判断它是9的概率还是有5%的（因为8和9都有上半部分的小圆，相当于拥有相似的特征），然后给予它代表其他数字的概率更小的值。

这是一个使用softmax回归（softmax regression）模型的经典案例。softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率，因为对于我们人类来说，这是相对来说比较直观的结果。

softmax回归（softmax regression）总结起来就两步。为了得到一张给定图片属于某个特定数字类的证据（evidence），我们对图片像素值进行加权求和。如果这个像素具有很强的证据说明这张图片不属于该类，那么相应的权值为负数，相反如果这个像素拥有有利的证据支持这张图片属于这个类，那么权值是正数。

下面的图片显示了一个模型学习到的图片上每个像素对于特定数字类的权值。红色代表负数权值，蓝色代表正数权值。

![Softmax-weights](/img/in-post/tensorflow-softmax-regression/softmax-weights.png)

我们也需要加入一个额外的偏置量（bias），因为输入往往会带有一些无关的干扰量。因此对于给定的输入图片 x 它代表的是数字 i 的证据可以表示为

![Softmax-evidence](/img/in-post/tensorflow-softmax-regression/softmax-evidence.png)

其中 Wi 表示权重，bi 表示数字 i 类的偏置量，j 代表给定图片 x 的像素索引用于像素求和。然后用softmax函数可以把这些证据转换成概率 y：

![Softmax-y](/img/in-post/tensorflow-softmax-regression/softmax-y.png)

这里的softmax可以看成是一个激励（activation）函数或者链接（link）函数，把我们定义的线性函数的输出转换成我们想要的格式，也就是关于10个数字类的概率分布。因此，给定一张图片，它对于每一个数字的吻合度可以被softmax函数转换成为一个概率值。softmax函数可以定义为：

![Softmax-normalize](/img/in-post/tensorflow-softmax-regression/softmax-normalize.png)

展开等式右边的子式，可以得到：

![Softmax-exp](/img/in-post/tensorflow-softmax-regression/softmax-exp.png)

但是更多的时候把softmax模型函数定义为前一种形式：把输入值当成幂指数求值，再正则化这些结果值。这个幂运算表示，更大的证据对应更大的假设模型（hypothesis）里面的乘数权重值。反之，拥有更少的证据意味着在假设模型里面拥有更小的乘数系数。假设模型里的权值不可以是0值或者负值。Softmax然后会正则化这些权重值，使它们的总和等于1，以此构造一个有效的概率分布。（更多的关于Softmax函数的信息，可以参考[Michael Nieslen的书里面的这个部分](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)，其中有关于softmax的可交互式的可视化解释。）

对于softmax回归模型可以用下面的图解释，对于输入的xs加权求和，再分别加上一个偏置量，最后再输入到softmax函数中：

![Softmax-regression-scalargraph](/img/in-post/tensorflow-softmax-regression/softmax-regression-scalargraph.png)

如果把它写成一个等式，我们可以得到：

![Softmax-regression-scalarequation](/img/in-post/tensorflow-softmax-regression/softmax-regression-scalarequation.png)

我们也可以用向量表示这个计算过程：用矩阵乘法和向量相加。这有助于提高计算效率。（也是一种更有效的思考方式）

![Softmax-regression-vectorequation](/img/in-post/tensorflow-softmax-regression/softmax-regression-vectorequation.png)

更进一步，可以写成更加紧凑的方式：

![Softmax-wx-b](/img/in-post/tensorflow-softmax-regression/softmax-wx-b.png)

## 实现回归模型
为了用python实现高效的数值计算，我们通常会使用函数库，比如NumPy，会把类似矩阵乘法这样的复杂运算使用其他外部语言实现。不幸的是，从外部计算切换回Python的每一个操作，仍然是一个很大的开销。如果你用GPU来进行外部计算，这样的开销会更大。用分布式的计算方式，也会花费更多的资源用来传输数据。

TensorFlow也把复杂的计算放在python之外完成，但是为了避免前面说的那些开销，它做了进一步完善。Tensorflow不单独地运行单一的复杂计算，而是让我们可以先用图描述一系列可交互的计算操作，然后全部一起在Python之外运行。（这样类似的运行方式，可以在不少的机器学习库中看到。）

接下来我们再复习一下前面几篇文章中出现的TensorFlow的基本使用方法：

使用TensorFlow之前，首先导入它：
```Python
import tensorflow as tf
```
我们通过操作符号变量来描述这些可交互的操作单元，可以用下面的方式创建一个：
```Python
x = tf.placeholder(tf.float32, [None, 784])
```
`x`不是一个特定的值，而是一个占位符`placeholder`，我们在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是`[None，784 ]`。（这里的`None`表示此张量的第一个维度可以是任何长度的。）

我们的模型也需要权重值和偏置量，当然我们可以把它们当做是另外的输入（使用占位符），但TensorFlow有一个更好的方法来表示它们：`Variable` 。 一个`Variable`代表一个可修改的张量，存在在TensorFlow的用于描述交互性操作的图中。它们可以用于计算输入值，也可以在计算中被修改。对于各种机器学习应用，一般都会有模型参数，可以用`Variable`表示。
```Python
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
```
我们赋予`tf.Variable`不同的初值来创建不同的`Variable`：在这里，我们都用全为零的张量来初始化`W`和`b`。因为我们要学习`W`和`b`的值，它们的初值可以随意设置。

注意，`W`的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，每一位对应不同数字类。`b`的形状是[10]，所以我们可以直接把它加到输出上面。

现在，我们可以实现我们的模型啦。只需要一行代码:
```Python
y = tf.nn.softmax(tf.matmul(x,W) + b)
```
首先，我们用tf.matmul(​​X，W)表示x乘以W，对应之前介绍Softmax回归的等式里面的Wx，这里`x`是一个2维张量拥有多个输入。然后再加上`b`，把和输入到`tf.nn.softmax`函数里面。

至此，我们先用了几行简短的代码来设置变量，然后只用了一行代码来定义我们的模型。TensorFlow不仅仅可以使softmax回归模型计算变得特别简单，它也用这种非常灵活的方式来描述其他各种数值计算，从机器学习模型对物理学模拟仿真模型。一旦被定义好之后，我们的模型就可以在不同的设备上运行,包括计算机的CPU，GPU，甚至是手机！

## 训练模型
为了训练我们的模型，我们首先需要定义一个指标来评估这个模型是好的。其实，在机器学习，我们通常定义指标来表示一个模型是坏的，这个指标称为成本（cost）或损失（loss），然后尽量最小化这个指标。但是，这两种方式是相同的。

一个非常常见的，非常漂亮的成本函数是“交叉熵”（cross-entropy）。交叉熵产生于信息论里面的信息压缩编码技术，但是它后来演变成为从博弈论到机器学习等其他领域里的重要技术手段。它的定义如下：

![Softmax-log](/img/in-post/tensorflow-softmax-regression/softmax-log.png)

y 是我们预测的概率分布, y' 是实际的分布（我们输入的one-hot vector)。比较粗糙的理解是，交叉熵是用来衡量我们的预测用于描述真相的低效性。更详细的关于交叉熵的解释超出本教程的范畴，但是你还是很有必要好好理解它，这里推荐一篇[colah的博客](http://colah.github.io/posts/2015-09-Visual-Information/)对于交叉熵的解释非常详尽，作者的其他文章对于本系列教程的贡献也很大。

为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
```Python
y_ = tf.placeholder("float", [None,10])
```
然后我们可以用

![Softmax-logy](/img/in-post/tensorflow-softmax-regression/softmax-logy.png)

计算交叉熵：
```Python
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
```
首先，用`tf.log`计算`y`的每个元素的对数。接下来，我们把`y_`的每一个元素和`tf.log(y)`的对应元素相乘。最后，用`tf.reduce_sum`计算张量的所有元素的总和。（注意，这里的交叉熵不仅仅用来衡量单一的一对预测和真实值，而是所有100幅图片的交叉熵的总和。对于100个数据点的预测表现比单一数据点的表现能更好地描述我们的模型的性能。

现在我们知道需要我们的模型做什么啦，用TensorFlow来训练它是非常容易的。因为TensorFlow拥有一张描述你各个计算单元的图，它可以自动地使用[反向传播算法(backpropagation algorithm)](http://colah.github.io/posts/2015-08-Backprop/)（具体也不展开细）来有效地确定你的变量是如何影响你想要最小化的那个成本值的。然后，TensorFlow会用你选择的优化算法来不断地修改变量以降低成本。
```Python
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
```
在这里，我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵。梯度下降算法（gradient descent algorithm）是一个简单的学习过程，TensorFlow只需将每个变量一点点地往使成本不断降低的方向移动。当然TensorFlow也提供了[其他许多优化算法](http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/train.html)：只要简单地调整一行代码就可以使用其他的算法。

TensorFlow在这里实际上所做的是，它会在后台给描述你的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法。然后，它返回给你的只是一个单一的操作，当运行这个操作时，它用梯度下降算法训练你的模型，微调你的变量，不断减少成本。

现在，我们已经设置好了我们的模型。在运行计算之前，我们需要添加一个操作来初始化我们创建的变量：
```Python
init = tf.initialize_all_variables()
```
现在我们可以在一个`Session`里面启动我们的模型，并且初始化变量：
```Python
sess = tf.Session()
sess.run(init)
```
然后开始训练模型，这里我们让模型循环训练1000次！
```Python
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```
该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行`train_step`。

使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）- 在这里更确切的说是随机梯度下降训练。在理想情况下，我们希望用我们所有的数据来进行每一步的训练，因为这能给我们更好的训练结果，但显然这需要很大的计算开销。所以，每一次训练我们可以使用不同的数据子集，这样做既可以减少计算开销，又可以最大化地学习到数据集的总体特性。

## 评估我们的模型