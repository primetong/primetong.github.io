---
layout:     post
title:      "TensorFlow Notes 5 | 【TensorFlow深度学习框架教程五】"
subtitle:   "MNIST数字数据集识别与TensorFlow模型持久化"
date:       2018-04-06
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - MNIST
    - 模型持久化
---

## MNIST数字数据集识别

### MNIST数据集
当我们开始学习编程的时候，第一件事往往是学习打印"Hello World"。就好比编程入门有Hello World，机器学习入门有MNIST。深度学习当然也是啦，现在几乎每个深度学习的入门样例都是MNIST数据集，可见其影响之深。
MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片，可以说是最出名的手写体数字识别数据集一点儿都不为过。

![Mnist_img](/img/in-post/tensorflow-mnist-model-persistence/mnist_img.png)

它也包含每一张图片对应的标签，告诉我们这个是数字几。比如，上面这四张图片的标签分别是5，0，4，1。
- MNIST提供60000张图片作为数据集（其中5.5W张图是train训练集，5K张图是validation验证集）
- 其中训练集用来估计模型，验证集用来确定网络结构或者控制模型复杂程度的参数，而测试集则检验最终选择最优的模型的性能如何。
- 此外还提供了10000张图作为测试集，与上述60000数据集彼此不可见
- 这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）。

正如前面提到的一样，每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为“xs”，把这些标签设为“ys”。训练数据集和测试数据集都包含xs和ys，比如训练数据集的图片是`mnist.train.images`，训练数据集的标签是`mnist.train.labels`。
每一张图片包含28X28个像素点。我们可以用一个数字数组来表示这张图片：

![Mnist-matrix](/img/in-post/tensorflow-mnist-model-persistence/mnist-matrix.png)

我们把这个数组展开成一个向量，长度是 28x28 = 784。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在784维向量空间里面的点, 并且拥有比较[复杂的结构](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) (提醒: 此类数据的可视化是计算密集型的)。

展平图片的数字数组会丢失图片的二维结构信息。这显然是不理想的，最优秀的计算机视觉方法会挖掘并利用这些结构信息，我们会在后续教程中介绍。但是在这个教程中我们忽略这些结构，所介绍的简单数学模型，softmax回归(softmax regression)，不会利用这些结构信息。

因此，在MNIST训练数据集中，`mnist.train.images`是一个形状为`[60000, 784]`的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。

![Mnist-train-xs](/img/in-post/tensorflow-mnist-model-persistence/mnist-train-xs.png)

相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。为了用于这个教程，我们使标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。因此在MNIST中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如说，标签0（即人类识别的数字0）表示成([1,0,0,0,0,0,0,0,0,0])。因此，`mnist.train.labels`是一个`[60000, 10]`的矩阵。

![Mnist-train-ys](/img/in-post/tensorflow-mnist-model-persistence/mnist-train-ys.png)

MNIST数据集的官网是[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)。可以通过官网下载，当然也可以通过[我的GitHub](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/5.tf_mnist_model_persistence/mnist_without_model_persistence/MNIST_data)下载，除此之外还有一些小DEMO提供给大家。

#### 导入数据集
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("保存数据集的路径/MNIST_data", one_hot=True)
```
这里的one_hot=True，表明使用标签“one-hot vectors”。

导入成功后，正常应该是可以看到4行解压提取成功的提示，如下图所示：

![Import_mnist](/img/in-post/tensorflow-mnist-model-persistence/import_mnist.png)

我们在导入了MNIST数据集以后，就可以使用mnist里的一些函数，比如`mnist.train.labels[0]`来查看第一张图的标签如下，就是一个one-hot形式的：

![Mnist_labels](/img/in-post/tensorflow-mnist-model-persistence/mnist_labels.png)

说明了第一张图的标签是7，此时可以使用`mnist.train.images[0]`会返回一个28 x 28 = 784的一个一维数组，当然光看数组并不是那么容易看出是数字7。

#### 返回对应集的样本数
在导入了MNIST数据集以后，就可以使用`mnist.train或validation或test.num_examples`可以返回对应训练集/验证集/测试集的样本数。

![Mnist_num](/img/in-post/tensorflow-mnist-model-persistence/mnist_num.png)

### MNIST数字数据集识别
下面给出了使用MNIST数字数据集识别的一个完整实例，直接运行可以看到每训练（train）1000步在验证集（validation）上得到的准确率（accuracy）并且在所有步数训练完（模型训练完毕）后，整个（网络）模型在测试集（test）上的准确率的表现情况。
程序只训练了5000轮，模型正确率就可以达到98.31%，当然继续训练会有更好的表现，可以修改各种模型相关参数来看看这些参数对于模型准确率的相应影响。
相应位置有详细的注释，可以知道程序中每一步或者每一个参数是起什么作用的。
```Python
#coding:utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784     # 输入节点
OUTPUT_NODE = 10     # 输出节点
LAYER1_NODE = 500    # 隐藏层数       
                              
BATCH_SIZE = 100     # 每次batch打包的样本个数        

# 模型相关的参数
LEARNING_RATE_BASE = 0.8      
LEARNING_RATE_DECAY = 0.99    
REGULARAZTION_RATE = 0.0001   
TRAINING_STEPS = 5000        
MOVING_AVERAGE_DECAY = 0.99  

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 使用滑动平均类
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # 不使用滑动平均类
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)  

def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数。
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数。
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果
    y = inference(x, None, weights1, biases1, weights2, biases2)
    
    # 定义训练轮数及相关的滑动平均类 
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    
    # 计算交叉熵及其平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    # 损失函数的计算
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    regularaztion = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularaztion
    
    # 设置指数衰减的学习率。
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # 初始化回话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels} 
        
        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" %(TRAINING_STEPS, test_acc)))
   
def main(argv=None):
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    train(mnist)
```
```
After 0 training step(s), validation accuracy using average model is 0.1086 
After 1000 training step(s), validation accuracy using average model is 0.9766 
After 2000 training step(s), validation accuracy using average model is 0.981 
After 3000 training step(s), validation accuracy using average model is 0.9816 
After 4000 training step(s), validation accuracy using average model is 0.982 
After 5000 training step(s), test accuracy using average model is 0.9831
```

## TensorFlow模型持久化