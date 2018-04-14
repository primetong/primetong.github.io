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
- 其中训练集用来估计模型，验证集用来确定网络结构或者控制模型复杂程度的参数  
- 此外还提供了10000张图作为测试集，与上述60000数据集彼此不可见。测试集检验最终选择最优的模型的性能如何   
- 这样的切分很重要，在机器学习模型设计时必须有一个单独的测试数据集不用于训练而是用来评估这个模型的性能，从而更加容易把设计的模型推广到其他数据集上（泛化）  

正如前面提到的一样，每一个MNIST数据单元有两部分组成：一张包含手写数字的图片和一个对应的标签。我们把这些图片设为“xs”，把这些标签设为“ys”。训练数据集和测试数据集都包含xs和ys，比如训练数据集的图片是`mnist.train.images`，训练数据集的标签是`mnist.train.labels`。
每一张图片包含28X28个像素点。我们可以用一个数字数组来表示这张图片：

![Mnist-matrix](/img/in-post/tensorflow-mnist-model-persistence/mnist-matrix.png)

我们把这个数组展开成一个向量，长度是 28x28 = 784。如何展开这个数组（数字间的顺序）不重要，只要保持各个图片采用相同的方式展开。从这个角度来看，MNIST数据集的图片就是在784维向量空间里面的点, 并且拥有比较[复杂的结构](http://colah.github.io/posts/2014-10-Visualizing-MNIST/) （注意: 此类数据的可视化是计算密集型的）。

展平图片的数字数组会丢失图片的二维结构信息。这显然是不理想的，优秀的计算机视觉方法会挖掘并利用这些结构信息。但是在这个例程中，可以忽略这些结构，因为所采用的简单数学模型：softmax回归(softmax regression)，不会利用这些结构信息。

因此，在MNIST训练数据集中，`mnist.train.images`是一个形状为`[60000, 784]`的张量，第一个维度数字用来索引图片，第二个维度数字用来索引每张图片中的像素点。在此张量里的每一个元素，都表示某张图片里的某个像素的强度值，值介于0和1之间。

![Mnist-train-xs](/img/in-post/tensorflow-mnist-model-persistence/mnist-train-xs.png)

相对应的MNIST数据集的标签是介于0到9的数字，用来描述给定图片里表示的数字。为了用于这个教程，我们使标签数据是"one-hot vectors"。 一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0。因此在MNIST中，数字n将表示成一个只有在第n维度（从0开始）数字为1的10维向量。比如说，标签0（即人类识别的数字0）表示成([1,0,0,0,0,0,0,0,0,0])。因此，`mnist.train.labels`是一个`[60000, 10]`的矩阵。

![Mnist-train-ys](/img/in-post/tensorflow-mnist-model-persistence/mnist-train-ys.png)

MNIST数据集的官网是[Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)。可以通过官网下载，当然也可以通过[我的GitHub](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/5.tf_mnist_model_persistence/mnist_without_model_persistence/MNIST_data)下载，除此之外还有一些小DEMO提供给大家。

#### 导入数据集
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("保存数据集的路径/MNIST_data", one_hot=True)
#载入MNIST数据集，如果指定路径"保存数据集的路径/MNIST_data"下没有已经下载好的数据集，那么TensorFlow会自动从Yann LeCun的官网下载数据集
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
当我们使用TensorFlow训练神经网络的时候，模型持久化对于我们的训练有很重要的作用。设想一下如下两个应用场景：
- 如果我们的神经网络比较复杂，训练数据比较多，那么我们的模型训练就会耗时很长，如果在训练过程中出现某些不可预计的错误，导致我们的训练意外终止，那么我们将会前功尽弃。为了避免这个问题，我们就可以通过模型持久化（保存为CKPT格式）来暂存我们训练过程中的临时数据。
- 如果我们训练的模型需要提供给用户做离线的预测，那么我们只需要前向传播的过程，只需得到预测值就可以了，这个时候我们就可以通过模型持久化（保存为PB格式）只保存前向传播中需要的变量并将变量的值固定下来，这个时候只需用户提供一个输入，我们就可以通过模型得到一个输出给用户。

首先，我们考虑到在训练网络、验证网络时都用到了相同的前向传播的过程。减少不必要的重复代码是非常重要的，因此我们把前向传播单独拎出来作为一个类，在需要用到他的时候记得`import`即可：
```Python
# -*- coding: utf-8 -*-
import tensorflow as tf

#定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

#通过tf.get_variable函数来获取变量，增加代码可读性
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    #如果给出了正则化生成函数，把当前变量的正则化损失加入losses集合
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights

#定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    #声明第一层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):

        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    #声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    #返回最后前向传播的结果
    return layer2
```

### 保存与加载 CKPT 格式的模型
如果我们的神经网络比较复杂，训练数据比较多，那么我们的模型训练就会耗时很长，如果在训练过程中出现某些不可预计的错误，导致我们的训练意外终止，那么我们将会前功尽弃。为了避免这个问题，我们就可以通过模型持久化（保存为CKPT格式）来暂存我们训练过程中的临时数据。  
保存的 CKPT 格式的模型介绍：
- checkpoint ： 记录目录下所有模型文件列表
- ckpt.data ： 保存模型中每个变量的取值
- ckpt.meta ： 保存整个计算图的结构

保存模型总结起来就3步：
1. 确定模型保存的路径和名字
2. 声明并得到一个 Saver
3. 通过 Saver.save 保存模型

加载模型总结起来就3步：
1. 确定要加载的模型的路径和名字
2. 声明并得到一个 Saver
3. 通过 Saver.restore 加载模型

下面先分步解析，之后会给上一份完整的DEMO

```Python
# 1. 确定要加载的模型的路径和名字
#先在开头定义变量的时候定义模型保存的路径（方便以后修改。不一定要在工程目录下，自己定）
MODEL_SAVE_PATH="./MNIST_model"
#模型保存的文件名
MODEL_NAME="mnist_model"
```
然后在你定义的运算函数方法体内，在Session（会话）打开之前，先声明并得到一个Saver：
```Python
# 2. 声明并得到一个 Saver
#初始化TensorFLow的持久化类，用于保存与加载模型（也可只声明一个类就行，以下只是为演示方便）
saver = tf.train.Saver()
loader = tf.train.Saver()
```
在Session打开之后，需要在模型run起来之前加载模型，然后每隔一定的轮数保存一下模型：
```Python
# 3. 通过 Saver.save 保存模型，通过通过 Saver.restore 加载模型
#根据模型保存路径找到保存的模型文件
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)  
#在开始计算之前先判断是否有模型可以加载
if ckpt and ckpt.model_checkpoint_path:
	#加载保存的模型
	loader.restore(sess, ckpt.model_checkpoint_path)

#每隔一定的轮数在对应路径上保存一下模型
saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
```

下面给出同时可以保存与加载 CKPT 格式的模型的训练MNIST数据集的完整代码：
```Python
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
#加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

#配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
#模型保存的路径
MODEL_SAVE_PATH="/home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_model"
#模型保存的文件名
MODEL_NAME="mnist_model"

def train(mnist):
    #定义输入输出placeholder
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #可以直接使用mnist_inference.py中定义的前向传播过程，并传入正则化参数
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    #定义损失函数、学习率、滑动平均、训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    #初始化TensorFLow的持久化类，用于保存与加载模型
    saver = tf.train.Saver()
    #variables_to_restore = variable_averages.variables_to_restore()#变量保存
    #loader = tf.train.Saver(variables_to_restore)#初始化用于加载的持久化类
    loader = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)#根据模型保存路径找到保存的模型文件
        if ckpt and ckpt.model_checkpoint_path:#在开始计算之前先判断是否有模型可以加载
            loader.restore(sess, ckpt.model_checkpoint_path)#加载保存的模型

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
#            if i == 0 or (i+1) % 1000 == 0:#test:使重新加载打印时的step不+1的方法1
#            if step % 1000 == 0:#test:使重新加载打印时的step不+1的方法2
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("/home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_data", one_hot=True)#训练集的路径
    train(mnist)
```
当然，光有训练是看不出训练的效果的啦，此时我们应该同时打开验证集的代码对这个模型（你保存的）的效果进行同步的验证（代码就不直接放出了占篇幅，只需在以上训练代码的基础上修改输入等操作即可，感兴趣的可以下载完整工程[MNIST数字数据集识别与TensorFlow模型持久化](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/5.tf_mnist_model_persistence)）运行结果如下：
```
Extracting /home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_data/train-images-idx3-ubyte.gz
Extracting /home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_data/train-labels-idx1-ubyte.gz
Extracting /home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting /home/witt/witt/Python/5.tf_mnist_model_persistence/MNIST_data/t10k-labels-idx1-ubyte.gz
After 14000 training step(s), validation accuracy = 0.9838
After 15000 training step(s), validation accuracy = 0.9846
After 16000 training step(s), validation accuracy = 0.985
After 17000 training step(s), validation accuracy = 0.985
After 18000 training step(s), validation accuracy = 0.9844
After 19000 training step(s), validation accuracy = 0.9848
After 20000 training step(s), validation accuracy = 0.9852
After 21000 training step(s), validation accuracy = 0.9846
After 22000 training step(s), validation accuracy = 0.9848
After 23000 training step(s), validation accuracy = 0.9854
After 24000 training step(s), validation accuracy = 0.9852
…
```

### 保存为 PB 格式模型
tf.train.Saver的缺点就是每次会保存程序的全部信息，但有时并不需要全部信息。比如在测试或离线预测时，只需要知道如何从神经网络的输入层经过前向传播计算得到输出层即可，而不需要类似于变量初始化、模型保存等辅助结点的信息。而且，将变量取值和计算图结构分成不同文件存储有时候也不方便，TensorFlow中提供了convert_variables_to_constants函数，可以将计算图中的变量及其取值通过常量的方式保存，这样可以将整个计算图统一存放在一个文件中。  
如果我们训练的模型需要提供给用户做离线的预测，那么我们只需要前向传播的过程，只需得到预测值就可以了，这个时候我们就可以通过模型持久化（保存为PB格式）只保存前向传播中需要的变量并将变量的值固定下来，这个时候只需用户提供一个输入，我们就可以通过模型得到一个输出给用户。  
总结起来有4步：
1. 定义运算过程
2. 通过 get_default_graph().as_graph_def() 得到当前图的计算节点信息
3. 通过 graph_util.convert_variables_to_constants 将相关节点的values固定
4. 通过 tf.gfile.GFile 进行模型持久化

```Python
# -*- coding: utf-8 -*-
import tensorflow as tf
import shutil
import os.path
from tensorflow.python.framework import graph_util


# MODEL_DIR = "model/pb"
# MODEL_NAME = "addmodel.pb"

# if os.path.exists(MODEL_DIR): 删除目录
#     shutil.rmtree(MODEL_DIR)
#
# if not tf.gfile.Exists(MODEL_DIR): #创建目录
#     tf.gfile.MakeDirs(MODEL_DIR)

output_graph = "model/pb/add_model.pb"

#下面的过程你可以替换成CNN、RNN等你想做的训练过程，这里只是简单的一个计算公式
input_holder = tf.placeholder(tf.float32, shape=[1], name="input_holder")
W1 = tf.Variable(tf.constant(5.0, shape=[1]), name="W1")
B1 = tf.Variable(tf.constant(1.0, shape=[1]), name="B1")
_y = (input_holder * W1) + B1
# predictions = tf.greater(_y, 50, name="predictions") #比50大返回true，否则返回false
predictions = tf.add(_y, 10,name="predictions") #做一个加法运算

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print "predictions : ", sess.run(predictions, feed_dict={input_holder: [10.0]})
    graph_def = tf.get_default_graph().as_graph_def() #得到当前的图的 GraphDef 部分，通过这个部分就可以完成重输入层到输出层的计算过程

    output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess,
        graph_def,
        ["predictions"] #需要保存节点的名字
    )
    with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
        f.write(output_graph_def.SerializeToString())  # 序列化输出
    print("%d ops in the final graph." % len(output_graph_def.node))
    print (predictions)

# for op in tf.get_default_graph().get_operations(): 打印模型节点信息
#     print (op.name)
```
- GraphDef：这个属性记录了TensorFlow计算图上节点的信息。
- add_model.pb ： 里面保存了从输入层到输出层这个计算过程的计算图和相关变量的值，我们得到这个模型后传入一个输入，既可以得到一个预估的输出值

### CKPT 转换成 PB格式
1. 通过传入 CKPT 模型的路径得到模型的图和变量数据
2. 通过 import_meta_graph 导入模型中的图
3. 通过 saver.restore 从模型中恢复图中各个变量的数据
4. 通过 graph_util.convert_variables_to_constants 将模型持久化

```Python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os.path
import argparse
from tensorflow.python.framework import graph_util

MODEL_DIR = "model/pb"
MODEL_NAME = "frozen_model.pb"

if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

def freeze_graph(model_folder):
    checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    output_graph = os.path.join(MODEL_DIR, MODEL_NAME) #PB模型保存路径

    output_node_names = "predictions" #原模型输出操作节点的名字
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True) #得到图、clear_devices ：Whether or not to clear the device field for an `Operation` or `Tensor` during import.

    graph = tf.get_default_graph() #获得默认的图
    input_graph_def = graph.as_graph_def()  #返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据

        print "predictions : ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}) # 测试读出来的模型是否正确，注意这里传入的是输出 和输入 节点的 tensor的名字，不是操作节点的名字

        output_graph_def = graph_util.convert_variables_to_constants(  #模型持久化，将变量值固定
            sess,
            input_graph_def,
            output_node_names.split(",") #如果有多个输出节点，以逗号隔开
        )
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

        for op in graph.get_operations():
            print(op.name, op.values())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_folder", type=str, help="input ckpt model dir") #命令行解析，help是提示符，type是输入的类型，
    # 这里运行程序时需要带上模型ckpt的路径，不然会报 error: too few arguments
    aggs = parser.parse_args()
    freeze_graph(aggs.model_folder)
    # freeze_graph("model/ckpt") #模型目录
```

如果想通过实战来加深对MNIST、模型持久化等的理解，可以参考这些[MNIST数字数据集识别与TensorFlow模型持久化的小DEMO](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.TensorFlow%26Python/5.tf_mnist_model_persistence)