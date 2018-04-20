---
layout:     post
title:      "TensorFlow Notes 7 | 【TensorFlow深度学习框架教程七】"
subtitle:   "深入MNIST与初识卷积神经网络"
date:       2018-04-12
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - Softmax
    - Regression
---

## 深入MNIST
经过前几篇教程的学习，我们对于MNIST和TensorFlow都有了一定的了解。因此本教程在前面几篇文章的基础上，进一步深入MNIST，并且通过一个深度卷积神经网络在MNIST数据集上的表现来引出CNN——初识卷积神经网络。

TensorFlow是一个非常强大的用来做大规模数值计算的库。其所擅长的任务之一就是实现以及训练深度神经网络。

在本教程中，我们将复习一下构建一个TensorFlow模型的基本步骤，以及MNIST的一些基本使用方法。并将通过这些步骤为MNIST构建一个深度卷积神经网络。