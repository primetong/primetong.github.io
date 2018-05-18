---
layout:     post
title:      "TensorFlow Notes 8 | 【TensorFlow深度学习框架教程八】"
subtitle:   "PIL的使用与MNIST识别手写数字"
date:       2018-04-28
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - PIL
    - MNIST
    - Handwriting-recognition
---

## MNIST识别手写数字
经过前几篇教程的学习，我们对于MNIST和TensorFlow已经有了很深入的了解了。这时候大家是不是有个疑问，我们光用这个网络模型来测试在MNIST数据集上识别的正确率有什么实际意义呢？在深度学习中，模型的任务一般分为分类和回归两类，训练出来的网络模型在实际中的应用大多是给他一个未知的输入，完成相应的分类或回归的任务。而我们就可以使用之前通过MNIST数据集训练的网络模型来完成对手写体数字的识别，你可别小瞧这个简单的任务，对人类来说是就是瞅一眼的事儿，对计算机来说可是要多动脑筋的呢！