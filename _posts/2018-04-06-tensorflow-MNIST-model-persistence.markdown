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
现在几乎每个深度学习的入门样例都是MNIST数据集，可见其影响之深，可以说是最出名的手写体数字识别数据集一点儿都不为过。
- MNIST提供60000张图片作为数据集（其中5.5W张图是train训练集，5K张图是validation验证集）
- 此外还提供了10000张图作为测试集，与上述60000数据集彼此不可见