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

接下来我们就来看看这次识别任务要识别的手写体是啥样的：

![Softmax-weights](/img/in-post/tensorflow-mnist-handwriting-recognition/handwriting0-9.jpg)

虽然和MNIST数据集的手写体数字有一定差别（感兴趣的可以看看我之前的第五篇里有，是属于比较飘逸的字体），但是这么工整的字体当然是难不倒我们的网络啦~

## PIL的使用
现在我们有了一个使用MNIST数据集训练好的网络，有了10张待识别的手写体数字图片，那么问题来了，我们怎么像PPAP一样把这个网络用于这些图片的识别？这时候我们的PIL就可以出场啦！

PIL：Python Imaging Library，已经是Python平台事实上的最基础的图像处理标准库了（当然量级更大些的OpenCV也是非常实用的，不过这次的任务暂时还用不到~）。该库支持多种文件格式，提供强大的图像处理功能，但API却非常简单易用。

##### Ubuntu中PIL的安装
Ubuntu中PIL的安装是非常简单的，通过apt：
```
$ sudo apt-get build-dep python-imaging
$ sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
$ sudo pip install Pillow
```
通过查看版本查看PIL是否安装成功：
```
$ python
>>> import PIL
>>> PIL.VERSION
'1.1.7'
```
如果没有安装成功的，根据提示先把缺失的包装上。如果是以下两个问题可以对应解决。
**Q&A**
1. Ubuntu使用apt-get时提示>”E: You must put some ‘source’ URIs in your sources.list”。
```
$ sudo sed -i -- 's/#deb-src/deb-src/g' /etc/apt/sources.list && sudo sed -i -- 
's/# deb-src/deb-src/g' /etc/apt/sources.list
$ sudo apt-get update
```
2. 执行apt-get update时提示W: Failed to fetch http://cn.archive.ubuntu.com/ubuntu/dists/natty/universe/source/Sources  404  Not Found  
```
$ cd /etc/apt/sources.list.d
$ sudo mv filename filename.bak   # 将提示对应的包改名备份即可（也就是删除该包）
```

##### 使用Image类
PIL中最重要的类是Image类，该类在Image模块中定义。

从文件加载图像：
```python
import Image
im = Image.open("lena.png")
```
如果成功，这个函数返回一个Image对象,可以使用该对象的属性来探索读入文件的内容。

##### 操作图像
来看看最常见的图像缩放操作，只需三四行代码：
```
import Image

# 打开一个jpg图像文件，注意路径要改成对应的，绝对相对都可以:
img = Image.open('./test.jpg')
# 获得图像尺寸:
w, h = img.size
# 缩放到50%:
img.thumbnail((w//2, h//2))
# 把缩放后的图像用jpeg格式保存:
img.save('./thumbnail.jpg', 'jpeg')
```
其他功能如切片、旋转、滤镜、输出文字、调色板等一应俱全。比如，模糊效果也只需几行代码：
```
import Image, ImageFilter

img = Image.open('./lena.jpg')
img2 = img.filter(ImageFilter.BLUR)
img2.save('./blur.jpg', 'jpeg')
```
PIL的`ImageDraw`提供了一系列绘图方法，让我们可以直接绘图。比如要生成字母验证码图片：
```
import Image, ImageDraw, ImageFont, ImageFilter
import random

# 随机字母:
def rndChar():
    return chr(random.randint(65, 90))

# 随机颜色1:
def rndColor():
    return (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2:
def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

# 240 x 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象:
font = ImageFont.truetype('Arial.ttf', 36)
# 创建Draw对象:
draw = ImageDraw.Draw(image)
# 填充每个像素:
for x in range(width):
    for y in range(height):
        draw.point((x, y), fill=rndColor())
# 输出文字:
for t in range(4):
    draw.text((60 * t + 10, 10), rndChar(), font=font, fill=rndColor2())
# 模糊:
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg');
```