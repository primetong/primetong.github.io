---
layout:     post
title:      "OpenCV Notes 4 | 【OpenCV图像处理入门学习教程四】"
subtitle:   "基于LoG算子的图像边缘检测"
date:       2018-03-27
author:     "Witt"
header-img: "img/post-bg-opencv.jpg"
tags:
    - OpenCV
    - LoG
    - 自定义算子
    - 边缘检测
---

# LoG边缘检测算子

LoG边缘检测算子是David Courtnay Marr和Ellen Hildreth（1980）共同提出的。因此，也称为边缘检测算法或Marr & Hildreth算子。该算法首先对图像做高斯滤波，然后再求其拉普拉斯（Laplacian）二阶导数。即图像与 Laplacian of the Gaussian function 进行滤波运算。最后，通过检测滤波结果的零交叉（Zero crossings）可以获得图像或物体的边缘。因而，也被业界简称为Laplacian-of-Gaussian (LoG)算子。

算法描述：LoG算子也就是 Laplace of Gaussian function（高斯拉普拉斯函数）。常用于数字图像的边缘提取和二值化。LoG 算子源于D.Marr计算视觉理论中提出的边缘提取思想, 即首先对原始图像进行最佳平滑处理, 最大程度地抑制噪声, 再对平滑后的图像求取边缘。
由于噪声点（灰度与周围点相差很大的像素点）对边缘检测有一定的影响，所以效果更好的边缘检测器是LoG算子，也就是Laplacian-Gauss算子。它把的Gauss平滑滤波器和Laplacian锐化滤波器结合了起来，先平滑掉噪声，再进行边缘检测，所以效果会更好。 

高斯函数和一级、二阶导数如下图所示：

![Gaussian-derivative](/img/in-post/opencv-log-edge-detection/gaussian-derivative.png)

LoG算子到中心的距离与位置加权系数的关系曲线象墨西哥草帽的剖面，所以LoG算子也叫墨西哥草帽滤波器。

![LoG-1](/img/in-post/opencv-log-edge-detection/log-one.png)

拉普拉斯算子是二阶差分算子，为什么要加入二阶的算子呢？试想一下，如果图像中有噪声，噪声在一阶导数处也会取得极大值从而被当作边缘。然而求解这个极大值也不方便，采用二阶导数后，极大值点就为0了，因此值为0的地方就是边界。

如下图所示，上面是一阶导数，下面是二阶导数：

![LoG-2](/img/in-post/opencv-log-edge-detection/log-two.png)

### 一、基于LoG算子的图像边缘检测

原图：

![Origin-lenna](/img/in-post/opencv-log-edge-detection/origin-lenna.jpg)

1.LoG算子与自定义滤波算子进行比较的结果：

![Log-running-result](/img/in-post/opencv-log-edge-detection/log-running-result.png)

2.LoG算子的结果：log-result

![Log-result](/img/in-post/opencv-log-edge-detection/log-result.png)

3.自定义3*3

|  1  |  1  |  1  |
|-----|-----|-----|
|  1  | -8  |  1  |
|  1  |  1  |  1  |

滤波结果：

![Custom-result](/img/in-post/opencv-log-edge-detection/custom-result.png)

### 二、代码解析

下面是一段基于LoG算子的图像边缘检测的代码，同时会生成两个结果，一个是LoG算子的结果，第二个是自定义3*3大小的一个算子的滤波结果（可以修改对应代码实现你自己想要的算子）

* IDE：Visual Studio 2013

* 语言：C++

* 依赖：OpenCV 2.4.9

程序是在VS2013和OpenCV2.4.9下运行的，部分参考代码如下，相应位置有详细注释，整个工程文件见下载页面：

```

#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  

using namespace cv;

int main()
{
	//使用LoG算子做边缘检测
	Mat src, src_gray;
	int kernel_size = 3;
	const char* window_name = "Laplacian-of-Gaussian Edeg Detection";

	src = imread("Lenna.jpg");
	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);	//先通过高斯模糊去噪声
	cvtColor(src, src_gray, CV_RGB2GRAY);
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	Mat dst, abs_dst;
	Laplacian(src_gray, dst, CV_16S, kernel_size);	//通过拉普拉斯算子做边缘检测
	convertScaleAbs(dst, abs_dst);

	imshow(window_name, abs_dst);

	//使用自定义滤波做边缘检测
	//自定义滤波算子 1  1  1
	//               1 -8  1
	//               1  1  1
	Mat custom_src, custom_gray, Kernel;
	custom_src = imread("Lenna.jpg");
	GaussianBlur(custom_src, custom_src, Size(3, 3), 0, 0, BORDER_DEFAULT);	//先通过高斯模糊去噪声
	cvtColor(custom_src, custom_gray, CV_RGB2GRAY);
	namedWindow("Custom Filter", CV_WINDOW_AUTOSIZE);

	Kernel = (Mat_<double>(3, 3) << 1, 1, 1, 1, -8, 1, 1, 1, 1);	//自定义滤波算子做边缘检测
	Mat custdst, abs_custdst;
	filter2D(custom_gray, custdst, CV_16S, Kernel, Point(-1, -1));
	convertScaleAbs(custdst, abs_custdst);

	imshow("Custom Filter", abs_custdst);
	waitKey(0);

	return 0;
}

```

从实验比较结果来看，OpenCV自带的LoG算子的图像边缘检测和自定义的算子的肉眼观测结果差距不大，大家可以自己试试改一改自定义算子来观察实验结果~

基于LoG算子的图像边缘检测，整个工程文件见下载页面