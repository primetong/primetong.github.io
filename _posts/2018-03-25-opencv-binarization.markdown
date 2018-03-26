---
layout:     post
title:      "OpenCV Notes 2 | 【OpenCV图像处理入门学习教程二】"
subtitle:   "不同阈值二值化图像"
date:       2018-03-25
author:     "Witt"
header-img: "img/post-bg-opencv.jpg"
tags:
    - OpenCV
    - 二值化
---

# 图像二值化介绍

图像二值化是图像预处理中非常重要的一部分。图像二值化简单来说就是将256个亮度等级的灰度图像通过适当的阈值选取而获得仍然可以反映图像整体和局部特征的二值化图像。
在数字图像处理中，二值图像占有非常重要的地位，首先，图像的二值化有利于图像的进一步处理，使图像变得简单，而且数据量减小，能凸显出感兴趣的目标的轮廓。其次，要进行二值图像的处理与分析，首先要把灰度图像二值化，得到二值化图像。

### 一、OpenCV中的图像二值化

在OpenCV中，图像二值化最关键的一个函数就是——cvThreshold()
下面简单地介绍一下这个函数，有关该函数的具体定义与说明，可以在OpenCV的imgproc\types_c.h中找到。
该函数的主要功能就是，采用Canny方法对图像进行边缘检测
函数原型及说明如下：

> void cvThreshold(               //函数说明：
const CvArr* src,		//第一个参数表示输入图像，必须为单通道灰度图。
CvArr* dst,			//第二个参数表示输出的边缘图像，为单通道黑白图。
double threshold,		//第三个参数表示阈值
double max_value,		//第四个参数表示最大值。
int threshold_type		//第五个参数表示运算方法。
);

上述的第五个参数也就是Threshold types如下：

> /* Threshold types ↓*/
enum
{	CV_THRESH_BINARY = 0,  /* value = value > threshold ? max_value : 0       */
	CV_THRESH_BINARY_INV = 1,  /* value = value > threshold ? 0 : max_value       */
	CV_THRESH_TRUNC = 2,  /* value = value > threshold ? threshold : value   */
	CV_THRESH_TOZERO = 3,  /* value = value > threshold ? value : 0           */
	CV_THRESH_TOZERO_INV = 4,  /* value = value > threshold ? 0 : value           */
	CV_THRESH_MASK = 7,
	CV_THRESH_OTSU = 8  /* use Otsu algorithm to choose the optimal threshold value; combine the flag with one of the above CV_THRESH_* values */
  //    最后一个是自适应算法取阈值（最大类间方差法），这样前面的第四个参数threshold会无效
};

### 二、基于OpenCV3.3的图像二值化（阈值可调）

下面是一段基于OpenCV3.3的图像二值化实例代码，阈值可以通过滑动条来调节，可以观察不同阈值：

* IDE：Visual Studio 2013

* 语言：C++

* 依赖：OpenCV 3.3.0

整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/Binarization)

```

#include <opencv.hpp>
using namespace std;

IplImage *g_pGrayImage = NULL;
IplImage *g_pBinaryImage = NULL;
const char *pstrWindowsBinaryTitle = "二值化之后的图像";

void on_trackbar(int val)
{
	// 转为二值图  
	cvThreshold(g_pGrayImage, g_pBinaryImage, val, 255, CV_THRESH_BINARY);
	// 显示二值图  
	cvShowImage(pstrWindowsBinaryTitle, g_pBinaryImage);
}

int main(int argc, char** argv)
{
	const char *pstrWindowsSrcTitle = "原图 - by Lenna";
	const char *pstrWindowsToolBarName = "二值化阈值";

	// 从文件中加载原图  
	IplImage *pSrcImage = cvLoadImage("Lenna.jpg", CV_LOAD_IMAGE_UNCHANGED);

	// 转为灰度图  Gray = R*0.299 + G*0.587 + B*0.114 （通道顺序B->G->R）
	g_pGrayImage = cvCreateImage(cvGetSize(pSrcImage), IPL_DEPTH_8U, 1);
	cvCvtColor(pSrcImage, g_pGrayImage, CV_BGR2GRAY);

	// 创建二值图  
	g_pBinaryImage = cvCreateImage(cvGetSize(g_pGrayImage), IPL_DEPTH_8U, 1);

	// 显示原图  
	cvNamedWindow(pstrWindowsSrcTitle, CV_WINDOW_AUTOSIZE);
	cvShowImage(pstrWindowsSrcTitle, pSrcImage);
	// 创建二值图窗口  
	cvNamedWindow(pstrWindowsBinaryTitle, CV_WINDOW_AUTOSIZE);

	// 滑动条    
	int nThreshold = 63;
	cvCreateTrackbar(pstrWindowsToolBarName, pstrWindowsBinaryTitle, &nThreshold, 254, on_trackbar);

	on_trackbar(63);		//初始阈值的设置，初步调试设为63

	cvWaitKey(0);

	cvDestroyWindow(pstrWindowsSrcTitle);
	cvDestroyWindow(pstrWindowsBinaryTitle);
	cvReleaseImage(&pSrcImage);
	cvReleaseImage(&g_pGrayImage);
	cvReleaseImage(&g_pBinaryImage);
	return 0;
}

```

运行结果如下：

![Jpeg-error](/img/in-post/opencv-binarization/binarization-origin.png)

![Jpeg-error](/img/in-post/opencv-binarization/binarization-result.png)

大家可以再拉一拉滑动条，看看不同的阈值二值化出来的图片怎么样~上图的值私以为是看起来比较舒服的~

当然除了cvThreshold()这个函数以外，OpenCV中还提供了cvAdaptiveThreshold()函数以及cvCanny()函数也是可以对图像进行二值化的，而cvAdaptiveThreshold()函数会使用Otsu算法(大律法或最大类间方差法)来进行自适应全局阈值，通过这个阈值对图像进行二值化，效果是和在调用cvThreshold()时传入参数CV_THRESH_OTSU是一样的，大家可以自己试一试~

基于OpenCV3.3的图像二值化（阈值可调），整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/Binarization)