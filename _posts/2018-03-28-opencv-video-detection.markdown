---
layout:     post
title:      "OpenCV Notes 5 | 【OpenCV图像处理入门学习教程五】"
subtitle:   "基于背景差分法的视频目标运动侦测"
date:       2018-03-28
author:     "Witt"
header-img: "img/post-bg-opencv.jpg"
tags:
    - OpenCV
    - 背景差分法
    - 视频
    - 运动侦测
---

## 运动目标检测

关于运动目标检测的方法总结，目前能够实现运动物体检测的方法主要有以下几种：
1. 背景差分法：能完整快速地分割出运动图像。其不足之处是易受光线变化影响，背景的更新是关键。不适用于摄像头运动的情况；
2. 光流法：能检测独立运动的图像，可用于摄像头运动的情况，但是计算复杂耗时，较难实现实时监测；
3. 帧差法：受光线变化影响较小，简单快速，但不能分割出完整的运动对象，需进一步运用目标分割算法。还有一些改进的算法，主要致力于减少光照影响和检测慢速物体变化。
以上是大多数文献中对这三种方法的评价，由于是初次接触，而且实验中的需求是静止相机，因此采用最简单的方法：背景差分法。

## 背景差分法实现步骤

将背景差分法的实现步骤总结如下：
1. 进行图像的预处理：主要包括对图像进行灰度化以及滤波。
灰度化的方法及其C语言实现可参考《Canny边缘检测算法原理及其VC实现详解(二)》一文；关于图像滤波，通常可采用的方法有中值滤波、均值滤波以及高斯滤波等。关于高斯滤波的实现详见《高斯图像滤波原理及其编程离散化实现方法》一文。
2. 背景建模：这是背景差法较为重要的第一步。目前大多的思路都是根据前N帧图像的灰度值进行区间统计从而得到一个具有统计意义的初始背景。在第一次的实现过程中，采用第一幅图片作为背景图，这样比较简单。
3. 前景提取：将当前最新的图像与背景做差，即可求得背景差图，然后根据一定的方法对改图进行二值化，最终获得运动前景区域，即实现图像分割。


## 运动检测和背景更新方法实现的步骤

所研究的运动检测和背景更新方法实现的步骤如下：  
(1)开辟静态内存，对图像进行初始化准备采集；  
(2)采集图像，定义参数k，作为图像序列计数。采集第1幅图像时，则根据第一帧的大小信息进行矩阵、图像的初始化，并且将第一帧图像进行灰度化处理，并转化为矩阵，作为背景图像及矩阵；如果k不等于1则把当前帧进行灰度化处理，并转化为矩阵，作为当前帧的图像及矩阵。用当前帧的图像矩阵和背景帧的图像矩阵做差算出前景图矩阵并对其进行二值化以便计算它与背景帧差别较大的像素个数，也就是二值化后零的个数。  
当第一帧的异物大于1W个像数点则需要将当前帧存储为第一帧，并且将系统的状态转为1——采集第二帧；  
第一帧和第二帧的异物都大于1W个像数点时，将当前帧存储为第二帧，通过判断第一帧和第二帧的差值来确定两帧是否连续，  
若连续则将系统状态转为2——采集第三帧，若不连续则报警，并把系统状态转为0——采集背景帧；
当第一帧和第二帧的异物都大于1W个像数点, 而第三帧没有时则报警；  
若连续3帧的异物都大于1W个像数点时，将当前帧存储为第三帧，通过判断第二帧和第三帧的差值来确定两帧是否连续，  
若连续则将更新背景，若不连续则报警。然后把系统状态转为0——采集背景帧。  
注意其中有一个0－1－2－0....的状态机。  

### 一、读取本地视频实现基于背景差分法的视频目标运动侦测

左上角是原视频，右上角是背景，下面是分割出来的前景，结果如下：

![Video-running-result](/img/in-post/opencv-video-detection/video-running-result.png)

### 二、打开笔记本摄像头实现基于背景差分法的视频目标运动侦测

这部分就不做演示了，实现起来也很简单，只需在上述读取本地视频实现基于背景差分法的视频目标运动侦测的工程项目中，修改属性页-配置属性-调试-命令参数-中传入主函数的参数，后面的处理步骤是一样的。
1.为空时为打开默认摄像头：

![Debug-argv-camera](/img/in-post/opencv-video-detection/debug-argv-camera.png)

2.非空时默认在工程目录下寻找对应同名的本地视频文件并开始读取处理：

![Debug-argv-video](/img/in-post/opencv-video-detection/debug-argv-video.png)

### 三、代码解析

下面是一段基于背景差分法的视频目标运动侦测的代码，当传入主函数的参数为空时（argc的值为1

> main(int argc, char *argv[])中的两个参数, argc表示参数个数，*argv则是具体的参数.  
默认情况下，project本身是作为第一个参数的(比如，我的应用输出是test.exe，则argv[0]对应的值为test.exe的绝对路径 - D:\program files\vs2013\vctest\debug\test.exe)，即默认情况下argc的值为1(该值无需手动改变),   
如果需设定其他参数，可以通过如下配置:  
<1>选择PROJECT—>Properties—>Configuration Properties—>Debugging—>Command Arguments  
<2>在Command Arguments中添加参数，假设 : 要设定argv[1] = ”23”, argv[2] = ”Hello”,   
那么输入值23  Hello即可(两个值之间空格隔开,要传空格就包括在" "中)然后保存即可。运行之后可发现参数值已经改变  
中文版：菜单[项目]->属性页->配置属性->调试，在[命令行参数]里填上即可。不同参数之前用空格隔开。

）  

打开当前运行环境计算机默认的摄像头，此时可在镜头前做小幅度动作即可看到分割出的前景与背景视频。

当传入主函数的参数不为空时（传入的是视频的全部文件名，包含.文件扩展名，例如traffic.flv），默认会进入工程目录下寻找对应的同名本地视频文件进行下一步的处理操作。

* IDE：Visual Studio 2013
* 语言：C++
* 依赖：OpenCV 2.4.9

程序是在VS2013和OpenCV2.4.9下运行的，部分参考代码如下，相应位置有详细注释，如果有fopen_s()报错的问题，请看：
> 解决VS2013中fopen替代为fopen_s的问题  
最普通的解决方法，就是使用fopen_s替代，这是fopen_s()函数的用法  
fopen_s(_Outptr_result_maybenull_ FILE ** _File, _In_z_ const char *   _Filename, _In_z_ const char * _Mode);  
这是fopen()函数：  
fopen(_In_z_ const char * _Filename, _In_z_ const char * _Mode);  
但fopen_s参数要比fopen多一个，并且返回的类型为：errno_t __cdecl，但fopen()返回的类型为：FILE * __cdecl  
因此，fopen_s函数可能并不适合自己的程序，解决方法有一比较好的方法：  
更改预处理定义：  
项目->属性->配置属性->C / C++->预处理器->预处理器定义，增加CRT_SECURE_NO_DEPRECATE
这样就可以解决vs2013报错的问题了。  

> cvCloneImage的原型是：  
IplImage* cvCloneImage(const IplImage* image);  
在使用函数之前，不用开辟内存。该函数会自己开一段内存，然后复制好image里面的数据，然后把这段内存中的数据返回给你。  
clone是把所有的都复制过来，也就是说不论你是否设置Roi, Coi等影响copy的参数，clone都会原封不动的克隆过来。  
copy就不一样，只会复制ROI区域等。  

整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/VideoDetection)：

```

#include <stdio.h>
#include <time.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

////调用以下函数可以对运动目标寻找轮廓并绘制矩形框，但是效果不佳暂且就不找问题所在了
//CvMemStorage *stor;
//CvSeq *cont;
//void DrawRec(IplImage* pImgFrame, IplImage* pImgProcessed, int MaxArea);

int main( int argc, char** argv )
{
//声明IplImage指针
  IplImage* pFrame = NULL;     //pFrame为视频截取的一帧
  IplImage* pFrame1 = NULL;      //第一帧
  IplImage* pFrame2 = NULL;//第二帧
  IplImage* pFrame3 = NULL;//第三帧
  IplImage* pFrImg = NULL;     //pFrImg为当前帧的灰度图
  IplImage* pBkImg = NULL;     //pBkImg为当前背景的灰度图
  IplImage* pBkImgTran = NULL;//pBkImgTran为当前背景处理过的图像
  IplImage* pFrImgTran = NULL;//pFrImgTran为当前前景处理过的图像
  CvMat* pFrameMat = NULL;     //pFrameMat为当前灰度矩阵
  CvMat* pFrMat = NULL;      //pFrMat为当前前景图矩阵，当前帧减去背景图
  CvMat* bg1 = NULL;
  CvMat* bg2 = NULL;
  CvMat* bg3 = NULL;
  CvMat* pFrMatB = NULL;     //pFrMatB为二值化（0,1）的前景图
  CvMat* pBkMat = NULL;
  CvMat* pZeroMat = NULL;               //用于计算bg1 - bg2 的值
  CvMat* pZeroMatB = NULL;//用于计算 pZeroMat阈值化后来判断有多少个零的临时矩阵
  CvCapture* pCapture = NULL;
  int warningNum = 0;      //检测到有异物入侵的次数
  int nFrmNum = 0;//帧计数
  int status = 0;        //状态标志位
//创建窗口
  cvNamedWindow("video", 1);
  cvNamedWindow("background",1);//背景
  cvNamedWindow("foreground",1);//前景
//使窗口有序排列
  cvMoveWindow("video", 30, 0);
  cvMoveWindow("background", 720, 0);
  cvMoveWindow("foreground", 365, 330);
  if ( argc > 2 )
    {
      fprintf(stderr, "Usage: bkgrd [video_file_name]\n");
      return -1;
    }
//打开摄像头，从摄像头取出码流可以使用海康、大唐等网络或者模拟摄像头，这里cvCaptureFromCAM(0)可直接打开笔记本摄像头
  if (argc ==1)
    if ( !(pCapture = cvCaptureFromCAM(0)))
      {
        fprintf(stderr, "Can not open camera.\n");
        return -2;
      }
//打开视频文件，是通过*argv传入的
  if (argc == 2)
    if ( !(pCapture = cvCaptureFromFile(argv[1])))
      {
        fprintf(stderr, "Can not open video file %s\n", argv[1]);
        return -2;
      }

////开始计时
//  time_t start,end;
//  time(&start);        //time() 返回从1970年1月1号00：00：00开始以来到现在的秒数（有10为数字）。
//  printf("%d\n",start);
//逐帧读取视频
  while (pFrame = cvQueryFrame( pCapture ))
    {
      nFrmNum++;
      //如果是第一帧，需要申请内存，并初始化
      if (nFrmNum == 1)
        {
          pBkImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U,1);
          pFrImg = cvCreateImage(cvSize(pFrame->width, pFrame->height), IPL_DEPTH_8U,1);
          pBkImgTran = cvCreateImage(cvSize(pFrame->width,pFrame->height), IPL_DEPTH_8U,1);
          pFrImgTran = cvCreateImage(cvSize(pFrame->width,pFrame->height), IPL_DEPTH_8U,1);
          pBkMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
          pZeroMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
          pFrMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
          pFrMatB = cvCreateMat(pFrame->height, pFrame->width, CV_8UC1);
          pZeroMatB = cvCreateMat(pFrame->height, pFrame->width, CV_8UC1);
          pFrameMat = cvCreateMat(pFrame->height, pFrame->width, CV_32FC1);
          cvZero(pZeroMat);
          //转化成单通道图像再处理
          cvCvtColor(pFrame, pBkImg, CV_BGR2GRAY);
          //转换为矩阵
          cvConvert(pFrImg, pBkMat);
        }
      else /* 不是第一帧的就这样处理 */
        {
          //pFrImg为当前帧的灰度图
          cvCvtColor(pFrame, pFrImg, CV_BGR2GRAY);
          //pFrameMat为当前灰度矩阵
          cvConvert(pFrImg, pFrameMat);
          //pFrMat为前景图矩阵，当前帧减去背景图
          cvAbsDiff(pFrameMat, pBkMat, pFrMat);
          //pFrMatB为二值化（0,1）的前景图
          cvThreshold(pFrMat,pFrMatB, 60, 1, CV_THRESH_BINARY);
          //将图像矩阵转化为图像格式，用以显示
          cvConvert(pBkMat, pBkImgTran);   
          cvConvert(pFrMat, pFrImgTran);  
          //显示图像
          cvShowImage("video", pFrame);
          cvShowImage("background", pBkImgTran); //显示背景
          cvShowImage("foreground", pFrImgTran); //显示前景

		  //DrawRec(pFrame, pFrImgTran, 16);		//对运动目标寻找轮廓并绘制矩形框，但是效果不佳暂且不找问题所在了

          //以上是每抓取一帧都要做的工作，下面进行危险检测
          if (cvCountNonZero(pFrMatB) > 10000 && status == 0) //表示是第一帧的异物大于1W个像数点
            {/* 则需要将当前帧存储为第一帧 */
              pFrame1 = cvCloneImage(pFrame);
              bg1 = cvCloneMat(pFrMat);
              status = 1;      //继续采集第2帧
            }
          else if (cvCountNonZero(pFrMatB) < 10000 && status == 1) // 表示第一帧的异物大于1W个像数点，而第二帧没有,则报警
            {
              printf("NO.%d warning!!!!\n\n",warningNum++);
              status = 0;
            }
          else if (cvCountNonZero(pFrMatB) > 10000 && status == 1)// 表示第一帧和第二帧的异物都大于1W个像数点
            {
              pFrame2 = cvCloneImage(pFrame);
              bg2 = cvCloneMat(pFrMat);
              cvAbsDiff(bg1, bg2, pZeroMat);
              cvThreshold(pZeroMat,pZeroMatB, 20, 1, CV_THRESH_BINARY);
              if (cvCountNonZero(pZeroMatB) > 3000 ) //表示他们不连续，这样的话要报警
                {
                  printf("NO.%d warning!!!!\n\n",warningNum++);
                  status = 0;
                }
              else
                {
                  status = 2;                   //继续采集第3帧
                }
            }
          else if (cvCountNonZero(pFrMatB) < 10000 && status == 2)//表示第一帧和第二帧的异物都大于1W个像数点,而第三帧没有
            {
              //报警
              printf("NO.%d warning!!!!\n\n",warningNum++);
              status = 0;
            }
          else if (cvCountNonZero(pFrMatB) > 10000 && status == 2)//表示连续3帧的异物都大于1W个像数点
            {
              pFrame3 = cvCloneImage(pFrame);
              bg3 = cvCloneMat(pFrMat);
              cvAbsDiff(bg2, bg3, pZeroMat);
              cvThreshold(pZeroMat,pZeroMatB, 20, 1, CV_THRESH_BINARY);
              if (cvCountNonZero(pZeroMatB) > 3000 ) //表示他们不连续，这样的话要报警
                {
                  printf("NO.%d warning!!!!\n\n",warningNum++);
                }
              else //表示bg2,bg3连续
                {
                  cvReleaseMat(&pBkMat);
                  pBkMat = cvCloneMat(pFrameMat); //更新背景
                }
                status = 0;                //进入下一次采集过程
            }
          //如果有按键事件，则跳出循环
          //此等待也为cvShowImage函数提供时间完成显示
          //等待时间可以根据CPU速度调整
          if ( cvWaitKey(2) >= 0 )
            break;
        }/* The End of the else */
    }/* The End of th while */
//销毁窗口
    cvDestroyWindow("video");
    cvDestroyWindow("background");
    cvDestroyWindow("foreground");
//释放图像和矩阵
    cvReleaseImage(&pFrImg);
    cvReleaseImage(&pBkImg);
    cvReleaseMat(&pFrameMat);
    cvReleaseMat(&pFrMat);
    cvReleaseMat(&pBkMat);
    cvReleaseCapture(&pCapture);
  return 0;
}

```

从实验结果来看，OpenCV中基于背景差分法的视频目标运动侦测可以很好地完成预期的视频目标运动检测的。

基于背景差分法的视频目标运动侦测，整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/VideoDetection)


