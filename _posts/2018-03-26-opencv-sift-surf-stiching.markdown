---
layout:     post
title:      "OpenCV Notes 3 | 【OpenCV图像处理入门学习教程三】"
subtitle:   "基于SIFT特征和SURF特征的微旋转图像拼接与融合生成全景图像的比较"
date:       2018-03-26
author:     "Witt"
header-img: "img/post-bg-opencv.jpg"
tags:
    - OpenCV
    - SIFT
    - SURF
    - 图像拼接
    - 图像融合
---

# 图像拼接与融合介绍

图像拼接在实际中的应用场景非常广泛，比如无人机的航拍，遥感图像等，甚至小到我们出去游玩用手机拍照时，无奈广角太小，没有办法一次将所有你要拍的景物全部拍下来（当然现在很多手机都自带全景拍照啦，但是不自己试试怎么拼接多不爽~）
那么现在假如你的手机是老爷机，没有广角镜头，没有全景功能，所以你对某处中国好山水从左往右依次拍了好几张照片，现在的你坐在电脑前，把手机插入电脑，总不能看着这些照片发呆吧！那么我们能不能把这些照片拼接成一个全景图像呢？现在利用OpenCV就可以做到图像拼接生成全景图像的效果！

比如我们有以下这样的两张图要进行拼接，还得考虑平时拍照时手抖拍斜了的情况嘛！

![Origin-1-left](/img/in-post/opencv-sift-surf-stiching/stiching-origin-one-left.jpg)![Origin-2-right](/img/in-post/opencv-sift-surf-stiching/stiching-origin-one-right.jpg)

大家看到可能会大喊：哎呀！这是哪个手残党拍的照片！拍斜了不说，光线还差了那么多，这拼起来多丑对不对！不要怕，只要满足图像拼接的基本需求——两张图有比较多的重叠部分，其他一切都不是问题~

图像拼接技术主要包括两个关键环节即图像配准和图像融合。对于图像融合部分，由于其耗时不太大，且现有的几种主要方法效果差别也不多，所以总体来说算法上比较成熟。
而图像配准部分是整个图像拼接技术的核心部分，它直接关系到图像拼接算法的成功率和运行速度，因此配准算法的研究是多年来研究的重点。
现在CV领域有很多特征点的定义，比如sift、surf、harris角点、ORB都是很有名的特征因子，都可以用来做图像拼接的工作，他们各有优势。本文将使用基于SIFT和SURF特征进行微旋转图像的图像拼接，用其他方法进行拼接也是类似的，简单来说都是由以下几步完成，区别在于特征点的提取不同：
* （1）特征点提取和描述。
* （2）特征点配对，找到两幅图像中匹配点的位置。
* （3）通过配对点，生成变换矩阵，并对图像1应用变换矩阵生成对图像2的映射图像，即图像配准。
* （4）图像2拼接拷贝到映射图像上，完成拼接。
通过以上4步之后，完成了基础的拼接过程。如果还想对重叠边界进行特殊处理，可以考虑图像融合（去裂缝处理）。


### 一、OpenCV2和3不同的环境配置

* IDE：Visual Studio 2013

* 语言：C++

* 依赖：OpenCV 2.4.9、3.3.0

安装教程可以参考本人之前的一篇博客：

[【OpenCV图像处理入门学习教程一】OpenCV2 + 3的安装教程与VS2013的开发环境配置 + JPEG压缩源码分析与取反运算修改](https://primetong.github.io/2018/03/24/opencv-configure-jpeg-analyse/)

可以使OpenCV2和OpenCV3共存。那么这里为什么又要提到OpenCV2和OpenCV3的区别了呢？其实本人也觉得挺奇葩的，因为从OpenCV3以来，一些比较新的功能都挪到了“opencv_contrib”库里，原因是他们觉得这些库“不安全”，因此并没有默认自带这些库，而图像拼接所要用到的很多特征算子，比如本文这次使用的SIFT和SURF特征又比较悲催地被列入到opencv_contrib库中。所以使用OpenCV2的童鞋们先无视这里，使用OpenCV3想要进行接下来的实验就必须先安装和配置好这个opencv_contrib库，可以参考以下的教程：

[Opencv3.1.0+opencv_contrib配置及使用SIFT测试](http://blog.csdn.net/nnnnnnnnnnnny/article/details/52182091)

在配置好之后，我们就可以愉快地进行图像拼接啦~

### 二、基于SIFT特征的微旋转图像拼接与融合生成全景图像

下面援引一些网上对于SIFT的描述方便大家理解。SIFT，即尺度不变特征变换（Scale-invariant feature transform，SIFT），是用于图像处理领域的一种描述。这种描述具有尺度不变性，可在图像中检测出关键点，是一种局部特征描述子。该方法于1999年由David Lowe首先发表于计算机视觉国际会议（International Conference on Computer Vision，ICCV），2004年再次经David Lowe整理完善后发表于International journal of computer vision（IJCV）。
SIFT特征是基于物体上的一些局部外观的兴趣点而与影像的大小和旋转无关。对于光线、噪声、微视角改变的容忍度也相当高。基于这些特性，它们是高度显著而且相对容易撷取，在母数庞大的特征数据库中，很容易辨识物体而且鲜有误认。使用SIFT特征描述对于部分物体遮蔽的侦测率也相当高，只需要少量的SIFT物体特征就足以计算出位置与方位。

SIFT特征检测主要包括以下4个基本步骤：
1. 尺度空间极值检测：
搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。
2. 关键点定位
在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。
3. 方向确定
基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。
4. 关键点描述
在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

SIFT特征匹配主要包括2个阶段：
* 第一阶段：SIFT特征的生成，即从多幅图像中提取对尺度缩放、旋转、亮度变化无关的特征向量。
* 第二阶段：SIFT特征向量的匹配。

SIFT特征的生成一般包括以下几个步骤：
1. 构建尺度空间，检测极值点，获得尺度不变性。
![Sift-1](/img/in-post/opencv-sift-surf-stiching/sift-one.jpg)
2. 特征点过滤并进行精确定位。
![Sift-2](/img/in-post/opencv-sift-surf-stiching/sift-two.jpg)
3. 为特征点分配方向值。
![Sift-3](/img/in-post/opencv-sift-surf-stiching/sift-three.jpg)
4. 生成特征描述子。
以特征点为中心取16×16的邻域作为采样窗口，将采样点与特征点的相对方向通过高斯加权后归入包含8个bin的方向直方图，最后获得4×4×8的128维特征描述子。示意图如下：
![Sift-4](/img/in-post/opencv-sift-surf-stiching/sift-four.jpg)

当两幅图像的SIFT特征向量生成以后，下一步就可以采用关键点特征向量的欧式距离来作为两幅图像中关键点的相似性判定度量。取图1的某个关键点，通过遍历找到图像2中的距离最近的两个关键点。在这两个关键点中，如果最近距离除以次近距离小于某个阈值，则判定为一对匹配点。
SIFT特征匹配的例子：
![Sift-5](/img/in-post/opencv-sift-surf-stiching/sift-five.jpg)

基于SIFT特征的微旋转图像拼接生成全景图像效果如下（还未融合，原图见上文章开头的两图）：![Result-1-without-mix](/img/in-post/opencv-sift-surf-stiching/stiching-result-one-without-mix.jpg)
融合后：![Result-1-with-mix](/img/in-post/opencv-sift-surf-stiching/stiching-result-one.jpg)

再来点测试结果，方便与之后的SURF特征作比较：

左图：![Origin-2-left](/img/in-post/opencv-sift-surf-stiching/stiching-origin-two-left.jpg)
右图：![Origin-2-right](/img/in-post/opencv-sift-surf-stiching/stiching-origin-two-right.jpg)
可以看到待拼接的左图、右图是有明显的、角度不大的旋转。现在给出基于SIFT特征的微旋转图像拼接与融合生成全景图像运行结果：![Running-2-sift](/img/in-post/opencv-sift-surf-stiching/sift-running-result-two.jpg)
以及效果：![Result-2-sift](/img/in-post/opencv-sift-surf-stiching/sift-stiching-result-two.jpg)

### 三、基于SURF特征的微旋转图像拼接与融合生成全景图像

用SIFT算法来实现图像拼接是很常用的方法，但是因为SIFT计算量很大，所以在速度要求很高的场合下不再适用。所以，它的改进方法SURF因为在速度方面有了明显的提高（速度是SIFT的3倍），所以在图像拼接领域还是大有作为（虽然说SURF精确度和稳定性不及SIFT，这点接下来就通过实际效果图进行比较）。下面将给出基于SIFT特征的微旋转图像拼接与融合生成全景图像的运行结果和效果图。

与SIFT比较的第一张图，拼接融合后的：![Surf-result-1](/img/in-post/opencv-sift-surf-stiching/surf-stiching-result-one.jpg)
第二张的运行结果：![Surf-running-2](/img/in-post/opencv-sift-surf-stiching/surf-running-result-two.jpg)
以及其效果图：![Surf-Result-2](/img/in-post/opencv-sift-surf-stiching/surf-stiching-result-two.jpg)

### 四、代码解析

刚刚使用到的基于SIFT特征和SURF特征的微旋转图像拼接与融合生成全景图像的代码如下：

* IDE：Visual Studio 2013

* 语言：C++

* 依赖：OpenCV 3.3.0

其实这两种特征算子的代码是一模一样的，只需要在提取特征点的时候稍微修改函数参数即可：

> //提取特征点（源代码第27行）     
Ptr<Feature2Df2d = xfeatures2d::SIFT::create();	//修改代码中的SIFT参数即可修改算法（比如SURF等）

部分参考代码如下，相应位置有详细注释，整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/ImgStiching)：

```

#include <opencv2/opencv.hpp>  
#include "highgui/highgui.hpp"    
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置  
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri);

int main(int argc, char *argv[])
{
	Mat image01 = imread("left2.jpg");
	Mat image02 = imread("right2.jpg");

	if (image01.data == NULL || image02.data == NULL)
		return 0;
	imshow("待拼接图像左图", image01);
	imshow("待拼接图像右图", image02);

	//灰度图转换  
	Mat image1, image2;
	cvtColor(image01, image1, CV_RGB2GRAY);
	cvtColor(image02, image2, CV_RGB2GRAY);

	//提取特征点    
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();	//修改SIFT参数即可修改算法（比如SURF）
	vector<KeyPoint> keyPoint1, keyPoint2;
	f2d->detect(image1, keyPoint1);
	f2d->detect(image2, keyPoint2);

	//特征点描述，为下边的特征点匹配做准备    
	Mat imageDesc1, imageDesc2;
	f2d->compute(image1, keyPoint1, imageDesc1);
	f2d->compute(image2, keyPoint2, imageDesc2);

	//获得匹配特征点，并提取最优配对     
	FlannBasedMatcher matcher;
	vector<DMatch> matchePoints;
	matcher.match(imageDesc1, imageDesc2, matchePoints, Mat());
	sort(matchePoints.begin(), matchePoints.end()); //特征点排序    
	//获取排在前N个的最优匹配特征点  
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<10; i++)
	{
		imagePoints1.push_back(keyPoint1[matchePoints[i].queryIdx].pt);
		imagePoints2.push_back(keyPoint2[matchePoints[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵，尺寸为3*3  
	Mat homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, image01.cols, 0, 1.0, 0, 0, 0, 1.0);
	Mat adjustHomo = adjustMat*homo;

	//获取最强配对点在原始图像和矩阵变换后图像上的对应位置，用于图像拼接点的定位  
	Point2f originalLinkPoint, targetLinkPoint, basedImagePoint;
	originalLinkPoint = keyPoint1[matchePoints[0].queryIdx].pt;
	targetLinkPoint = getTransformPoint(originalLinkPoint, adjustHomo);
	basedImagePoint = keyPoint2[matchePoints[0].trainIdx].pt;

	//图像配准  
	Mat imageTransform1;
	warpPerspective(image01, imageTransform1, adjustMat*homo, Size(image02.cols + image01.cols + 110, image02.rows));

	//在最强匹配点左侧的重叠区域进行累加，是衔接稳定过渡，消除突变  
	Mat image1Overlap, image2Overlap; //图1和图2的重叠部分     
	image1Overlap = imageTransform1(Rect(Point(targetLinkPoint.x - basedImagePoint.x, 0), Point(targetLinkPoint.x, image02.rows)));
	image2Overlap = image02(Rect(0, 0, image1Overlap.cols, image1Overlap.rows));
	Mat image1ROICopy = image1Overlap.clone();  //复制一份图1的重叠部分  
	for (int i = 0; i<image1Overlap.rows; i++)
	{
		for (int j = 0; j<image1Overlap.cols; j++)
		{
			double weight;
			weight = (double)j / image1Overlap.cols;  //随距离改变而改变的叠加系数  
			image1Overlap.at<Vec3b>(i, j)[0] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[0] + weight*image2Overlap.at<Vec3b>(i, j)[0];
			image1Overlap.at<Vec3b>(i, j)[1] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[1] + weight*image2Overlap.at<Vec3b>(i, j)[1];
			image1Overlap.at<Vec3b>(i, j)[2] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[2] + weight*image2Overlap.at<Vec3b>(i, j)[2];
		}
	}
	Mat ROIMat = image02(Rect(Point(image1Overlap.cols, 0), Point(image02.cols, image02.rows)));  //图2中不重合的部分  
	ROIMat.copyTo(Mat(imageTransform1, Rect(targetLinkPoint.x, 0, ROIMat.cols, image02.rows))); //不重合的部分直接衔接上去  
	namedWindow("拼接结果-SIFT", 0);
	imshow("拼接结果-SIFT", imageTransform1);
	imwrite("拼接结果-SIFT.jpg", imageTransform1);
	waitKey();
	return 0;
}

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置  
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri)
{
	Mat originelP, targetP;
	originelP = (Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
	targetP = transformMaxtri*originelP;
	float x = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
	float y = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
	return Point2f(x, y);
}

```

### 五、两种特征的结果比较与总结

##### 1.测试图片不宜过大。

测试用的图片不能过大，如果直接输入拍好的照片（比如3456*4608），不但导致程序需要运行数分钟才出结果，还容易导致拼接失败（主要是特征点匹配太多）。

##### 2.尽量使用静态图片（没有动态因素干扰的）

在第一张测试图片中，我们可以很清楚地看到中间拼接处有一辆黑色车辆的“鬼影”，这是为什么？因为两幅图中的黑色车辆移动刚好在中间接缝处了啊！所以要做图像拼接，尽量保证使用的是静态图片，不要加入一些动态因素干扰拼接。


##### 3.图像融合（去裂缝处理）：

在拼接图的交界处，两图因为光照色泽的原因使得两图交界处的过渡很糟糕，所以需要特定的处理解决这种不自然。本文的处理思路是加权融合，在重叠部分由前一幅图像慢慢过渡到第二幅图像，即将图像的重叠区域的像素值按一定的权值相加合成新的图像。

如果没有做去裂缝处理，（SIFT）效果如下：![Result-2-sift-no-mix](/img/in-post/opencv-sift-surf-stiching/sift-stiching-result-two-without-mix.jpg)

与有做图像融合处理的效果（见二、的最后一张效果图）相比，拼接处明显很突兀，不自然，有断裂的感觉。

##### 4.基于SIFT特征和SURF特征的微旋转图像拼接与融合生成全景图像的比较。

通过刚刚的处理效果图的比较，可以明显地比较出SIFT的优势在于对待拼接图片小幅度旋转的适应性，精准度较高；而SURF算法对于待拼接图片的平直性要求很高，稍微旋转的图片拼接后已经失真。查阅资料得知SURF算法的优势在于速度方面有明显的提高（速度是SIFT的3倍）。

基于SIFT特征和SURF特征的微旋转图像拼接与融合生成全景图像，整个工程文件见[下载页面](https://github.com/primetong/LearningCollectionOfWitt/tree/master/2017.DigitalImageProcessing/ImgStiching)