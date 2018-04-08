---
layout:     post
title:      "TensorFlow Notes 1 | 【TensorFlow深度学习框架教程一】"
subtitle:   "在UEFI模式下安装Ubuntu16.04与Windows双启动"
date:       2018-03-30
author:     "Witt"
header-img: "img/post-bg-tensorflow.jpg"
tags:
    - TensorFlow
    - UEFI
    - Ubuntu
    - Windows
    - 双系统启动
---

## 在UEFI模式下安装Ubuntu16.04与Windows双启动

从 Windows8 开始，微软用 UEFI（全称统一的可扩展固件接口，Unified Extensible Firmware Interface）取代了BIOS，UEFI有“安全启动”这个特点，引导程序只会启动那些得到 UEFI 固件签署的引导装载程序。此安全功能可以防止Rootkit类的恶意软件，并提供了额外的安全层。但它有一个缺点，如果你想在Win8/10的电脑上双引导Linux，安全机制会阻止这样做。

win7以下的系统可以跳过4、5步骤。

#### 如何判断电脑是否是uefi启动

按下win+r打开运行，输入msinfo32，确定，打开系统信息可以看到

![Confirm-uefi](/img/in-post/tensorflow-ubuntu-windows/confirm-uefi.png)

##### 1、做个备份
做个备份总是个好的选择，不至于丢失重要数据，方法很多不做介绍

##### 2、创建一个Ubuntu的USB启动盘
- [按需下载ubuntu镜像](http://cn.ubuntu.com/download/)

![Download-ubuntu](/img/in-post/tensorflow-ubuntu-windows/download-ubuntu.png)

- 使用[UltraISO下载](https://cn.ultraiso.net/xiazai.html)写入ubuntu镜像

  ![UltraISO-ubuntu](/img/in-post/tensorflow-ubuntu-windows/ultraiso-ubuntu.png)
  
  [详细过程可参见](http://jingyan.baidu.com/article/19020a0a396b6e529d2842cb.html)

##### 3、为Ubuntu划分一块安装分区
假设你有一个全新的系统，我们要做的第一件事是创建一个分区来安装Linux。你可以通过在控制面板中搜索‘磁盘’找到磁盘管理工具。

在磁盘管理工具中，右键点击你想划分并缩小的卷：

![Compress-disk](/img/in-post/tensorflow-ubuntu-windows/compress-disk.png)

缩小后出现的未分配空间就放在那里好了，不用对其分区和格式化，我们会在安装Ubuntu时用到它。

##### 4、在Windows中禁用快速启动 [可选]

为了实现快速启动，Windows8/10引进了叫做“快速启动”的新特性。尽管不强制要求，最好还是将其禁用。

步骤如下：打开控制面板 > 硬件与声音 > 电源选项 > 选择电源按钮的功能 > 更改当前不可用的设置，取消选中启用快速启动（推荐）。

##### 5、禁用Windows的安全启动（secure boot）
这是非常重要的步骤，为了实现Windows和Linux的双启动，我们必须在UEFI中禁用安全启动（secure boot）。

虽然在 BIOS 时代，访问BIOS是相当简单的，在启动的时候按F10或F12键即可。但是在 UEFI 的世界里，就不一样了。要访问 UEFI 设置，你就必须进入到 Windows 中去，让我们来看看如何在 Windows 8 中访问 UEFI 设置来禁用安全启动。

- 进入PC设置，点击 Windows+I 按钮进入 Windows 设置界面，点击更新和安全选项
- 进入高级启动

![Advanced-startup](/img/in-post/tensorflow-ubuntu-windows/advanced-startup.png)

这之后并不会立刻重新启动，而是会在下一步看到一些选项。

- 进入UEFI设置

-->疑难解答
![UEFI-one](/img/in-post/tensorflow-ubuntu-windows/uefi-one.jpg)
-->高级选项
![UEFI-two](/img/in-post/tensorflow-ubuntu-windows/uefi-two.jpg)
-->UEFI固件设置
![UEFI-three](/img/in-post/tensorflow-ubuntu-windows/uefi-three.jpg)

接下来，在UEFI设置里，点击重新启动按钮重新启动您的系统，就会看到类似BIOS一样的界面。

- 在UEFI中禁用安全启动

这个时候，你一定已经启动到UEFI工具界面，移动到启动选项卡，在那里你会发现安全引导选项被设置为启用，将其改为Disabled并保存即可。

![Boot-disabled](/img/in-post/tensorflow-ubuntu-windows/boot-disabled.jpg)

接下来将正常引导到Windows，现在就支持双启动 Windows8/10 与 Ubuntu 或其它 Linux 操作系统了。

##### 6、安装Ubuntu

点击重新启动并按住shift，在类似UEFI的界面上选择从USB启动的选项。

当你用USB启动盘启动后，你会看到试用（try）或者安装（install）Ubuntu的选择，这里要点击“安装”。

![Install-ubuntu-0](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-0.jpg)

安装窗口中你需要注意的是安装类型（Installation Type）。选择这里的其它选项（Something else）：

![Install-ubuntu-1](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-1.jpg)

我们将用之前创建的分区来创建根分区（/），交换空间（Swap）以及家目录（Home）。选择空闲（free space）然后点击加号（+）。

![Install-ubuntu-2](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-2.jpg)

根分区 / 10到20GB空间就足够了，选择大小（Size），然后选择Ext4作为文件系统以及 /（意思是根）作为挂载点（Mount point）。

![Install-ubuntu-3](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-3.png)

点击确定会回到分区界面，下一步我们创建交换空间（Swap）。像之前一样，再次点击加号（+），这次我们选择作为交换空间（Swaparea），建议的交换空间大小是物理内存的两倍。

![Install-ubuntu-4](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-4.png)

以同样的方式创建家目录（Home）。给它分配最大的空间（实际上是给它分配剩余的所有空间），因为这是你会用来存储音乐，图片以及下载的文件的位置。

![Install-ubuntu-5](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-5.png)

分配好了根分区（ / ），交换空间（Swap）和家目录（Home）之后，点击现在安装（Install Now）：

![Install-ubuntu-6](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-6.jpg)

接下来的就是设置用户名密码等等，基本上就是只需点击下一步。

![Install-ubuntu-7](/img/in-post/tensorflow-ubuntu-windows/install-ubuntu-7.jpg)

一旦安装完成，重新启动电脑，你应该会看到紫色的grub欢迎界面，表明Ubuntu和Windows 8的双启动模式安装成功了。