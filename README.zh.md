# Witt Blog 

* [这里](https://github.com/primetong/primetong.github.io)仅仅是博客的源码和文档，想直接看博客内容的请移步[胃痛的博客](https://primetong.github.io/)
* 博客还在开发中，后期会考虑陆续开发一些新功能，博客内容也是一直在更新的
* 采用jekyll搭建
* 使用了Hux的模板，在其基础上增删了一些功能，喜欢可以打Star~
* 如果仅仅是喜欢模板请看[分割线](#boilerplate)中的内容

## 前言

好嘞各位看官，Witt 的GitHub博客就这么开通了。

现在的代码世界真是太棒啦，搭个GitHub博客有无数的轮子。再次感谢Hux提供的模板！

[跳过废话，直接看技术实现 ](#build)

2018 年，Witt 总算有个地方可以好好写点东西了。

作为一个程序员， Blog 这种轮子要是挂在大众博客程序上就太没意思了。一是觉得大部分 Blog 服务都太丑，二是觉得不能随便定制不好玩。但是之前都没太折腾。

在写了几篇博客记录学习生活之后，写博客的快感以及上述两点原因又激起了我开GitHub博客的冲动。之前的一些博客后期会陆陆续续整理进来，能够通过Git Workflow风格的Commit来Post博客感觉实在棒，整个人都Geek了起来哈哈。

<p id = "build"></p>

---

## 正文

接下来说说搭建这个博客的技术细节。其实对于做技术的来说不算是什么技术了哈哈。

之前有关注过 [GitHub Pages](https://pages.github.com/) + [Jekyll](http://jekyllrb.com/) 快速 Building Blog 的技术方案，当时就感觉非常轻松时尚。

其优点非常明显：

* **Markdown** 带来的优雅写作体验。
* 非常熟悉的 Git workflow ，**Git Commit 即 Blog Post**。
* 利用 GitHub Pages 的域名和免费无限空间，不用自己折腾主机。
	* 如果需要自定义域名，也只需要简单改改 DNS 加个 CNAME 就好了。
* Jekyll 的自定制非常容易，基本就是个模版引擎。

***

### 安装流程
* 1.要用github pages，首先要在github中建立一个基于你的用户名的repository: 比如说你的用户名叫username，就要建立名为username.github.io的repo。在以前的github版本中还需要在后台开启pages的功能，现在系统检测到这样的repo名称之后，会在setting中自动开启GitHub Pages的功能。
* 2.这样之后你就可以把这个repo克隆到本地随意进行修改了，在这个里面上传的网页就是你的网站的内容了，可以上传一个index.html试一试，这就是你的网站主页了。 关于GitHub的使用，我想我不用多说吧，都看到这里的还不会用Git，我就[黑人问号脸]了我。
* 3.之后我们就要在本地部署jekyll，jekyll的原理很简单。这是一个已经合成好的静态html网站结构，你用这个结构在username.github.io文件夹里面粘帖好所有文件。再把更新完的本地repo推送到GitHub的master branch里面，你的网站就更新建设完毕了。 
首先你需要ruby来使用本地jekyll。Mac和Linux可以用Terminal配合yum或者brew这样的包管理器很方便的安装ruby。Windows下更是方便，可以直接用集成好的Ruby installer来进行安装，安装包下载以及安装过程也不用多说了。

	* 安装完ruby，之后就是要安装RubyGems，gem是一个ruby的包管理系统，可以用gem很方便的在本地安装ruby应用。

	安装方法  

	> //在RubyGems官网上下载压缩包，解压到你的本地任意位置  
	//在Terminal中  
	cd yourpath to RubyGems //你解压的位置  
	ruby setup.rb  

* 4.有了gem之后安装jekyll就很容易了，其实用过nodejs和npm的同学应该很熟悉这样的包安装，真是这个世界手残脑残们的救星（情不自禁地摸了摸自己快残了的手） 安装jekyll，有了gem，直接在Terminal里面输入以下代码：
 
	> $ gem install jekyll 

* 5.好了，现在你的电脑已经准备完毕了。如果你是想自己捣鼓，可以根据这样的目录结构在你的username.github.io文件夹下建立以下目录结构：

> ├── _config.yml

> ├── _drafts

> |   ├── begin-with-the-crazy-ideas.textile
|   └── on-simplicity-in-technology.markdown
├── _includes

> |   ├── footer.html
|   └── header.html
├── _layouts

> |   ├── default.html
|   └── post.html
├── _posts

> |   ├── 2018-03-23-running.markdown
|   └── 2018-02-28-start.markdown
├── _site

> └── index.html

你可以一个个依次建立起来，然后在自己编写一个你想要的博客。
* 6.如果你只是个普通用户，只是想要一个模板然后开始写自己的博客。那就很容易了，有几个可以简单开始的模板。
	*  [极简，或者说极客风格模板](https://github.com/huxpro/huxpro.github.io/)
	*  [jekyll的模板网站](http://jekyllthemes.org/)，可以找到各式各样你喜欢的模板。
* 7.下载完了模板，可以吧里面的内容解压到你自己的网站目录底下。这时候你可以测试一下：
	
	> $ cd you website path //cd到你的网站目录下  
	$ jekyll serve  
	//一个开发服务器将会运行在 http://localhost:4000/  
	//你就能在本地服务器看到你用模板搭建的网站了  

* 8.这时候可以看一下jekyll的设置，让你把模板变成你自己个性化的内容。在网站根目录下面找到 _config.yml,这里会有几个比较关键的设置： 里面的permalink 就是你博客文章的目录结构，可以用pretty来简单的设置成日期+文章标题.html，也可以用自己喜欢的结构来设置。 记得把encoding 设置成utf-8，这样有利于中英文双语的写作和阅读。
* 9.到这里你就可以开始写博客了，所有的文章直接放在_posts文件夹下面，格式就是我们之前提到的markdown文件，默认的格式是.md和.markdown文件。Jekyll对于博文，都是要求放在_posts目录下面，同时对博文的文件名有严格的规定，必须保持格式YEAR-MONTH-DAY-title.MARKUP，通常情况下，咱们采用推荐的Markdown撰写博文，基于该格式，本博文的文件名为2018-03-23-Welcome-to-Witt-Blog.markdown。每篇文章的开始处需要使用yml格式来写明这篇文章的简单介绍，格式如下：

	```
	---
	author: Witt
	date: 2018-03-23 13:42:24+00:00
	layout: post
	title: Welcome to Witt Blog | 胃痛的GitHub博客开通啦
	tags:
	- life
	- begin
	- javascript
	---
	```

* 10.到此为止可以开始尽情的写博客了，用GitHub软件同步到你的repository里面，网站上面就可以进行正常的显示了。

***

配置的过程中也没遇到什么坑，基本就是 Git 的流程，相当顺手

大的 Jekyll 主题上直接 clone 了 Hux Blog（这个主题也相当有名，就不多赘述了。在他的模板基础上增删了一些功能，以后会考虑陆续开放什么的）。

Theme 的 CSS 是基于 Bootstrap 定制的，看得不爽的地方直接在 Less 里改就好了（平时更习惯 SCSS 些）。

* 这篇README主要讲这个博客是搭建过程，想看其他博客内容的请移步[胃痛的博客](https://primetong.github.io/)。
* 采用jekyll搭建。
* 博客还在开发中，后期会考虑陆续开发一些新功能，博客内容也是一直在更新的。
* 使用了Hux的模板，在其基础上增删了一些功能，喜欢可以打Star~
* 如果仅仅是喜欢模板请看[模板的使用教程](https://github.com/primetong/primetong.github.io/blob/master/README.zh.md)。

***

<p id = "boilerplate"></p>

***

我是可爱的分割线

[黄玄的博客模板（中文）](https://github.com/Huxpro/huxpro.github.io/blob/master/README.zh.md)

***

## 致谢

1. 这个模板是从这里[黄玄的博客](https://github.com/huxpro/huxpro.github.io/)  cloned 的。感谢这个作者！

2. 感谢 Jekyll、Github Pages、Bootstrap和Hux!