# 1、安装环境

* **CUDA10.2**
* **TensorRT7.2**
* **OpenCV3.4（工程中已给出，不需安装）**
* **vs2015**
下载相关工程：[https://github.com/wang-xinyu/tensorrtx.git](https://github.com/wang-xinyu/tensorrtx.git)

# 2、生成yolov5s.wts文件

在生成`yolov5s.wts`前，首先需要[下载](https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt)模型。同时，需要我们安装[ultralytics/yolov5](https://github.com/ultralytics/yolov5)环境。这里可以参考网上其它文章或github教程进行配置安装，这里不详加说明。

* 将`tensorrtx-master\yolov5`文件夹下的`gen_wts.py`拷贝到`ultralytics/yolov5`文件夹下

![<img src="D:\工作\yolov3\image\32.png" alt="32" style="zoom:50%;" />](https://img-blog.csdnimg.cn/20201221010339605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d4cGxvbA==,size_16,color_FFFFFF,t_70#pic_center)


在当前目录下执行：

```bash
python gen_wts.py
```

最终我们会在当前目录下得到一个`yolov5s.wts`文件。

# 3、vs2015环境搭建

这里我们使用别人已经编好的库，下载连接：**[ tensorrtx](https://github.com/wang-xinyu/tensorrtx)**。（注意：部分头文件、lib文件已在我之后的工程中给出，有需要的同学可以直接下载）

* 创建vs工程，命名为yolov5_Trt,并将`tensorrtx-master\yolov5`文件夹下的文件拷贝到创建的项目工程中

![<img src="D:\工作\yolov3\image\31.png" alt="31" style="zoom:50%;" />](https://img-blog.csdnimg.cn/20201221010359749.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d4cGxvbA==,size_16,color_FFFFFF,t_70#pic_center)


* 添加头文件

```bash
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\include
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\include\tensorrt
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\include\cudnn
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\include
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\include\opencv\opencv2
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\include\opencv
```



* 添加lib

```bash
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\lib\x64
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\lib\trt
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\lib\opencv
C:\Users\Administrator\Documents\Visual Studio 2015\Projects\yolov5_Trt\yolov5_Trt\lib\cudnn
```

* 添加依赖项

```bash
cudart.lib
cublas.lib
cudnn.lib
cudnn64_8.lib
myelin64_1.lib
nvinfer.lib
nvinfer_plugin.lib
nvonnxparser.lib
nvparsers.lib
opencv_world340.lib
```

# 4、TensorRt加速实现

* 修改`yolov5.cpp`源码

由于源码需要在命令行下执行，所以我们需要改改，能够直接执行

将421~449行的代码：

```c++
    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        std::ifstream file(engine_name, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov5 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }
```

替换为：

```c++
if (true) {
		IHostMemory* modelStream{ nullptr };
		APIToModel(BATCH_SIZE, &modelStream);
		assert(modelStream != nullptr);
		std::ofstream p(engine_name, std::ios::binary);
		if (!p) {
			std::cerr << "could not open plan output file" << std::endl;
			return -1;
		}
		p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
		modelStream->destroy();
		return 0;
	}
	else {
		std::ifstream file(engine_name, std::ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtModelStream = new char[size];
			assert(trtModelStream);
			file.read(trtModelStream, size);
			file.close();
		}
	}
```

* 生成`yolov5s.engine`文件

直接运行代码，会出现以下问题

1.  <font color='red'>fatal error C1083: 无法打开包括文件: “dirent.h”: No such file or directory</font>

解决方法：[VS2017/2019 无法打开包括文件: “dirent.h”: No such file or directory](https://blog.csdn.net/weixin_39956356/article/details/108555345)

* 运行yolov5s 下的TensorRT

将423行的`if(true)`改为`if(false)`,直接运行。这里我们测试了`"./test_data/3.jpg"`下的图片，该图片大小为：640\*480

![<img src="D:\工作\yolov3\image\3_1111.jpg" alt="3_1111" style="zoom:50%;" />](https://img-blog.csdnimg.cn/20201221010438305.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d4cGxvbA==,size_16,color_FFFFFF,t_70#pic_center)


最终测试的结果为：

![<img src="D:\工作\yolov3\image\res.jpg" alt="res" style="zoom:50%;" />](https://img-blog.csdnimg.cn/20201221010454240.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3d4cGxvbA==,size_16,color_FFFFFF,t_70#pic_center)


不包括第一次加载模型的时间，在我自己的笔记本GeForce MX250上进行一次推理大概耗时30ms左右。


参考链接：[win10下在vs2015上进行yolov5 TensorRT加速实践](https://blog.csdn.net/wxplol/article/details/111466155)




