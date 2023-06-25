# YOLOV7+LPRnet实现车牌识别

yolov7采用[bubbliiiing](https://github.com/bubbliiiing),B导的[yolov7pytorch](https://github.com/WongKinYiu/yolov7)实现版本，有很详细的中文注释，
采用[CCPD2019](https://github.com/detectRecog/CCPD)中的很少一部分数据集（12k张）进行了车牌检测训练，
所以效果一般，并且LPRnet没有进行专门训练，采用的是[这个仓库](https://github.com/sirius-ai/LPRNet_Pytorch)自带的LPRnet训练权重

yolov7的训练在这个仓库[yolov7pytorch](https://github.com/WongKinYiu/yolov7)有很详细的教程，我就不过多赘述，训练好后在yolocarid.py
这个文件中修改了路径就行

LPRnet的训练也很简单，将CCPD数据集转换成LPRnet训练数据集格式网上有很多教程，主要就是已图片文件名做标签，
训练好后的模型，将main.py中的超参数pretrained_model改成相应模型路径即可。

## 运行


### 1.环境：如果环境从0开始，运行指令
```shell
pip install -r zero2allrequirements.txt 
```
安装文件中的库（可能不全，报错提示模块缺少可以再安装）

推荐：
使用conda先搭建安装好pytorch的虚拟环境，包括numpy,matplotlib，scipy这些库之后，再运行下面这条指令：
```shell
pip install -r notzero2allrequirements.txt
```
这样错误会少，容易安装

安装慢请使用镜像源

### 2.运行
保证权重模型路径正确后，直接运行 main.py 即可，里面有超参数，检测模式参数等等可以调，看注释调整就行

## 预训练权重网盘链接：
百度网盘：
best_epoch_weights.pth(放根目录model_data文件夹中)：

    链接: https://pan.baidu.com/s/1OlljN60S1ZJBc0rdCROFPA 提取码: i4x6 

Final_LPRNet_model.pth（放到LPRNet文件夹中的weight文件夹中）：

     链接: https://pan.baidu.com/s/1Ukq4NWMBbUEKZtD7iq-FwQ 提取码: aztp



## Reference
https://github.com/WongKinYiu/yolov7

https://github.com/bubbliiiing/yolov7-pytorch

https://github.com/sirius-ai/LPRNet_Pytorch

https://github.com/detectRecog/CCPD
