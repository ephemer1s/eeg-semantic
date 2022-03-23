# eeg-semantic

 Sleep Level Electroencephalogram

## 目录

### 可执行代码

`eegcnn.py` - 定义网络结构

`preprocess.py` - 预处理

`showsignal.py` - 显示信号频谱

`train.py` - 训练/测试

### 其他

`./figures` - 生成图像目录

`./pickles` - 模型保存目录

`./processed` - 预处理后的数据集

`./sc4002e0` - 原始数据集

`eegnet_model` - 卷积神经网络的graphviz可视化模型

`environment.yaml` - anaconda环境文件

## 环境配置

在当前工程目录打开控制台，并使用如下代码：

```
conda env create -f environment.yaml
```

即可自动配置环境。

确保当前设备使用Nvidia独立显卡，且CUDA版本不低于10.2

## 执行

```
python train.py
```

