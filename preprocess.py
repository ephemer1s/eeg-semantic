import numpy as np
from scipy import signal
from scipy import fft
import os


def bw_filter(data, fs=3000, lowcut=40, highcut=260, order=2):
    '''
    对list型输入data进行butterworth滤波，返回滤波后的数据
    '''
    nyq = 0.5*fs  # nyquist frequency of sample
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order, [low,high], btype='bandpass')
    data = signal.lfilter(b, a, data)
    return data
    

def preprocess(X_noise=0.0):
    '''
    数据加载和预处理，X_noise为高斯噪声参数，默认为0（不加噪声）
    '''
    with open("./sc4002e0/sc4002e0_data.txt") as f:
        data = f.readlines()
    data = [float(s) for s in data]
    data = np.array(data)
    # data = bw_filter(data)
    data = data.reshape(2830, 3000)

    with open("./sc4002e0/sc4002e0_label.txt") as f:
        label = f.readlines()
    label = [float(s) for s in label]
    label = np.array(label)[0:2830]

    if not X_noise == 0.0:
        print("adding noise, sigma is {}".format(X_noise))
        noise = np.random.normal(0, X_noise, size=data.shape)
        data = data + noise

    return data, label



def split_dataset(data, index, percentile=0.8):
    """
    划分数据集，data为数据集，index为外部生成的随机序列索引，percentile为划分比例
    """
    rand_data = []
    for i in range(len(data)):
        rand_data.append(data[index[i]])
    rand_data = np.array(rand_data)
    pi = int(data.shape[0] * percentile)
    stro = "pi=" + str(pi)
    train_set = rand_data[:pi]
    test_set = rand_data[pi:-1]
    return train_set, test_set


def fastft(data):
    """
    根据FFT处理特征值
    """
    fjw = np.array([fft(data[i, :]) for i in range(len(data))])
    gjw = np.array([np.abs(fjw[i, :]) for i in range(len(data))])
    gjw = gjw / (gjw.max() - gjw.min())
    phijw = np.array([np.angle(fjw[i, :]) for i in range(len(fjw))])

    return gjw, phijw


if __name__ == "__main__":
    if not os.path.exists(os.path.dirname("./processed")):
        os.mkdir("./processed")
    data, label = preprocess()
    np.savetxt("./processed/data.csv", data, delimiter=",")
    np.savetxt("./processed/label.csv", label, delimiter=",")
    