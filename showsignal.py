import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess, fastft


def draw(s1, s1f, figdir):
    '''
    对信号s1和信号s1f进行对比绘图，保存在figdir路径
    '''
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    plt.plot(s1,color='r')
    ax1.set_title('Original Signal')
    plt.ylabel('Amplitude')

    ax2 = fig.add_subplot(212)
    plt.plot(s1f,color='r')
    ax2.set_title('Processed Signal')
    plt.ylabel('Amplitude')

    plt.savefig(figdir)
    plt.close('all')


if __name__ == "__main__":
    data, label = preprocess()
    data_noised, _ = preprocess(X_noise=3)
    f_data, _ = fastft(data)
    f_noised, _ = fastft(data_noised)
    # 画图
    draw(data[0], data_noised[0], "./figures/preprocess_gn_slice.png")
    vis = [898,945,974,1048,1300,1477,1688]
    fig = plt.figure()
    x = []
    for i in range(7):
        x.append(fig.add_subplot(331 + i))
        # freq = np.abs(np.array(fft(train[rep[i], :]), dtype=complex))
        x[i].plot(f_noised[vis[i]])
        title = "Label" + str(i)
        x[i].set_title(title)
    # fig.subplots_adjust(left=0, top=20, right=5, bottom=5, wspace=0.01, hspace=0.01)
    fig.savefig("./figures/fft-inclass.png")