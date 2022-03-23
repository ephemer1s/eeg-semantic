import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import eegcnn # import my cnn model
from preprocess import preprocess, split_dataset, fastft # import my preprocess funcs
from sklearn.metrics import confusion_matrix, f1_score
import pickle
from torchviz import make_dot
import os
os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz/bin'

if __name__ == "__main__":
    net = eegcnn.Net()  # 加载训练网络
    print(net)  # 输出网络结构
    net = net.cuda()  # 网络使用GPU
    # 定义超参数
    batch_size = 16  # 批大小
    learning_rate = 0.02  # 学习率
    num_epoches = 20  # 迭代次数
    percentile = 0.7  # 数据集分割比例
    noise = 1.2  # 高斯噪声
    # 定义优化器、loss
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)  # 定义随机梯度下降优化器
    loss_func = torch.nn.CrossEntropyLoss()  # 定义交叉熵
    losses, acces, eval_losses, eval_acces = [], [], [], []  # 实例化loss和acc序列
    best_acc = 0
    best_epoch = 0
    # 载入数据集
    data, label = preprocess(X_noise=noise)  # 载入数据集，噪声为设定值
    index = np.arange(len(data))
    np.random.shuffle(index)  # 随机打乱数据index
    x0, x1 = split_dataset(data, index, percentile=percentile)  # 随机划分数据
    y0, y1 = split_dataset(label, index, percentile=percentile)  # 随机划分标签
    x0 = torch.from_numpy(x0).long().cuda()  # GPU化
    y0 = torch.from_numpy(y0).long().cuda()
    deal_set = TensorDataset(x0, y0)  # 训练集装载到tensor
    train_data = DataLoader(
        dataset=deal_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0)  # 定义训练集加载器
    # 开始迭代
    net.train()  # 网络调整为训练模式
    for e in range(num_epoches):
        train_loss, train_acc = 0, 0  # 每次迭代清零loss和acc
        for im, label in train_data:  # 分批载入数据和标签
            im = Variable(torch.unsqueeze(im, dim=1).float())  # 数据矩阵降维
            label = Variable(torch.unsqueeze(label, dim=1).long())  # 标签矩阵降维
            out = net(im).cuda()  # 喂数据给网络
            # 迭代
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计结果
            train_loss += loss.item()
            _, pred = out.max(1)  # 得到预测值
            num_correct = (pred == label).sum().item()  # 计算准确个数
            acc = num_correct / im.shape[0]  # 计算准确率
            # acc = f1_score(label.cpu(), pred.cpu(), average="macro")
            train_acc += acc  # 更新总准确率
        # 打印本轮迭代结果
        print("epoch %d finished: loss=%6f,  acc=%6f"
            % (e, train_loss / len(train_data), train_acc / len(train_data)))
        losses.append(train_loss / len(train_data))  # 更新loss
        acces.append(train_acc / len(train_data))  # 更新acc
        # 保存当前训练的最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = e
            if best_acc > 0.95:
                best_model = net
                best_out = out.cpu()
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)
    ax.set_title('train loss')
    ax.plot(np.arange(len(losses)), losses)
    bx.set_title('train acc')
    bx.plot(np.arange(len(acces)), acces)
    acc_path = ("./figures/Acc_lr="
        + str(learning_rate) + "_bs=" + str(batch_size) + "_percentile=" + str(percentile) 
        + "_epoch" + str(best_epoch) + "_noise=" + str(noise) + ".png")
    plt.savefig(acc_path)
    # 可视化模型
    # 这一步如果报错是因为第13行graphviz的路径不对，需要安装graphviz
    g = make_dot(best_out)
    g.render('eegnet_model', view=False)
    # 保存模型
    model_path = ("./pickles/conv1d_lr="
        + str(learning_rate) + "_bs=" + str(batch_size) + "_percentile=" + str(percentile) 
        + "_epoch" + str(best_epoch) + "_noise=" + str(noise))
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)  # 保存训练好的模型
    # 载入测试集
    x1 = torch.from_numpy(x1).float().cuda()  # 载入测试数据GPU化
    deal_set = TensorDataset(x1)  # 装载测试集
    test_data = DataLoader(
        dataset=deal_set, batch_size=64, 
        shuffle=False, num_workers=0)  # 定义测试集载入器
    # 进行测试
    net.eval()  # 网络调整为测试模式
    res = []  # 预测结果
    for im in test_data:
        im = Variable(torch.unsqueeze(im[0], dim=1))  # 数据矩阵降维
        out = net(im).cuda()  # 预测
        _, pred = out.max(1)  # 得到预测值
        pred.unsqueeze_(1)
        res.append(pred)
    # 整理测试结果
    res = torch.cat(res, dim=0)
    res = res.cpu().numpy().squeeze()  # 整理结果
    num_correct = 0
    for i in range(len(y1)):
        if res[i] == y1[i]:
            num_correct += 1
    acc = num_correct / len(y1)  # 统计准确率
    print(acc, f1_score(y1, res, average="macro"))  # 得到预测结果accuracy和F1-macro分数
    # 画混淆矩阵
    confusion = confusion_matrix(y1, res)
    figure, ax = plt.subplots()
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # 画色卡
    plt.colorbar()     
    # 画坐标轴刻度
    y_tick = ['W', 'S1', 'S2', 'S3', 'S4', 'R', 'M']
    classes = list(y_tick)
    indices = range(len(confusion))
    plt.tick_params(labelsize=11)  
    plt.xticks(indices, classes)
    plt.yticks(indices, classes, rotation=90)
    # 画坐标轴标题
    font1 = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 15,}
    plt.xlabel('prediction', font1)
    plt.ylabel('real label', font1)
    # 画图的标题
    font2 = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 23,}
    plt.title('Confusion Matrix', font2)
    # 画混淆矩阵上的文字标注
    # for first_index in range(len(confusion)):
    #     for second_index in range(len(confusion[first_index])):
    #         plt.text(
    #             first_index,
    #             second_index, 
    #             confusion[first_index][second_index])
    # 保存
    cfsmat_path = ("./figures/ConfusionMatrix_lr="
        + str(learning_rate) + "_bs=" + str(batch_size) + "_percentile=" + str(percentile) 
        + "_epoch" + str(best_epoch) + "_noise=" + str(noise) + ".png")
    plt.savefig(cfsmat_path)