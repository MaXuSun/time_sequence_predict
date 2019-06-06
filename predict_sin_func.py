# 本文件是预测sin函数的main文件
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch

import matplotlib.pyplot as plt
import numpy as np
import rnn

def predict_sin(net,test_input,index,pre_num=1000,input_size=3):
    """
    通过传入的test_input进行后续pre_num个点的预测，并将预测结果保存下来
    :param net: 使用的模型
    :param test_input: 进行预测的依据，起始数据
    :param index(int): 用户输入的序列，用来分别保存的图片
    :param pre_num(int): 向前预测的点数
    :param input_size:
    """
    delimer = test_input.size(1)-input_size+1

    with torch.no_grad():
        pre = net.predict(test_input, pre_num)
        y = pre.cpu().detach().numpy()

    # 画结果
    plt.figure(figsize=(35, 15))
    plt.title("Predict values for sin function", fontsize=25)
    plt.xlabel('x', fontsize=25)
    plt.ylabel('y', fontsize=25)

    colors = ['r','b','c','y','c','m','lime','silver','brown','chartreuse']
    for i in range(test_input.size(0)):
        # print(len(np.arange(delimer)),len(y[i]))
        # print(len(np.arange(delimer, delimer + pre_num)),len(y[i][delimer:]))
        plt.plot(np.arange(delimer), y[i][:delimer], linewidth=2.0,color=colors[i])
        plt.plot(np.arange(delimer, delimer + pre_num), y[i][delimer:],
                 linewidth=2.0, linestyle=":",color=colors[i])

    plt.savefig('my_pre_sin_function %d.pdf' % index)
    plt.close()


if __name__ == '__main__':
    # 加载数据
    data = torch.from_numpy(torch.load('traindata.pt'))

    # 通过设置 input_size 决定如何怎样预测：
    # 如果input_size = 3,则使用 n-2,n-1,n -> n+1
    # 如果input_size = 2,则使用 n-1,n -> n+1
    input_size = 3

    # 后 97 行作为训练数据
    train_input = Variable(data[3:,:-1]).cuda()            # 取前999列进行预测
    train_target = Variable(data[3:,input_size:]).cuda()   # 取后（length - input_size）个作为比对的target
    # 前 3 行作为测试数据
    test_input = data[:3,:-1].cuda()
    test_target = data[:3,input_size:].cuda()

    # 初始化训练用到的模型,损失函数,优化器
    sinnet = rnn.SinNet(input_size,50)
    sinnet = sinnet.cuda()
    sinnet.double()
    loss_fc = nn.MSELoss().cuda()
    optimizer = optim.LBFGS(sinnet.parameters(),lr = 0.8)

    # 开始训练
    for i in range(15):
        print('step:',i)
        def closure():
            # step = 0
            optimizer.zero_grad()
            out = sinnet(train_input)
            loss = loss_fc(out,train_target)
            print('loss',loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

        # 每次优化后都进行预测
        predict_sin(sinnet,test_input,i,input_size=input_size)

    # torch.save(sinnet,'predict_sin_func.pkl') # 将训练的网络保存下来

    # net = torch.load('sinnet.pkl')
