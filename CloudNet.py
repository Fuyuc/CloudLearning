import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from confusion_matrix import draw_matrix
from torch.optim import lr_scheduler
from torchvision import models
from torchsummary import summary


def loadtraindata():
    path = r"C:/Users/76505/Desktop/cloud_data/CCSN/CCSN_v2/train/"                                         # 路径
    trainset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((227, 227)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.CenterCrop(227),
                                                    transforms.ToTensor()])
                                                )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                              shuffle=True, num_workers=2)
    return trainloader


class Net(nn.Module):  # 定义网络，继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11,stride=4)  # 卷积层
        self.pool = nn.MaxPool2d(3, 2)  # 池化层
        self.conv2 = nn.Conv2d(96, 256, 5,stride=1,padding=2)  # 卷积层
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)  # 卷积层
        self.conv4 = nn.Conv2d(384, 256, 3, stride=1, padding=1)  # 卷积层
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 11)  # 11个输出



    def forward(self, x):  # 前向传播

        out_put = []
        x = self.pool(F.relu(self.conv1(x)))  # F就是torch.nn.functional
        out_put.append(x)
        x = self.pool(F.relu(self.conv2(x)))
        out_put.append(x)
        x = F.relu(self.conv3(x))
        out_put.append(x)
        x = self.pool(F.relu(self.conv4(x)))
        out_put.append(x)
        x = x.view(-1, 256 * 6 * 6)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        # 从卷基层到全连接层的维度转换
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x,out_put

classes = ('0','1', '2', '3', '4',
           '5', '6', '7', '8', '9','10')

# classes = ('卷云','卷层云', '卷积云', '高积云', '高层云', '积云',
#                '积雨云', '雨层云', '层积云', '层云', '航迹云')

def loadtestdata():
    path = r"D:/practice/leecode/swimcat(5)/"
    testset = torchvision.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((227, 227)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=True, num_workers=2)
    return testloader


def trainandsave():
    trainloader = loadtraindata()
    torch.cuda.empty_cache()
    # print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() != True else 'cpu')

    # 神经网络结构
    net = Net().to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001

    #动态更新学习率
    # Assuming optimizer uses lr = 0.05 for all groups
    # lr = 0.05     if epoch < 30
    # lr = 0.005    if 30 <= epoch < 60
    # lr = 0.0005   if 60 <= epoch < 90m
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(100):  # 训练的数据量为20000个epoch，每个epoch为一个循环
        i = 0
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for inputs,labels in trainloader:  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            i += 1
            # enumerate是python的内置函数，既获得索引也获得数据
            # get the inputs
            inputs,labels = inputs.to(device),labels.to(device)

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)[0]  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            # scheduler.step()
            # lr = scheduler.get_lr()
            running_loss += loss.item()  # loss累加
            if i % 100 == 99:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 100))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用
    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'netfiles/net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'netfiles/net_params.pkl')  # 只保存神经网络的模型参数

def reload_net():
    trainednet = torch.load('netfiles/net.pkl')
    return trainednet

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test():
    sum = 0
    testloader = loadtestdata()
    net = reload_net()
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print('GroundTruth: '
    #       , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    # print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))
    Real = [classes[labels[j]] for j in range(130)]
    Predicted = [classes[predicted[j]] for j in range(130)]
    # draw_matrix(Real,Predicted)
    count = 0
    for i in range(len(Real)):
        if Real[i] == Predicted[i]:
            count += 1
    return count/130

def draw_feat_map(out_put):
    testloader = loadtestdata()
    net = reload_net()
    for inputs,_ in testloader:
        imshow(np.squeeze(inputs))
        out_put = net(inputs)[1]
        for id, feature_map in enumerate(out_put):
            print("卷积层%d" % (id + 1))
            print(feature_map.shape)
            # [N, C, H, W] -> [C, H, W]
            im = np.squeeze(feature_map.detach().cpu().numpy())
            im = im / 2 + 0.5  # unnormalize
            # [C, H, W] -> [H, W, C]
            im = np.transpose(im, [1, 2, 0])
            print(feature_map.shape)
            # show top 12 feature maps
            plt.figure()
            for i in range(16):
                # img_show(im[:, :, ])
                ax = plt.subplot(4, 4, i + 1)
                # [H, W, C]
                plt.imshow(im[:, :, i], cmap=plt.cm.gray)
            plt.show()


# def test():
#     testloader = loadtestdata()
#     net = reload_net()
#     dataiter = iter(testloader)
#     sum = 0
#     while True:
#         try:
#             images, labels = dataiter.next()                  #
#             outputs = net(Variable(images))
#             _, predicted = torch.max(outputs.data, 1)
#             Real = [classes[labels[j]] for j in range(0,len(predicted))]
#             Predicted = [classes[predicted[j]] for j in range(0,len(predicted))]
#             count  = len(set(Real) & set(Predicted))
#             sum += count
#         except StopIteration:
#             break
#     return sum/330

if __name__ == '__main__':
    trainandsave()
    # print(test())

    #查看模型各层形状
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     # vgg = Net().to(device)
    #     # summary(vgg, (3, 227, 227))