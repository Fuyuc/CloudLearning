import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
from torchvision import models
from torchsummary import summary

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 7, 2)   #109*109
        self.conv2 = nn.Conv2d(96, 256, 5, 4)  #27*27
        self.conv3 = nn.Conv2d(256, 512, 3, 2)  # 13*13
        self.conv4 = nn.Conv2d(512, 512, 3, stride = 1,padding = 1)  # 13*13
        self.conv5 = nn.Conv2d(512, 512, 1)  # 13*13
        self.fc1 = nn.Linear(13*13*512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 5)

    def forward(self, x):
        per_out = []
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1,13*13*512)
        feat_fc1 = F.relu(self.fc1(x))
        x = F.relu(self.fc2(feat_fc1))
        x = self.fc3(x)
        return x
        # return x,feat_fc1

def loadtraindata():
    path = r"C:/Users/76505/Desktop/cloudset/swimcat/"
    trainset = tv.datasets.ImageFolder(path,
                                       transform=transforms.Compose([
                                           transforms.Resize((224, 224)),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),  # 转为Tensor
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
                                       ])
                                                )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)
        # print(trainset.class_to_idx)
        # imshow(trainset[0][0])

    return trainloader

def trainandsave():
    trainloader = loadtraindata()
    # 神经网络结构
    torch.cuda.empty_cache()
    # print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() != True else 'cpu')
    net = Net().to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(5):  # 训练的数据量为5个epoch，每个epoch为一个循环
        i = 0
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for inputs,labels in trainloader:  # 这里我们遇到了第一步中出现的trailoader，代码传入数据
            i = i + 1
            inputs,labels = inputs.to(device),labels.to(device)
            # wrap them in Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = net(inputs)  # 把数据输进CNN网络net

            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.item()  # loss累加
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i, running_loss / 100))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # 保存神经网络
    torch.save(net, 'netfiles/net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(net.state_dict(), 'netfiles/net_params.pkl')  # 只保存神经网络的模型参数



"""测试"""
classes = ('1', '2', '3', '4',
           '5', '6', '7', '8', '9','10','11')

def reload_net():
    trainednet = torch.load('netfiles/net.pkl')
    return trainednet


def loadtestdata():
    path = r"C:/Users/76505/Desktop/cloudset/swimcat/"
    testset = tv.datasets.ImageFolder(path,
                                                transform=transforms.Compose([
                                                    transforms.Resize((224, 224)),  # 将图片缩放到指定大小（h,w）或者保持长宽比并缩放最短的边到int大小
                                                    transforms.ToTensor()])
                                                )
    testloader = torch.utils.data.DataLoader(testset, batch_size=25,
                                             shuffle=True, num_workers=2)
    return testloader

def test():
    feature_list = []
    testloader = loadtestdata()
    net = reload_net()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() != True else 'cpu')

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    # imshow(tv.utils.make_grid(images, nrow=5))  # nrow是每行显示的图片数量，缺省值为8
    print('GroundTruth: '
          , " ".join('%5s' % classes[labels[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = net(images)
    _, predicted = torch.max(outputs.data,1)
    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25))) #打印前25个预测值

    # for i, data in enumerate(testloader, 0):
    #     image, label = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
    #     image, label = image.to(device), label.to(device)
    #     outputs = net(image)[0]
    #     print(outputs,label)
    #     feature_list.append(outputs.detach().cpu().numpy()[0])
    # return np.array(feature_list)

if __name__ == '__main__':
    # feature_list, label_list = test()
    test()
    # trainandsave()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = Net().to(device)
    # print(device)
    # print(torch.cuda.get_device_name(0))
    # print(torch.rand(3, 3).cuda())