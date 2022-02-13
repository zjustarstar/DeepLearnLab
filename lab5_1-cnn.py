import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
import os

MODEL_NAME = "./Modules/lenet_fashion_mnist.pth"

# 将数值标签转成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 下载数据
def load_data(batch_size=256, display=False):
    mnist_train = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                                    train=True, download=False,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='Datasets/FashionMNIST',
                                                   train=False, download=False,
                                                   transform=transforms.ToTensor())
    print(len(mnist_train), len(mnist_test))

    # 小批量数目
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True)
    # num_workers=0,不开启多线程读取。
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size,
                                            shuffle=False)

    # 显示10张图
    if display:
        features, labels = iter(train_iter).next()
        showFashionImg(features[0:9], get_fashion_mnist_labels(labels[0:9]))

    return train_iter, test_iter


class myLeNet(nn.Module):
    def __init__(self):
        super(myLeNet, self).__init__()
        self.conv = nn.Sequential(
            # in_channels, out_channels, kernel_size
            # 28*28->24*24
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            # kernal_size, stride
            # 24*24->12*12
            nn.MaxPool2d(2, 2),
            # 12*12->8*8
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            # 8*8->4*4
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        # flatten features as input
        x = self.fc(x.view(x.shape[0], -1))

        return x


def showFashionImg(imgs, titles):
    for i in range(len(imgs)):
        img = imgs[i]
        plt.subplot(1, len(imgs), i+1)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(img.view(28, 28))

        plt.title(titles[i])

    plt.show()


def train_FashionData():
    lr = 0.15
    epoch = 200
    batch_size = 200

    train_iter, test_iter = load_data(batch_size)
    use_gpu = torch.cuda.is_available()
    print("gpu available {0}".format(use_gpu))
    model = myLeNet()
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # 设置了momentum的SGD比未设置的效果好多了。
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8)

    print("=="*10 + "start training" + "=="*10)
    loss_data = []
    for e in range(epoch):
        accuracy = 0
        loss_total = 0
        for step, (features, label) in enumerate(train_iter):
            if use_gpu:
                features = features.cuda()
                label = label.cuda

            y_hat = model(features)
            loss = criterion(y_hat, label)
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 精确度
            # 1 表示每行的最大值
            _, y_pred = torch.max(y_hat, 1)
            # 每批次的准确度。因为都是0和1，可利用mean求每次的准确度
            accuracy += (y_pred == label).float().mean()

        accuracy = accuracy / (step + 1)
        loss_data.append(loss_total/(step+1))
        if e % 10 == 0:
            print("epoch {0}: accuracy={1}, batch_average_loss={2}".format(e, accuracy, loss_total/(step+1)))
    print("Final: accuracy={0}, loss={1}".format(accuracy, loss_total))

    # 保存模型
    torch.save(model.state_dict(), MODEL_NAME)

    # show train_loss curve
    x = torch.arange(0, epoch, 1)
    plt.title('train loss')
    plt.plot(x, loss_data)
    plt.show()

    print("==" * 10 + "start testing" + "==" * 10)
    # 测试集上的数据
    model.eval()
    eval_loss = 0
    eval_accury = 0
    for step, (features, y) in enumerate(test_iter):
        if use_gpu:
            features = features.cuda()
            label = label.cuda

        with torch.no_grad():
            y_hat = model(features)
            loss = criterion(y_hat, label)

        eval_loss += loss.item()
        _, y_pred = torch.max(y_hat, 1)
        eval_accury += (y_pred == y).float().mean()
    print("test average accuray:{0}, average loss:{1}".format(eval_accury/len(test_iter), eval_loss/len(test_iter)))


train_FashionData()


