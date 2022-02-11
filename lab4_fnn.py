import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
import os

MODEL_NAME = "./Modules/fashion_mnist.pth"

def show_activation_curve():
    x = torch.linspace(-6, 6, 100)
    s = nn.Sigmoid()
    y_sigmoid = s(x)
    t = nn.Tanh()
    y_tanh = t(x)
    r = nn.ReLU()
    y_relu = r(x)
    soft = nn.Softplus()
    y_soft = soft(x)

    titles = ["Sigmoid", "Tanh", "ReLU", "Softplus"]
    y = [y_sigmoid, y_tanh, y_relu, y_soft]
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.plot(x, y[i], 'r-')
        plt.title(titles[i])
        plt.grid()

    plt.show()


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


class fashionNet(nn.Module):
    def __init__(self):
        n_in_dim = 28 * 28
        n_hidden1_dim = 50
        n_hidden2_dim = 100
        n_out_dim = 10
        super(fashionNet, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(n_in_dim, n_hidden1_dim),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(n_hidden1_dim, n_hidden2_dim),
            nn.ReLU(inplace=True)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(n_hidden2_dim, n_out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)

        return self.linear3(x)


def showFashionImg(imgs, titles):
    for i in range(len(imgs)):
        img = imgs[i]
        plt.subplot(1, len(imgs), i+1)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(img.view(28, 28))

        plt.title(titles[i])

    plt.show()


def test_FashionNet():
    if not os.path.exists(MODEL_NAME):
        print("模型文件不存在")

    # 加载网络..
    model = fashionNet()
    model.load_state_dict(torch.load(MODEL_NAME))

    # 测试模式
    model.eval()
    train_iter, test_iter = load_data(batch_size=20)
    features, labels = iter(test_iter).next()

    # 预测标签
    features = features.view(features.size(0), -1)
    _pred = model(features)
    _, index = torch.max(_pred, 1)
    pred_labels = get_fashion_mnist_labels(index)

    # 显示10个图和标题
    # 标题. \n表示回车，第一行是真标签，第二行是预测标签
    true_labels = get_fashion_mnist_labels(labels)
    titles = [true_title + "\n" + pred_title for true_title, pred_title in zip(true_labels, pred_labels)]
    showFashionImg(features[0:10], titles[0:10])


def train_FashionData():
    lr = 0.15
    epoch = 200
    batch_size = 200

    train_iter, test_iter = load_data(batch_size)
    use_gpu = torch.cuda.is_available()
    model = fashionNet()
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_data = []
    for e in range(epoch):
        accuracy = 0
        loss_total = 0
        for step, (features, label) in enumerate(train_iter):
            features = features.view(features.size(0), -1)

            if use_gpu:
                features = features.cuda()
                label = label.cuda

            y_hat = model(features)
            loss = criterion(y_hat, label)
            loss_total += loss.item()
            # print("batch loss: {0}".format(loss.item()))

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

    # 测试集上的数据
    model.eval()
    eval_loss = 0
    eval_accury = 0
    for step, (features, y) in enumerate(test_iter):
        features = features.view(features.size(0), -1)
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


# load_data(batch_size=256, display=True)
# train_FashionData()
test_FashionNet()

