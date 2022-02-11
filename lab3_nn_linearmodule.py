import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt

num_inputs = 2
num_examples = 100
true_w = torch.tensor([3, 5.2])
true_b = 3.7
# 创建x和y数据, 用于简单的线性回归
def create_data(display=True):
    x = torch.randn(num_examples, num_inputs, dtype=torch.float32)
    y = true_w @ x.t() + true_b
    y += torch.normal(0, 0.01, size=y.size(), dtype=torch.float32)
    if display:
        fig = plt.figure()
        fig.add_subplot(1, 2, 1)
        plt.scatter(x[:, 0], y, 1)
        fig.add_subplot(1, 2, 2)
        plt.scatter(x[:, 1], y, 1)
        plt.show()
    return x, y


# 创建用于LR回归的数据;
def create_data_LR(display=True):
    sample_size = 1000
    bias = 0.8
    data1 = torch.normal(1.0, 0.2, (sample_size, 2)) + bias
    lable1 = torch.zeros((sample_size, 1))
    data2 = torch.normal(0.2, 0.25, (sample_size, 2)) + bias
    lable12 = torch.ones((sample_size, 1))
    train_feature = torch.cat((data1, data2))
    train_label = torch.cat((lable1, lable12))

    if display:
        plt.scatter(data1[:, 0], data1[:, 1], c='r', label='class 0')
        plt.scatter(data2[:, 0], data2[:, 1], c='b', label='class 1')
        plt.show()

    return train_feature, train_label


def test_dataloader():
    train_features = torch.Tensor([[1.1, 2.1, 3.1],
                                   [4.1, 5.1, 6.1],
                                   [7.1, 8.1, 9.1],
                                   [10.1, 11.1, 12.1]])
    train_labels = torch.Tensor([[1.1], [2.1], [3.1], [4.1]])
    dataset = Data.TensorDataset(train_features, train_labels)
    for i in range(len(dataset)):
        print(dataset[i])


class testModule(nn.Module):
    def __init__(self):
        super(testModule, self).__init__()
        # sub-module(child module)
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        # child module
        self.features = nn.Sequential(nn.Linear(4, 2),
                                     nn.Linear(2, 1))
        self.features.add_module("classifier", nn.Sigmoid())
        # 将参数类型转换为float16
        self.features.to(torch.float16)
        # 将参数转换到cpu
        self.features.to(torch.device('cpu'))

    def forward(self, x):
        return None

def show_module_info():
    tmodule = testModule()
    # 将模型参数和buffer转移到cpu
    tmodule.cpu()
    # 将模型参数和buffer数据类型变为float
    tmodule.float()
    # 将模型所有参数的梯度清零
    tmodule.zero_grad()
    for name, param in tmodule.named_parameters():
        print("parameter name:{0}, data:{1}".format(name, param.data))
    for name, child in tmodule.named_children():
        print("children name:{0}, children:{1}".format(name, child))
    for name, modules in tmodule.named_modules():
        print("sub_modules name:{0}, module:{1}".format(name, modules))
    print(tmodule)


class myLinearNet(nn.Module):

    # in_feature_num是myLinearNet
    def __init__(self, in_feature_num) -> None:
        # 父类初始化
        super(myLinearNet, self).__init__()
        # 定义模型
        self.linear = nn.Linear(in_feature_num, 1)

    # 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


class myLRNet(nn.Module):
    def __init__(self):
        super(myLRNet, self).__init__()
        #定义模型
        self.features = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.features(x)


# 测试线性模型
def test_LinearModule():
    # 一些参数
    lr = 0.03
    batch_size = 10
    num_epochs = 10

    # 选择模型
    module = myLinearNet(num_inputs)
    # 初始化模型参数
    nn.init.normal_(module.linear.weight, mean=0, std=0.01)
    # 也可以直接修改bias的data: module.linear.bias.data.fill_(0)
    nn.init.constant_(module.linear.bias, val=0)
    layer_state_dict = module.state_dict()
    for layer_name, value in layer_state_dict.items():
        print("layer_name:" + layer_name)
        print("layer_value:" + str(value.data.tolist()))

    # loss定义
    loss = nn.MSELoss()
    # 优化算法, 将module的参数传入梯度下降算法
    optimizer = optim.SGD(module.parameters(), lr)
    for param_name, param_value in optimizer.state_dict().items():
        print("param_name:" + param_name)
        print("param_value:" + str(param_value))

    # 开始训练模型
    X, Y = create_data(False)
    dataset = Data.TensorDataset(X, Y)
    # 随机读取小批量.shuffle控制是否随机读取
    batch_samples = Data.DataLoader(dataset, batch_size,
                                    shuffle=True)
    loss_data = []
    for epoch in range(num_epochs):
        # 小批量训练
        for x, y in batch_samples:
            y_hat = module(x)
            l = loss(y_hat.squeeze(), y.squeeze())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            l.backward()
            # 使用随机梯度下降更新参数
            optimizer.step()

        epoch_loss = loss(module(X).squeeze(), Y)
        loss_data.append(epoch_loss.item())
        print('epoch:%d, loss=%.2f' % (epoch + 1, epoch_loss.mean().item()))

    # 打印最终的参数
    w = module.linear.weight.data
    b = module.linear.bias.data
    print(w, b)
    # 准确度
    print(torch.div(w, true_w.squeeze()), b / true_b)
    plt.plot(range(num_epochs), loss_data)
    plt.show()


def test_LRModule():
    # 一些参数
    lr = 0.03
    batch_size = 10
    num_epochs = 10

    # 选择模型
    module = myLRNet()
    module_Momentum = myLRNet()
    module_RMSProp = myLRNet()
    module_Adam = myLRNet()
    nets = [module, module_Momentum, module_RMSProp, module_Adam]

    layer_state_dict = module.state_dict()
    for module_name, value in layer_state_dict.items():
        print("module_name:" + module_name)
        print("module_value:" + str(value.data.tolist()))

    # loss定义
    loss = nn.BCELoss()

    # 优化算法, 将module的参数传入梯度下降算法
    optimizer = optim.SGD(module.parameters(), lr)
    opt_momentum = optim.SGD(module_Momentum.parameters(), lr, momentum=0.8)
    opt_rmsprop = optim.RMSprop(module_RMSProp.parameters(), lr, alpha=0.8)
    opt_adam = optim.Adam(module_Adam.parameters(), lr)
    opts = [optimizer, opt_momentum, opt_rmsprop, opt_adam]

    # 开始训练模型
    X, Y = create_data_LR(False)
    dataset = Data.TensorDataset(X, Y)
    # 随机读取小批量.shuffle控制是否随机读取
    batch_samples = Data.DataLoader(dataset, batch_size, shuffle=True)
    loss_data = [[], [], [], []]
    # 用于记录每个epoch 表示sgd的平均损失;
    base = 0
    for epoch in range(num_epochs):
        # 小批量训练
        sgd_loss = 0
        for step, (x, y) in enumerate(batch_samples):
            for (net, opt, ld) in zip(nets, opts, loss_data):
                y_hat = net(x)
                l = loss(y_hat.squeeze(), y.squeeze())
                # 梯度清零
                opt.zero_grad()
                # 反向传播
                l.backward()
                # 使用随机梯度下降更新参数
                opt.step()
                # 记录损失
                ld.append(l.item())
            sgd_loss += loss_data[0][base + step]
        base += (step+1)
        # 标准sgd在每个epoch的平均损失
        print('epoch:%d, loss=%.2f' % (epoch + 1, sgd_loss/(step+1)))

    final_y_hat = module(X)
    mask = final_y_hat.ge(0.5).int()
    correct_num = (mask == Y).sum()
    acc = correct_num.item() / X.size(0)
    print("accuracy={0}".format(acc))
    # 打印最终的参数
    w = module.features[0].weight.data
    b = module.features[0].bias.data
    w0, w1 = w.squeeze()[0], w.squeeze()[1]
    plot_x = torch.arange(-0.5, 3, 0.1)
    plot_y = (-w0.item() * plot_x - b.item()) / w1.item()

    # 画出分类曲线.两类放在一起，大小各一半
    datasize = int(len(X) / 2)
    plt.scatter(X[0:datasize-1, 0], X[0:datasize-1, 1], c='r', label='class 0')
    plt.scatter(X[datasize:datasize*2-1, 0], X[datasize:datasize*2-1, 1], c='b', label='class 1')
    plt.plot(plot_x, plot_y)
    plt.show()

    # 不同模型的loss曲线
    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_data):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('batchs')
    plt.ylabel('Loss')
    plt.show()


# test_dataloader()
# show_module_info()
# test_LinearModule()
test_LRModule()

