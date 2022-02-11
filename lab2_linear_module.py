import torch
import math
import random
from matplotlib import pyplot as plt

num_inputs = 2
num_examples = 100
true_w = torch.tensor([3, 5.2])
true_b = 3.7
# 创建x和y数据
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


# 随机小批量读取数据
def data_iter(x, y, batch_size = 10):
    input_size = len(x)
    index = list(range(input_size))
    random.shuffle(index)
    for i in range(0, input_size, batch_size):
        j = torch.LongTensor(index[i:min(i+batch_size, input_size)])
        yield x.index_select(0, j), y.index_select(0, j)


# 参数初始化
def init_param():
    w = torch.normal(0, 0.01, (num_inputs, 1), requires_grad=True, dtype=torch.float32)
    b = torch.zeros(1, requires_grad=True, dtype=torch.float32)
    return w, b


# 定义线性模型
def _model(x, w, b):
    return torch.mm(x, w) + b


# 定义简单的MSE损失函数
def loss_fun(y_hat, y):
    se = (y_hat.squeeze() - y) ** 2
    return se.sum()


# 定义随机梯度下降,注意使用.data直接改变
# 叶子张量的值
def sgd(params, lr, batchsize = 10):
    for p in params:
        p.data -= lr * p.grad / batchsize


X, Y = create_data(False)

# 开始训练模型
lr = 0.03
batch_size = 10
num_epochs = 10
loss = loss_fun
w, b = init_param()
loss_data = []
for epoch in range(num_epochs):
    # 小批量训练
    for x, y in data_iter(X, Y, batch_size):
        l = loss(_model(x, w, b), y)
        # print(l)
        # 反向传播
        l.backward()
        # 使用随机梯度下降更新参数
        sgd([w, b], lr, batch_size)

        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()

    epoch_loss = loss(_model(X, w, b), Y)
    loss_data.append(epoch_loss.item())
    print('epoch:%d, loss=%.2f' % (epoch+1, epoch_loss.mean().item()))

print(w, b)
# 准确度
print(torch.div(w.squeeze(), true_w.squeeze()), b/true_b)
plt.plot(range(num_epochs), loss_data)
plt.show()
