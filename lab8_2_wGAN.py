# cDCGAN

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
import PIL.Image as Image
import os
import numpy as np
import random
import common as comm
import albumentations


MODEL_NAME = "./Modules/wGAN.pth"



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
    n = 10
    if display:
        imgs = [mnist_train[i][0] for i in range(10)]
        labels = [mnist_train[i][1] for i in range(10)]
        images = list(map(transforms.ToPILImage(), imgs))
        comm.show_imgs(images, get_fashion_mnist_labels(labels), 4)

    return train_iter, test_iter


class generator(nn.Module):
    def __init__(self, input_dim=100, out_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * input_size//4 * input_size//4),
            nn.BatchNorm1d(128 * input_size//4 * input_size//4),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.out_dim, 4, 2, 1),
            nn.Tanh()
        )
        # 初始化网络权重
        comm.initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, self.input_size//4, self.input_size//4)
        x = self.deconv(x)
        return x


class discriminator(nn.Module):
    def __init__(self, input_dim=1, out_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.input_size = input_size
        self.out_dim = out_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4),
                      1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.out_dim),
            # nn.Sigmoid(),
        )
        # 初始化网络权重
        comm.initialize_weights(self)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 128 * self.input_size // 4 * self.input_size // 4)
        return self.fc(x)


def save_image(G, z_dim, epoch):
    G.eval()

    batch_size = 10
    z = torch.rand((batch_size, z_dim))

    out = G(z)
    images = list(map(transforms.ToPILImage(), out))
    for i in range(len(images)):
        filename = "./Test/" + str(epoch) + "_" + str(i) + ".png"
        images[i].save(filename)


def train_gan():
    # 超参
    batch_size = 200
    lr = 0.0005
    # 潜变量维度
    z_dim = 70
    epoch = 30

    # clipping value
    clip_c = 0.01
    nCritic = 3

    # 定义网络和优化器等
    train_iter, _ = load_data(batch_size, False)
    # shape=[N, C, H, W]
    data_shape = iter(train_iter).next()[0].shape
    # data_shape[1] = C, data_shape[2] = H 假设宽和高一样
    D = discriminator(input_dim=data_shape[1], out_dim=1, input_size=data_shape[2])
    G = generator(input_dim=z_dim, out_dim=data_shape[1], input_size=data_shape[2])
    criterion = nn.BCELoss()
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    # 记录每个epoch的网络损失
    GLoss = []
    DLoss = []
    real_label = torch.ones(batch_size, 1)
    fake_label = torch.zeros(batch_size, 1)

    print("*"*20 + "start training" + "*"*20)
    D.train()
    for e in range(epoch):
        G.train()
        G_loss_epoch = 0
        D_loss_epoch = 0
        for step, (features, _) in enumerate(train_iter):

            # ===================判别器网络训练====================
            optimizer_D.zero_grad()

            D_real_out = D(features)
            D_real_loss = -torch.mean(D_real_out)
            # 真实样本的平均D(x)值,从1开始, 最终=0.5为最佳
            D_X = D_real_out.mean().item()

            # 生成随机的潜在变量
            z = torch.randn((batch_size, z_dim))
            z_out = G(z)
            D_z_out = D(z_out)
            D_fake_loss = torch.mean(D_z_out)
            # 虚假样本的平均D(G(z))值,从0开始, 最终升到0.5左右最佳
            D_G_Z1= D_z_out.mean().item()

            L_D = torch.div(D_real_loss + D_fake_loss, 2)
            L_D.backward()
            optimizer_D.step()

            for p in D.parameters():
                p.data.clip(-clip_c, clip_c)

            # ===================生成器网络训练====================
            if ((step+1) % nCritic) == 0:
                optimizer_G.zero_grad()
                z_out = G(z)
                D_fake_out = D(z_out)
                L_G = -torch.mean(D_fake_out)

                L_G.backward()
                optimizer_G.step()
                D_G_Z2 = D_fake_out.mean().item()

            # 记录损失
            DLoss.append(L_D.item())
            GLoss.append(L_G.item())
            # 记录epoch内平均每个batch的损失
            G_loss_epoch += L_G.item()
            D_loss_epoch += L_D.item()

            if (step+1) % 50 == 0:
                print("epoch={0},progress={1}/{2}, "
                      "D(x)={3}, D(G(z))={4}, "
                      "G_loss={5}, D_loss={6}".format(
                    (e+1), step+1, (60000 / batch_size),
                    D_X, D_G_Z1,
                    L_G.item(), L_D.item()
                ))

        if (e+1) % 10 == 0:
            print("epoch={0}, D_epoch_avgloss={1}, G_epoch_avgloss={2}".format(
                (e+1), D_loss_epoch/(step+1), G_loss_epoch/(step+1)))
            with torch.no_grad():
                save_image(G, z_dim, e)


train_gan()


