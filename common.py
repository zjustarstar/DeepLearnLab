# 用于所有文件的公共类
import torch
import math
import PIL.Image as Image
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


# 按照每行显示cols_num个图像
def show_imgs(imgs, titles, cols_num):
    t = len(imgs)
    rows = math.floor(t / cols_num) + 1
    for i in range(t):
        img = imgs[i]
        plt.subplot(rows, cols_num, i+1)
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.imshow(img)
        if len(str(titles[i])):
            plt.title(titles[i])
    plt.show()


def test_showImg():
    img = Image.open("bear.jpg")
    imgs = [img, img, img, img, img]
    show_imgs(imgs, ["bear", "bear", "bear", "bear", "bear"], 3)

# test_showImg()
