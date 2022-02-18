# 迁移学习

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


MODEL_NAME = "./Modules/finetune_resnet18_hotdog.pth"

def load_data(display=False):
    input_imgsize = 224
    # 定义数据增强
    trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    train_aug = transforms.Compose([
        # 稍微裁剪的大一点
        transforms.RandomResizedCrop(size=input_imgsize + 32),
        transforms.CenterCrop(input_imgsize),
        transforms.ToTensor(),
        trans_norm
    ])
    test_aug = transforms.Compose([
        transforms.Resize(input_imgsize + 32),
        transforms.CenterCrop(input_imgsize),
        transforms.ToTensor(),
        trans_norm
    ])

    dir = "./Datasets/hotdog/"
    train_data = ImageFolder(os.path.join(dir, "train"),
                             transform=train_aug)
    test_data = ImageFolder(os.path.join(dir, "test"),
                            transform=test_aug)

    pos_img = [train_data[i][0] for i in range(4)]
    neg_img = [train_data[-i-1][0] for i in range(4)]
    if display:
        lables = ["pos"]*4 + ["neg"]*4
        comm.show_imgs(pos_img+neg_img, lables, 4)

    return train_data, test_data


def fine_tune_train():
    batch_size = 200
    lr = 0.01

    # step 1: Load Data
    train_data, test_data = load_data()
    train_iter = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    print("train_data:{0}, batch_size:{1}".format(len(train_data), batch_size))

    # step 2: define net
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 2)
    # for name, child in trained_resnet.named_children():
    #     print("children name:{0}, children:{1}".format(name, child))

    # step 3： 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    output_params = list(map(id, model.fc.parameters()))
    backbone_params = filter(lambda p: id(p) not in output_params, model.parameters())

    # 不同参数需要不同的学习率
    optimizer = optim.SGD([{'params': backbone_params},
                           {'params': model.fc.parameters(), 'lr': lr*10}],
                          lr=lr, weight_decay=0.001)

    print("==" * 10 + "start training" + "==" * 10)
    epoch = 50
    loss_data = []
    final_accuracy = 0
    for e in range(epoch):
        accuracy = 0
        loss_total = 0
        for step, (features, label) in enumerate(train_iter):
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
        if accuracy > final_accuracy:
            final_accuracy = accuracy
            torch.save(model, MODEL_NAME)
        loss_data.append(loss_total / (step + 1))
        if e % 10 == 0:
            print("epoch {0}: accuracy={1}, batch_average_loss={2}".format(e, accuracy, loss_total / (step + 1)))
    print("Final: accuracy={0}, loss={1}".format(accuracy, loss_total))


# model = torchvision.models.ResNet()
# model.load_state_dict(torch.load("./Modules/pretrained_resnet18.pth"))

def test_model():
    input_imgsize = 224
    # 定义数据增强
    trans_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    test_aug = transforms.Compose([
        transforms.Resize(input_imgsize + 32),
        transforms.CenterCrop(input_imgsize),
        transforms.ToTensor(),
        trans_norm
    ])

    # 随便读取前10个正负类图片
    posdir = "./Datasets/hotdog/test/hotdog/"
    negdir = "./Datasets/hotdog/test/not-hotdog/"
    posfile = [posdir + str(i)+".png" for i in range(1000, 1010, 1)]
    negfile = [negdir + str(i)+".png" for i in range(1000, 1010, 1)]
    files = posfile + negfile
    imgs = [Image.open(f) for f in files]
    features = list(map(test_aug, imgs))

    model = torch.load(MODEL_NAME)
    # for name, child in model.named_children():
    #     print("children name:{0}, children:{1}".format(name, child))

    model.eval()
    result = []
    for i in range(len(features)):
        with torch.no_grad():
            # 输入必须是N*C*H*W, featuresp[i]的结构是C*H*W
            y_hat = model(features[i].unsqueeze(0))
            _, y_pred = torch.max(y_hat, 1)
            if y_pred.item()==0:
                result.append("hotdog")
            else:
                result.append("not-hotdog")

    comm.show_imgs(imgs, result, 4)


load_data(True)
# test_model()


