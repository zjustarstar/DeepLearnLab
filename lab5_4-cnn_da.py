import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as Data
from matplotlib import pyplot as plt
import PIL.Image as Image
import cv2
import numpy as np
import random
import common as comm
import albumentations

MODEL_NAME = "./Modules/lenet_fashion_mnist.pth"


# 测试data augmentation
def test_da():
    img = Image.open("bear.jpg")
    titles, imgs = [], []
    # 改变大小
    newsize = (int(img.size[0]*0.8), int(img.size[1]*1.2))
    da1_img = transforms.Resize(newsize)(img)
    titles.append("resize")
    # 随机裁剪
    da2_img = transforms.RandomCrop(100, 100)(img)
    titles.append("randomCrop_100")
    # 色度、亮度、饱和度、对比度的变化
    da3_img = transforms.ColorJitter(hue=0.5, brightness=0.8)(img)
    titles.append("ColorJitter")
    # 以概率p随机灰度化,转换后的图片可以选1或者3 通道的
    da4_img = transforms.RandomGrayscale(p=0.7)(img)
    titles.append("RandomGrayscale")
    # 随机进行水平/垂直翻转
    da5_img = transforms.RandomVerticalFlip(p=1)(img)
    titles.append("RandomVerticalFlip")
    # 随机旋转15-30度
    da6_img = transforms.RandomRotation(random.randint(15, 30))(img)
    titles.append("RandomRotation")
    # 随机周围填充.下例将图填充为正方形
    da7_img = transforms.Pad((0, (img.size[0]-img.size[1])//2))(img)
    titles.append("Pad")
    # 随机遮挡.随机遮挡是对 (c, h, w) 形状的 tensor 进行操作，
    # 一般在 ToTensor 之后进行，而使用PIL.Image读取图片后的形式是(H, W, C)
    # 形状的图片，所以直接用RandomErasing时会出现错误。
    # transpose (H, W, C) -> (C, H, W)
    img_array = np.array(img).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_array)
    erased_img_tensor = transforms.RandomErasing(p=1, value=(0,0,0))(img_tensor)
    # (C, H, W) -> (H, W, C)
    erased_img_array = erased_img_tensor.numpy().transpose(1, 2, 0)
    da8_img = Image.fromarray(erased_img_array)
    titles.append("RandomErase")
    imgs = [da1_img,da2_img,da3_img,da4_img,da5_img,
            da6_img,da7_img,da8_img]
    # 组合
    da_compose = transforms.Compose([transforms.RandomHorizontalFlip(p=0.6),
                                     transforms.ColorJitter(hue=0.5, saturation=0.5),
                                     transforms.RandomAutocontrast(p=0.7),
                                     transforms.RandomEqualize(p=0.5)])
    for i in range(6):
        imgs.append(da_compose(img))
        titles.append(" ")

    t = [transforms.Pad(100, fill=(0, 255 ,255)),
         transforms.CenterCrop(300),
         transforms.RandomRotation(10)]
    imgs.append(transforms.RandomApply(t, p=1)(img))
    titles.append("RandomApply")

    # albumentations, 使用opencv读取
    img = cv2.imread("bear.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    al_compose = albumentations.Compose([
        # 高斯噪声
        albumentations.GaussNoise(p=1),
        # 随机通道丢弃
        albumentations.ChannelDropout(fill_value=0, p=1),
        # 随机添加小块, cutout效果
        albumentations.CoarseDropout(max_holes=20, min_holes=10,
                                     fill_value=64, p=1),
        # albumentation的天气增强-随机降雨效果
        albumentations.RandomRain(p=1),
        albumentations.OneOf([
            albumentations.MotionBlur(p=1),
            albumentations.OpticalDistortion(p=1)])
        ])
    al_img = al_compose(image=img)["image"]
    imgs.append(al_img)
    titles.append("Albument")

    comm.show_imgs(imgs,titles,4)


# train_FashionData()
test_da()


