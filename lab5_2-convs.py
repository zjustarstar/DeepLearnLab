# 各种卷积计算
import torch
import torch.nn as nn
import torch.nn.functional as F


# 测试转置卷积
def test_convT():
    input = torch.arange(1, 10).reshape(3, 3).unsqueeze(0).unsqueeze(0)
    filter = torch.arange(1, 5).reshape(2, 2).unsqueeze(0).unsqueeze(0)
    output = F.conv_transpose2d(input, filter)
    print(output)

    # stride = 2
    output = F.conv_transpose2d(input, filter, stride=2)
    print(output)

    # filter = [3, 3], padding=1
    filter2 = torch.arange(1, 10).reshape(3, 3).unsqueeze(0).unsqueeze(0)
    output = F.conv_transpose2d(input, filter2, padding=1)
    print(output)


class myInception(nn.Module):
    def __init__(self, in_c, c0, c1, c2, c3):
        super(myInception, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_c, c0, kernel_size=1),
            nn.ReLU()
        )
        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_c, c1[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c1[0], c1[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_1_5 = nn.Sequential(
            nn.Conv2d(in_c, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.pool_conv1=nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_c, c3, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_1_3(x)
        x3 = self.conv_1_5(x)
        x4 = self.pool_conv1(x)
        out = torch.cat((x1, x2, x3, x4), 1)
        return out


def test_inception():
    net = myInception(2, 4, (6, 8), (4, 8), 4)
    x = torch.randn(2, 28, 28).unsqueeze(0)
    out = net(x)
    print(out.shape)

# test_convT()
test_inception()



