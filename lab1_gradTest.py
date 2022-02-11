import torch


# 各种创建tensor的操作
def test_create_tensor():
    # 创建一个没有初始化的5*3矩阵
    a1 = torch.empty(5, 3)
    print(a1)
    # 创建一个随机初始化的5*3矩阵
    a2 = torch.rand(5, 3)
    print(a2)
    # 返回的tensor默认具有相同的torch.dtype和torch.device.
    # 系列函数new_
    a2_1 = a2.new_ones(5, 3, dtype=torch.float64)
    print(a2_1)
    # 大小一样, 指定新的数据类型. 系列函数*_like
    a2_2 = torch.randn_like(a2_1, dtype=torch.float)
    print(a2_2)
    # 构造一个填满0且数据类型为long的矩阵
    a3 = torch.zeros(5, 3, dtype=torch.long)
    print(a3)
    # 直接从一个张量构造
    a4 = torch.tensor([5.5, 3])
    print(a4)


def test_broadcast():
    # 都有一个长度为1的维度
    x = torch.arange(1, 3).view(1, 2)
    y = torch.arange(1, 4).view(3, 1)
    print(x + y)
    # 有一方为1
    m = torch.tensor([[0, 0, 0],[1, 1, 1],
                      [2, 2, 2],[3, 3, 3]])
    n = torch.tensor([[1],[2], [3], [4]])
    print(m+n)
    # 数组维度不同,但是后缘维度的长度相等.[4, 3]
    x = torch.tensor([[0, 0, 0], [1, 1, 1],
                      [2, 2, 2], [3, 3, 3]])
    # shape=([1, 3])
    y = torch.tensor([1, 2, 3])
    print(x + y)

def test_grad():
    # 默认requires_grad=False
    # 也可以初始化为True
    # x = torch.ones(2, 2, requires_grad=True)
    x = torch.ones(2, 2)
    t = x * 2 + 3
    print(t.grad_fn)

    x.requires_grad_(True)
    for i in range(2):
        y = x + 2
        z = y * y * 3
        out = z.mean()
        print(out, out.grad_fn)

        out.backward()
        print(x.grad)

        x.grad.zero_()


def test_mutation():
    x = torch.tensor([[1, 1], [2, 3],
                      [6, 8], [0, 9],
                      [8, 10],[2, 2]])
    #t()是transpose用于2D矩阵的便捷转置函数
    print(x.t())
    # -1表示该维度由系统自动计算
    y = x.reshape(-1, 3)
    print(y)


test_mutation()

