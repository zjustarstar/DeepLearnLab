import os
import glob
import unicodedata
import string
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import common


HIDDEN_SIZE = 128
MODEL_NAME = "./Modules/6_0_charRnn.pth"

# 所有可能的字符
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# 从all_letters中查找字母索引，例如 "a" = 0
def letter2index(letter):
    return all_letters.find(letter)


def read_data():
    path = 'Datasets\\name_data\\names\\*.txt'

    lang_category = []  # 语言类型
    category_names = {} # 每种语言对应的名字examples

    for pathfile in glob.glob(path):
        # 文件名就是语言类型. splitext[0]文件名，第二个是扩展名
        lang_name = os.path.splitext(os.path.basename(pathfile))[0]
        lang_category.append(lang_name)
        # 按行读取
        lines = open(pathfile, encoding='utf-8').read().strip().split('\n')
        format_lines = [unicodeToAscii(line) for line in lines]
        category_names[lang_name] = format_lines

        #print(category_names['Chinese'][:5])

    return lang_category, category_names


def name2tensor(name, vocab_size):
    for ind, ch in enumerate(name):
        # one-hot接受long-type类型
        index = torch.tensor(letter2index(ch)).long()
        letter = F.one_hot(index, vocab_size).unsqueeze(dim=0)
        if ind == 0:
            t = letter
        else:
            t = torch.cat((t, letter), dim=0)
    return t


class myCharRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myCharRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, h):
        h_state = self.i2h(torch.cat((x, h)))
        h_state = self.relu(h_state)
        o = self.h2o(h_state)
        y = self.softmax(o)
        return y, h_state


def iter_examples(lang_category, category_names):
    samples = []
    for ind in range(len(lang_category)):
        lang = lang_category[ind]
        names = category_names[lang]
        sample = list(zip([ind] * len(names), names))
        samples = samples + sample
        # ind_tensor = F.one_hot(torch.tensor(ind).long(), len(lang_category)).float()
        # yield names, ind_tensor

    index = list(range(0, len(samples)))
    random.shuffle(index)
    for ind in index:
        y = samples[ind][0]  # 语言类别
        x = samples[ind][1]  # 单词
        x_tensor = F.one_hot(torch.tensor(y).long(), len(lang_category)).float()
        yield  x, x_tensor


def train():
    epoch_size = 100
    lr = 0.0005

    # 数据读取
    lang_category, category_names = read_data()
    lang_types = len(lang_category)
    print("lang category:%d" % lang_types)

    # 网络
    module = myCharRnn(n_letters, HIDDEN_SIZE, lang_types)
    optimizer = optim.SGD(module.parameters(), lr=lr, momentum=0.8)
    criterion = nn.CrossEntropyLoss()

    module.train()
    loss_list = []
    for e in range(epoch_size):
        batch_loss = 0
        for ind, (name, lang_index) in enumerate(iter_examples(lang_category, category_names)):
            hstate = torch.zeros(HIDDEN_SIZE)
            name_tensor = name2tensor(name, n_letters)
            # hstate不参与梯度计算
            hstate = hstate.detach()
            # 一个单词一次loss
            for i in range(len(name)):
                y_hat, hstate = module(name_tensor[i, :], hstate)
            loss = criterion(y_hat.unsqueeze(dim=0), lang_index.unsqueeze(dim=0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss += loss

        loss_list.append(batch_loss/(ind+1))

        if (e+1) % 1 == 0:
            print("epoth={}, avg batch loss={}".format(e+1, batch_loss/(ind+1)))

    # 保存模型，画loss曲线;
    torch.save(module.state_dict(), MODEL_NAME)
    common.show_loss_curve(loss_list)


def predict(name, lange_category):
    if not os.path.exists(MODEL_NAME):
        print("模型文件不存在")

    # 加载网络..
    model = myCharRnn(n_letters, HIDDEN_SIZE, len(lange_category))
    model.load_state_dict(torch.load(MODEL_NAME))

    model.eval()
    name_tensor = name2tensor(name, n_letters)
    hstate = torch.zeros(HIDDEN_SIZE)
    for i in range(len(name)):
        y_hat, hstate = model(name_tensor[i, :], hstate)

    k = 3
    y_value, y_index = y_hat.topk(k)
    for i in range(k):
        category = lange_category[y_index[i]]
        print("top {} prediction={}, value={}".format(i+1, category, y_value[i]))

train()