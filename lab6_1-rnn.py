# 时序神经网络
import torch
import random
import math
import os
import string
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256
MODEL_NAME = "./Modules/6_1_rnn.pth"

def grad_clipping(params, theta):
    norm = torch.tensor([0.0])
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def get_corpus():
    file_path = "Datasets\\jaychou_lyrics\\jaychou_lyrics.txt"
    f = open(file_path, encoding='utf-8')
    corpus = f.read()
    f.close()

    corpus_chars = ''
    for ch in corpus:
        # 去掉歌词中的字符和空格;
        if ch==' ':
            continue
        if ch not in string.ascii_letters:
            corpus_chars = corpus_chars + ch
    print("original size={}, new size={}".format(len(corpus), len(corpus_chars)))

    #把换行符换成空格
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    #将两个空格换为1个
    corpus_chars = corpus_chars.replace('  ', ' ')

    return corpus_chars


# 建立字符索引
def gen_char_index(corpus):
    # 为数据集中的每个字符建立索引. set无序, 用sort保持原有顺序.
    # 这样train和test时能保持一致;
    index2char = list(set(corpus))
    index2char.sort(key=corpus.index)
    char2index = dict([(ch, i) for i, ch in enumerate(index2char)])
    vocab_size = len(index2char)

    corpus_index = [char2index[ch] for ch in corpus]
    # print(corpus[:20])
    # print(corpus_index[:20])

    return index2char, char2index, corpus_index, vocab_size


# def iter_data_loader2(corpus_index, seq_size, batch_size):
#     corpus_indices = torch.tensor(corpus_index, dtype=torch.float32)
#     data_len = len(corpus_indices)
#     batch_len = data_len // batch_size
#     indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
#     epoch_size = (batch_len - 1) // seq_size
#     for i in range(epoch_size):
#         i = i * seq_size
#         X = indices[:, i: i + seq_size]
#         Y = indices[:, i + 1: i + seq_size + 1]
#         yield X, Y


# 加载一组训练数据, 每组batch_size个长度为seq_size的样本
# rand=True表示随机读取
def iter_data_loader(corpus_index, seq_size, batch_size, rand=True):
    corpus_index = torch.tensor(corpus_index, dtype=torch.float32)
    # 多少个epoch.减一是因为Y从X后一位开始算
    epoch_size = (len(corpus_index) -1) // (batch_size * seq_size)
    # 取一定整数个数的序列作为输入. Y从后一位算起，要+1。
    samples = corpus_index[0:batch_size * seq_size * epoch_size+1]

    unit_size = batch_size * seq_size
    # 顺序读取模式
    if not rand:
        for e in range(epoch_size):
            start = e * unit_size
            # x往后顺延一位的同等长度序列作为y
            # 每次输入的shape是[batch_size, seq_size]
            x = samples[start:start+unit_size].view(batch_size, seq_size)
            y = samples[start+1:start+unit_size+1].view(batch_size, seq_size)
            yield x, y
    # 随机读取模式
    else:
        epoch_index = list(range(epoch_size))
        random.shuffle(epoch_index)
        for e in epoch_index:
            start = e * unit_size
            x = samples[start:start + unit_size].view(batch_size, seq_size)
            y = samples[start + 1:start + unit_size + 1].view(batch_size, seq_size)
            yield x, y


class myModel(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(myModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state):
        # 首先将输入转为one-hot向量. one_hot只接受LongTensor数据类型
        features = F.one_hot(x.type(torch.LongTensor), self.vocab_size)
        # 转为[seqsize, batchsize, vocabsize]格式
        features = features.transpose(1, 0)
        # 隐藏层的输出out_h[seqsize, batchsize, hiddensize]
        out_h, state = self.rnn(features.type(torch.float32), state)
        # [seqsize*batchsize, hiddensize]->[seqsize*batchsize, vocab_size]
        # shape[-1]表示最后的维度hiddensize
        # 输出的y是[seqsize*batchsize, vocabsize]
        y = self.dense(out_h.view(-1, out_h.shape[-1]))
        return y, state


def train():
    # 时间步数大小和批大小
    seq_size, batch_size = 64, 32
    # 隐藏层大小
    h_size = HIDDEN_SIZE
    epoch = 150

    corpus_char = get_corpus()
    index2char, char2index, corpus_index, vocab_size = gen_char_index(corpus_char)
    print("vocab size is=%d" % vocab_size)

    # 定义模型和损失
    model = myModel(h_size, vocab_size)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = None
    model.train()
    for e in range(epoch):
        total_loss = 0
        for index, (x, y) in enumerate(iter_data_loader(corpus_index, seq_size, batch_size)):
            # 一定要有这个，否则会出错.
            # 使用detach函数从计算图分离隐藏状态, 这是为了使模型参数的梯度
            # 计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
            if state is not None:
                state = state.detach()

            # print("X={}, Y={}".format(x, y))
            n_x = x.reshape(1, -1).squeeze().type(torch.int)
            n_y = y.reshape(1, -1).squeeze().type(torch.int)
            # 对应的句子
            # c1 = ''.join(index2char[i] for i in n_x)
            # c2 = ''.join(index2char[i] for i in n_y)
            # print(c1)
            # print(c2)

            y_hat, state = model(x, state)
            onehot_y = F.one_hot(y.type(torch.LongTensor), vocab_size)
            # 将y转为和y_hat一样的大小
            onehot_y = onehot_y.view(-1, y_hat.shape[-1]).type(torch.float)
            l = loss(y_hat, onehot_y)

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            # grad_clipping(model.parameters(), 1e-2)
            optimizer.step()

            # 总的损失.
            total_loss += l.item()

        if (e+1) % 1 == 0:
            try:
                perplexity = math.exp(total_loss / (index+1))
            except OverflowError:
                perplexity = float('inf')
            print("epoth={}, avg batch loss={}, perplexity={}".format
                  (e+1, total_loss/(index+1), perplexity))

            output = predict_online(model, corpus_char, '思念', 10)
            print(output)

    torch.save(model.state_dict(), MODEL_NAME)


def predict_offline(prefix, out_len):
    if not os.path.exists(MODEL_NAME):
        print("模型文件不存在")

    corpus_char = get_corpus()
    index2char, char2index, corpus_index, vocab_size = gen_char_index(corpus_char)

    # 加载网络..
    model = myModel(HIDDEN_SIZE, vocab_size)
    model.load_state_dict(torch.load(MODEL_NAME))
    return predict_online(model, corpus_char, prefix, out_len)


def predict_online(model, corpus_char, prefix, out_len):
    index2char, char2index, corpus_index, vocab_size = gen_char_index(corpus_char)

    # # 测试模式
    model.eval()
    # 转为数字编码
    prefix_index = [char2index[ch] for ch in prefix]
    output = [prefix_index[0]]
    state = None
    for t in range(out_len + len(prefix) - 1):
        # 永远读取最后一位;因为最后一位是上一轮循环生成的结果
        x = torch.tensor(output[-1]).view(1, 1)
        y, state = model(x, state)
        out = y.view(1, -1).argmax().item()
        # 先读取prefix内容，再续上新生成的
        if t<len(prefix)-1:
            output.append(prefix_index[t+1])
        else:
            output.append(out)

    # 转化为字符串
    output_string = ''.join([index2char[ind] for ind in output])
    return output_string

train()
# output = predict_offline("思念在", 30)
# print(output)





