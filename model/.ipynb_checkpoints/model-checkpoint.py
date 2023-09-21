import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)

# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4):
         #原+原的全局向量  原+扩 原+原  原+扩 乱+原 乱+扩    c是全局特征向量
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


def sample_negative_index(negative_number=0, epoch=0, epochs=0, Number = 0):

    lamda = 1 / 2
    # lamda = 1
    # lamda = 1 / 4
    lower, upper = 0, Number-1
                      # 总结点数

    mu = ((epoch) / epochs) ** lamda * (upper - lower) # 设置上界
    # 512 递增
    # if epoch == 80:
    #     print(mu)
    X = stats.uniform(1,mu) # 均匀连续随机变量。 1-mu_2
    # print(X) # 1 - mu 的均匀分布函数  29  34
    # print(mu)
    index = X.rvs(negative_number)  # 采样 随机采样negative_number=50个
    # index = np.floor(index)
    index = index.astype(np.int64)
    return index

def sample_node_negative_index(negative_number=0, epoch=0, epochs=0, Number = 0):
    lamda = 1 / 2  #一开始全为 1/2
    # lamda = 1 / 4
    # lamda = 1 / 6
    lower, upper = 0, Number-1
                      # 总结点数

    mu = ((epoch) / epochs) ** lamda * (upper - lower) # 设置上界
    # 512 递增
    X = stats.uniform(1,mu) # 均匀连续随机变量。 1-mu_2
    index = X.rvs(negative_number)  # 采样 随机采样negative_number=50个

    ################

    mu_1 = ((epoch - 1) / epochs) ** lamda * (upper - lower)
    if epoch > 10:
        mu_1 = ((epoch - 10) / epochs) ** lamda * (upper - lower)
    #print('mu_1:', mu_1)
    mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
    #print('mu_2:', mu_2)
    #print('mu_2-mu_1:', mu_2-mu_1)
    #sigma = negative_number / 6
    # X表示含有最大最小值约束的正态分布
    # X = stats.truncnorm(
    #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数 正态分布采样
    X = stats.uniform(mu_1,mu_2-mu_1)  # 均匀分布采样
    # print(mu_2) #42.03807797699604
    #X = stats.uniform(1, mu_2)  # 均匀连续随机变量。 1-mu_2
    #print(X) # 1 - mu_2 的均匀分布函数
    index = X.rvs(negative_number)  # 采样 随机采样16个
    #print(index)

    ###############

    index = index.astype(np.int64)
    return index

def Sim_feat_loss_selfpace(Z , temperature = 1.0, epoch = 0, epochs = 0,s=10):

    N = Z.shape[0] # 512
    index = sample_negative_index(negative_number=s, epoch=epoch, epochs=epochs, Number=N)
    index = torch.tensor(index).cuda()   # [50]   10-100

    sim_matrix = torch.exp(torch.pow(Z, 2) / temperature)  # Z平方  [-1,1]->[0,1]
    positive_samples_ii_jj = torch.diag(sim_matrix).reshape(N, 1) #对角线余弦相似度全是1 #[512,1]
    positive_samples = torch.sum(positive_samples_ii_jj,1)  # [512]

    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False)
    # 竖着排  升序   [512,512]
    negative_samples = sim_matrix_sort.index_select(0, index)
    # print(negative_samples.size())   # torch.Size([50,512])

    negative_samples = torch.sum(negative_samples, 0)  # [512]

    # print(negative_samples)    选趋近于0的 负样本正交    矩阵论线性无关

    loss = (- torch.log(positive_samples / negative_samples)).mean()
    # loss = (- torch.log(positive_samples / (positive_samples + negative_samples))).mean()


    return loss

def Sim_node_loss_selfpace(Z, adj_label12 , temperature = 1.0, epoch = 0, epochs = 0, s=10):
    N = Z.shape[0]
    index = sample_node_negative_index(negative_number=s,epoch=epoch, epochs=epochs, Number=N)
     # 270个
    index = torch.tensor(index).cuda()
    sim_matrix = torch.exp(torch.pow(Z, 2) / temperature)

    positive_samples = torch.sum(sim_matrix * adj_label12, 1) # 是邻居的相似度更高
     #print('positive_samples:',positive_samples.max())
     #############
    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False)
    negative_samples = sim_matrix_sort.index_select(0, index)
    # print(negative_samples.size())                     # torch.Size([270, 2708])
    negative_samples = torch.sum(negative_samples, 0)  # 不是邻居的相似度应该更小
    # print(negative_samples.size()) # torch.Size([2708])
    #############
    # negative_samples = torch.sum(sim_matrix, 0)
    # print('negative_samples:',negative_samples.min())
    
    loss = (- torch.log(positive_samples / negative_samples)).mean() # 拉大分子 拉小分母
    return loss


#def Sim_node_loss_selfpace(Z, adj_label12 , temperature = 1.0, epoch = 0, epochs = 0):
#    N = Z.shape[0]
#
#
#    sim_matrix = Z
#    #print(adj_label12)
#    positive_samples = F.sigmoid(sim_matrix * adj_label12) # 是邻居的相似度更高  torch.Size([1, 2708, 2708])
#    index = adj_label12>0  # 不为0的索引  torch.Size([99596, 3])
#
#    # list_X = []
#    #
#    # for i in range(index.size()[0]):
#    #     list_X.append(positive_samples[0][index[i][1], index[i][2]])
#    new_x = positive_samples.masked_select(index)
#    # print(new_x.size())
#    # print(new_x)
#    # time.sleep(10000)
#    loss = (- torch.log(new_x)).mean() # 拉大分子 拉小分母
#    return loss

class Model(nn.Module):
    def __init__(self, n_in, n_h, tau_feat, tau_node):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(n_h)

        self.tau_feat = tau_feat
        self.tau_node = tau_node

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def batched_feat_loss(self, z1, z2, batch_size, epoch, epochs):

        losses = []
        for i in range(batch_size):
            feat = self.sim(z1[i].T, z2[i].T)
            losses.append(Sim_feat_loss_selfpace(feat, temperature=self.tau_feat, epoch=epoch,
                                                 epochs=epochs))

        return torch.stack(losses).mean()
    def batched_node_loss(self, z1, z2, adj_label12, batch_size, epoch, epochs):

        losses = []
        for i in range(batch_size):
            node = self.sim(z1[i]+z2[i], z1[i]+z2[i])
            losses.append(Sim_node_loss_selfpace(node, adj_label12, temperature=self.tau_node, epoch=epoch, epochs=epochs))
        return torch.stack(losses).mean()


    def forward(self, seq1, seq2, adj, diff, sparse, epoch, epochs, adj_label12, batchsize = 1,s=10, use_feat_loss = True, use_node_loss = False):
                 # 原特征  打乱特征  矩阵  扩散矩阵  False  None
        h_1 = self.gcn1(seq1, adj, sparse)
        h_2 = self.gcn2(seq1, diff, sparse)

        c_1 = self.read(h_1)
        c_1 = self.sigm(c_1)
        c_2 = self.read(h_2)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4)
        if batchsize == 1:
            Z_1 = torch.squeeze(h_1, 0)  # Z = [2708 , 512]
            Z_2 = torch.squeeze(h_2, 0)  # Z = [2708 , 512]
            if use_feat_loss:
                feat = torch.mm(F.normalize((Z_1+Z_2).T), F.normalize((Z_1+Z_2).T).T)  # [512,512]
                feat_loss = Sim_feat_loss_selfpace(feat, temperature=self.tau_feat, epoch=epoch, epochs=epochs,s=s)
            else:
                feat_loss = 0.
            if use_node_loss:
                node = torch.mm(F.normalize(Z_1+Z_2), F.normalize(Z_1+Z_2).T)   # [2708,2708]
                node_loss = Sim_node_loss_selfpace(node, adj_label12, temperature=self.tau_node, epoch=epoch, epochs=epochs,s=s)
            else:
                node_loss = 0.
        else:
            if use_feat_loss:
                feat_loss = self.batched_feat_loss(h_1+h_2, h_1+h_2, batchsize, epoch, epochs)
            else:
                feat_loss = 0.
            if use_node_loss:
                node_loss = self.batched_node_loss(h_1, h_2, adj_label12, batchsize, epoch, epochs)
            else:
                node_loss = 0.
        return ret, feat_loss, node_loss

    def embed(self, seq, adj, diff, sparse):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1)
        h_2 = self.gcn2(seq, diff, sparse)

        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret

