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
    lower, upper = 0, Number-1
    mu = ((epoch) / epochs) ** lamda * (upper - lower) 
    X = stats.uniform(1,mu) 
    index = X.rvs(negative_number)
    index = index.astype(np.int64)
    return index

def sample_node_negative_index(negative_number=0, epoch=0, epochs=0, Number = 0):
    lamda = 1 / 2 
    lower, upper = 0, Number-1
    mu = ((epoch) / epochs) ** lamda * (upper - lower)
    X = stats.uniform(1,mu) 
    index = X.rvs(negative_number) 

    mu_1 = ((epoch - 1) / epochs) ** lamda * (upper - lower)
    if epoch > 10:
        mu_1 = ((epoch - 10) / epochs) ** lamda * (upper - lower)
    mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
    X = stats.uniform(mu_1,mu_2-mu_1) 
    index = X.rvs(negative_number) 
    index = index.astype(np.int64)
    return index

def Sim_feat_loss_selfpace(Z , temperature = 1.0, epoch = 0, epochs = 0,s=10):

    N = Z.shape[0] 
    index = sample_negative_index(negative_number=s, epoch=epoch, epochs=epochs, Number=N)
    index = torch.tensor(index).cuda()  

    sim_matrix = torch.exp(torch.pow(Z, 2) / temperature)  
    positive_samples_ii_jj = torch.diag(sim_matrix).reshape(N, 1) 
    positive_samples = torch.sum(positive_samples_ii_jj,1) 

    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False)
    negative_samples = sim_matrix_sort.index_select(0, index)
    negative_samples = torch.sum(negative_samples, 0)  

    loss = (- torch.log(positive_samples / negative_samples)).mean()


    return loss

def Sim_node_loss_selfpace(Z, adj_label12 , temperature = 1.0, epoch = 0, epochs = 0, s=10):
    N = Z.shape[0]
    index = sample_node_negative_index(negative_number=s,epoch=epoch, epochs=epochs, Number=N)
    index = torch.tensor(index).cuda()
    sim_matrix = torch.exp(torch.pow(Z, 2) / temperature)

    positive_samples = torch.sum(sim_matrix * adj_label12, 1)
    sim_matrix_sort, _ = torch.sort(sim_matrix, dim=0, descending=False)
    negative_samples = sim_matrix_sort.index_select(0, index)
         
    negative_samples = torch.sum(negative_samples, 0) 
    loss = (- torch.log(positive_samples / negative_samples)).mean() 
    return loss



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


    def forward(self, seq1, seq2, adj, diff, sparse, epoch, epochs, adj_label12, batchsize = 1,s=10):
              
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
            Z_1 = torch.squeeze(h_1, 0)
            Z_2 = torch.squeeze(h_2, 0) 
            feat = torch.mm(F.normalize((Z_1+Z_2).T), F.normalize((Z_1+Z_2).T).T)  # [512,512]
            feat_loss = Sim_feat_loss_selfpace(feat, temperature=self.tau_feat, epoch=epoch, epochs=epochs,s=s)
            node = torch.mm(F.normalize(Z_1+Z_2), F.normalize(Z_1+Z_2).T)   # [2708,2708]
            node_loss = Sim_node_loss_selfpace(node, adj_label12, temperature=self.tau_node, epoch=epoch, epochs=epochs,s=s)

        else:
            feat_loss = self.batched_feat_loss(h_1+h_2, h_1+h_2, batchsize, epoch, epochs)

            node_loss = self.batched_node_loss(h_1, h_2, adj_label12, batchsize, epoch, epochs)

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

