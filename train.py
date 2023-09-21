import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from utils import sparse_mx_to_torch_sparse_tensor, get_A_r
from dataset import load
from model.model import *

def Train(args):
    dataset = args.dataset
    nb_epochs = args.nb_epochs
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2_coef
    hid_units = args.hid_units
    sample_size = args.sample_size
    batch_size = args.batch_size
    cnt_wait = args.cnt_wait
    best = args.best
    best_t = args.best_t
    gamma = args.gamma
    sigma = args.sigma
    sparse = args.sparse
    verbose = args.verbose
    tau_feat = args.tau_feat
    tau_node = args.tau_node
    s = args.self_sample
    max_epoch = args.max_epoch

    
    adj, adj_label12, diff, features, labels, idx_train, idx_val, idx_test = load(dataset)


    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units, tau_feat, tau_node)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if torch.cuda.is_available():
        model.cuda()
        labels = labels.cuda()
        lbl = lbl.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
        

    for epoch in range(1,nb_epochs+1):
        
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bal, bf = [], [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size]) # i到i+2000
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bal.append(adj_label12[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size) 
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size) 
        bal = np.array(bal).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)    

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)
            adj_label = torch.FloatTensor(bal)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)

        shuf_fts = bf[:, idx, :]

        if torch.cuda.is_available():
            bf = bf.cuda()
            ba = ba.cuda()
            bd = bd.cuda()
            shuf_fts = shuf_fts.cuda()
            adj_label = adj_label.cuda()

        model.train()
        optimiser.zero_grad()

        logits, feat_loss, node_loss = model(bf, shuf_fts, ba, bd, sparse, epoch, nb_epochs, adj_label, batch_size,s)
        info_loss = b_xent(logits, lbl)

        loss = info_loss + gamma * feat_loss + sigma * node_loss


        loss.backward()
        optimiser.step()
        
        
        if verbose:
            print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()))
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'model_{dataset}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(f'model_{dataset}.pkl'))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    embeds, _ = model.embed(features, adj, diff, sparse)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    val_lbls = labels[idx_val]
    test_lbls = labels[idx_test]

    wd = 0.01 if dataset == 'citeseer' else 0.0


    accs_test = []
    for _ in range(50):

        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=wd)
        log.cuda()
        for _ in range(max_epoch):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc_test = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs_test.append(acc_test * 100)
    accs_test = torch.stack(accs_test)
    print('acc and std：', accs_test.mean().item(), accs_test.std().item())
    return accs_test.mean()
