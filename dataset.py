from dgl.data import CoraFullDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset
from dgl.data import AmazonCoBuyPhotoDataset
from utils import preprocess_features, normalize_adj, compute_ppr, select_DatasSet, get_A_r
from dgl.data import CitationGraphDataset,load_data
from sklearn.preprocessing import MinMaxScaler

import scipy.sparse as sp
import networkx as nx
import numpy as np
import os


def download(dataset):
    if dataset in ['citeseer', 'pubmed', 'cora']:
        return CitationGraphDataset(name=dataset)
    elif dataset == 'amac':
        return AmazonCoBuyComputerDataset()
    elif dataset == 'amap':
        return AmazonCoBuyPhotoDataset()
    elif dataset == 'coauthorCS':
        return CoauthorCSDataset()
    elif dataset == 'coauthorP':
        return CoauthorPhysicsDataset()
    elif dataset == 'corafull':
        return CoraFullDataset()
    else:
        return None


def load(dataset):
    datadir = os.path.join('data', dataset)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        ds = download(dataset)[0]
        adj = ds.adjacency_matrix().to_dense().numpy()
        diff = compute_ppr(adj, 0.2)
        feat = ds.ndata['feat'].numpy()
        labels = ds.ndata['label'].numpy()
 

        if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
            train_mask = ds.ndata['train_mask']
            val_mask = ds.ndata['val_mask']
            test_mask = ds.ndata['test_mask']

            idx_train = np.where(np.array(train_mask) == True)
            idx_val = np.where(np.array(val_mask) == True)
            idx_test = np.where(np.array(test_mask) == True)
            np.save(f'{datadir}/idx_train.npy', idx_train)
            np.save(f'{datadir}/idx_val.npy', idx_val)
            np.save(f'{datadir}/idx_test.npy', idx_test)

        np.save(f'{datadir}/adj.npy', adj)
        np.save(f'{datadir}/diff.npy', diff)
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)
    else:
        adj = np.load(f'{datadir}/adj.npy')
        diff = np.load(f'{datadir}/diff.npy')
        feat = np.load(f'{datadir}/feat.npy')
        labels = np.load(f'{datadir}/labels.npy')

    if dataset == "cora" or dataset == "citeseer" or dataset == "pubmed":
        idx_train = np.load(f'{datadir}/idx_train.npy')
        idx_val = np.load(f'{datadir}/idx_val.npy')
        idx_test = np.load(f'{datadir}/idx_test.npy')
    else:

        idx_train, idx_val, idx_test = select_DatasSet(labels)


    if dataset == 'citeseer':
        feat = preprocess_features(feat)

        epsilons = [1e-5, 1e-4, 1e-3, 1e-2]
        avg_degree = np.sum(adj) / adj.shape[0]
        epsilon = epsilons[np.argmin([abs(avg_degree - np.argwhere(diff >= e).shape[0] / diff.shape[0])
                                      for e in epsilons])]

        diff[diff < epsilon] = 0.0
        scaler = MinMaxScaler()
        scaler.fit(diff)
        diff = scaler.transform(diff)

    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    

    adjdatadir = os.path.join(datadir, 'adj_12.npy')
    if not os.path.exists(adjdatadir):
        adj_12 = get_A_r(adj, r=2)
        np.save(f'{datadir}/adj_12.npy', adj_12)
    else:
        adj_12 = np.load(f'{datadir}/adj_12.npy')

    return adj, adj_12, diff, feat, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    adj, adj_label12, diff, features, labels, idx_train, idx_val, idx_test = load('cora')
    
