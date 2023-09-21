import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp
import random
import yaml
import logging

def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        print("Best args not found")
        return args

    # print("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args

import argparse
def build_args():
    parser = argparse.ArgumentParser(description='model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--seed', type=int, default=999)
    parser.add_argument('--sample_size', type=int, default=2708)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--sigma', type=float, default=0.001)
    parser.add_argument('--tau_feat', type=float, default=1.0)
    parser.add_argument('--tau_node', type=float, default=10.0)
    parser.add_argument('--nb_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--cnt_wait', type=int, default=0)
    parser.add_argument('--best_t', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_coef', type=float, default=0.0)
    parser.add_argument('--best', type=float, default=1e9)
    parser.add_argument('--hid_units', type=int, default=512)
    parser.add_argument('--self_sample', type=int, default=50)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--sparse', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False) 
    args = parser.parse_args()
    return args

def get_A_r(adj_label, r):
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label @ adj_label 
    elif r == 3:
        adj_label = adj_label @ adj_label @ adj_label
    elif r == 4:
        adj_label = adj_label @ adj_label @ adj_label @ adj_label
    return adj_label


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_ppr(a, alpha=0.2, self_loop=True):
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))  # a(I_n-(1-a)A~)^-1

def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)

def select_DatasSet(labels):
    label_set = {}
    for i, label in enumerate(labels, 0):
        if label not in label_set.keys():
            label_set[label] = [i]
        else:
            label_set[label].extend([i])

    train_id = []
    valid_id = []
    test_id = []

    for key, value in label_set.items():

        train_id.extend(value[:int(len(value) * 0.1)])
        valid_id.extend(value[int(len(value) * 0.1): int(len(value) * 0.2)])
        test_id.extend(value[int(len(value) * 0.2):])


    return np.array(train_id), np.array(valid_id), np.array(test_id)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)