from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch.nn.functional as F
import os.path as osp
import os
import argparse
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)



def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())




def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if data.train_mask is None:
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
            else:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    else:  # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data


class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


def l2_normalize(x):
    return x / torch.sqrt(torch.sum(x**2, dim=1).unsqueeze(1))


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','root','epochs','isAnneal','dropout','warmup_step','clus_num_iters']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def fill_ones(x):
    n_data = x.shape[0]
    x = torch.sparse_coo_tensor(x._indices(), torch.ones(x._nnz()).to(x.device), [n_data, n_data])

    return x