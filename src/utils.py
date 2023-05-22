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


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)

