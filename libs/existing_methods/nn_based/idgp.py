'''
Cite from: 
    Qiao, C., Xu, N., & Geng, X. (2023). Decompositional generation process for instance-dependent partial label learning. ICLR.
    https://github.com/palm-ml/idgp/blob/main/main_idgp_benchmark.py
'''

import argparse
from copy import deepcopy
import os
import pickle
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
import torcheval.metrics.functional as mF
import torcheval.metrics.classification as mC
from timm.scheduler import CosineLRScheduler


from .nn_utils import *

sys.path.append('../../')
from utils import *
from nn_models import *

def weighted_crossentropy_f(f_outputs, weight):
    l = weight * torch.log(f_outputs + 1e-20)
    loss = (-torch.sum(l)) / l.size(0)
    
    return loss

def weighted_crossentropy_g(g_outputs, weight):
    l = weight * torch.log(g_outputs + 1e-20) + (1 - weight) * torch.log(1 - g_outputs + 1e-20)
    loss = ( - torch.sum(l)) / (l.size(0))

    return loss


def weighted_crossentropy_f_with_g(f_outputs, g_outputs, targets):
    weight = g_outputs.clone().detach() * targets
    weight[weight == 0] = 1.0
    logits1 = (1 - weight) / (weight + 1e-20)
    logits2 = weight.prod(dim=1, keepdim=True)
    weight = logits1 * logits2
    weight = weight * targets
    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
    weight = weight.clone().detach()
    
    l = weight * torch.log(f_outputs + 1e-20)
    loss = (-torch.sum(l)) / l.size(0)
    
    return loss

def weighted_crossentropy_g_with_f(g_outputs, f_outputs, targets):
 
    weight = f_outputs.clone().detach() * targets
    weight = weight / (weight.sum(dim=1, keepdim=True) + 1e-20)
    l = weight * ( torch.log((1 - g_outputs +1e-20) / (g_outputs + 1e-20)))
    l = weight * (torch.log(1 - g_outputs + 1e-20))
    loss = ( - torch.sum(l)) / ( l.size(0)) + \
        ( - torch.sum(targets * torch.log(g_outputs + 1e-20) + (1 - targets) * torch.log(1 - g_outputs + 1e-20))) / (l.size(0))
    
    return loss

def update_d(f_outputs, targets):
    new_d = f_outputs.clone().detach() * targets.clone().detach()
    new_d = new_d / (new_d.sum(dim=1, keepdim=True) + 1e-20)
    return new_d

def update_b(g_outputs, targets):
    new_b = g_outputs.clone().detach() * targets.clone().detach()
    
    return new_b

# 続き：ここから　d_arrayは外部で確保しておく．indicesはchatGPTにしたがって取得する
def warm_up(warmup_ep, T_1, 
            f, f_opt, train_loader, test_loader, d_array, device, valid_loader=None):

    print("Begin warm-up, warm up epoch {}".format(warmup_ep))
    
    d_array = d_array.to(device)

    for epoch in range(0, warmup_ep):
        f.train()

        for features, targets, indices in train_loader:
            features, targets = features.to(device), targets.to(device)
            f_logits = f(features)
            f_outputs = F.softmax(f_logits / T_1, dim=1)
            L_f = weighted_crossentropy_f(f_outputs, d_array[indices, :])
            f_opt.zero_grad()
            L_f.backward()
            f_opt.step()
            d_array[indices,:] = update_d(f_outputs, targets)
        f.eval()

        test_acc  = accuracy_check(loader=test_loader,  model=f, device=device)
        if valid_loader is not None:
            valid_acc = accuracy_check(loader=valid_loader, model=f, device=device)
            print("Warm_up Epoch {:>3d}, valid acc: {:.2f}, test acc: {:.2f}. ".format(epoch+1, valid_acc, test_acc))
        else:
            print("Warm_up Epoch {:>3d}, test acc: {:.2f}. ".format(epoch+1, test_acc))

    return f, d_array

def warm_up_g(warmup_ep, T_2, g, g_opt, train_loader, b_array, device):
    b_array = b_array.to(device)
    print("Begin warm-up, warm up epoch {}".format(warmup_ep))

    for epoch in range(0, warmup_ep):
        g.train()
        for features, targets, indexes in train_loader:
            features, targets = features.to(device), targets.to(device)
            g_logits = g(features)
            g_outputs = torch.sigmoid(g_logits / T_2)
            L_g = weighted_crossentropy_g(g_outputs, b_array[indexes,:])

            g_opt.zero_grad()
            L_g.backward()
            g_opt.step()
            b_array[indexes,:] = update_b(g_outputs, targets)
        g.eval()
        
    return g, b_array


def accuracy_check_g(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for features, targets, indices in loader:
            features, targets = features.to(device), targets.to(device)
            outputs = torch.sigmoid(model(features))
            _, pred = torch.max(outputs.data, dim=1)
            total += (pred == targets).sum().item()
            num_samples += targets.size(0)

    return 100*(total/num_samples)

def noisy_output(outputs, d_array, targets):
    _, true_labels = torch.max(d_array * targets, dim=1)
    pseudo_matrix  = F.one_hot(true_labels, outputs.shape[1]).float().cuda().detach()
    return pseudo_matrix * (1 - outputs) + (1 - pseudo_matrix) * outputs



def train_idgp(lr, bs, ep, wd, device,
               lr_g, wd_g, warmup_ep, T_1, T_2, # idgp specific
               alpha, beta, delta, theta, gamma, eta, # idgp specific
               est_target_name, 
               comp_train_df, comp_test_df, ord_train_df, ord_test_df,
               arch=None,
               f=None, g=None,
               hidden_dim=None,
               seed=42,
               verbose=True,
               test_size_for_loop=-1,):
    
    dl_generator = set_seed_torch(seed, return_g=True)

    comp_train_loader, comp_test_loader, num_classes = prepare_dataloader(train_df=comp_train_df,
                                                                            test_df=comp_test_df,
                                                                            batch_size=bs,
                                                                            target_name=est_target_name,
                                                                            dl_generator=dl_generator,
                                                                            return_idx=True)

    ord_train_loader, ord_test_loader, num_classes = prepare_dataloader(train_df=ord_train_df,
                                                                        test_df=ord_test_df,
                                                                        batch_size=bs,
                                                                        target_name=est_target_name,
                                                                        dl_generator=dl_generator,
                                                                        return_idx=True)
    
    if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
        use_test_index = sorted(random.sample(ord_test_df.index.tolist(), test_size_for_loop))
        loop_comp_train_loader, loop_comp_test_loader, num_classes = prepare_dataloader(train_df=comp_train_df,
                                                                                        test_df=comp_test_df.iloc[use_test_index].reset_index(drop=True),
                                                                                        batch_size=bs,
                                                                                        target_name=est_target_name,
                                                                                        dl_generator=dl_generator,
                                                                                        return_idx=True)
                        
        loop_ord_train_loader, loop_ord_test_loader, num_classes = prepare_dataloader(train_df=ord_train_df,
                                                                                        test_df=ord_test_df.iloc[use_test_index].reset_index(drop=True),
                                                                                        batch_size=bs,
                                                                                        target_name=est_target_name,
                                                                                        dl_generator=dl_generator,
                                                                                        return_idx=True)
                        
    else:
        loop_comp_test_loader = copy.deepcopy(comp_test_loader)
        loop_ord_test_loader = copy.deepcopy(ord_test_loader)


    input_dim = len(comp_train_df.columns) - len([c for c in list(comp_train_df.columns) if re.search(rf'^{est_target_name}_', c)])

    if f is None or g is None:
        assert arch is not None
        if arch == 'mlp':
            assert hidden_dim is not None
            f = mlp_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
            g = mlp_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)


    f = f.to(device)
    g = g.to(device)

    f_opt = torch.optim.Adam(f.parameters(), weight_decay=wd, lr=lr)
    g_opt = torch.optim.Adam(g.parameters(), weight_decay=wd_g, lr=lr_g)
    consistency_criterion_f = nn.KLDivLoss(reduction='batchmean').cuda()
    consistency_criterion_g = nn.KLDivLoss(reduction='batchmean').cuda()

    # warm up
    target_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{est_target_name}_', c)]
    d_array = torch.tensor(comp_train_df[target_cols].values, dtype=torch.float32)
    b_array = torch.tensor(comp_train_df[target_cols].values, dtype=torch.float32)
    f, d_array = warm_up(warmup_ep=warmup_ep, T_1=T_1, f=f, f_opt=f_opt, train_loader=comp_train_loader, test_loader=loop_ord_test_loader, 
                         d_array=d_array, device=device)
    g, b_array = warm_up_g(warmup_ep=warmup_ep, T_2=T_2, g=g, g_opt=g_opt, train_loader=comp_train_loader, b_array=b_array, device=device)

    d_array = d_array.to(device)
    b_array = b_array.to(device)
    save_table = np.zeros(shape=(ep, 5))
    for epoch in range(ep):
        f.train()
        g.train()

        for features, targets, indices in comp_train_loader:
            features, targets = features.to(device), targets.to(device)
            f_logits = f(features)
            g_logits = g(features)

            f_outputs = F.softmax(f_logits / T_1, dim=1)
            g_outputs = torch.sigmoid(g_logits / T_2)

            L_f = weighted_crossentropy_f(f_outputs, d_array[indices,:])
            L_f_g = weighted_crossentropy_f_with_g(f_outputs, 
                                                   noisy_output(g_outputs, d_array[indices, :], targets), 
                                                   targets)
            L_g = weighted_crossentropy_g(g_outputs, b_array[indices,:])
            L_g_f = weighted_crossentropy_g_with_f(g_outputs, f_outputs, targets)


            f_outputs_log = torch.log_softmax(f_logits, dim=-1)
            f_consist_loss = consistency_criterion_f(f_outputs_log, d_array[indices,:])
            g_outputs_log = nn.LogSigmoid()(g_logits)
            g_consist_loss = consistency_criterion_g(g_outputs_log, b_array[indices,:])
            lam = min(epoch / 100, 1)

            L_F = alpha * L_f + beta  * L_f_g + lam * delta * f_consist_loss
            L_G = theta * L_g + gamma * L_g_f + lam * eta   * g_consist_loss
            f_opt.zero_grad()
            L_F.backward()
            f_opt.step()
            g_opt.zero_grad()
            L_G.backward()
            g_opt.step()
            d_array[indices,:] = update_d(f_outputs, targets)
            b_array[indices,:] = update_b(g_outputs, targets)

        f.eval()
        g.eval()
        with torch.no_grad():
            train_accuracy = accuracy_check(loader=ord_train_loader, model=f, device=device)
            test_accuracy = accuracy_check(loader=loop_ord_test_loader, model=f, device=device)
            train_loss = loss_check(loader=ord_train_loader, model=f, device=device, loss_func='log', num_classes=num_classes)
            test_loss = loss_check(loader=loop_ord_test_loader, model=f, device=device, loss_func='log', num_classes=num_classes)
        if verbose:
            print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss

    f.eval()
    g.eval()
    with torch.no_grad():
        train_score_dict = evaluation_torch(loader=ord_train_loader, model=f, device=device, num_classes=num_classes)
        test_score_dict = evaluation_torch(loader=ord_test_loader, model=f, device=device, num_classes=num_classes)
        train_score_dict['loss'] = loss_check(loader=ord_train_loader, model=f, device=device, loss_func='log', num_classes=num_classes)
        test_score_dict['loss'] = loss_check(loader=ord_test_loader, model=f, device=device, loss_func='log', num_classes=num_classes)

    return f, save_table, train_score_dict, test_score_dict