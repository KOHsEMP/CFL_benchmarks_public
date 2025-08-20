import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import requests
import pickle
import arff
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse

import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
import torcheval.metrics.functional as mF
import torcheval.metrics.classification as mC
from timm.scheduler import CosineLRScheduler

#sys.path.append('./')
from .nn_loss import *
from .nn_utils import *

sys.path.append('../../')
from utils import *
from nn_models import *


def train_risk_based(lr, bs, ep, wd, device, 
                    est_method, est_target_name, 
                    comp_train_df, comp_test_df, ord_train_df, ord_test_df,
                    arch=None, model=None,
                    nn_beta=None, # method specific parameters
                    hidden_dim=None,
                    seed=42,
                    test_size_for_loop=-1,
                    verbose=False):

    
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

    # estimate class prior
    if est_method in ['free', 'nn', 'ga']:
        target_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{est_target_name}_', c)]
        hard_comp_labels = np.argmin(comp_train_df[target_cols].values, axis=1)
        ccp = class_prior(hard_comp_labels)
    else:
        ccp=None

    # initial weight for PRODEN
    if est_method == 'proden':
        target_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{est_target_name}_', c)]
        train_final_labels = torch.from_numpy(comp_train_df[target_cols].values).clone()


                                                    
    
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

    if model is None:
        assert arch is not None
        if arch == 'mlp':
            assert hidden_dim is not None
            model = mlp_model(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=ep, lr_min=1e-6, 
                                warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    
    model.eval()
    with torch.no_grad():
        train_accuracy = accuracy_check(loader=ord_train_loader, model=model, device=device)
        test_accuracy = accuracy_check(loader=loop_ord_test_loader, model=model, device=device)
        train_loss = loss_check(loader=comp_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes, 
                                ccp=ccp, nn_beta=nn_beta)
        test_loss = loss_check(loader=loop_comp_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes,
                               ccp=ccp, nn_beta=nn_beta)
    if verbose:
        print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(0, train_accuracy, test_accuracy, train_loss, test_loss))
    
    save_table = np.zeros(shape=(ep, 5))
    for epoch in range(ep):
        scheduler.step(epoch)
        model.train()
        for i, (instances, labels, indices) in enumerate(comp_train_loader):
            instances, labels = instances.to(device), labels.to(device) # labels is represetated as scaled one-hot encoded partial labels
            optimizer.zero_grad()
            outputs = model(instances)

            if est_method == 'rc':
                loss = rc_loss(outputs, labels)
            elif est_method == 'cc':
                labels[labels != 0] = 1 # enable non-scaled
                loss = cc_loss(outputs, labels)
            elif est_method == 'pc':
                loss = pc_loss(outputs, torch.min(labels, dim=1)[1], num_classes=num_classes) 
            elif est_method == 'forward':
                labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                loss = forward_loss(outputs, num_classes=num_classes, labels=labels, device=device)
            elif est_method == 'free':
                labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                loss, _ = assump_free_loss(outputs, num_classes=num_classes, labels=labels, ccp=ccp, device=device)
            elif est_method == 'nn':
                labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                loss, _ = non_negative_loss(outputs, num_classes=num_classes, labels=labels, ccp=ccp, beta=nn_beta, device=device)
            elif est_method == 'ga':
                labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                loss, loss_vector = chosen_loss_c(f=outputs, num_classes=num_classes, labels=labels, ccp=ccp, meta_method='free', device=device)
                if torch.min(loss_vector).item() < 0:
                    loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(num_classes, requires_grad=True).view(-1,1).to(device)), 1)
                    min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                    loss = torch.sum(min_loss_vector)
            
            elif est_method == 'proden':
                labels = train_final_labels[indices, :].detach().to(device)
                loss, new_label = partial_loss(outputs, labels)
                train_final_labels[indices, :] = new_label.detach().cpu()
                
            else:
                raise NotImplementedError
            
            loss.backward()

            if est_method == 'ga':
                if torch.min(loss_vector).item() < 0:
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            p.grad = -1*p.grad
            
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_accuracy = accuracy_check(loader=ord_train_loader, model=model, device=device)
            test_accuracy = accuracy_check(loader=loop_ord_test_loader, model=model, device=device)
            train_loss = loss_check(loader=comp_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes,
                                    ccp=ccp, nn_beta=nn_beta)
            test_loss = loss_check(loader=loop_comp_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes,
                                   ccp=ccp, nn_beta=nn_beta)
        if verbose:
            print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss
    
    model.eval()
    with torch.no_grad():
        train_score_dict = evaluation_torch(loader=ord_train_loader, model=model, device=device, num_classes=num_classes)
        test_score_dict = evaluation_torch(loader=ord_test_loader, model=model, device=device, num_classes=num_classes)
        train_score_dict['loss'] = loss_check(loader=comp_train_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes,
                                              ccp=ccp, nn_beta=nn_beta)
        test_score_dict['loss'] = loss_check(loader=comp_test_loader, model=model, device=device, loss_func=est_method, num_classes=num_classes,
                                             ccp=ccp, nn_beta=nn_beta)


    return model, save_table, train_score_dict, test_score_dict





