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
import yaml
import argparse
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval
import argparse
import logging
from logging import getLogger, Logger
from pytz import timezone
from datetime import datetime
import math
import copy
import gc

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
import jax

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import * 
from evaluation import *
from load_data import *
from learning import *
from utils_processing import *
from helpers import *
from nn_models import *

from existing_methods.nn_based.risk_based import *
from existing_methods.nn_based.idgp import *
from existing_methods.nn_based.plsp import *



class CFLDataset(torch.utils.data.Dataset):
    def __init__(self, ord_features, feat_est_input_features, comp_features_dict, targets):
        self.ord_features=ord_features
        self.feat_est_input_features = feat_est_input_features
        self.comp_features_dict = comp_features_dict.copy()
        self.targets = targets

        self.comp_features_names = list(comp_features_dict.keys())

    def __len__(self):
        return len(self.ord_features)

    def __getitem__(self, idx):
        tmp_comp_features_dict = {}
        for cf_name in self.comp_features_names:
            tmp_comp_features_dict[cf_name] = self.comp_features_dict[cf_name][idx]
        return self.ord_features[idx], self.feat_est_input_features[idx], tmp_comp_features_dict, self.targets[idx], idx  # OFs, CFs, Labels, Indices
    
def cfl_dataloader(train_df, test_df, batch_size,
                   cf_names, downstream_target_name,
                   dl_generator=None, return_dataset=False):

    train_comp_feature_dict = {}
    test_comp_feature_dict = {}
    cf_ohe_cols_all = []
    for cf_name in cf_names:
        cf_ohe_cols = [c for c in train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
        train_comp_feature_dict[cf_name] = torch.tensor(train_df[cf_ohe_cols].values, dtype=torch.float32)
        test_comp_feature_dict[cf_name] = torch.tensor(test_df[cf_ohe_cols].values, dtype=torch.float32)
        cf_ohe_cols_all.extend(cf_ohe_cols)

    estimated_cf_cols =[c for c in train_df.columns.tolist() if re.search(rf'^estimated_', c)]

    train_dataset = CFLDataset(ord_features=torch.tensor(train_df.drop([downstream_target_name]+cf_ohe_cols_all+estimated_cf_cols, axis=1).values, dtype=torch.torch.float32),
                               feat_est_input_features=torch.tensor(train_df.drop([downstream_target_name]+cf_ohe_cols_all, axis=1).values, dtype=torch.torch.float32),
                               comp_features_dict=train_comp_feature_dict,
                               targets=F.one_hot(torch.tensor(train_df[downstream_target_name].values, dtype=torch.int64)).to(torch.float32) )   
    test_dataset = CFLDataset(ord_features=torch.tensor(test_df.drop([downstream_target_name]+cf_ohe_cols_all+estimated_cf_cols, axis=1).values, dtype=torch.torch.float32),
                               feat_est_input_features=torch.tensor(test_df.drop([downstream_target_name]+cf_ohe_cols_all, axis=1).values, dtype=torch.torch.float32),
                               comp_features_dict=test_comp_feature_dict,
                               targets=F.one_hot(torch.tensor(test_df[downstream_target_name].values, dtype=torch.int64)).to(torch.float32) )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                                               worker_init_fn=seed_worker, generator=dl_generator)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    num_classes_dict = {}
    num_classes_dict[downstream_target_name] = len(test_df[downstream_target_name].value_counts())
    for cf_name in cf_names:
        num_classes_dict[cf_name] = len([c for c in test_df.columns.tolist() if re.search(rf'^{cf_name}_', c)])

    if return_dataset:
        return train_loader, test_loader, train_dataset, test_dataset, num_classes_dict
    else:
        return train_loader, test_loader, num_classes_dict



def iterative_learning_feat_est(feat_est_models:dict, lr, bs, ep, wd, device, 
                                obj_lam,
                                est_method, cf_names,
                                comp_train_df, comp_test_df, ord_train_df, ord_test_df,  
                                label_pred_model, pred_loss_func='log',
                                nn_beta=None, # method specific parameters
                                seed=42,
                                test_size_for_loop=-1,
                                verbose=False):
    
    dl_generator = set_seed_torch(seed, return_g=True)

    # prepare dataloader
    cfl_comp_train_loader, cfl_comp_test_loader, num_classes_dict = cfl_dataloader(train_df=comp_train_df,
                                                                                test_df=comp_test_df,
                                                                                batch_size=bs,
                                                                                cf_names=cf_names,
                                                                                downstream_target_name='target',
                                                                                dl_generator=dl_generator)
    
    cfl_ord_train_loader, cfl_ord_test_loader, num_classes_dict = cfl_dataloader(train_df=ord_train_df,
                                                                                test_df=ord_test_df,
                                                                                batch_size=bs,
                                                                                cf_names=cf_names,
                                                                                downstream_target_name='target',
                                                                                dl_generator=dl_generator)
    
    if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
        use_test_index = sorted(random.sample(ord_test_df.index.tolist(), test_size_for_loop))

    comp_onehot_names_list = []
    # comp OneHot column names
    for cf_name in cf_names:
        comp_onehot_names_list += [col for col in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', col)]

    comp_train_loaders, comp_test_loaders = {}, {}
    loop_comp_test_loaders ={}
    ord_train_loaders, ord_test_loaders = {}, {}
    loop_ord_test_loaders ={}
    for cf_name in cf_names:
        est_target_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
        comp_train_loaders[cf_name], comp_test_loaders[cf_name], _ = prepare_dataloader(train_df=comp_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            test_df=comp_test_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            batch_size=bs,
                                                                            target_name=cf_name,
                                                                            dl_generator=dl_generator,
                                                                            return_idx=True)

        ord_train_loaders[cf_name], ord_test_loaders[cf_name], _ = prepare_dataloader(train_df=ord_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            test_df=ord_test_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            batch_size=bs,
                                                                            target_name=cf_name,
                                                                            dl_generator=dl_generator,
                                                                            return_idx=True)

    
        if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
            _, loop_comp_test_loaders[cf_name], _ = prepare_dataloader(train_df=comp_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                test_df=comp_test_df.iloc[use_test_index].reset_index(drop=True).drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                batch_size=bs,
                                                                                target_name=cf_name,
                                                                                dl_generator=dl_generator,
                                                                                return_idx=True)

            _, loop_ord_test_loaders[cf_name], _ = prepare_dataloader(train_df=ord_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                test_df=ord_test_df.iloc[use_test_index].reset_index(drop=True).drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                batch_size=bs,
                                                                                target_name=cf_name,
                                                                                dl_generator=dl_generator,
                                                                                return_idx=True)
                                
        else:
            loop_comp_test_loaders[cf_name] = comp_test_loaders[cf_name]
            loop_ord_test_loaders[cf_name] = ord_test_loaders[cf_name]
    
    cf_ohe_cols_all = []
    for cf_name in cf_names:
        cf_ohe_cols_all.extend([c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)])

    # estimate class prior
    ccp_dict ={}
    for cf_name in cf_names:
        if est_method in ['free', 'nn', 'ga']:
            cf_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
            hard_comp_labels = np.argmin(comp_train_df[cf_ohe_cols].values, axis=1)
            ccp_dict[cf_name] = class_prior(hard_comp_labels)
        else:
            ccp_dict[cf_name] = None

    # initial weight for PRODEN
    if est_method == 'proden':
        train_final_labels_dict = {}
        for cf_name in cf_names:
            cf_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
            train_final_labels_dict[cf_name] = torch.from_numpy(comp_train_df[cf_ohe_cols].values).clone()



    # preparation for feature estimation models
    feat_est_optimizers = {}
    feat_est_schedulers = {}
    for cf_name in cf_names:
        feat_est_models[cf_name] = feat_est_models[cf_name].to(device)
        feat_est_optimizers[cf_name] = torch.optim.Adam(feat_est_models[cf_name].parameters(), weight_decay=wd, lr=lr)
        feat_est_schedulers[cf_name] = CosineLRScheduler(feat_est_optimizers[cf_name], t_initial=ep, lr_min=1e-6, 
                                                        warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)


    # preparation for the label prediction model
    label_pred_model = label_pred_model.to(device)

    if pred_loss_func == 'logistic':
        if num_classes_dict['target'] ==2:
            pred_criteria = nn.SoftMarginLoss()
        else:
            pred_criteria = nn.MultiMarginLoss()
    elif pred_loss_func == 'log':
        pred_criteria = nn.CrossEntropyLoss() # uses softmax function inside
    else:
        raise NotImplementedError

    with torch.no_grad():
        for cf_name in cf_names:
            feat_est_models[cf_name].eval()
            train_accuracy = accuracy_check(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device)
            test_accuracy = accuracy_check(loader=loop_ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device)
            train_loss = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name], 
                                    ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            test_loss = loss_check(loader=loop_comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
        if verbose:
            print('{} Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(cf_name, 0, train_accuracy, test_accuracy, train_loss, test_loss))
    
    
    save_table_dict = {}
    for cf_name in cf_names:
        save_table_dict[cf_name] = np.zeros(shape=(ep, 5))

    # training ==================================================================================================================================================================================
    for epoch in range(ep):
        for cf_name in cf_names:
            feat_est_schedulers[cf_name].step(epoch)
            feat_est_models[cf_name].train()
        
        for i, (ord_features, feat_est_input_features, comp_feature_dict, labels, indices) in enumerate(cfl_comp_train_loader):
            ord_features, feat_est_input_features, labels = ord_features.to(device), feat_est_input_features.to(device), labels.to(device)
            for cf_name in cf_names:
                comp_feature_dict[cf_name] = comp_feature_dict[cf_name].to(device)

            outputs_list = []
            for cf_name in cf_names:
                feat_est_optimizers[cf_name].zero_grad()
                outputs_list.append(feat_est_models[cf_name](feat_est_input_features))
            
            feat_est_loss_dict = {}
            for cf_idx, cf_name in enumerate(cf_names):
                if est_method == 'rc':
                    feat_est_loss_dict[cf_name] = rc_loss(outputs_list[cf_idx], comp_feature_dict[cf_name])
                elif est_method == 'cc':
                    comp_feature_dict[cf_name][comp_feature_dict[cf_name] != 0] = 1 # enable non-scaled
                    feat_est_loss_dict[cf_name] = cc_loss(outputs_list[cf_idx], comp_feature_dict[cf_name])
                elif est_method == 'pc':
                    feat_est_loss_dict[cf_name] = pc_loss(outputs_list[cf_idx], torch.min(comp_feature_dict[cf_name], dim=1)[1], num_classes=num_classes_dict[cf_name]) 
                elif est_method == 'forward':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name] = forward_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], device=device)
                elif est_method == 'free':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], _ = assump_free_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], device=device)
                elif est_method == 'nn':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], _ = non_negative_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], beta=nn_beta, device=device)
                elif est_method == 'ga':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], loss_vector = chosen_loss_c(f=outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], meta_method='free', device=device)
                    if torch.min(loss_vector).item() < 0:
                        loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(num_classes_dict[cf_name], requires_grad=True).view(-1,1).to(device)), 1)
                        min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                        feat_est_loss_dict[cf_name] = torch.sum(min_loss_vector)
                
                elif est_method == 'proden':
                    est_labels = train_final_labels_dict[cf_name][indices, :].detach().to(device)
                    feat_est_loss_dict[cf_name], new_label = partial_loss(outputs_list[cf_idx], est_labels)
                    train_final_labels_dict[cf_name][indices, :] = new_label.detach().cpu()
                    
                else:
                    raise NotImplementedError

 
            pred_outputs = label_pred_model(torch.cat([ord_features]+outputs_list  , dim=1))
            if pred_loss_func == 'logistic':
                if num_classes_dict['target'] == 2:
                    pred_loss = pred_criteria(pred_outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
                else:
                    pred_loss = pred_criteria(pred_outputs, torch.argmax(labels, axis=1).to(torch.int64))
            elif pred_loss_func == 'log':
                pred_loss = pred_criteria(pred_outputs, labels)

            all_loss = 0
            for cf_name in cf_names:
                all_loss += feat_est_loss_dict[cf_name]
            all_loss = pred_loss + obj_lam * all_loss
            
            all_loss.backward()

            for cf_name in cf_names:
                if est_method == 'ga':
                    if torch.min(loss_vector).item() < 0:
                        for group in feat_est_optimizers[cf_name].param_groups:
                            for p in group['params']:
                                p.grad = -1*p.grad

                feat_est_optimizers[cf_name].step()

    

        for cf_name in cf_names:
            with torch.no_grad():
                feat_est_models[cf_name].eval()
                train_accuracy = accuracy_check(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device)
                test_accuracy = accuracy_check(loader=loop_ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device)
                train_loss = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name], 
                                        ccp=ccp_dict[cf_name], nn_beta=nn_beta)
                test_loss = loss_check(loader=loop_comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                    ccp=ccp_dict[cf_name], nn_beta=nn_beta)

            if verbose:
                print('[{}] Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(cf_name, epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        
            save_table_dict[cf_name][epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss
    
    train_score_dicts, test_score_dicts = {}, {}
    for cf_name in cf_names:
        feat_est_models[cf_name].eval()
        with torch.no_grad():
            train_score_dicts[cf_name] = evaluation_torch(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, num_classes=num_classes_dict[cf_name])
            test_score_dicts[cf_name] = evaluation_torch(loader=ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, num_classes=num_classes_dict[cf_name])
            train_score_dicts[cf_name]['loss'] = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            test_score_dicts[cf_name]['loss'] = loss_check(loader=comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            
        
    return feat_est_models, save_table_dict, train_score_dicts, test_score_dicts




def accuracy_check_joint_label_pred(cfl_loader, label_pred_model, feat_est_models, device):
    soft_max = F.softmax

    label_pred_model = label_pred_model.to(device)
    for key in feat_est_models.keys():
        feat_est_models[key] = feat_est_models[key].to(device)

    total, num_samples = 0, 0
    for ord_features, feat_est_input_features, comp_feature_dict, labels, _ in cfl_loader:
        ord_features, feat_est_input_features, labels = ord_features.to(device), feat_est_input_features.to(device), labels.to(device)
        for cf_name in comp_feature_dict.keys():
            comp_feature_dict[cf_name] = comp_feature_dict[cf_name].to(device)

        outputs_list = []
        for cf_name in comp_feature_dict.keys():
            outputs_list.append(feat_est_models[cf_name](feat_est_input_features))

        pred_outputs = label_pred_model(torch.cat([ord_features]+outputs_list, dim=1))


        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)

        sm_outputs = soft_max(pred_outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, dim=1)
        total += torch.tensor(predicted == labels, dtype=torch.int8).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples


def loss_check_joint_label_pred(cfl_loader, label_pred_model, feat_est_models, device, loss_func, num_classes):
    total, num_samples = 0, 0

    label_pred_model = label_pred_model.to(device)
    for key in feat_est_models.keys():
        feat_est_models[key] = feat_est_models[key].to(device)
    
    
    # for label prediction models    
    if loss_func == 'logistic':
        if num_classes ==2:
            criteria = nn.SoftMarginLoss()
        else:
            criteria = nn.MultiMarginLoss()
    elif loss_func == 'log':
        criteria = nn.CrossEntropyLoss() # uses softmax function inside

    
    for ord_features, feat_est_input_features, comp_feature_dict, labels, _ in cfl_loader:
        ord_features, feat_est_input_features, labels = ord_features.to(device), feat_est_input_features.to(device), labels.to(device)
        for cf_name in comp_feature_dict.keys():
            comp_feature_dict[cf_name] = comp_feature_dict[cf_name].to(device)

        outputs_list = []
        for cf_name in comp_feature_dict.keys():
            outputs_list.append(feat_est_models[cf_name](feat_est_input_features))

        pred_outputs = label_pred_model(torch.cat([ord_features]+outputs_list, dim=1))

        # for label prediction models    
        if loss_func == 'logistic':
            if num_classes ==2:
                loss = criteria(pred_outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
            else:
                loss = criteria(pred_outputs, torch.argmax(labels, axis=1).to(torch.int64))
        elif loss_func == 'log':
            loss = criteria(pred_outputs, labels)

        total += loss.detach().cpu() * ord_features.size(0)
        num_samples += ord_features.size(0)

    return total / num_samples


def evaluation_torch_joint_label_pred(cfl_loader, label_pred_model, feat_est_models, device, num_classes):
    soft_max = F.softmax

    label_pred_model = label_pred_model.to(device)
    for key in feat_est_models.keys():
        feat_est_models[key] = feat_est_models[key].to(device)

    all_labels = torch.tensor([], dtype=torch.int64)
    all_pred_labels = torch.tensor([], dtype=torch.int64)
    all_pred_probs = torch.tensor([])
    for ord_features, feat_est_input_features, comp_feature_dict, labels, _ in cfl_loader:
        ord_features, feat_est_input_features, labels = ord_features.to(device), feat_est_input_features.to(device), labels.to(device)
        for cf_name in comp_feature_dict.keys():
            comp_feature_dict[cf_name] = comp_feature_dict[cf_name].to(device)

        feat_outputs_list = []
        for cf_name in comp_feature_dict.keys():
            feat_outputs_list.append(feat_est_models[cf_name](feat_est_input_features))

        outputs = label_pred_model(torch.cat([ord_features]+feat_outputs_list, dim=1))

        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)
        
        sm_outputs = soft_max(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, dim=1)

        all_labels = torch.cat([all_labels, labels.detach().cpu().to(torch.int64)])
        all_pred_probs = torch.cat([all_pred_probs, sm_outputs.detach().cpu()], dim=0)
        all_pred_labels = torch.cat([all_pred_labels, predicted.detach().cpu().to(torch.int64)])

    score_dict = {}
    if num_classes == 2:
        score_dict['acc'] = mF.binary_accuracy(input=all_pred_labels, target=all_labels)
        score_dict['f1'] = mF.binary_f1_score(input=all_pred_labels, target=all_labels)
        #score_dict['prec'] = mF.binary_precision(input=all_pred_labels, target=all_labels)
        #score_dict['rec'] = mF.binary_recall(input=all_pred_labels, target=all_labels)
        score_dict['auroc'] = mF.binary_auroc(input=all_pred_labels, target=all_labels)
    else: # multiclass
        score_dict['acc'] = mF.multiclass_accuracy(input=all_pred_probs, target=all_labels, num_classes=num_classes)
        score_dict['f1'] = mF.multiclass_f1_score(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        #score_dict['prec'] = mF.multiclass_precision(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        #score_dict['rec'] = mF.multiclass_recall(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')
        score_dict['auroc'] = mF.multiclass_auroc(input=all_pred_probs, target=all_labels, num_classes=num_classes, average='macro')

    return score_dict



def joint_learning(feat_est_models:dict, label_pred_model,
                    lr, bs, ep, wd, device, 
                    obj_lam,
                    est_method, cf_names,
                    comp_train_df, comp_test_df, ord_train_df, ord_test_df,  
                    pred_loss_func='log',
                    nn_beta=None, # method specific parameters
                    seed=42,
                    test_size_for_loop=-1,
                    verbose=False):

    dl_generator = set_seed_torch(seed, return_g=True)

    if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
        use_test_index = sorted(random.sample(ord_test_df.index.tolist(), test_size_for_loop))

    # prepare dataloader
    cfl_comp_train_loader, cfl_comp_test_loader, num_classes_dict = cfl_dataloader(train_df=comp_train_df,
                                                                                test_df=comp_test_df,
                                                                                batch_size=bs,
                                                                                cf_names=cf_names,
                                                                                downstream_target_name='target',
                                                                                dl_generator=dl_generator)
    
    cfl_ord_train_loader, cfl_ord_test_loader, num_classes_dict = cfl_dataloader(train_df=ord_train_df,
                                                                                test_df=ord_test_df,
                                                                                batch_size=bs,
                                                                                cf_names=cf_names,
                                                                                downstream_target_name='target',
                                                                                dl_generator=dl_generator)
    
    if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
        _, loop_cfl_comp_test_loader, _ = cfl_dataloader(train_df=comp_train_df,
                                                            test_df=comp_test_df.iloc[use_test_index].reset_index(drop=True),
                                                            batch_size=bs,
                                                            cf_names=cf_names,
                                                            downstream_target_name='target',
                                                            dl_generator=dl_generator)

                                    
        _, loop_cfl_ord_test_loader, _ = cfl_dataloader(train_df=ord_train_df,
                                                        test_df=ord_test_df.iloc[use_test_index].reset_index(drop=True),
                                                        batch_size=bs,
                                                        cf_names=cf_names,
                                                        downstream_target_name='target',
                                                        dl_generator=dl_generator)
    else:
        loop_cfl_comp_test_loader = cfl_comp_test_loader
        loop_cfl_ord_test_loader = cfl_ord_test_loader

    comp_onehot_names_list = []
    # comp OneHot column names
    for cf_name in cf_names:
        comp_onehot_names_list += [col for col in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', col)]

    comp_train_loaders, comp_test_loaders = {}, {}
    loop_comp_test_loaders ={}
    ord_train_loaders, ord_test_loaders = {}, {}
    loop_ord_test_loaders ={}
    for cf_name in cf_names:
        est_target_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
        comp_train_loaders[cf_name], comp_test_loaders[cf_name], _ = prepare_dataloader(train_df=comp_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            test_df=comp_test_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            batch_size=bs,
                                                                            target_name=cf_name,
                                                                            dl_generator=dl_generator,
                                                                            return_idx=True)

        ord_train_loaders[cf_name], ord_test_loaders[cf_name], _ = prepare_dataloader(train_df=ord_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            test_df=ord_test_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                            batch_size=bs,
                                                                            target_name=cf_name,
                                                                            dl_generator=dl_generator,
                                                                            return_idx=True)

    
        if test_size_for_loop > 0 and test_size_for_loop < ord_test_df.shape[0]:
            _, loop_comp_test_loaders[cf_name], _ = prepare_dataloader(train_df=comp_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                test_df=comp_test_df.iloc[use_test_index].reset_index(drop=True).drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                batch_size=bs,
                                                                                target_name=cf_name,
                                                                                dl_generator=dl_generator,
                                                                                return_idx=True)

            _, loop_ord_test_loaders[cf_name], _ = prepare_dataloader(train_df=ord_train_df.drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                test_df=ord_test_df.iloc[use_test_index].reset_index(drop=True).drop(['target'] + list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1),
                                                                                batch_size=bs,
                                                                                target_name=cf_name,
                                                                                dl_generator=dl_generator,
                                                                                return_idx=True)
                                
        else:
            loop_comp_test_loaders[cf_name] = comp_test_loaders[cf_name]
            loop_ord_test_loaders[cf_name] = ord_test_loaders[cf_name]
    
    cf_ohe_cols_all = []
    for cf_name in cf_names:
        cf_ohe_cols_all.extend([c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)])

    # estimate class prior
    ccp_dict ={}
    for cf_name in cf_names:
        if est_method in ['free', 'nn', 'ga']:
            cf_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
            hard_comp_labels = np.argmin(comp_train_df[cf_ohe_cols].values, axis=1)
            ccp_dict[cf_name] = class_prior(hard_comp_labels)
        else:
            ccp_dict[cf_name] = None

    # initial weight for PRODEN
    if est_method == 'proden':
        train_final_labels_dict = {}
        for cf_name in cf_names:
            cf_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{cf_name}_', c)]
            train_final_labels_dict[cf_name] = torch.from_numpy(comp_train_df[cf_ohe_cols].values).clone()



    # preparation for feature estimation models
    feat_est_optimizers = {}
    feat_est_schedulers = {}
    for cf_name in cf_names:
        feat_est_models[cf_name] = feat_est_models[cf_name].to(device)
        feat_est_optimizers[cf_name] = torch.optim.Adam(feat_est_models[cf_name].parameters(), weight_decay=wd, lr=lr)
        feat_est_schedulers[cf_name] = CosineLRScheduler(feat_est_optimizers[cf_name], t_initial=ep, lr_min=1e-6, 
                                                        warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)


    # preparation for the label prediction model
    label_pred_model = label_pred_model.to(device)
    label_pred_optimizer = torch.optim.Adam(label_pred_model.parameters(), weight_decay=wd, lr=lr)
    label_pred_scheduler = CosineLRScheduler(label_pred_optimizer, t_initial=ep, lr_min=1e-6, 
                                                        warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    

    if pred_loss_func == 'logistic':
        if num_classes_dict['target'] ==2:
            pred_criteria = nn.SoftMarginLoss()
        else:
            pred_criteria = nn.MultiMarginLoss()
    elif pred_loss_func == 'log':
        pred_criteria = nn.CrossEntropyLoss() # uses softmax function inside
    else:
        raise NotImplementedError
    

    with torch.no_grad():
        for cf_name in cf_names:
            feat_est_models[cf_name].eval()
            train_accuracy = accuracy_check(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device)
            test_accuracy = accuracy_check(loader=loop_ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device)
            train_loss = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name], 
                                    ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            test_loss = loss_check(loader=loop_comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            if verbose:
                print('[{}] Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(cf_name, 0, train_accuracy, test_accuracy, train_loss, test_loss))
        
        train_accuracy = accuracy_check_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device)
        test_accuracy = accuracy_check_joint_label_pred(cfl_loader=loop_cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device)
        train_loss = loss_check_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])
        test_loss = loss_check_joint_label_pred(cfl_loader=loop_cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])

        if verbose:
                print('[{}] Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format('target', 0, train_accuracy, test_accuracy, train_loss, test_loss))


    save_table_dict = {}
    for cf_name in cf_names:
        save_table_dict[cf_name] = np.zeros(shape=(ep, 5))
    save_table_dict['target'] = np.zeros(shape=(ep, 5))

    # training ==================================================================================================================================================================================
    for epoch in range(ep):
        for cf_name in cf_names:
            feat_est_schedulers[cf_name].step(epoch)
            feat_est_models[cf_name].train()
        label_pred_scheduler.step(epoch)
        label_pred_model.train()

        for i, (ord_features, feat_est_input_features, comp_feature_dict, labels, indices) in enumerate(cfl_comp_train_loader):
            ord_features, feat_est_input_features, labels = ord_features.to(device), feat_est_input_features.to(device), labels.to(device)
            for cf_name in cf_names:
                comp_feature_dict[cf_name] = comp_feature_dict[cf_name].to(device)

            # calc output and loss for feature estimation models -------------------------------------------------------------------------------------------------------------------------
            outputs_list = []
            for cf_name in cf_names:
                feat_est_optimizers[cf_name].zero_grad()
                outputs_list.append(feat_est_models[cf_name](feat_est_input_features))

            feat_est_loss_dict = {}
            for cf_idx, cf_name in enumerate(cf_names):
                if est_method == 'rc':
                    feat_est_loss_dict[cf_name] = rc_loss(outputs_list[cf_idx], comp_feature_dict[cf_name])
                elif est_method == 'cc':
                    comp_feature_dict[cf_name][comp_feature_dict[cf_name] != 0] = 1 # enable non-scaled
                    feat_est_loss_dict[cf_name] = cc_loss(outputs_list[cf_idx], comp_feature_dict[cf_name])
                elif est_method == 'pc':
                    feat_est_loss_dict[cf_name] = pc_loss(outputs_list[cf_idx], torch.min(comp_feature_dict[cf_name], dim=1)[1], num_classes=num_classes_dict[cf_name]) 
                elif est_method == 'forward':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name] = forward_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], device=device)
                elif est_method == 'free':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], _ = assump_free_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], device=device)
                elif est_method == 'nn':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], _ = non_negative_loss(outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], beta=nn_beta, device=device)
                elif est_method == 'ga':
                    comp_feature_dict[cf_name] = torch.argmin(comp_feature_dict[cf_name], dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
                    feat_est_loss_dict[cf_name], loss_vector = chosen_loss_c(f=outputs_list[cf_idx], num_classes=num_classes_dict[cf_name], labels=comp_feature_dict[cf_name], ccp=ccp_dict[cf_name], meta_method='free', device=device)
                    if torch.min(loss_vector).item() < 0:
                        loss_vector_with_zeros = torch.cat((loss_vector.view(-1,1), torch.zeros(num_classes_dict[cf_name], requires_grad=True).view(-1,1).to(device)), 1)
                        min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
                        feat_est_loss_dict[cf_name] = torch.sum(min_loss_vector)
                
                elif est_method == 'proden':
                    est_labels = train_final_labels_dict[cf_name][indices, :].detach().to(device)
                    feat_est_loss_dict[cf_name], new_label = partial_loss(outputs_list[cf_idx], est_labels)
                    train_final_labels_dict[cf_name][indices, :] = new_label.detach().cpu()
                    
                else:
                    raise NotImplementedError
                

            # calc output and loss of label prediction model -----------------------------------------------------------------------------------------------------

            pred_outputs = label_pred_model(torch.cat([ord_features]+outputs_list  , dim=1))
            if pred_loss_func == 'logistic':
                if num_classes_dict['target'] == 2:
                    pred_loss = pred_criteria(pred_outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
                else:
                    pred_loss = pred_criteria(pred_outputs, torch.argmax(labels, axis=1).to(torch.int64))
            elif pred_loss_func == 'log':
                pred_loss = pred_criteria(pred_outputs, labels)

            # backward loss -------------------------------------------------------------------------------------------------------------------------------------------
            all_loss = 0
            for cf_name in cf_names:
                all_loss += feat_est_loss_dict[cf_name]
            all_loss = pred_loss + obj_lam * all_loss
            
            all_loss.backward()
            for cf_name in cf_names:
                if est_method == 'ga':
                    if torch.min(loss_vector).item() < 0:
                        for group in feat_est_optimizers[cf_name].param_groups:
                            for p in group['params']:
                                p.grad = -1*p.grad
                feat_est_optimizers[cf_name].step()

            label_pred_optimizer.step()

            

                
        for cf_name in cf_names:
            with torch.no_grad():
                feat_est_models[cf_name].eval()
                train_accuracy = accuracy_check(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device)
                test_accuracy = accuracy_check(loader=loop_ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device)
                train_loss = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name], 
                                        ccp=ccp_dict[cf_name], nn_beta=nn_beta)
                test_loss = loss_check(loader=loop_comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                    ccp=ccp_dict[cf_name], nn_beta=nn_beta)

            if verbose:
                print('[{}] Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(cf_name, epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        
            save_table_dict[cf_name][epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss

        train_accuracy = accuracy_check_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device)
        test_accuracy = accuracy_check_joint_label_pred(cfl_loader=loop_cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device)
        train_loss = loss_check_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])
        test_loss = loss_check_joint_label_pred(cfl_loader=loop_cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])

        if verbose:
            print('[{}] Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format('target', epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
        save_table_dict['target'][epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss

        
    train_score_dicts, test_score_dicts = {}, {}
    for cf_name in cf_names:
        feat_est_models[cf_name].eval()
        with torch.no_grad():
            train_score_dicts[cf_name] = evaluation_torch(loader=ord_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, num_classes=num_classes_dict[cf_name])
            test_score_dicts[cf_name] = evaluation_torch(loader=ord_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, num_classes=num_classes_dict[cf_name])
            train_score_dicts[cf_name]['loss'] = loss_check(loader=comp_train_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            test_score_dicts[cf_name]['loss'] = loss_check(loader=comp_test_loaders[cf_name], model=feat_est_models[cf_name], device=device, loss_func=est_method, num_classes=num_classes_dict[cf_name],
                                                ccp=ccp_dict[cf_name], nn_beta=nn_beta)
            
    train_score_dicts['target'] = evaluation_torch_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, num_classes=num_classes_dict['target'])
    test_score_dicts['target'] = evaluation_torch_joint_label_pred(cfl_loader=cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, num_classes=num_classes_dict['target'])
    train_score_dicts['target']['loss'] = loss_check_joint_label_pred(cfl_loader=cfl_comp_train_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])
    test_score_dicts['target']['loss'] = loss_check_joint_label_pred(cfl_loader=loop_cfl_comp_test_loader, label_pred_model=label_pred_model, feat_est_models=feat_est_models, device=device, loss_func=pred_loss_func, num_classes=num_classes_dict['target'])
    
    return label_pred_model, feat_est_models, save_table_dict, train_score_dicts, test_score_dicts

