import os
import sys
import re
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torcheval.metrics.functional as mF
import torcheval.metrics.classification as mC


from .nn_loss import *

sys.path.append('../../')
from utils import *

class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], idx  # feature, label, index




def prepare_dataloader(train_df, test_df, batch_size, target_name='target', dl_generator=None, return_idx=False, return_dataset=False):
    
    # when target is one-hot encoded, target_cols is list, otherwise, target_cols is str (taget_name) 
    target_cols = [c for c in train_df.columns.tolist() if re.search(rf'^{target_name}_', c)]
    if len(target_cols) == 0: # target col is not one-hot encoded
        target_cols = [target_name]
    
    if len(target_cols) == 1: # target col is not one-hot encoded
        target_cols = target_name
        if return_idx:
            train_dataset = IdxDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                            F.one_hot(torch.tensor(train_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                            )
            test_dataset  = IdxDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                            F.one_hot(torch.tensor(test_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                            )
        else:
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                            F.one_hot(torch.tensor(train_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                            )
            test_dataset  = torch.utils.data.TensorDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                            F.one_hot(torch.tensor(test_df[target_cols].values, dtype=torch.int64)).to(torch.float32)
                                                            )
    else: # target col is one-hot encoded
        if return_idx:
            train_dataset = IdxDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        torch.tensor(train_df[target_cols].values, dtype=torch.float32))
            test_dataset  = IdxDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        torch.tensor(test_df[target_cols].values, dtype=torch.float32))
        else:
            train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        torch.tensor(train_df[target_cols].values, dtype=torch.float32))
            test_dataset  = torch.utils.data.TensorDataset(torch.tensor(test_df.drop(target_cols, axis=1).values, dtype=torch.float32),
                                                        torch.tensor(test_df[target_cols].values, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                                               worker_init_fn=seed_worker, generator=dl_generator)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    if target_cols == target_name:
        num_classes = len(test_df[target_name].value_counts())
    else:
        num_classes = len(target_cols)

    if return_dataset:
        return train_loader, test_loader, train_dataset, test_dataset, num_classes
    else:
        return train_loader, test_loader, num_classes



def accuracy_check(loader, model, device):
    soft_max = F.softmax
    total, num_samples = 0, 0
    for instances, labels, _ in loader:
        instances, labels = instances.to(device), labels.to(device)
        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)
        outputs = model(instances)
        sm_outputs = soft_max(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, dim=1)
        total += torch.tensor(predicted == labels, dtype=torch.int8).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples


def loss_check(loader, model, device, loss_func, num_classes, ccp=None, nn_beta=None):
    total, num_samples = 0, 0
    
    # for label prediction models    
    if loss_func == 'logistic':
        if num_classes ==2:
            criteria = nn.SoftMarginLoss()
        else:
            criteria = nn.MultiMarginLoss()
    elif loss_func == 'log':
        criteria = nn.CrossEntropyLoss() # uses softmax function inside
    
    for instances, labels, _ in loader:
        instances, labels = instances.to(device), labels.to(device)
        outputs = model(instances)

        # for label prediction models    
        if loss_func == 'logistic':
            if num_classes ==2:
                loss = criteria(outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
            else:
                loss = criteria(outputs, torch.argmax(labels, axis=1).to(torch.int64))
        elif loss_func == 'log':
            loss = criteria(outputs, labels)
        # for feature estimation models
        elif loss_func == 'rc':
            loss = rc_loss(outputs, labels)
        elif loss_func == 'cc':
            loss = cc_loss(outputs, labels)
        elif loss_func == 'pc':
            loss = pc_loss(outputs, torch.min(labels, dim=1)[1], num_classes=num_classes) 
        elif loss_func == 'forward':
            labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
            loss = forward_loss(outputs, num_classes=num_classes, labels=labels, device=device)
        elif loss_func in ['free', 'ga']:
            labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
            loss, _ = assump_free_loss(outputs, num_classes=num_classes, labels=labels, ccp=ccp, device=device)
        elif loss_func == 'nn':
            labels = torch.argmin(labels, dim=1)  # ohe label that candidate is 1 -> non-ohe label that complementary is 1
            loss, _ = non_negative_loss(outputs, num_classes=num_classes, labels=labels, ccp=ccp, beta=nn_beta, device=device)
        elif loss_func == 'proden':
            loss, _ = partial_loss(outputs, labels)

        total += loss.detach().cpu() * instances.size(0)
        num_samples += instances.size(0)

    return total / num_samples

def evaluation_torch(loader, model, device, num_classes):
    soft_max = F.softmax

    all_labels = torch.tensor([], dtype=torch.int64)
    all_pred_labels = torch.tensor([], dtype=torch.int64)
    all_pred_probs = torch.tensor([])
    for instances, labels, _ in loader:
        instances, labels = instances.to(device), labels.to(device)
        if labels.dim() == 2: # when label is one-hot encoded
            _, labels = torch.max(labels, dim=1)
        outputs = model(instances)
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

