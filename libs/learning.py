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
import copy
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

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import jax
import jaxopt
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler import CosineLRScheduler

from utils import * 
from evaluation import *
from nn_models import *
from existing_methods.nn_based.nn_utils import *




# learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss)

# TODO: remove this function
def training_and_eval(df, target_col, algorithm, test_size, task, seed):

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    model = training(train_df=train_df, 
                     target_col=target_col,
                     algorithm=algorithm,
                     task=task,
                     seed=seed)
    
    scores = evaluating(test_df=test_df,
                        model=model,
                        target_col=target_col,
                        task=task,
                        n_classes=df[target_col].nunique()
                        )

    
    print(scores)

    return model, scores


def training(train_df, target_col, algorithm, task, seed, verbose=False):
    '''
    Training label prediction model
    Args:
        train_df: training data (pd.DataFrame) includes input features (instances) and output feature (output label)
        target_col: output feature name
        algorithm: training algorithm for creating label prediction model (choices: "logistic", "random_forest", "adaboost", "mlp1221")
        task: "classification" ("regression" is not implemented)
        seed: random seed for training algorithms
    Returns:
        model: trained label prediction model
    '''
    # training
    if algorithm == "logistic":
        model = LogisticRegression(random_state=seed, solver="lbfgs", multi_class='auto', n_jobs=1)    
    elif algorithm == "svm":
        model = SVC(kernel='rbf', gamma='auto', random_state=seed, probability=True)
    elif algorithm == "knn":
        if task == "classification":
            model = KNeighborsClassifier(n_neighbors=train_df["target"].nunique(), n_jobs=1)
        else: # regression
            raise NotImplementedError
    elif algorithm == "decision_tree":
        model =  DecisionTreeClassifier(random_state=seed)
    elif algorithm == "gaussian_naive_bayes":
        model = GaussianNB()
    elif algorithm == "qda":
        model = QuadraticDiscriminantAnalysis()
    elif algorithm == "gaussian_process":
        if task == "classification":
            _kernel = 1.0 * RBF(1.0)
            model = GaussianProcessClassifier(kernel=_kernel, random_state=seed, n_jobs=1)
        else: # regression
            raise NotImplementedError
    elif algorithm == "random_forest":
        if task == "classification":
            model = RandomForestClassifier(random_state=seed, n_jobs=1)
        else: # regression
            raise NotImplementedError
    elif algorithm == "adaboost":
        if task == "classification":
            model = AdaBoostClassifier(random_state=seed,
                                    n_estimators=10,
                                    base_estimator= DecisionTreeClassifier(max_depth=10))
        else: # regression
            raise NotImplementedError
    
    elif algorithm == "mlp1221":
        if task == "classification":
            #model = MLPClassifier(hidden_layer_sizes=(100, 200, 200, 100), activation='relu', solver='adam',
            #                    shuffle=True, random_state=seed, early_stopping=True)
            input_dim = train_df.drop(target_col, axis=1).shape[1]
            model = mlp_4layers(input_dim=input_dim, hidden_dims=[100,200,200,100], output_dim=len(np.unique(train_df['target'].values)))
            model = train_mlp(model=model, lr=0.00005, bs=512, ep=100, wd=0.0002, device='cuda',  
                              train_df=train_df, seed=seed, target_col=target_col, num_classes=len(np.unique(train_df['target'].values)),
                              return_eval=False, verbose=verbose)
            model = model.cpu()
        else:
            raise NotImplementedError
    elif algorithm == "mlp_2layer":
        if task == "classification":
            input_dim = train_df.drop(target_col, axis=1).shape[1]
            model = mlp_model(input_dim=input_dim, hidden_dim=500, output_dim=len(np.unique(train_df['target'].values)))
            model = train_mlp(model=model, lr=0.00005, bs=512, ep=100, wd=0.0002, device='cuda',
                              train_df=train_df, seed=seed, target_col=target_col, num_classes=len(np.unique(train_df['target'].values)),
                              return_eval=False, verbose=verbose)
            model = model.cpu()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    if algorithm not in ["mlp1221", 'mlp_2layer']:
        model.fit(train_df.drop(target_col,axis=1), train_df[target_col])  
    
    return model
    

def evaluating(test_df, model, target_col, task, n_classes=None, is_torch=False, device='cuda'):
    '''
    Evaluating the prediction results of the givin label prediction model.
    Args:
        test_df: test data (pd.DataFrame) includes input features (instances) and output feature (output label)
        model: label prediction model (by created "training" function)
        target_col: output feature name
        n_classes: the number of classes when the problem is classification problem
    Returns:
        scores: dict (key: score name, val: score value)
    '''

    if task == "classification":
        if is_torch:
            soft_max = F.softmax
            model.to(device)
            test_pred_prob = soft_max(model(torch.tensor(test_df.drop(target_col,axis=1).values, dtype=torch.float32).to(device)))
            test_pred_label = torch.argmax(test_pred_prob, axis=1).to(torch.int64).detach().cpu().numpy()
            test_pred_prob = test_pred_prob.detach().cpu().numpy()
        else:
            test_pred_label = model.predict(test_df.drop(target_col,axis=1))
            test_pred_prob  = model.predict_proba(test_df.drop(target_col,axis=1))
    elif task == "regression":
        raise NotImplementedError

    else:
        raise NotImplementedError
    
    # evaluate
    if task == "classification":
        scores = evaluation_classification_detail(true_value=test_df[target_col].values, 
                                                    pred_label=test_pred_label, 
                                                    pred_prob=test_pred_prob,
                                                    n_classes=n_classes)
    elif task == "regression":
        raise NotImplementedError
    else:
        raise NotImplementedError
    

    return scores


def train_mlp(model, lr, bs, ep, wd, device, train_df, seed, target_col, num_classes, loss_func='log',
              test_df=None, test_size_for_loop=-1, verbose=False, return_eval=True):

    
    dl_generator = set_seed_torch(seed, return_g=True)

    train_dataset = IdxDataset(torch.tensor(train_df.drop(target_col, axis=1).values, dtype=torch.float32),
                                                   F.one_hot(torch.tensor(train_df[target_col].values, dtype=torch.int64)).to(torch.float32))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, 
                                               worker_init_fn=seed_worker, generator=dl_generator)
    
    if test_df is not None:
        test_dataset = IdxDataset(torch.tensor(test_df.drop(target_col, axis=1).values, dtype=torch.float32),
                                                    F.one_hot(torch.tensor(test_df[target_col].values, dtype=torch.int64)).to(torch.float32))
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
    
        if test_size_for_loop > 0 and test_size_for_loop < test_df.shape[0]:
            use_test_index = sorted(random.sample(test_df.index.tolist(), test_size_for_loop))
            loop_test_dataset = IdxDataset(torch.tensor(test_df.iloc[use_test_index].reset_index(drop=True).drop(target_col, axis=1).values, dtype=torch.float32),
                                                        F.one_hot(torch.tensor(test_df.loc[use_test_index, target_col].reset_index(drop=True).values, dtype=torch.int64)).to(torch.float32))
            loop_test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True, 
                                                    worker_init_fn=seed_worker, generator=dl_generator)
            
        else:
            loop_test_loader = copy.deepcopy(test_loader)


    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=wd, lr=lr)
    scheduler = CosineLRScheduler(optimizer, t_initial=ep, lr_min=1e-6, 
                                warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    
    if loss_func == 'logistic':
        if num_classes ==2:
            criteria = nn.SoftMarginLoss()
        else:
            criteria = nn.MultiMarginLoss()
    elif loss_func == 'log':
        criteria = nn.CrossEntropyLoss() # uses softmax function inside
    else:
        raise NotImplementedError
    
    model.eval()
    with torch.no_grad():
        train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
        if test_df is not None:
            test_accuracy = accuracy_check(loader=loop_test_loader, model=model, device=device)
        train_loss = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        if test_df is not None:
            test_loss = loss_check(loader=loop_test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
    if verbose:
        if test_df is not None:
            print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(0, train_accuracy, test_accuracy, train_loss, test_loss))
        else:
            print('Epoch: {}. Tr Acc: {}. Tr Loss: {}.'.format(0, train_accuracy, train_loss))
    
    if test_df is not None:
        save_table = np.zeros(shape=(ep, 5))
    else:
        save_table = np.zeros(shape=(ep, 3))
    
    
    for epoch in range(ep):
        scheduler.step(epoch)
        model.train()
        for i, (instances, labels, _) in enumerate(train_loader):
            instances, labels = instances.to(device), labels.to(device) # labels is represetated as scaled one-hot encoded partial labels
            optimizer.zero_grad()
            outputs = model(instances)

            if loss_func == 'logistic':
                if num_classes == 2:
                    loss = criteria(outputs[:, 1], torch.argmax(labels, axis=1).mul_(2).sub_(1).to(torch.int64)) # label \in {+1, -1}
                else:
                    loss = criteria(outputs, torch.argmax(labels, axis=1).to(torch.int64))
            elif loss_func == 'log':
                loss = criteria(outputs, labels)
            
            loss.backward()            
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
            if test_df is not None:
                test_accuracy = accuracy_check(loader=loop_test_loader, model=model, device=device)
            
            train_loss = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
            if test_df is not None:
                test_loss = loss_check(loader=loop_test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        if verbose:
            if test_df is not None:
                print('Epoch: {}. Tr Acc: {}. Te Acc: {}. Tr Loss: {}. Te Loss: {}'.format(epoch+1, train_accuracy, test_accuracy, train_loss, test_loss))
            else:
                print('Epoch: {}. Tr Acc: {}. Tr Loss: {}.'.format(epoch+1, train_accuracy, train_loss))

        if test_df is not None:
            save_table[epoch, :] = epoch+1, train_accuracy, test_accuracy, train_loss, test_loss
        else:
            save_table[epoch, :] = epoch+1, train_accuracy, train_loss

    model.eval()
    with torch.no_grad():
        train_score_dict = evaluation_torch(loader=train_loader, model=model, device=device, num_classes=num_classes)
        if test_df is not None:
            test_score_dict = evaluation_torch(loader=test_loader, model=model, device=device, num_classes=num_classes)
        train_score_dict['loss'] = loss_check(loader=train_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
        if test_df is not None:
            test_score_dict['loss'] = loss_check(loader=test_loader, model=model, device=device, loss_func=loss_func, num_classes=num_classes)
    

    if return_eval: 
        return model, save_table, train_score_dict, test_score_dict
    else:
        return model
