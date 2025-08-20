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

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch



def set_seed(seed=42):
    random.seed(seed) # python's seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # numpy's seed

def set_seed_torch(seed=42, return_g=False):
    random.seed(seed) # python's seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) # numpy's seed
    torch.manual_seed(seed) # torch's seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    if return_g:
        return g


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def abbrev_alg(alg_name):
    
    abbrev_dict = {
        'logistic':'LR',
        'svm': 'SVM',
        'knn': 'KNN',
        'decision_tree': 'DTree',
        'gaussian_naive_bayes': 'GNB',
        'qda': 'QDA',
        'random_forest':'RF',
        'adaboost':'AdaBoost',
        'mlp1221':'4hidden-MLP',
        'mlp_2layer':'2hidden-MLP',
    }

    return abbrev_dict[alg_name]

def show_score_name(score_name):
    show_score_dict = {
        'accuracy_score': 'Accuracy',
        'f1_score': 'F1',
        'f1_score_macro': 'F1 (macro)',
        'recall_score': 'Recall',
        'recall_score_macro': 'Recall (macro)',
        'precision_score': 'Precision',
        'precision_score_macro': 'Precision (macro)',
        'cross_entropy': 'Cross Entropy',
        'entropy': 'Entropy',
        'roc_auc_score': 'ROCAUC',
        'log_loss':'Log Loss',
    }


def show_score_abbrev(score_name):
    show_score_dict = {
        'accuracy_score': 'Acc',
        'f1_score': 'F1',
        'f1_score_macro': 'F1',
        'recall_score': 'Rec',
        'recall_score_macro': 'Rec',
        'precision_score': 'Prec',
        'precision_score_macro': 'Prec',
        'cross_entropy': 'CE',
        'entropy': 'SE',
        'roc_auc_score': 'ROCAUC',
        'log_loss':'Log',
    }

    return show_score_dict[score_name]

def show_dataset_name(dataset_name):
    show_dataset_dict = {
        'bank':'Bank',
        'adult': 'Adult',
        'diabetes': 'Diabetes',
    }

    return show_dataset_dict[dataset_name]


def show_method_name(method_name):
    show_method_dict={
        'ord': 'Ord',
        'comp': 'Comp',
        'ipal': 'IPAL',
        'rc': 'RC',
        'cc': 'CC',
        'proden': 'PRODEN',
        'idgp': 'IDGP',
        'forward': 'Forward',
        'free': 'Free',
        'nn': 'NN',
        'ga': 'GA',
    }
    return show_method_dict[method_name]


def return_task(dataset_name):

    if dataset_name in ['bank', 'adult']:
        return 'classification'
    else:
        return 'regression'
    

def return_uniq_val_dict(dataset_name):
    uniq_val_data_dict = {
        'adult':{
                "workclass": 9, 
                "education": 16, 
                "marital-status": 7, 
                "occupation": 15, 
                "relationship": 6, 
                "race": 5, 
                "native-country": 42,
                },
        'bank':{
                'job':12, 
                'marital':3, 
                'education':4, 
                'contact':3, 
                'poutcome':4,
                }
    }
    return uniq_val_data_dict[dataset_name]

def return_cat_cols_list(dataset_name):
    cat_cols_dict = {
    'adult':["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"],
    'bank': ['job', 'marital', 'education', 'contact', 'poutcome']
    }
    return cat_cols_dict[dataset_name]


def show_feature_name(feature_name):
    show_feature_dict = {
        'job':'job',
        'marital':'marital',
        'education': 'education',
        'contact':'contact',
        'poutcome':'poutcome',

        'workclass':'workclass',
        'marital-status':'marital',
        'occupation':'occupation',
        'relationship':'relationship',
        'race':'race',
        'native-country':'native-country'
    }

    return show_feature_dict[feature_name]


def show_strategy_name(strategy_name):
    show_strategy_dict = {
        'sequential':'Sequential',
        'iterative': 'Iterative',
        'joint': 'Joint'
    }

    return show_strategy_dict[strategy_name]