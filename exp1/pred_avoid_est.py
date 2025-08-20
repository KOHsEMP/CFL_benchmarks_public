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

sys.path.append("../libs")
from utils import * 
from evaluation import *
from load_data import *
from learning import *
from utils_processing import *
from helpers import *


from existing_methods.nn_based.risk_based import *
from existing_methods.nn_based.idgp import *




def run(args, logger):
    # set seed
    set_seed_torch(args.seed)

    # data preparation =====================================================================================
    # load data
    logger.info("data loading...")
    data_df, cat_cols = load_data(data_name=args.dataset_name, data_path=args.data_dir,
                                  sample_size=args.sample_size, seed=args.seed)
    
    if args.task == 'classification':
        n_classes = data_df['target'].nunique()
    else:
        n_classes = None

    # convert some ordinary features to complementary features
    if 'all' in args.comp_cols:
        comp_feature_list = cat_cols
    else:
        comp_feature_list = args.comp_cols

    comp_data_df = privacy_transform(df=data_df, mask_feature_list=comp_feature_list, mode="comp", seed=args.seed)
    
    # one hot encoding
    data_df = exec_ohe(data_df, cat_cols, is_comp=False)

    cat_cols_ord = [c for c in cat_cols if c not in comp_feature_list]
    comp_data_df = exec_ohe(comp_data_df, cat_cols_ord, is_comp=False) 
    comp_data_df = exec_ohe(comp_data_df, comp_feature_list, is_comp=True)

    # normalization of comp cols
    col_names = comp_data_df.columns.tolist()
    for col in comp_feature_list:
        col_ohe_idx = [i for i, c in enumerate(col_names) if col in c]
        comp_data_df.iloc[:, col_ohe_idx] *= 1.0/(len(col_ohe_idx) -1) 

    
    comp_onehot_names_list = []
    # comp OneHot column names
    for cat_col in comp_feature_list:
        comp_onehot_names_list += [col for col in comp_data_df.columns.tolist() if re.search(rf'^{cat_col}_', col)]


    # train test split
    test_index = sorted(random.sample(data_df.index.tolist(), int(data_df.shape[0] * args.test_rate)))
    train_index = sorted(list(set(data_df.index.tolist()) - set(test_index)))

    ord_train_df = data_df.iloc[train_index].reset_index(drop=True)
    ord_test_df = data_df.iloc[test_index].reset_index(drop=True)
    comp_train_df = comp_data_df.iloc[train_index].reset_index(drop=True)
    comp_data_df = exec_ohe(comp_data_df, comp_feature_list, is_comp=True) # Order: OFs, Target, Non CF Categorical cols, CFs

    data_df.reindex(comp_data_df.columns, axis=1)

    # training and evaluation
    ## for log
    pred_model_score_dict = {} # Save the model's prediction score when using data where the CFs' exact values is estimated only from the training data.
    if args.method not in ['ord', 'comp']:
        pred_model_score_dict= {'soft': {}, 'hard':{}}


    # disambiguation (estimating CFs' exact values)=========================================================================================================================================
    # load first estimated result
    logger.info("load estimated results...")
    with open(os.path.join(args.output_dir, args.exp_name, args.method, args.estimated_log_name, args.estimated_log_name +"_df.pkl"), "rb") as f:
        estimated_log_df_dict = pickle.load(f)
    disamb_soft_df = estimated_log_df_dict['disamb_soft_df']
    disamb_hard_df = estimated_log_df_dict['disamb_hard_df']

    if len(args.avoid_estimate_cols) > 0 and args.pred_aec:
        for cat_col in args.avoid_estimate_cols:
            avoid_estimate_ohe_cols =  [col for col in comp_data_df.columns.tolist() if re.search(rf'^{cat_col}_', col)]
            disamb_soft_df[avoid_estimate_ohe_cols] = comp_data_df[avoid_estimate_ohe_cols].values
            disamb_hard_df[avoid_estimate_ohe_cols] = comp_data_df[avoid_estimate_ohe_cols].values

    # prediction of output labels ================================================================================================================================
    logger.info("predicting downstream task...")

    # set seed
    set_seed_torch(args.seed)

    if args.task == "classification":
        algorithm_list = ['logistic', 'random_forest', 'adaboost', 'mlp1221', 'mlp_2layer']
    elif args.task == "regression":
        raise NotImplementedError
    else:
        raise NotImplementedError
    

    for algorithm in algorithm_list:
        logger.info(f"algorithm = {algorithm}")

        is_torch=False
        if algorithm in ['mlp1221', 'mlp_2layer']:
            is_torch=True

        if args.method == "ord":
            pred_model = training(train_df=ord_train_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            scores = evaluating(test_df=ord_test_df, model=pred_model, target_col='target', task=args.task, n_classes=n_classes, is_torch=is_torch)
            pred_model_score_dict[algorithm] = scores.copy()

            del scores['confusion_matrix']
            logger.info(scores)

        elif args.method == "comp":
            pred_model = training(train_df=comp_train_df, target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            scores = evaluating(test_df=comp_test_df, model=pred_model, target_col='target', task=args.task, n_classes=n_classes, is_torch=is_torch)
            pred_model_score_dict[algorithm] = scores.copy()

            del scores['confusion_matrix']
            logger.info(scores)

        elif args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga', 'proden', 'idgp', ]:
            pred_model_soft = training(train_df=disamb_soft_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            pred_model_hard = training(train_df=disamb_hard_df.loc[train_index, :], target_col='target', algorithm=algorithm, task=args.task, seed=args.seed)
            
            pred_model_scores_soft = evaluating(test_df=disamb_soft_df.loc[test_index, :], model=pred_model_soft, target_col='target', task=args.task, n_classes=n_classes, is_torch=is_torch)
            pred_model_scores_hard = evaluating(test_df=disamb_hard_df.loc[test_index, :], model=pred_model_hard, target_col='target', task=args.task, n_classes=n_classes, is_torch=is_torch)

            pred_model_score_dict['soft'][algorithm] = pred_model_scores_soft.copy()
            pred_model_score_dict['hard'][algorithm] = pred_model_scores_hard.copy()

            del pred_model_scores_soft['confusion_matrix']
            del pred_model_scores_hard['confusion_matrix']
            logger.info(f"pred_model_scores [soft]: {pred_model_scores_soft}")
            logger.info(f"pred_model_scores [hard]: {pred_model_scores_hard}")

    # save log =======================================================================================================================================

    # save 
    log_dict = {}
    log_dict['args'] = args
    log_dict['pred_model_score'] = pred_model_score_dict


    with open(os.path.join(args.output_dir, args.exp_name, args.method, args.log_name, args.log_name +"_log.pkl"), "wb") as f:
        pickle.dump(log_dict, f)
    



if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args()
    if args.config_file is not None and os.path.exists(args.config_file):
        config_file = args.config_file
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            config_args = argparse.Namespace(**config)
            for k, v in config_args.__dict__.items():
                if v is not None:
                    setattr(args, k, v)
        print(f"Loaded config from {config_file}!")

    # load default config
    default_config_dict = get_args_default()
    for key, val in config_args.__dict__.items():
        if key not in default_config_dict.keys():
            setattr(args, key, val)

    if len(args.avoid_estimate_cols) == 0 and args.pred_aec:
        print(f"Please choose pred avoid estimate cols")
        exit()


    if args.dataset_name in ['bank', 'adult']:
        args.task = 'classification'
    else:
        args.task = 'regression'

    args.pred_aec = True

    args.exp_name += f"_{args.dataset_name}"

    args.estimated_log_name = get_log_filename(args)

    if args.iter_idx > 0:
        args.estimated_log_name = args.estimated_log_name + f'_iter{args.iter_idx}'

    # if there is no first iteration log, then, stop this program.
    if not os.path.isfile(os.path.join(args.output_dir, args.exp_name, args.method, args.estimated_log_name, args.estimated_log_name +"_df.pkl")):
        print(f"There is not estimated_log_name: {os.path.join(args.output_dir, args.exp_name, args.method, args.estimated_log_name, args.estimated_log_name +'_df.pkl')}")
        exit()
    
    args.log_name = args.estimated_log_name + '_pAEC' + str(comp_cols_code(args.dataset_name, args.avoid_estimate_cols))


    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.method), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.method, args.log_name), exist_ok=True)
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        #filename=os.path.join(args.output_dir, args.exp_name, log_filename),
    )

    logger=getLogger(args.dataset_name)

    # https://qiita.com/r1wtn/items/d615f19e338cbfbfd4d6
    # Set handler to output to files
    fh = logging.FileHandler(os.path.join(args.output_dir, args.exp_name, args.method, args.log_name, args.log_name + ".log"))
    fh.setLevel(logging.DEBUG)
    def customTime(*args):
        return datetime.now(timezone('Asia/Tokyo')).timetuple()
    formatter = logging.Formatter(
        fmt='%(levelname)s : %(asctime)s : %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S %z"
    )
    formatter.converter = customTime
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
   
    # logging args
    for k, v in config_args.__dict__.items():
        logger.info(f"args[{k}] = {v}")

    run(args, logger)