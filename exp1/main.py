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
        col_ohe_idx = [i for i, c in enumerate(col_names) if re.search(rf'^{col}_', c)]
        comp_data_df.iloc[:, col_ohe_idx] *= 1.0/(len(col_ohe_idx) -1) 

    comp_onehot_names_list = []
    # comp OneHot column names
    for comp_col in comp_feature_list:
        comp_onehot_names_list += [col for col in comp_data_df.columns.tolist() if re.search(rf'^{comp_col}_', col)]


    # train test split
    test_index = sorted(random.sample(data_df.index.tolist(), int(data_df.shape[0] * args.test_rate)))
    train_index = sorted(list(set(data_df.index.tolist()) - set(test_index)))

    ord_train_df = data_df.iloc[train_index].reset_index(drop=True)
    ord_test_df = data_df.iloc[test_index].reset_index(drop=True)
    comp_train_df = comp_data_df.iloc[train_index].reset_index(drop=True)
    comp_test_df = comp_data_df.iloc[test_index].reset_index(drop=True)

    # training and evaluation
    ## for log
    pred_model_score_dict = {} # Save the model's prediction score when using data where the CFs' exact values is estimated only from the training data.
    est_model_score_dict = {} # Save the score of estimation for CF's exact values when using data where the CFs' exact values is estimated only from the training data.
    if args.method not in ['ord', 'comp']:
        pred_model_score_dict= {'soft': {}, 'hard':{}}

    # disambiguation (estimating CFs' exact values)=========================================================================================================================================
    logger.info("disambiguation...")

    # set seed
    set_seed_torch(args.seed)
    
    disamb_soft_df = comp_data_df.copy() 
    disamb_hard_df = comp_data_df.copy() 

    
    # NN-based ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga', 'proden', 'idgp']:
        soft_max = F.softmax

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_dict = {}
        save_table_dict, train_score_dict_dict, test_score_dict_dict = {}, {}, {}

        logger.info("training models...")
        for comp_col in comp_feature_list:
            logger.info(f"training and disambiguating '{comp_col}' ...")
            est_target_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{comp_col}_', c)]

            # prepare dataset
            if args.use_bar_feature:
                nn_comp_train_df = comp_train_df.drop(['target'], axis=1).copy(deep=True)
                nn_comp_test_df = comp_test_df.drop(['target'], axis=1).copy(deep=True)
                nn_ord_train_df = ord_train_df.drop(['target'], axis=1).copy(deep=True)
                nn_ord_test_df = ord_test_df.drop(['target'], axis=1).copy(deep=True)

                # CF as a feature
                rename_dict = {}
                rename_list = []
                for comp_ohehot_name in comp_onehot_names_list:
                    rename_dict[comp_ohehot_name] = 'estimated_' + comp_ohehot_name
                    rename_list.append('estimated_' + comp_ohehot_name)
                nn_comp_train_df.rename(columns=rename_dict, inplace=True)
                nn_comp_test_df.rename(columns=rename_dict, inplace=True)
                nn_ord_train_df.rename(columns=rename_dict, inplace=True)
                nn_ord_test_df.rename(columns=rename_dict, inplace=True)

                nn_ord_train_df[rename_list] = nn_comp_train_df[rename_list].values
                nn_ord_test_df[rename_list] = nn_comp_test_df[rename_list].values
                # CF as a target
                nn_comp_train_df[est_target_ohe_cols] = comp_train_df[est_target_ohe_cols].values
                nn_comp_test_df[est_target_ohe_cols] = comp_test_df[est_target_ohe_cols].values
                nn_ord_train_df[est_target_ohe_cols] = ord_train_df[est_target_ohe_cols].values
                nn_ord_test_df[est_target_ohe_cols] = ord_test_df[est_target_ohe_cols].values

            else:
                nn_comp_train_df = comp_train_df.drop(['target']+list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1).copy(deep=True)
                nn_comp_test_df = comp_test_df.drop(['target']+list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1).copy(deep=True)
                nn_ord_train_df = ord_train_df.drop(['target']+list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1).copy(deep=True)
                nn_ord_test_df = ord_test_df.drop(['target']+list(set(comp_onehot_names_list) - set(est_target_ohe_cols)), axis=1).copy(deep=True)

            


            # Risk-based ----------------------------------------------------------------------------------------------------------------------------
            if args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga', 'proden']:
                if args.arch == 'mlp':
                    assert args.hd is not None
                    model = mlp_model(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dim=args.hd, output_dim=len(est_target_ohe_cols))
                elif args.arch == 'mlp4':
                    assert len(args.hds) == 4
                    model = mlp_4layers(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dims=args.hds, output_dim=len(est_target_ohe_cols))

                model, save_table, train_score_dict, test_score_dict = \
                    train_risk_based(model=model, lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=device, 
                                        est_method=args.method, est_target_name=comp_col, 
                                        comp_train_df=nn_comp_train_df, 
                                        comp_test_df=nn_comp_test_df, 
                                        ord_train_df=nn_ord_train_df, 
                                        ord_test_df=nn_ord_test_df, 
                                        nn_beta=args.nn_beta, # method specific parameters
                                        hidden_dim=args.hd,
                                        seed=args.seed,
                                        test_size_for_loop=args.test_size_for_loop,
                                        verbose=args.verbose)
            # IDGP ----------------------------------------------------------------------------------------------------------------------------
            elif args.method == 'idgp':
                if args.arch == 'mlp':
                    assert args.hd is not None
                    f = mlp_model(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dim=args.hd, output_dim=len(est_target_ohe_cols))
                    g = mlp_model(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dim=args.hd, output_dim=len(est_target_ohe_cols))
                elif args.arch == 'mlp4':
                    assert len(args.hds) == 4
                    f = mlp_4layers(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dims=args.hds, output_dim=len(est_target_ohe_cols))
                    g = mlp_4layers(input_dim=len(nn_comp_train_df.columns) - len(est_target_ohe_cols), 
                                        hidden_dims=args.hds, output_dim=len(est_target_ohe_cols))
                model, save_table, train_score_dict, test_score_dict = \
                    train_idgp(f=f, g=g, lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=device, 
                                lr_g=args.idgp_lr_g, wd_g=args.idgp_wd_g, warmup_ep=args.idgp_warmup_ep, T_1=args.idgp_T_1, T_2=args.idgp_T_2,
                                alpha=args.idgp_alpha, beta=args.idgp_beta, delta=args.idgp_delta, theta=args.idgp_theta, gamma=args.idgp_gamma, eta=args.idgp_eta,
                                est_target_name=comp_col, 
                                comp_train_df=nn_comp_train_df, 
                                comp_test_df=nn_comp_test_df, 
                                ord_train_df=nn_ord_train_df, 
                                ord_test_df=nn_ord_test_df, 
                                hidden_dim=args.hd,
                                seed=args.seed,
                                test_size_for_loop=args.test_size_for_loop,
                                verbose=args.verbose)



            model_dict[comp_col] = copy.deepcopy(model)
            save_table_dict[comp_col] = copy.deepcopy(save_table)
            train_score_dict_dict[comp_col] = train_score_dict.copy()
            test_score_dict_dict[comp_col] = test_score_dict.copy()

            # disambiguation
            model.eval()
            with torch.no_grad():
                # train data
                outputs = model(torch.tensor(nn_comp_train_df.drop(est_target_ohe_cols, axis=1).values, dtype=torch.float32).to(device))
                pred_confs = soft_max(outputs).detach().cpu().numpy()
                disamb_soft_df.loc[train_index, est_target_ohe_cols] = pred_confs.copy()
                disamb_hard_df.loc[train_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]
                # test data
                outputs = model(torch.tensor(nn_comp_test_df.drop(est_target_ohe_cols, axis=1).values, dtype=torch.float32).to(device))
                pred_confs = soft_max(outputs).detach().cpu().numpy()
                disamb_soft_df.loc[test_index, est_target_ohe_cols] = pred_confs.copy()
                disamb_hard_df.loc[test_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]

        # evaluation of disambiguation
        ## inductive (create graph using only train data)
        disamb_scores_per_col_train, disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                    df_pred_prob=disamb_soft_df.iloc[train_index],
                                                                                                    df_pred_label=disamb_hard_df.iloc[train_index],
                                                                                                    comp_cols=comp_feature_list)
        disamb_scores_per_col_test, disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                                    df_pred_prob=disamb_soft_df.iloc[test_index], # for calculating Cross Entropy
                                                                                                    df_pred_label=disamb_hard_df.iloc[test_index],
                                                                                                    comp_cols=comp_feature_list)
        
        est_model_score_dict['scores_per_col (train)'] = disamb_scores_per_col_train.copy()
        est_model_score_dict['scores_average (train)'] = disamb_scores_average_train.copy()
        est_model_score_dict['scores_per_col (test)'] = disamb_scores_per_col_test.copy()
        est_model_score_dict['scores_average (test)'] = disamb_scores_average_test.copy()

        # show result
        del disamb_scores_average_train['confusion_matrix']
        del disamb_scores_average_test['confusion_matrix']
        logger.info(f"[train] disamb_scores_average: {disamb_scores_average_train}")
        logger.info(f"[test] disamb_scores_average: {disamb_scores_average_test}")

    


    # comp ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif args.method == 'comp':
        # evaluation using only training data
        disamb_scores_per_col_train, disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                        df_pred_prob=comp_data_df.iloc[train_index],
                                                                                        comp_cols=comp_feature_list,
                                                                                        labeling_strategy='random')    
        
        # evaluation using only test data
        disamb_scores_per_col_test, disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                        df_pred_prob=comp_data_df.iloc[test_index],
                                                                                        comp_cols=comp_feature_list,
                                                                                        labeling_strategy='random')
        
        est_model_score_dict['scores_per_col (train)'] = disamb_scores_per_col_train.copy()
        est_model_score_dict['scores_average (train)'] = disamb_scores_average_train.copy()
        est_model_score_dict['scores_per_col (test)'] = disamb_scores_per_col_test.copy()
        est_model_score_dict['scores_average (test)'] = disamb_scores_average_test.copy()

        # show result
        del disamb_scores_average_train['confusion_matrix']
        del disamb_scores_average_test['confusion_matrix']
        logger.info(f"[train] disamb_scores_average: {disamb_scores_average_train}")
        logger.info(f"[test] disamb_scores_average: {disamb_scores_average_test}")


    # save estimated result ================================================================================================================================
    # save estimated dataframe
    log_df_dict = {}
    if args.method not in ['ord', 'comp']:
        log_df_dict['disamb_soft_df'] = disamb_soft_df.copy()
        log_df_dict['disamb_hard_df'] = disamb_hard_df.copy()
    log_name = get_log_filename(args)
    with open(os.path.join(args.output_dir, args.exp_name, args.method, log_name, log_name +"_df.pkl"), "wb") as f:
        pickle.dump(log_df_dict, f)



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

        elif args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga', 'proden', 'idgp']:
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
    log_dict['est_model_score'] = est_model_score_dict

    if args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga', 'proden', 'idgp']:
        log_dict['models'] = model_dict
        log_dict['save_table_dict'] = save_table_dict
        log_dict['train_score_dict_dict'] = train_score_dict_dict
        log_dict['test_score_dict_dict'] = test_score_dict_dict

    log_name = get_log_filename(args)
    with open(os.path.join(args.output_dir, args.exp_name, args.method, log_name, log_name +"_log.pkl"), "wb") as f:
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
    
        

    if args.dataset_name in ['bank', 'adult']:
        args.task = 'classification'
    else:
        args.task = 'regression'

    args.exp_name += f"_{args.dataset_name}"

    log_name = get_log_filename(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.method), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, args.exp_name, args.method, log_name), exist_ok=True)
    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        #filename=os.path.join(args.output_dir, args.exp_name, log_filename),
    )

    logger=getLogger(args.dataset_name)

    # https://qiita.com/r1wtn/items/d615f19e338cbfbfd4d6
    # Set handler to output to files
    fh = logging.FileHandler(os.path.join(args.output_dir, args.exp_name, args.method, log_name, log_name + ".log"))
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
