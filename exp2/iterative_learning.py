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
from nn_models import *
from cfl_learning import *

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
    comp_data_df = exec_ohe(comp_data_df, comp_feature_list, is_comp=True) # Order: OFs, Target, Non CF Categorical cols, CFs

    data_df.reindex(comp_data_df.columns, axis=1)


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


    # training and evaluation ====================================================================================================================================
    ## for log

    if args.start_round == 1:
        pred_model_score_dicts = {} # Save the model's prediction score when using data where the CFs' exact values is estimated only from the training data.
        est_model_score_dicts = {} # Save the score of estimation for CF's exact values when using data where the CFs' exact values is estimated only from the training data.
        est_save_tables, est_train_score_dicts, est_test_score_dicts = {}, {}, {}
        pred_save_tables, pred_train_score_dicts, pred_test_score_dicts = {}, {}, {}
        est_model_dicts, pred_model_dicts = {}, {}

    else:
        # load score log data
        with open(os.path.join(args.output_dir, args.exp_name, args.method, args.log_name, args.log_name +"_log.pkl"), "rb") as f:
            log_dict = pickle.load(f)
        pred_model_score_dicts = log_dict['pred_model_score'].copy()
        est_model_score_dicts = log_dict['est_model_score'].copy()

        est_save_tables = log_dict['est_save_tables'].copy()
        est_train_score_dicts = log_dict['est_train_score_dicts'].copy()
        est_test_score_dicts = log_dict['est_test_score_dicts'].copy()
        
        pred_save_tables = log_dict['pred_save_tables'].copy()
        pred_train_score_dicts = log_dict['pred_train_score_dicts'].copy()
        pred_test_score_dicts = log_dict['pred_test_score_dicts'].copy()

        est_model_dicts = log_dict['est_model_dicts'].copy()
        pred_model_dicts = log_dict['pred_model_dicts'].copy()

        executed_rounds = np.max(list(pred_model_score_dicts.keys()))

        logger.info(f'loaded score log data (~ round {executed_rounds})')

        if args.start_round > executed_rounds+1:
            logger.info(f'changed start_round from {args.start_round} to {executed_rounds+1}')
            args.start_round = executed_rounds+1


    soft_max = F.softmax
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    for round_idx in range(args.start_round, args.iterative_round+1):

        logger.info(f"start round {round_idx} ...")

        pred_model_score_dicts[round_idx] = {'soft': {}, 'hard':{}}
        est_model_score_dicts[round_idx] = {}
        est_save_tables[round_idx],  est_train_score_dicts[round_idx], est_test_score_dicts[round_idx] = {}, {}, {}
        pred_save_tables[round_idx], pred_train_score_dicts[round_idx], pred_test_score_dicts[round_idx] = {}, {}, {}
        est_model_dicts[round_idx], pred_model_dicts[round_idx] = {}, {}

        disamb_soft_df = comp_data_df.copy() 
        disamb_hard_df = comp_data_df.copy() 

        # learn feature estimation models ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        logger.info(f"[round {round_idx}] training feature estimation models...")
        # set seed
        set_seed_torch(args.seed)
            
        if round_idx == 1: ## 1st round .........................................................................................................................
            
            for comp_col in comp_feature_list:
                logger.info(f"[round {round_idx}] training and disambiguating '{comp_col}' ...")
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

            
                ## save result
                est_model_dicts[round_idx][comp_col] = copy.deepcopy(model)
                est_save_tables[round_idx][comp_col] = copy.deepcopy(save_table)
                est_train_score_dicts[round_idx][comp_col] = train_score_dict.copy()
                est_test_score_dicts[round_idx][comp_col] = test_score_dict.copy()

                ## disambiguation
                est_model_dicts[round_idx][comp_col].eval()
                with torch.no_grad():
                    # train data
                    outputs = est_model_dicts[round_idx][comp_col](torch.tensor(nn_comp_train_df.drop(est_target_ohe_cols, axis=1).values, dtype=torch.float32).to(device))
                    pred_confs = soft_max(outputs).detach().cpu().numpy()
                    disamb_soft_df.loc[train_index, est_target_ohe_cols] = pred_confs.copy()
                    disamb_hard_df.loc[train_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]
                    # test data
                    outputs = est_model_dicts[round_idx][comp_col](torch.tensor(nn_comp_test_df.drop(est_target_ohe_cols, axis=1).values, dtype=torch.float32).to(device))
                    pred_confs = soft_max(outputs).detach().cpu().numpy()
                    disamb_soft_df.loc[test_index, est_target_ohe_cols] = pred_confs.copy()
                    disamb_hard_df.loc[test_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]


            ## evaluate feature estimation models ...................................................................................................................
            disamb_scores_per_col_train, disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                        df_pred_prob=disamb_soft_df.iloc[train_index],
                                                                                                        df_pred_label=disamb_hard_df.iloc[train_index],
                                                                                                        comp_cols=comp_feature_list)
            disamb_scores_per_col_test, disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                                        df_pred_prob=disamb_soft_df.iloc[test_index], # for calculating Cross Entropy
                                                                                                        df_pred_label=disamb_hard_df.iloc[test_index],
                                                                                                        comp_cols=comp_feature_list)
            
            est_model_score_dicts[round_idx]['scores_per_col (train)'] = disamb_scores_per_col_train.copy()
            est_model_score_dicts[round_idx]['scores_average (train)'] = disamb_scores_average_train.copy()
            est_model_score_dicts[round_idx]['scores_per_col (test)'] = disamb_scores_per_col_test.copy()
            est_model_score_dicts[round_idx]['scores_average (test)'] = disamb_scores_average_test.copy()

            # show result
            del disamb_scores_average_train['confusion_matrix']
            del disamb_scores_average_test['confusion_matrix']
            logger.info(f"[round {round_idx}] train disamb_scores_average: {disamb_scores_average_train}")
            logger.info(f"[round {round_idx}] test disamb_scores_average: {disamb_scores_average_test}")


        # 2nd round ~ .........................................................................................................................
        else: 
        
            # prepare dataset
            nn_comp_train_df = comp_train_df.copy(deep=True)
            nn_comp_test_df = comp_test_df.copy(deep=True)
            nn_ord_train_df = ord_train_df.copy(deep=True)
            nn_ord_test_df = ord_test_df.copy(deep=True)
            if args.use_bar_feature:
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
                for comp_col in comp_feature_list:
                    est_target_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{comp_col}_', c)]
                    nn_comp_train_df[est_target_ohe_cols] = comp_train_df[est_target_ohe_cols].values
                    nn_comp_test_df[est_target_ohe_cols] = comp_test_df[est_target_ohe_cols].values
                    nn_ord_train_df[est_target_ohe_cols] = ord_train_df[est_target_ohe_cols].values
                    nn_ord_test_df[est_target_ohe_cols] = ord_test_df[est_target_ohe_cols].values

            feat_est_input_dim = len(nn_comp_train_df.columns) -1 - len(comp_onehot_names_list)
            num_classes_dict = {}
            num_classes_dict['target'] = len(data_df['target'].value_counts())
            for comp_col in comp_feature_list:
                num_classes_dict[comp_col] = len([c for c in comp_data_df.columns.tolist() if re.search(rf'^{comp_col}_', c)])


            # train feature estimation models
            feat_est_models = {}
            for comp_col in comp_feature_list:
                if args.arch == 'mlp':
                    assert args.hd is not None
                    feat_est_models[comp_col] = mlp_model(input_dim=feat_est_input_dim, hidden_dim=args.hd, output_dim=num_classes_dict[comp_col])
                    #est_model_dicts[round_idx][comp_col] = mlp_model(input_dim=feat_est_input_dim, hidden_dim=args.hd, output_dim=num_classes_dict[comp_col])
                elif args.arch == 'mlp4':
                    assert len(args.hds) == 4
                    feat_est_models[comp_col] = mlp_4layers(input_dim=feat_est_input_dim, hidden_dims=args.hds, output_dim=num_classes_dict[comp_col])
                    #est_model_dicts[round_idx][comp_col] = mlp_4layers(input_dim=feat_est_input_dim, hidden_dims=args.hds, output_dim=num_classes_dict[comp_col])
            est_model_dicts[round_idx], save_table_dict, train_score_dicts, test_score_dicts = \
                            iterative_learning_feat_est(feat_est_models=feat_est_models, 
                                                        lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=device, 
                                                        obj_lam=args.obj_lam,
                                                        est_method=args.method, cf_names=comp_feature_list,
                                                        comp_train_df=nn_comp_train_df, comp_test_df=nn_comp_test_df, 
                                                        ord_train_df=nn_ord_train_df, ord_test_df=nn_ord_test_df,  
                                                        label_pred_model=pred_model_dicts[round_idx-1]['soft'], 
                                                        pred_loss_func=args.pred_loss_func,
                                                        nn_beta=args.nn_beta,
                                                        seed=args.seed,
                                                        test_size_for_loop=args.test_size_for_loop,
                                                        verbose=args.verbose)
            
            # save model
            for comp_col in comp_feature_list:
                est_model_dicts[round_idx][comp_col] = copy.deepcopy(feat_est_models[comp_col])
            est_save_tables[round_idx] = save_table_dict.copy()
            est_train_score_dicts[round_idx] = train_score_dicts.copy()
            est_test_score_dicts[round_idx] = test_score_dicts.copy()


            # disambiguation
            for comp_col in comp_feature_list:
                feat_est_models[comp_col].eval()
                est_target_ohe_cols = [c for c in comp_train_df.columns.tolist() if re.search(rf'^{comp_col}_', c)]
                with torch.no_grad():
                    # train data
                    outputs = feat_est_models[comp_col](torch.tensor(nn_comp_train_df.drop(['target']+comp_onehot_names_list, axis=1).values, dtype=torch.float32).to(device))
                    pred_confs = soft_max(outputs).detach().cpu().numpy()
                    disamb_soft_df.loc[train_index, est_target_ohe_cols] = pred_confs.copy()
                    disamb_hard_df.loc[train_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]
                    # test data
                    outputs = feat_est_models[comp_col](torch.tensor(nn_comp_test_df.drop(['target']+comp_onehot_names_list, axis=1).values, dtype=torch.float32).to(device))
                    pred_confs = soft_max(outputs).detach().cpu().numpy()
                    disamb_soft_df.loc[test_index, est_target_ohe_cols] = pred_confs.copy()
                    disamb_hard_df.loc[test_index, est_target_ohe_cols] = np.identity(len(est_target_ohe_cols))[np.argmax(pred_confs, axis=1).tolist()]

            # evaluate feature estimation models
            ## inductive (create graph using only train data)
            disamb_scores_per_col_train, disamb_scores_average_train = evaluation_disamb_cls(df_true=data_df.iloc[train_index],
                                                                                                        df_pred_prob=disamb_soft_df.iloc[train_index],
                                                                                                        df_pred_label=disamb_hard_df.iloc[train_index],
                                                                                                        comp_cols=comp_feature_list)
            disamb_scores_per_col_test, disamb_scores_average_test = evaluation_disamb_cls(df_true=data_df.iloc[test_index],
                                                                                                        df_pred_prob=disamb_soft_df.iloc[test_index], # for calculating Cross Entropy
                                                                                                        df_pred_label=disamb_hard_df.iloc[test_index],
                                                                                                        comp_cols=comp_feature_list)
            
            est_model_score_dicts[round_idx]['scores_per_col (train)'] = disamb_scores_per_col_train.copy()
            est_model_score_dicts[round_idx]['scores_average (train)'] = disamb_scores_average_train.copy()
            est_model_score_dicts[round_idx]['scores_per_col (test)'] = disamb_scores_per_col_test.copy()
            est_model_score_dicts[round_idx]['scores_average (test)'] = disamb_scores_average_test.copy()

            # show result
            del disamb_scores_average_train['confusion_matrix']
            del disamb_scores_average_test['confusion_matrix']
            logger.info(f"[round {round_idx}] train disamb_scores_average: {disamb_scores_average_train}")
            logger.info(f"[round {round_idx}] test disamb_scores_average: {disamb_scores_average_test}")




        ## learn label prediction models :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        logger.info(f"[round {round_idx}] predicting downstream task...")
        # set seed
        set_seed_torch(args.seed)
        if args.task == 'classification':
            num_classes = len(np.unique(data_df['target'].values))
        elif args.task == 'regression':
            num_classes = 1
        else:
            raise NotImplementedError
        
        if args.arch == 'mlp':
            assert args.hd is not None
            pred_model_soft = mlp_model(input_dim=disamb_soft_df.drop(['target'], axis=1).shape[1], hidden_dim=args.hd, output_dim=num_classes)
            pred_model_hard = mlp_model(input_dim=disamb_hard_df.drop(['target'], axis=1).shape[1], hidden_dim=args.hd, output_dim=num_classes)
        elif args.arch == 'mlp4':
            assert len(args.hds) == 4
            pred_model_soft = mlp_4layers(input_dim=disamb_soft_df.drop(['target'], axis=1).shape[1], hidden_dims=args.hds, output_dim=num_classes)
            pred_model_hard = mlp_4layers(input_dim=disamb_hard_df.drop(['target'], axis=1).shape[1], hidden_dims=args.hds, output_dim=num_classes)
    

        pred_model_soft, pred_save_table_soft, pred_train_score_dict_soft, pred_test_score_dict_soft \
                    = train_mlp(model=pred_model_soft, lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=device,
                                    seed=args.seed, target_col='target', num_classes=num_classes, loss_func=args.pred_loss_func,
                                    train_df=disamb_soft_df.loc[train_index, :],
                                    test_df=disamb_soft_df.loc[test_index, :], 
                                    return_eval=True, verbose=args.verbose, test_size_for_loop=args.test_size_for_loop)
        
        pred_model_hard, pred_save_table_hard, pred_train_score_dict_hard, pred_test_score_dict_hard \
                        = train_mlp(model=pred_model_hard, lr=args.lr, bs=args.bs, ep=args.ep, wd=args.wd, device=device,
                                    seed=args.seed, target_col='target', num_classes=num_classes, loss_func=args.pred_loss_func,
                                    train_df=disamb_hard_df.loc[train_index, :],
                                    test_df=disamb_hard_df.loc[test_index, :], 
                                    return_eval=True, verbose=args.verbose, test_size_for_loop=args.test_size_for_loop)
        
        pred_model_scores_soft = evaluating(test_df=disamb_soft_df.loc[test_index, :], model=pred_model_soft, target_col='target', task=args.task, n_classes=n_classes, is_torch=True)
        pred_model_scores_hard = evaluating(test_df=disamb_hard_df.loc[test_index, :], model=pred_model_hard, target_col='target', task=args.task, n_classes=n_classes, is_torch=True)

        # save
        pred_model_dicts[round_idx]['soft'] = copy.deepcopy(pred_model_soft)
        pred_model_dicts[round_idx]['hard'] = copy.deepcopy(pred_model_hard)

        pred_save_tables[round_idx]['soft'] = pred_save_table_soft.copy()
        pred_save_tables[round_idx]['hard'] = pred_save_table_hard.copy()
        pred_train_score_dicts[round_idx]['soft'] = pred_train_score_dict_soft.copy()
        pred_train_score_dicts[round_idx]['hard'] = pred_train_score_dict_hard.copy()
        pred_test_score_dicts[round_idx]['soft'] = pred_test_score_dict_soft.copy()
        pred_test_score_dicts[round_idx]['hard'] = pred_test_score_dict_hard.copy()

        pred_model_score_dicts[round_idx]['soft'] = pred_model_scores_soft.copy()
        pred_model_score_dicts[round_idx]['hard'] = pred_model_scores_soft.copy()

        del pred_model_scores_soft['confusion_matrix']
        del pred_model_scores_hard['confusion_matrix']
        logger.info(f"[round {round_idx}] pred_model_scores [soft]: {pred_model_scores_soft}")
        logger.info(f"[round {round_idx}] pred_model_scores [hard]: {pred_model_scores_hard}")



    log_dict = {}
    log_dict['args'] = args

    log_dict['pred_model_score'] = pred_model_score_dicts
    log_dict['est_model_score'] = est_model_score_dicts

    log_dict['est_save_tables'] = est_save_tables
    log_dict['est_train_score_dicts'] = est_train_score_dicts
    log_dict['est_test_score_dicts'] = est_test_score_dicts

    log_dict['pred_save_tables'] = pred_save_tables
    log_dict['pred_train_score_dicts'] = pred_train_score_dicts
    log_dict['pred_test_score_dicts'] = pred_test_score_dicts

    log_dict['est_model_dicts'] = est_model_dicts
    log_dict['pred_model_dicts'] = pred_model_dicts

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
    
        

    if args.dataset_name in ['bank', 'adult']:
        args.task = 'classification'
    else:
        args.task = 'regression'

    args.exp_name += f"_{args.dataset_name}"

    args.log_name = get_log_filename(args)
    
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