from argparse import ArgumentParser

from load_data import *

def arg_parser():
    parser = ArgumentParser()

    parser.add_argument("--config_file", help="config file name")

    parser.add_argument("--exp_name")
    parser.add_argument("--dataset_name", choices=['diabetes', 'bank', 'adult', 'census'])
    parser.add_argument("--main_dir", default="../")
    parser.add_argument("--data_dir", default="../../../opt/nas/data")
    parser.add_argument("--output_dir", default="../output")
    parser.add_argument("--method", choices=['ord', 'comp', 'ipal', 'rc', 'cc', 'forward', 'free', 'nn', 'ga', 'idgp', 'plsp'])
    parser.add_argument("--iter_idx", type=int, default=0)
    parser.add_argument("--use_bar_feature", type=bool, default=False)

    parser.add_argument("--sample_size", type=int, help="all data size. if sample_size = -1, we use all data")
    parser.add_argument("--test_rate", type=float) 
    parser.add_argument("--comp_cols", type=str, nargs="+", default=['all'], help="list of features to be CFs") 
    parser.add_argument("--avoid_estimate_cols", type=str, nargs="+", default=[], help="list of features that don't be estimated by estimation methods")
    parser.add_argument("--iter_aec", type=bool, default=False)
    parser.add_argument("--pred_aec", type=bool, default=True)

    parser.add_argument("--n_parallel", type=int, help="the number of using cores")
    parser.add_argument("--measure_time", type=bool, help="measure each execution times")
    parser.add_argument("--use_jax", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=bool, default=False)

    # for iterative learning or joint learning
    parser.add_argument("--is_iterative", type=bool, default=False)
    parser.add_argument("--is_joint", type=bool, default=False) 
    parser.add_argument("--obj_lam", type=float)
    parser.add_argument("--start_round", type=int, default=1)
    parser.add_argument("--iterative_round", type=int)
    parser.add_argument("--pred_loss_func", type=str, choices=['log', 'logistic'])


    # for nn_based
    parser.add_argument("--arch", type=str, default='mlp')
    parser.add_argument("--lr", type=float)
    parser.add_argument("--bs", type=int)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--hd", type=int)
    parser.add_argument("--hds", type=int, nargs='+', default=[])
    parser.add_argument("--test_size_for_loop", type=int, default=-1)

    # for nn (non-negative)
    parser.add_argument("--nn_beta", type=float, default=None)

    # for idgp
    parser.add_argument("--idgp_lr_g", type=float, default=None)
    parser.add_argument("--idgp_wd_g", type=float, default=None)
    parser.add_argument("--idgp_warmup_ep", type=int, default=None)
    parser.add_argument("--idgp_T_1", type=float, default=None)
    parser.add_argument("--idgp_T_2", type=float, default=None)
    parser.add_argument("--idgp_alpha", type=float, default=None)
    parser.add_argument("--idgp_beta", type=float, default=None)
    parser.add_argument("--idgp_delta", type=float, default=None)
    parser.add_argument("--idgp_theta", type=float, default=None)
    parser.add_argument("--idgp_gamma", type=float, default=None)
    parser.add_argument("--idgp_eta", type=float, default=None)


    return parser

def get_args_default():
    '''
    get a dict has default args' values
    '''
    default_dict = {
        'main_dir': '../',
        'data_dir': '../../../opt/nas/data',
        'output_dir': '../output',

        'sample_size': -1,
        'test_rate': 0.5,
        'comp_cols': ['all'],
        'avoid_estimate_cols':[],


        'n_parallel': 4,
        'measure_time': True,
        'use_jax': True,
        'seed': 42,

        # for IPAL
        'ipal_alpha' : 1.0,
        'ipal_k': 20,
        'iapl_T': 100,
        
    }

    return default_dict

def get_log_filename(args):
    '''
    get log file name
    '''
    name = ""

    name += args.dataset_name

    if 'is_iterative' in dir(args):
        if args.is_iterative:
            name += '_iterative'
            name += '_OL'+ str(args.obj_lam)
            name += '_PL'+ str(args.pred_loss_func)
    if 'is_joint' in dir(args):
        if args.is_joint:
            name += '_joint'
            name += '_OL'+ str(args.obj_lam)
            name += '_PL'+ str(args.pred_loss_func)

    name += '_' + args.method

    if args.method not in ['comp', 'ord']:
        if args.use_bar_feature:
            name += '_ubf'

        if len(args.avoid_estimate_cols) > 0:
            if args.iter_aec:
                name += '_iAEC' + str(comp_cols_code(args.dataset_name, args.avoid_estimate_cols))

    # hyperparameters of methods
    if args.method == 'ipal':
        name += '_k' + str(args.ipal_k)
        name += '_alpha' + str(args.ipal_alpha)
        name += '_T' + str(args.ipal_T)

    elif args.method in ['rc', 'cc', 'pc', 'forward', 'free', 'nn', 'ga']:
        name += '_arch' + str(args.arch)
        name += '_lr' + str(args.lr)
        name += '_bs' + str(args.bs)
        name += '_ep' + str(args.ep)
        name += '_wd' + str(args.wd)
        if args.arch == 'mlp':
            name += '_hd' + str(args.hd)


        if args.method == 'nn':
            name += '_nnbeta' + str(args.nn_beta)

    
    elif args.method == 'idgp':
        name += '_lrg' + str(args.idgp_lr_g)
        name += '_wdg' + str(args.idgp_wd_g)
        name += '_wu' + str(args.idgp_warmup_ep)
        name += '_T1' +str(args.idgp_T_1)
        name += '_T2' +str(args.idgp_T_2)
        name += '_a' + str(args.idgp_alpha)
        name += '_b' + str(args.idgp_beta)
        name += '_d' + str(args.idgp_delta)
        name += '_t' + str(args.idgp_theta)
        name += '_g' + str(args.idgp_gamma)
        name += '_e' + str(args.idgp_eta)

    elif args.method == 'plsp':
        name += '_stt' + str(args.plsp_st_threshold)
        name += '_prep' + str(args.plsp_pretrain_epoch)
        name += '_nl' + str(args.plsp_num_labeled_instances)
        name += '_ti' + str(args.plsp_train_iterations)
        name += '_twu' + str(args.plsp_threshold_warmup)
        name += '_lam' + str(args.plsp_lambda_0)
        name += '_gam' + str(args.plsp_gamma_0)


    # dataset settings
    name += '_size' + str(args.sample_size)
    name += '_test'+ str(args.test_rate)

    if args.method != 'ord':
        if args.comp_cols == ['all']:
            name += '_CColAll'
        else:
            name += '_CCol' + str(comp_cols_code(args.dataset_name, args.comp_cols))
    


    # computing settings
    name += '_seed' + str(args.seed)

    return name


def pseudo_args(args_dict):
    class Args():
        tmp = "ttt"
    args = Args()
    for k, v in args_dict.items():
        if v is not None:
            setattr(args, k, v)
    return args
