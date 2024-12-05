#!/usr/bin/env python3

"""
Script for evaluating the operational capacity of factorizers
"""
import argparse
import numpy as np
import os,time
from tqdm import tqdm
import json
import sys
import torch as t
import collections
from dotmap import DotMap
import cProfile, pstats
import pickle
import time

from models.densebipolarbatched import densebipolarbatched
from models.blockcodefactorizer import blockcodefactorizer



# recursivley expand all the iterable parameters of the config 
def configExpansion(config):
    experiments = []
    experiments.append(config)

    for k,v in config.items():
        # item is dict: do recursive expansion
        if isinstance(v, dict):
            v = configExpansion(v)

        # item is iteratable
        if isinstance(v, list):
            experiments_iter = []
            for i in range(len(v)):
                if i == 0:
                    for e in experiments:
                        e[k] = v[i]
                else:
                    for e in experiments:
                        e_iter = e.copy()
                        e_iter[k] = v[i]
                        experiments_iter.append(e_iter)
            experiments.extend(experiments_iter)
    
    return experiments

# recursively merge two dicts
def dict_merge(dct, merge_dct):
    for k, v in merge_dct.iteritems():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

parser = argparse.ArgumentParser(description='Resonator Capacity Test')
parser.add_argument('--r', type=str, nargs="?", metavar='EXPERIMENT ID')
parser.add_argument('--id', type=str, nargs="?", metavar='EXPERIMENT ID')
parser.add_argument('--custom-config', type=str, nargs="?", metavar='PATH')
argsCLI = parser.parse_args()

# check if input is valid
if argsCLI.id == None and argsCLI.custom_config == None:
    parser.print_help(sys.stderr)
    sys.exit(1)

# find experiments folder
root = os.path.dirname(os.path.realpath(__file__))
experiments = os.path.join(root, "experiments")

# find path for config.json file        
if argsCLI.id != None:
    for path in os.listdir(experiments):
        if(not path.startswith(argsCLI.id+"_")):
            continue
        configPath = os.path.join(experiments, path, "config.json")
else:
    configPath = argsCLI.custom_config

# open config file
with open(configPath, 'r') as _fp:
    data = json.load(_fp)
    config = DotMap(data)

# ack reading of config
data["running"] = True
with open(configPath, 'w') as _fp:
    _fp.write(json.dumps(data, indent=4))

if (not 'savedir' in config) or config.savedir == None or config.savedir == "":
    # use default file location
    config.savedir = os.path.dirname(configPath)
os.makedirs(config.savedir, exist_ok=True)
print("Save to directory: {}".format(config.savedir))

experiments = configExpansion(config)


###############################################################################
### Iterate through all experiments
###############################################################################
for args in experiments:
    # Define the problem size
    if args.M.fixed is not None: 
        M_vec = [args.M.fixed]
    else: 
        npoints = (args.M.log_stop - args.M.log_start) * args.M.nDecade + 1
        M_vec = np.round(np.logspace(args.M.log_start, args.M.log_stop, npoints) ** (1/args.num_factors)).astype(int).tolist()


    frame_errors_vec = np.zeros(len(M_vec))
    idx_errors_vec = np.zeros((len(M_vec),args.ntrial//args.batchsize))
    time_vec = np.zeros(len(M_vec))
    niter_vec = np.zeros(len(M_vec))
    idx_errors_vec[:]=np.nan
    time_vec[:] = np.nan
    niter_vec[:] = np.nan
    niter_stat_vec = np.zeros((len(M_vec), args.ntrial))
    niter_stat_vec[:] = np.nan

    for idx,M in enumerate(M_vec):
        max_iter = int((M**(args.num_factors - 1)) / args.num_factors * args.iter.fac) if (args.iter.max==-1 or args.iter.max==None) else args.iter.max

        print("Search space: {}, max iterations: {}".format(M**args.num_factors, max_iter))
        
        if argsCLI.r:
            args.id = argsCLI.r
            with open(os.path.join(config.savedir, 'resonator_{}.pickle'.format(args.id)), 'rb') as handle:
                factor_code = pickle.load(handle)
        else:
            if args.arch=="densebipolarbatched":
                seq_dec = True
                if args.decoding == "parallel":
                    seq_dec = False
                factor_code = densebipolarbatched(args.d, args.num_factors, M, decodingSequential=seq_dec, **args) 
            elif args.arch=="blockcodefactorizer": 
                factor_code = blockcodefactorizer(F=args.num_factors, Mx=M, **args)
            else:
                raise ValueError(f"Invalid arch, got {args.arch}")


        idx_errors_tot = 0  
        frame_errors = 0    
        niter_tot = 0
        niter_min = max_iter
        niter_max = 0
        t1 = time.time()

        timestr = time.strftime("%Y%m%d-%H%M%S")
        for trial in tqdm(range(args.ntrial//args.batchsize)):
            if args.prod_vec:
                vecs = np.load(os.path.join(config.savedir, args.prod_vec))
                x = vecs["prod_vec"]
                x[x<0] = -1
                x[x>=0] = 1
                
                u = vecs["u"]

                x = t.tensor(x).type(t.long)
                u = t.tensor(u).type(t.long)

                x = x[trial * args.batchsize : (trial+1) * args.batchsize]
                u = u[trial * args.batchsize : (trial+1) * args.batchsize]
            else:
                if argsCLI.r:
                    # to reload particular instance of problem
                    Uvec = t.tensor(np.load(os.path.join(config.savedir, "u.npz"))["Uvec"])
                    u = Uvec[trial:trial+args.batchsize]
                    u = t.squeeze(u)
                else:
                    # produce random signal 
                    u = t.randint(0,M,(args.batchsize,args.num_factors)).to(factor_code._device)
                    if args.arch=="densebipolar": 
                        u = t.squeeze(u)

                x = factor_code.encode(u)
            
            u_hat  = factor_code.decode(x, max_iter)

            # Error computation
            idx_errors = (t.sum((u != u_hat))).item()
            idx_errors_tot = idx_errors_tot+idx_errors
            frame_errors =frame_errors+ t.sum(t.sum(u!=u_hat,-1)!=0).item()

            # count number of iterations
            conv_idx = factor_code._get_number_iter()
            niter_tot = niter_tot + conv_idx.sum().item()        
            niter_stat_vec[idx,trial*args.batchsize:(trial+1)*args.batchsize] = factor_code._get_number_iter().cpu().squeeze().numpy()        

            if conv_idx.min().item() < niter_min:
                niter_min = conv_idx.min().item()
            
            if conv_idx.max().item() > niter_max:
                niter_max = conv_idx.max().item()

            # collect intermediate statics
            idx_errors_vec[idx,trial] = idx_errors_tot/((trial+1)*args.num_factors*args.batchsize)
            frame_errors_vec[idx] = frame_errors/((trial+1)*args.batchsize)
            niter_vec[idx]=niter_tot/((trial+1)*args.batchsize)

            save_name = os.path.join(args.savedir, factor_code.filename())
            save_name = save_name + timestr
            np.savez(save_name, M_vec = np.array(M_vec), frame_errors_vec=frame_errors_vec,
                idx_errors_vec = idx_errors_vec, time_vec=time_vec, niter_vec=niter_vec,
                niter_stat_vec=niter_stat_vec,args=args)


        t2 = time.time()
        trial = trial+1
        t_pertrial=(t2-t1)/(trial*args.batchsize)
        time_vec[idx] = t_pertrial

        print("Search Space: {:} \t \tAcc: {:.5f} \tNiter={:} \tNmax={:} \tNmin={:}".format(M**args.num_factors,1-idx_errors_tot/(trial*args.num_factors*args.batchsize),niter_vec[idx], niter_max, niter_min))
        print("Time per trial: {:.4f}s".format(t_pertrial))

"""
#code snippet to iterate file_list and print mean errors
for file in file_list:
    #np.load() file
    a = np.load(file, allow_pickle=True)
    error_vec = a['idx_errors_vec']
    arrgs = a['args']
    a = arrgs.item()
    if (np.isnan(error_vec).sum() ==0):
        print(a.G0_noise_std_rel,error_vec.mean())


#list all files in dir
import glob
file_list = glob.glob("experiments/100e_prnoise/noise_dense*.npz")
"""
