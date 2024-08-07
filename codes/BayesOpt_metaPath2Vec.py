import sys, os
import numpy as np
from bayes_opt import BayesianOptimization
from metaPath2Vec import parse_args_metapath2vec, trainMetaPath2Vec
from util import *

args = parse_args_metapath2vec()

metaPath_params_list = {
    'learning_rate': (1, 50)  # learning rate from 1e-3 to 5e-2
}

def metaPath_best_loss(learning_rate, return_res=False):
    args.learning_rate = learning_rate * 1e-3
    res, best_loss = trainMetaPath2Vec(args)
    if return_res:
        return res
    else:
        return best_loss
    

def main():
    drug_list = get_drug_list()
    for drug in drug_list:
        if os.path.exists(f'../data/intermediate/{drug}_emb.pickle'):
            continue
        args.drug = drug
        res_list = []
        # train embeddings for each split
        for split_num in range(20):
            args.split_num = split_num
            print(f'Split {args.split_num} for {args.drug} using MetaPath2Vec')

            hyper_opt = BayesianOptimization(metaPath_best_loss, metaPath_params_list)
            hyper_opt.maximize(10, 5)
            best_hyper = hyper_opt.max['params']
            res_emb = metaPath_best_loss(**best_hyper, return_res=True)
            res_list.append(res_emb)
        with open(f'../data/intermediate/{drug}_emb.pickle', 'wb') as f:
            pickle.dump(res_list, f)

    # train embeddings for all data for each drug
    for drug in drug_list:
        if os.path.exists(f'../data/intermediate/{drug}_emb_all.pickle'):
            continue
        args.drug = drug
        args.split_num = -1
        print(f'All for {args.drug} using MetaPath2Vec')
        hyper_opt = BayesianOptimization(metaPath_best_loss, metaPath_params_list)
        hyper_opt.maximize(10, 5)
        best_hyper = hyper_opt.max['params']
        res_emb = metaPath_best_loss(**best_hyper, return_res=True)
        with open(f'../data/intermediate/{drug}_emb_all.pickle', 'wb') as f:
            pickle.dump(res_emb, f)


if __name__ == '__main__':
    main()