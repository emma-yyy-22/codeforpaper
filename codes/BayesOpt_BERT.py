import sys, os
import numpy as np
from bayes_opt import BayesianOptimization
from BERT import parse_args_bert, train_bert
from util import *

args = parse_args_bert()

# hyper parameters in params:
# 'batch_size': 8, 16, 32;
# 'num_hidden_layers': 1, 2, 3;
# 'num_attention_heads': 1, 2, 4;
# 'learning_rate': 2e-5 to 1e-4
params_list = {
    'batch_size': (-0.5, 2.49),
    'num_hidden_layers': (-0.5, 2.49),
    'num_attention_heads': (-0.5, 2.49),
    'learning_rate': (2, 10)
}

def best_val_auroc(batch_size, num_hidden_layers, num_attention_heads, learning_rate, return_res=False):
    args.batch_size = [8, 16, 32][round(batch_size)]
    args.num_encoder_layers = [1, 2, 3][round(num_hidden_layers)]
    args.num_attention_heads = [1, 2, 4][round(num_attention_heads)]
    args.learning_rate = learning_rate * 1e-5
    res, best_val_auroc = train_bert(args)
    if return_res:
        return res
    else:
        return best_val_auroc


def main():
    if not args.no_pretrained and args.freeze:
        model_name = 'bert1'
    elif args.no_pretrained and not args.freeze:
        model_name = 'bert1_no_pretrained'
    elif not args.no_pretrained and not args.freeze:
        model_name = 'bert1_no_freeze'
    else:
        raise ValueError('Invalid model name!')

    if os.path.exists(f'../results/{args.bert}_{model_name}.pkl'):
        sys.exit(0)

    for split_num in range(20): # 20 splits in total
        if os.path.exists(f'../results/{args.drug}_{model_name}_{split_num}.pkl'):
            continue

        args.split_num = split_num
        print(f'Train and test for {args.drug} in split {args.split_num} using {model_name}...')

        hyper_opt = BayesianOptimization(best_val_auroc, params_list, random_state=42)
        hyper_opt.maximize(10, 5)
        best_hyper = hyper_opt.max['params']
        best_res = best_val_auroc(**best_hyper, return_res=True)
        with open(f'../results/{args.drug}_{model_name}_{split_num}.pkl', 'wb') as f:
            pickle.dump((best_res, best_hyper), f)

    # Combine results from 20 splits
    res_list, best_param_list = [], []
    for split_num in range(20):
        with open(f'../results/{args.drug}_{model_name}_{split_num}.pkl', 'rb') as f:
            best_res, best_hyper = pickle.load(f)
        res_list.append(best_res)
        best_param_list.append(best_hyper)
    with open(f'../results/{args.drug}_{model_name}.pkl', 'wb') as f:
        pickle.dump((res_list, best_param_list), f)
    
    for i in range(20):
        if os.path.exists(f'../results/{args.drug}_{model_name}_{i}.pkl'):
            os.remove(f'../results/{args.drug}_{model_name}_{i}.pkl')


if __name__ == '__main__':
    main()