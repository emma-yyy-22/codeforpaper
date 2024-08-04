import ast
import os
import gc
import random
import warnings
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset

import pickle
from utils import *

warnings.filterwarnings('ignore')

DATASET = '' # Global dataset name for training (Walker or CRyPTIC).
DRUG_NAME = ''  # Global drug name for training.
SPLIT_NUM = 0  # Global split number for training.
EPOCHS = 100  # Global number of epochs for training.

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# neural network binary classifier
class NNbinaryClassifier(nn.Module):
    def __init__(self, input_size, n_hidden_layers, hidden_size, dropout_prob):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.extend([nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_prob)])
        for _ in range(n_hidden_layers - 1):
            self.layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout_prob)])
        self.layers.extend([nn.Linear(hidden_size, 1), nn.Sigmoid()])

    def forward(self, inputs, labels):
        logits = inputs
        for layer in self.layers:
            logits = layer(logits)
        loss = BCELoss()(logits.view(-1), labels)
        return loss, logits
    


def train_binary_classifier(drug, split_num, params):
    # load the dataset
    if DATASET=='Walker':
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}.pkl', 'rb') as f:
            X = pickle.load(f)[split_num]
        y = getWalkerLabels(drug)[split_num]
    elif DATASET=='CRyPTIC':
        with open(f'../Data/idx_splits/CRyPTIC_single_binary/{drug}_pca_1024d.pkl', 'rb') as f:
            X = pickle.load(f)[split_num]
        y = getCrypticBinaryLabels(drug)[split_num]
    
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    # convert to torch dataset
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=params['batch_size'])
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=params['batch_size'])
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=params['batch_size'])

    input_size = X_train.shape[1]
    model = NNbinaryClassifier(input_size, params['n_hidden_layers'], params['hidden_size'], dropout_prob=0.1)

    if torch.cuda.is_available():
        model.cuda()
    
    # Set the optimizer
    optimizer = Adam(model.parameters(), lr=params['learning_rate'])
    # Set a linear learning rate scheduler
    # Careful: this scheduler has bugs and does not work properly
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch / EPOCHS)

    # Early stopping when the validation AUROC does not improve for 5 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            # model.zero_grad()
            loss, logits = model(inputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
        
        # Evaluate the model on the validation set
        model.eval()
        val_logits = []
        val_labels = []
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            with torch.no_grad():
                _, logits = model(inputs, labels)
            val_logits.append(logits.detach().cpu().numpy())
            val_labels.append(labels.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_auroc = roc_auc_score(val_labels, val_logits)
        # print(f'Epoch {epoch + 1} validation AUROC: {val_auroc:.4f}')

        # Save the best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        
        # Early stopping
        if epoch - best_val_epoch >= 5:
            break
    
    # Load the best model and evaluate on the test set
    model.load_state_dict(best_model)
    model.eval()

    train_logits = []
    train_labels = []
    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        with torch.no_grad():
            _, logits = model(inputs, labels)
        train_logits.append(logits.detach().cpu().numpy())
        train_labels.append(labels.detach().cpu().numpy())
    train_logits = np.concatenate(train_logits, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    val_logits = []
    val_labels = []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        with torch.no_grad():
            _, logits = model(inputs, labels)
        val_logits.append(logits.detach().cpu().numpy())
        val_labels.append(labels.detach().cpu().numpy())
    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    test_logits = []
    test_labels = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        with torch.no_grad():
            _, logits = model(inputs, labels)
        test_logits.append(logits.detach().cpu().numpy())
        test_labels.append(labels.detach().cpu().numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    res = (train_logits, val_logits, test_logits, train_labels, val_labels, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res, best_val_auroc


nn_params_list = {
    'batch_size': (-0.5, 2.49),
    'n_hidden_layers': (-0.5, 3.49),
    'hidden_size': (-0.5, 3.49),
    'learning_rate': (1, 10)
}

def nn_val_auroc(batch_size, n_hidden_layers, hidden_size, learning_rate, return_res=False):
    params = {}
    params['batch_size'] = [8, 16, 32][round(batch_size)]
    params['n_hidden_layers'] = [1, 2, 4, 8][round(n_hidden_layers)]
    params['hidden_size'] = [64, 128, 256, 512][round(hidden_size)]
    params['learning_rate'] = learning_rate * 1e-4
    res, best_val_auroc = train_binary_classifier(DRUG_NAME, SPLIT_NUM, params)
    if return_res:
        return res
    else:
        return best_val_auroc


    
if __name__ == "__main__":
    DATASET = str(sys.argv[1])  # 'Walker' or 'CRyPTIC'
    DRUG_NAME = str(sys.argv[2])
    model_name = 'nn'

    for split_num in range(20): # 20 splits in total
        if os.path.exists(f'../Results/single_binary_classification/{DATASET}/{DRUG_NAME}_{model_name}_{split_num}.pkl'):
            continue

        SPLIT_NUM = split_num
        print(f'Train and test for {DRUG_NAME} in split {SPLIT_NUM} using {model_name}...')

        hyper_opt = BayesianOptimization(globals()[f'{model_name}_val_auroc'], globals()[f'{model_name}_params_list'], random_state=42)
        hyper_opt.maximize(10, 5)
        best_hyper = hyper_opt.max['params']
        best_res = globals()[f'{model_name}_val_auroc'](**best_hyper, return_res=True)
        with open(f'../Results/single_binary_classification/{DATASET}/{DRUG_NAME}_{model_name}_{split_num}.pkl', 'wb') as f:
            pickle.dump((best_res, best_hyper), f)

    # Combine results from 20 splits
    res_list, best_param_list = [], []
    for split_num in range(20):
        with open(f'../Results/single_binary_classification/{DATASET}/{DRUG_NAME}_{model_name}_{split_num}.pkl', 'rb') as f:
            best_res, best_hyper = pickle.load(f)
        res_list.append(best_res)
        best_param_list.append(best_hyper)
    with open(f'../Results/single_binary_classification/{DATASET}/{DRUG_NAME}_{model_name}.pkl', 'wb') as f:
        pickle.dump((res_list, best_param_list), f)
    
    deleteInterimResult(DATASET, DRUG_NAME, model_name, 'binary')

