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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset

import pickle
import argparse
import wandb
from util import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Train and test comparison models')
parser.add_argument('--model', type=str, help='Model name (NN, WDNN, DeepAMR, CNNGWP or MutEmbAblation)')
parser.add_argument('--drug', type=str, help='Drug name')
parser.add_argument('--split_num', type=int, help='Split number')
parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--early_stopping', type=int, default=5, help='Number of epochs for early stopping')
parser.add_argument('--dropout_prob', type=float, default=0.1, help='Dropout probability')
args = parser.parse_args()


# define a function to train a model for one epoch
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for step, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        optimizer.zero_grad()
        loss, _ = model(inputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# define a function to evaluate a model
def evaluate_model(model, dataloader, device):
    model.eval()
    res_logits, res_labels = [], []
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        with torch.no_grad():
            _, logits = model(inputs, labels)
        res_logits.append(logits.detach().cpu().numpy())
        res_labels.append(labels.detach().cpu().numpy())
    res_logits = np.concatenate(res_logits, axis=0)
    res_labels = np.concatenate(res_labels, axis=0)
    return res_logits, res_labels


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
    

def train_binary_nn(args):
    wandb.init(project='Airframe', config=args)
    wandb.run.name = f'{args.drug}_{args.model}_{args.split_num}'
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    train_dataloader, val_dataloader, test_dataloader = load_data_loader(args.drug, args.split_num, args.batch_size)

    input_size = train_dataloader.dataset.tensors[0].shape[1]
    model = NNbinaryClassifier(input_size, args.n_hidden_layers, args.hidden_size, dropout_prob=args.dropout_prob)
    model.to(device)
    
    # Set the optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Early stopping
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    # Train the model
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})
        
        # Evaluate the model on the validation set
        val_logits, val_labels = evaluate_model(model, val_dataloader, device)
        val_auroc = roc_auc_score(val_labels, val_logits)
        wandb.log({'epoch': epoch, 'val_auroc': val_auroc})

        # Save the best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        
        # Early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break
    
    # Load the best model and evaluate on the test set
    model.load_state_dict(best_model)
    model.eval()

    # train_logits, train_labels = evaluate_model(model, train_dataloader, device)
    # val_logits, val_labels = evaluate_model(model, val_dataloader, device)
    test_logits, test_labels = evaluate_model(model, test_dataloader, device)
    wandb.log({'test_auroc': roc_auc_score(test_labels, test_logits)})

    res = {'test_preds': test_logits, 'test_labels': test_labels, 'best_model': best_model}
    wandb.finish()
    # delete the data and model to save memory
    del model, train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    return res, best_val_auroc


def bayesOpt_nn(args):
    best_res = None
    best_hyperparams = None
    best_val_auroc = 0

    space = {
        'batch_size': (-0.5, 2.49),
        'n_hidden_layers': (-0.5, 3.49),
        'hidden_size': (-0.5, 3.49),
        'learning_rate': (1, 10) # 1e-4 to 1e-3
    }

    list_batch_size = [8, 16, 32]
    list_n_hidden_layers = [1, 2, 4, 8]
    list_hidden_size = [64, 128, 256, 512]  

    def objective(batch_size, n_hidden_layers, hidden_size, learning_rate):
        nonlocal best_res, best_hyperparams, best_val_auroc
        args.batch_size = list_batch_size[round(batch_size)]
        args.n_hidden_layers = list_n_hidden_layers[round(n_hidden_layers)]
        args.hidden_size = list_hidden_size[round(hidden_size)]
        args.learning_rate = learning_rate * 1e-4
        res, val_auroc = train_binary_nn(args)
        if val_auroc > best_val_auroc:
            best_res = res
            best_hyperparams = {
                'batch_size': args.batch_size,
                'n_hidden_layers': args.n_hidden_layers,
                'hidden_size': args.hidden_size,
                'learning_rate': args.learning_rate
            }
            best_val_auroc = val_auroc
        return val_auroc

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    res = best_res
    res['best_hyperparams'] = best_hyperparams
    return res


# WDNN
class WDNN(nn.Module):
    def __init__(self, input_dim, dropout_prob):
        super(WDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(input_dim + 256, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.l2_reg = 1e-8

    def forward(self, x, labels):
        input_data = x
        x = F.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        x = torch.cat([input_data, x], dim=1)
        x = torch.sigmoid(self.output(x))

        loss = nn.BCELoss()(x.view(-1), labels)
        
        return loss, x
    

def train_WDNN(args):
    wandb.init(project='Airframe', config=args)
    wandb.run.name = f'{args.drug}_{args.model}_{args.split_num}'

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    _, val_dataloader, test_dataloader = load_data_loader(args.drug, args.split_num, args.batch_size)

    # Because there is a batch_norm layer in the model, we need to drop the last batch if the number of samples is not divisible by the batch size
    X_train, _, _ = get_features(drug, split_num)
    y_train, _, _ = get_labels(drug, split_num)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size, drop_last=True)
    

    input_size = train_dataloader.dataset.tensors[0].shape[1]
    model = WDNN(input_size, args.dropout_prob)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Early stopping
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})

        # Evaluate the model on the validation set
        val_logits, val_labels = evaluate_model(model, val_dataloader, device)
        val_auroc = roc_auc_score(val_labels, val_logits)
        wandb.log({'epoch': epoch, 'val_auroc': val_auroc})

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()

        # Early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
    test_logits, test_labels = evaluate_model(model, test_dataloader, device)
    wandb.log({'test_auroc': roc_auc_score(test_labels, test_logits)})

    res = {'test_preds': test_logits, 'test_labels': test_labels, 'best_model': best_model}
    wandb.finish()
    # delete the data and model to save memory
    del model, train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    return res, best_val_auroc


def bayesOpt_WDNN(args):
    best_res = None
    best_hyperparams = None
    best_val_auroc = 0

    space = {
        'batch_size': (-0.5, 2.49), # 8, 16, 32
        'learning_rate': (1, 10)  # 1e-4 to 1e-3
    }

    def objective(batch_size, learning_rate):
        nonlocal best_res, best_hyperparams, best_val_auroc
        args.batch_size = [8, 16, 32][round(batch_size)]
        args.learning_rate = learning_rate * 1e-4
        res, val_auroc = train_WDNN(args)
        if val_auroc > best_val_auroc:
            best_res = res
            best_hyperparams = {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
            best_val_auroc = val_auroc
        return val_auroc

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    res = best_res
    res['best_hyperparams'] = best_hyperparams
    return res


# DeepAMR
class DeepAMR(nn.Module):
    def __init__(self, input_dim, dropout_prob):
        super(DeepAMR, self).__init__() 

        # Encoder
        self.encoder = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(input_dim, 500),
            nn.ReLU(True),
            nn.Linear(500, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 20),
            nn.ReLU(True)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(20, 4),
            nn.ReLU(True),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(20, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 500),
            nn.ReLU(True),
            nn.Linear(500, input_dim)
        )
        
    def forward(self, x, labels):
        encoded = self.encoder(x)
        
        # Task outputs
        task_out = self.classifier(encoded)
        decoded = self.decoder(encoded)
        loss_reconstruction = nn.MSELoss()(decoded, x)
        loss_classification = nn.BCELoss()(task_out.view(-1), labels)
        loss = loss_reconstruction + loss_classification
        return loss, task_out


def train_DeepAMR(args):
    wandb.init(project='Airframe', config=args)
    wandb.run.name = f'{args.drug}_{args.model}_{args.split_num}'

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    train_dataloader, val_dataloader, test_dataloader = load_data_loader(args.drug, args.split_num, args.batch_size)

    input_size = train_dataloader.dataset.tensors[0].shape[1]
    model = DeepAMR(input_size, args.dropout_prob)
    model.to(device)

    # The original code does not mention the learning rate, so I will use 0.001 as the default learning rate
    optimizer = Adam(model.parameters(), lr=args.learning_rate) 

    # Early stopping when the validation AUROC does not improve for 5 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    # Train the model
    for epoch in range(100):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})
        
        # Evaluate the model on the validation set
        val_logits, val_labels = evaluate_model(model, val_dataloader, device)
        val_auroc = roc_auc_score(val_labels, val_logits)
        wandb.log({'epoch': epoch, 'val_auroc': val_auroc})

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()

        # early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break
    
    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()

    test_logits, test_labels = evaluate_model(model, test_dataloader, device)
    wandb.log({'test_auroc': roc_auc_score(test_labels, test_logits)})

    res = {'test_preds': test_logits, 'test_labels': test_labels, 'best_model': best_model}
    wandb.finish()
    # delete the data and model to save memory
    del model, train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    return res, best_val_auroc


def bayesOpt_DeepAMR(args):
    best_res = None
    best_hyperparams = None
    best_val_auroc = 0

    space = {
        'batch_size': (-0.5, 2.49), # 8, 16, 32
        'learning_rate': (1, 10)  # 1e-4 to 1e-3
    }

    def objective(batch_size, learning_rate):
        nonlocal best_res, best_hyperparams, best_val_auroc
        args.batch_size = [8, 16, 32][round(batch_size)]
        args.learning_rate = learning_rate * 1e-4
        res, val_auroc = train_DeepAMR(args)
        if val_auroc > best_val_auroc:
            best_res = res
            best_hyperparams = {
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
            best_val_auroc = val_auroc
        return val_auroc

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    res = best_res
    res['best_hyperparams'] = best_hyperparams
    return res


# Change some settings of CNN-GWP to make it fair to compare with other models
class CNNGWP(nn.Module):
    def __init__(self, input_dim, filters, kernel_size):
        super(CNNGWP, self).__init__()
        self.conv1d = nn.Conv1d(1, filters, kernel_size, stride=2)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()

        # Calculate the output dimension after conv1d and pooling
        conv_output_dim = (input_dim - kernel_size) // 2 + 1
        pool_output_dim = conv_output_dim // 2
        self.dense = nn.Linear(filters * pool_output_dim, 1)
        # self.dense = nn.Linear(filters * ((input_dim - kernel_size + 1) // 2 // 2), 1)

    def forward(self, x, labels):
        # Ensure input tensor has the correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a channel dimension

        x = torch.relu(self.conv1d(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = torch.sigmoid(x)
        loss = nn.BCELoss()(x.view(-1), labels)
        return loss, x

def train_CNNGWP(args):
    wandb.init(project='Airframe', config=args)
    wandb.run.name = f'{args.drug}_{args.model}_{args.split_num}'

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    train_dataloader, val_dataloader, test_dataloader = load_data_loader(args.drug, args.split_num, args.batch_size)

    input_size = train_dataloader.dataset.tensors[0].shape[1]
    model = CNNGWP(input_size, args.filters, args.kernel_size)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Early stopping when the validation AUROC does not improve for 10 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})
    
        # Evaluate the model on validation set for hyperparameter tuning
        val_logits, val_labels = evaluate_model(model, val_dataloader, device)
        val_auroc = roc_auc_score(val_labels, val_logits)
        wandb.log({'epoch': epoch, 'val_auroc': val_auroc})

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        # early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
    test_logits, test_labels = evaluate_model(model, test_dataloader, device)
    wandb.log({'test_auroc': roc_auc_score(test_labels, test_logits)})

    res = {'test_preds': test_logits, 'test_labels': test_labels, 'best_model': best_model}
    wandb.finish()
    # delete the data and model to save memory
    del model, train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    return res, best_val_auroc


# Use Bayesian Optimization for hyperparameter tuning
def bayes_optimise_CNNGWP(args):
    best_res = None
    best_hyperparams = None
    best_val_auroc = 0

    space = {
        'filters': (20, 100),
        'kernel_size': (10, 50),
        'batch_size': (-0.5, 2.49),  # 8, 16, 32
        'learning_rate': (1, 10)  # 1e-4 to 1e-3
    }

    def objective(filters, kernel_size, batch_size, learning_rate):
        nonlocal best_res, best_hyperparams, best_val_auroc
        args.filters = int(filters)
        args.kernel_size = int(kernel_size)
        args.batch_size = [8, 16, 32][round(batch_size)]
        args.learning_rate = learning_rate * 1e-4
        res, val_auroc = train_CNNGWP(args)
        if best_res is None or val_auroc > best_val_auroc:
            best_res = res
            best_val_auroc = val_auroc
            best_hyperparams = {
                'filters': args.filters,
                'kernel_size': args.kernel_size,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate
            }
        return val_auroc


    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    res = best_res
    res['best_hyperparams'] = best_hyperparams
    return res


# Dataset for the mutation embeddings ablation study
class MutEmbDataset(Dataset):
    def __init__(self, drug, split_num, mode='train', emb_type='max'):

        # load the mutation dictionary, with the index starting from 0
        mut_dict = get_mut_dict(drug, split_num)
        # load the embeddings
        with open(f'../data/intermediate/{drug}_emb.pickle', 'rb') as f:
            embeddings = pickle.load(f)[split_num] # embeddings shape: (num_mutations, emb_dim(128))
        assert len(mut_dict) == embeddings.shape[0]

        self.inputs = []
        # Load the dataset for inputs
        train_idx, val_idx, test_idx = get_data_splits(drug)[split_num]
        geno_pheno = get_geno_pheno()
        assert mode in ['train', 'val', 'test']
        indices = train_idx if mode == 'train' else val_idx if mode == 'val' else test_idx
        mutations = geno_pheno['MUTATIONS'][indices]

        assert emb_type in ['max', 'mean', 'sum', 'min']
        for muts in mutations:
            if emb_type == 'max':
                emb = np.zeros(embeddings.shape[1])
                for mut in muts:
                    if mut in mut_dict:
                        emb = np.maximum(emb, embeddings[mut_dict[mut]])
                self.inputs.append(emb)
            elif emb_type == 'mean':
                emb = np.zeros(embeddings.shape[1])
                count = 0
                for mut in muts:
                    if mut in mut_dict:
                        emb += embeddings[mut_dict[mut]]
                        count += 1
                if count > 0:
                    emb /= count
                self.inputs.append(emb)
            elif emb_type == 'sum':
                emb = np.zeros(embeddings.shape[1])
                for mut in muts:
                    if mut in mut_dict:
                        emb += embeddings[mut_dict[mut]]
                self.inputs.append(emb)
            elif emb_type == 'min':
                emb = np.zeros(embeddings.shape[1])
                for mut in muts:
                    if mut in mut_dict:
                        emb = np.minimum(emb, embeddings[mut_dict[mut]])
                self.inputs.append(emb)
        
        y_train, y_val, y_test = get_labels(drug, split_num)
        self.labels = y_train if mode == 'train' else y_val if mode == 'val' else y_test
        assert len(self.inputs) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


def train_MutEmbAblation(args):
    wandb.init(project='Airframe', config=args)
    wandb.run.name = f'{args.drug}_{args.model}_{args.split_num}'
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # convert to torch dataset
    train_dataset = MutEmbDataset(args.drug, split_num, mode='train', emb_type=args.emb_type)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size)
    val_dataset = MutEmbDataset(args.drug, split_num, mode='val', emb_type=args.emb_type)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size)
    test_dataset = MutEmbDataset(args.drug, split_num, mode='test', emb_type=args.emb_type)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size)

    input_size = train_dataset.inputs[0].shape[0]  # The input size is the number of features in the embeddings (128)
    model = NNbinaryClassifier(input_size, args.n_hidden_layers, args.hidden_size, dropout_prob=args.dropout_prob)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Early stopping when the validation AUROC does not improve for 5 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(args.max_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        wandb.log({'epoch': epoch, 'train_loss': train_loss})
        
        # Evaluate the model on validation set for hyperparameter tuning
        val_logits, val_labels = evaluate_model(model, val_dataloader, device)
        val_auroc = roc_auc_score(val_labels, val_logits)
        wandb.log({'epoch': epoch, 'val_auroc': val_auroc})

        # Save the best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        
        # Early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
    test_logits, test_labels = evaluate_model(model, test_dataloader, device)
    wandb.log({'test_auroc': roc_auc_score(test_labels, test_logits)})

    res = {'test_preds': test_logits, 'test_labels': test_labels, 'best_model': best_model}
    wandb.finish()
    # delete the data and model to save memory
    del model, train_dataloader, val_dataloader, test_dataloader
    gc.collect()

    return res, best_val_auroc


def bayesOpt_MutEmbAblation(args):
    best_res = None
    best_hyperparams = None
    best_auroc = 0

    space = {
        'batch_size': (-0.5, 2.49),  # 8, 16, 32
        'n_hidden_layers': (-0.5, 3.49),  # 1, 2, 4, 8
        'hidden_size': (-0.5, 3.49),  # 64, 128, 256, 512
        'learning_rate': (1, 10)  # 1e-4 to 1e-3
    }

    list_batch_size = [8, 16, 32]
    list_n_hidden_layers = [1, 2, 4, 8]
    list_hidden_size = [64, 128, 256, 512]

    def objective(batch_size, n_hidden_layers, hidden_size, learning_rate):
        nonlocal best_res, best_hyperparams, best_auroc
        args.batch_size = list_batch_size[round(batch_size)]
        args.n_hidden_layers = list_n_hidden_layers[round(n_hidden_layers)]
        args.hidden_size = list_hidden_size[round(hidden_size)]
        args.learning_rate = learning_rate * 1e-4
        res, val_auroc = train_MutEmbAblation(args)
        if best_res is None or val_auroc > best_auroc:
            best_res = res
            best_auroc = val_auroc
            best_hyperparams = {
                'batch_size': args.batch_size,
                'n_hidden_layers': args.n_hidden_layers,
                'hidden_size': args.hidden_size,
                'learning_rate': args.learning_rate
            }
        return val_auroc


    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    res = best_res
    res['best_hyperparams'] = best_hyperparams
    return res


# Train the models
if __name__ == '__main__':
    # if os.path.exists(f'../results/{args.drug}_{args.model}.pkl'):
    #     sys.exit()
    
    drugs = get_drug_list()
    for drug in drugs:
        args.drug = drug
        if os.path.exists(f'../results/{args.drug}_{args.model}.pkl'):
            continue
        for split_num in range(20):
            if os.path.exists(f'../results/{args.drug}_{args.model}_{split_num}.pkl'):
                continue

            args.split_num = split_num 

            if args.model == 'WDNN':
                res = bayesOpt_WDNN(args)
            elif args.model == 'DeepAMR':
                res = bayesOpt_DeepAMR(args)
            elif args.model == 'CNNGWP':
                res = bayes_optimise_CNNGWP(args)
            elif args.model == 'MutEmbAblation_max':
                args.emb_type = 'max'
                res = bayesOpt_MutEmbAblation(args)
            elif args.model == 'MutEmbAblation_mean':
                args.emb_type = 'mean'
                res = bayesOpt_MutEmbAblation(args)
            elif args.model == 'MutEmbAblation_sum':
                args.emb_type = 'sum'
                res = bayesOpt_MutEmbAblation(args)
            elif args.model == 'MutEmbAblation_min':
                args.emb_type = 'min'
                res = bayesOpt_MutEmbAblation(args)
            else:
                raise ValueError(f'Invalid model name: {args.model}')

            with open(f'../results/{args.drug}_{args.model}_{split_num}.pkl', 'wb') as f:
                pickle.dump(res, f)
        
        # Combine results from 20 splits
        res_list = []
        for split_num in range(20):
            with open(f'../results/{args.drug}_{args.model}_{split_num}.pkl', 'rb') as f:
                res = pickle.load(f)
            res_list.append(res)
        with open(f'../results/{args.drug}_{args.model}.pkl', 'wb') as f:
            pickle.dump(res_list, f)
        
        for split_num in range(20):
            os.remove(f'../results/{args.drug}_{args.model}_{split_num}.pkl')