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
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset

import pickle
import argparse
from utils import *
from NNbinaryClassification import NNbinaryClassifier

warnings.filterwarnings('ignore')

DATASET = 'Walker' 

parser = argparse.ArgumentParser(description='Train and test comparison models')
parser.add_argument('--model', type=str, help='Model name (WDNN, DeepAMR, CNNGWP or MutEmbAblation)')
parser.add_argument('--drug', type=str, help='Drug name')

args = parser.parse_args()

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WDNN
class WDNN(nn.Module):
    def __init__(self, input_dim):
        super(WDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.output = nn.Linear(input_dim + 256, 1)
        
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.batch_norm3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.5)
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
    

def train_WDNN(drug, split_num):
    # load the dataset
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}.pkl', 'rb') as f:
        X = pickle.load(f)[split_num]
    y = getWalkerLabels(drug)[split_num]
    
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    # convert to torch dataset
    # The original code does not mention the batch size, so I will use 32 as the default batch size
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)

    input_size = X_train.shape[1]
    model = WDNN(input_size)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr=np.exp(-1.0 * 9))

    for epoch in range(100):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            optimizer.zero_grad()
            loss, _ = model(inputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
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

    print(f'{drug} Split{split_num} test AUROC:', roc_auc_score(test_labels, test_logits))

    res = (test_logits, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res


# DeepAMR
class DeepAMR(nn.Module):
    def __init__(self, input_dim):
        super(DeepAMR, self).__init__()
        
        dropout_prob = 0.3

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
        
    def forward(self, x):
        encoded = self.encoder(x)
        
        # Task outputs
        task_out = self.classifier(encoded)
        
        decoded = self.decoder(encoded)
        
        return decoded, task_out

def train_DeepAMR(drug, split_num):
    # load the dataset
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}.pkl', 'rb') as f:
        X = pickle.load(f)[split_num]
    y = getWalkerLabels(drug)[split_num]
    
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    # convert to torch dataset
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=64)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=64)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=64)

    input_size = X_train.shape[1]
    model = DeepAMR(input_size)

    if torch.cuda.is_available():
        model.cuda()

    criterion_reconstruction = nn.MSELoss()
    criterion_classification = nn.BCELoss()
    # The original code does not mention the learning rate, so I will use 0.001 as the default learning rate
    optimizer = Adam(model.parameters(), lr=0.001) 

    # Early stopping when the validation AUROC does not improve for 5 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    # Train the model
    for epoch in range(100):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            optimizer.zero_grad()
            decoded, logits = model(inputs)
            loss_reconstruction = criterion_reconstruction(decoded, inputs)
            loss_classification = criterion_classification(logits.view(-1), labels)
            loss = loss_reconstruction + loss_classification
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation set
        model.eval()
        val_logits = []
        val_labels = []
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            with torch.no_grad():
                _, logits = model(inputs)
            val_logits.append(logits.detach().cpu().numpy())
            val_labels.append(labels.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_auroc = roc_auc_score(val_labels, val_logits)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        # early stopping
        if epoch - best_val_epoch >= 5:
            break
    
    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()

    test_logits = []
    test_labels = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch
        with torch.no_grad():
            _, logits = model(inputs)
        test_logits.append(logits.detach().cpu().numpy())
        test_labels.append(labels.detach().cpu().numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    print(f'{drug} Split{split_num} test AUROC:', roc_auc_score(test_labels, test_logits))

    res = (test_logits, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res


# CNNGWP
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

def train_CNNGWP(drug, split_num, filters, kernel_size):
    # load the dataset
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}.pkl', 'rb') as f:
        X = pickle.load(f)[split_num]
    y = getWalkerLabels(drug)[split_num]
    
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    # convert to torch dataset
    # The original code does not mention the batch size, so I will use 32 as the default batch size
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=48)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=48)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=48)

    input_size = X_train.shape[1]
    model = CNNGWP(input_size, filters, kernel_size)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr=0.00025)

    # Early stopping when the validation AUROC does not improve for 10 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(250):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            optimizer.zero_grad()
            loss, _ = model(inputs, labels)
            loss.backward()
            optimizer.step()
    
        # Evaluate the model on validation set for hyperparameter tuning
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

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        # early stopping
        if epoch - best_val_epoch >= 10:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
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

    # print('AUROC:', roc_auc_score(test_labels, test_logits))

    res = (test_logits, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res, best_val_auroc


# Use Bayesian Optimization for hyperparameter tuning

def bayes_optimise_CNNGWP(drug, split_num):
    best_res = None
    best_auroc = 0

    def objective(filters, kernel_size):
        nonlocal best_res, best_auroc
        res, val_auroc = train_CNNGWP(drug, split_num, int(filters), int(kernel_size))
        if best_res is None or val_auroc > best_auroc:
            best_res = res
            best_auroc = val_auroc
        return val_auroc

    space = {
        'filters': (20, 100),
        'kernel_size': (10, 50)
    }

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=15,
        n_iter=25
    )

    best_auroc = optimiser.max['target']
    print(f'{drug} Split{split_num} AUROC:', best_auroc)

    return best_res


# Change some settings of CNN-GWP to make it fair to compare with other models
class CNNGWP2(nn.Module):
    def __init__(self, input_dim, filters, kernel_size):
        super(CNNGWP2, self).__init__()
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

def train_CNNGWP2(drug, split_num, filters, kernel_size, batch_size, learning_rate):
    # load the dataset
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}.pkl', 'rb') as f:
        X = pickle.load(f)[split_num]
    y = getWalkerLabels(drug)[split_num]
    
    X_train, X_val, X_test = X
    y_train, y_val, y_test = y

    # convert to torch dataset
    # The original code does not mention the batch size, so I will use 32 as the default batch size
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    input_size = X_train.shape[1]
    model = CNNGWP2(input_size, filters, kernel_size)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping when the validation AUROC does not improve for 10 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(100):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            optimizer.zero_grad()
            loss, _ = model(inputs, labels)
            loss.backward()
            optimizer.step()
    
        # Evaluate the model on validation set for hyperparameter tuning
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

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        # early stopping
        if epoch - best_val_epoch >= 5:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
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

    # print('AUROC:', roc_auc_score(test_labels, test_logits))

    res = (test_logits, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res, best_val_auroc


# Use Bayesian Optimization for hyperparameter tuning

def bayes_optimise_CNNGWP2(drug, split_num):
    best_res = None
    best_auroc = 0

    def objective(filters, kernel_size, batch_size, learning_rate):
        nonlocal best_res, best_auroc

        res, val_auroc = train_CNNGWP2(drug, split_num, int(filters), int(kernel_size), [8, 16, 32][round(batch_size)], learning_rate*1e-4)
        if best_res is None or val_auroc > best_auroc:
            best_res = res
            best_auroc = val_auroc
        return val_auroc

    space = {
        'filters': (20, 100),
        'kernel_size': (10, 50),
        # batch_size: 8, 16, 32
        'batch_size': (-0.5, 2.49),
        # learning_rate: 1e-4 to 1e-3
        'learning_rate': (1, 10)
    }

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    best_auroc = optimiser.max['target']
    print(f'{drug} Split{split_num} AUROC:', best_auroc)

    return best_res


# Pure embeddings with BERT
# The dataset needs to be modified because the mutations not present in the training set should be removed
class MutEmbDataset(Dataset):
    def __init__(self, drug, split_num, mode='train', emb_type='max'):

        # load the mutation dictionary, with the index starting from 0
        mut_dict = getMutDict(drug, split_num)
        # load the embeddings
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_emb.pickle', 'rb') as f:
            embeddings = pickle.load(f)[split_num] # embeddings shape: (num_mutations, emb_dim(128))
        assert len(mut_dict) == embeddings.shape[0]

        self.inputs = []
        # Load the dataset for inputs
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_split.pickle', 'rb') as f:
            train_idx, val_idx, test_idx = pickle.load(f)[split_num]
        geno_pheno = pd.read_pickle('../Data/Walker2015Lancet.pkl')
        indices = train_idx if mode == 'train' else val_idx if mode == 'val' else test_idx
        mutations = geno_pheno['MUTATIONS'][indices]

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
        
        y_train, y_val, y_test = getWalkerLabels(drug)[split_num]
        self.labels = y_train if mode == 'train' else y_val if mode == 'val' else y_test
        assert len(self.inputs) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


def train_MutEmbAblation(drug, split_num, emb_type, batch_size, n_hidden_layers, hidden_size, learning_rate):
    # convert to torch dataset
    train_dataset = MutEmbDataset(drug, split_num, mode='train', emb_type=emb_type)
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataset = MutEmbDataset(drug, split_num, mode='val', emb_type=emb_type)
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataset = MutEmbDataset(drug, split_num, mode='test', emb_type=emb_type)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    input_size = train_dataset.inputs[0].shape[0]  # The input size is the number of features in the embeddings (128)
    model = NNbinaryClassifier(input_size, n_hidden_layers, hidden_size, dropout_prob=0.1)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping when the validation AUROC does not improve for 5 epochs
    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    for epoch in range(100):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            inputs, labels = batch
            optimizer.zero_grad()
            loss, _ = model(inputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on validation set for hyperparameter tuning
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

        # Save the best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        
        # Early stopping
        if epoch - best_val_epoch >= 5:
            break

    # Evaluate the model
    model.load_state_dict(best_model)
    model.eval()
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

    # print(f'{drug} Split{split_num} test AUROC:', roc_auc_score(test_labels, test_logits))

    res = (test_logits, test_labels)

    # delete the data and model to save memory
    del model, train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    gc.collect()

    return res, best_val_auroc

def bayes_optimise_MutEmbAblation(drug, split_num, emb_type):
    best_res = None
    best_auroc = 0

    def objective(batch_size, n_hidden_layers, hidden_size, learning_rate):
        nonlocal best_res, best_auroc
        param_batch_size = [8, 16, 32][round(batch_size)]
        param_n_hidden_layers = [1, 2, 4, 8][round(n_hidden_layers)]
        param_hidden_size = [64, 128, 256, 512][round(hidden_size)]
        param_learning_rate = learning_rate * 1e-4
        res, val_auroc = train_MutEmbAblation(drug, split_num, emb_type, param_batch_size, param_n_hidden_layers, param_hidden_size, param_learning_rate)
        if best_res is None or val_auroc > best_auroc:
            best_res = res
            best_auroc = val_auroc
        return val_auroc

    # batch_size: 8, 16, 32
    # n_hidden_layers: 1, 2, 4, 8
    # hidden_size: 64, 128, 256, 512
    # learning_rate: 1e-4 to 1e-3
    space = {
        'batch_size': (-0.5, 2.49),
        'n_hidden_layers': (-0.5, 3.49),
        'hidden_size': (-0.5, 3.49),
        'learning_rate': (1, 10)
    }

    optimiser = BayesianOptimization(
        f=objective,
        pbounds=space,
        random_state=42
    )

    optimiser.maximize(
        init_points=10,
        n_iter=5
    )

    best_auroc = optimiser.max['target']
    print(f'{drug} Split{split_num} AUROC:', best_auroc)

    return best_res


# Train the models
if __name__ == '__main__':
    if os.path.exists(f'../Results/single_binary_classification/{DATASET}/{args.drug}_{args.model}.pkl'):
        sys.exit()
    
    for split_num in range(20):
        if os.path.exists(f'../Results/single_binary_classification/{DATASET}/{args.drug}_{args.model}_{split_num}.pkl'):
            continue

        if args.model == 'WDNN':
            res = train_WDNN(args.drug, split_num)
        elif args.model == 'DeepAMR':
            res = train_DeepAMR(args.drug, split_num)
        elif args.model == 'CNNGWP':
            res = bayes_optimise_CNNGWP(args.drug, split_num)
        elif args.model == 'CNNGWP2':
            res = bayes_optimise_CNNGWP2(args.drug, split_num)
        elif args.model == 'MutEmbAblation_max':
            res = bayes_optimise_MutEmbAblation(args.drug, split_num, 'max')
        elif args.model == 'MutEmbAblation_mean':
            res = bayes_optimise_MutEmbAblation(args.drug, split_num, 'mean')
        elif args.model == 'MutEmbAblation_sum':
            res = bayes_optimise_MutEmbAblation(args.drug, split_num, 'sum')
        elif args.model == 'MutEmbAblation_min':
            res = bayes_optimise_MutEmbAblation(args.drug, split_num, 'min')
        else:
            raise ValueError(f'Invalid model name: {args.model}')

        with open(f'../Results/single_binary_classification/{DATASET}/{args.drug}_{args.model}_{split_num}.pkl', 'wb') as f:
            pickle.dump(res, f)
    
    # Combine results from 20 splits
    res_list = []
    for split_num in range(20):
        with open(f'../Results/single_binary_classification/{DATASET}/{args.drug}_{args.model}_{split_num}.pkl', 'rb') as f:
            res = pickle.load(f)
        res_list.append(res)
    with open(f'../Results/single_binary_classification/{DATASET}/{args.drug}_{args.model}.pkl', 'wb') as f:
        pickle.dump(res_list, f)
    
    deleteInterimResult(DATASET, args.drug, args.model, 'binary')