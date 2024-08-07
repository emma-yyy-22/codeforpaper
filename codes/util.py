import pickle
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold

# Return the list of drugs for the dataset
def get_drug_list(dataset=1):
    if dataset == 1:
        return ['RIF', 'INH', 'EMB', 'PZA', 'SM', 'OFX', 'CAP', 'AK', 'KAN', 'MOX', 'CIP']
    else:
        raise ValueError('Dataset not found')
    
# Return the raw dataset 1
def get_geno_pheno():
    return pd.read_pickle('../data/Walker2015Lancet.pkl')


# Return the splitted dataset indices for a drug
def get_data_splits(drug):
    if not os.path.exists(f'../data/idx_splits/{drug}_split.pickle'):
        save_split()
    with open(f'../data/idx_splits/{drug}_split.pickle', 'rb') as f:
        splits = pickle.load(f)
    return splits   


# Return the dataset indices for a drug
def get_data_indices(drug):
    if not os.path.exists(f'../data/idx_splits/{drug}_index.pickle'):
        save_split()
    with open(f'../data/idx_splits/{drug}_index.pickle', 'rb') as f:
        indices = pickle.load(f)
    return indices


# Make data splits for single drug binary classification
def singleBinarySplit(drug):
    geno_pheno = get_geno_pheno()
    x, y = [], []
    for i in range(len(geno_pheno)):
        if geno_pheno[drug][i] == 'R':
            x.append(i)
            y.append(1)
        elif geno_pheno[drug][i] == 'S':
            x.append(i)
            y.append(0)
    x, y = np.array(x), np.array(y)
    res_split = []
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)
    ssp = StratifiedShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
    for idx1, idx2 in rskf.split(x, y):
        train_val_index, test_index = x[idx1], x[idx2]
        y_train_val = y[idx1]
        for idx3, idx4 in ssp.split(train_val_index, y_train_val):
            res_split.append((np.sort(train_val_index[idx3]), np.sort(train_val_index[idx4]), np.sort(test_index)))
    
    return res_split, x


# Save the splits for single drug binary classification
def save_split():
    drug_list = get_drug_list()
    for drug in drug_list:
        res_split, x = singleBinarySplit(drug)
        with open(f"../data/idx_splits/{drug}_split.pickle", "wb") as fp:
            pickle.dump(res_split, fp)
        with open(f"../data/idx_splits/{drug}_index.pickle", "wb") as fp:
            pickle.dump(x, fp)


# function to get the labels for Walker dataset from the data splits
def get_labels(drug, split_num):
    geno_pheno = get_geno_pheno()
    splits = get_data_splits(drug)
    
    train_idx, val_idx, test_idx = splits[split_num]
    train_pheno = geno_pheno[drug][train_idx]
    y_train = np.array([1 if y == 'R' else 0 for y in train_pheno])
    val_pheno = geno_pheno[drug][val_idx]
    y_val = np.array([1 if y == 'R' else 0 for y in val_pheno])
    test_pheno = geno_pheno[drug][test_idx]
    y_test = np.array([1 if y == 'R' else 0 for y in test_pheno])
    
    return y_train, y_val, y_test


# function to get the input features for the dataset 1
def get_features(drug, split_num):
    geno_pheno = get_geno_pheno()
    splits = get_data_splits(drug)
    train_idx, val_idx, test_idx = splits[split_num]
    
    # create a dictionary of mutations
    mutation = geno_pheno['MUTATIONS']
    mut_set = set()
    for i in range(len(mutation)):
        mut_set = mut_set.union(set(mutation[i]))
    mut_list = list(mut_set)
    mut_dict = dict(zip(mut_list, range(1, len(mut_list)+1)))  # 0 is reserved for padding

    # create the input features
    mut_matrix = np.zeros((len(geno_pheno), len(mut_dict)+1))
    for i in range(len(geno_pheno)):
        for mut in geno_pheno['MUTATIONS'][i]:
            mut_matrix[i][mut_dict[mut]] = 1
    
    return mut_matrix[train_idx], mut_matrix[val_idx], mut_matrix[test_idx]


# Function to return all available labels for a drug, irrespective of the split
def get_labels_all(drug):
    geno_pheno = get_geno_pheno()
    indices = get_data_indices(drug)
    pheno = geno_pheno[drug][indices]
    y = np.array([1 if y == 'R' else 0 for y in pheno])
    return y


# Function to return the dictionary of mutations from the Dataset 1
def get_mut_dict(drug, split_num):
    # Load the split
    if split_num != None:
        train_idx, val_idx, test_idx = get_data_splits(drug)[split_num]
    else:
        train_idx = get_data_indices(drug)

    # Load the dataset
    geno_pheno = get_geno_pheno()
    mutation = geno_pheno['MUTATIONS'][train_idx]
    
    mut_set = set()
    for i in range(len(mutation)):
        mut_set = mut_set.union(set(mutation[i]))
    mut_list = list(mut_set)
    mut_list.sort()
    
    mut_dict = dict(zip(mut_list, range(0, len(mut_list))))
    return mut_dict


# Function to load the standard dataloader for the dataset 1
def load_data_loader(drug, split_num, batch_size):
    X_train, X_val, X_test = get_features(drug, split_num)
    y_train, y_val, y_test = get_labels(drug, split_num)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float))
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader


