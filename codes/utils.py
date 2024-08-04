import pickle
import pandas as pd
import numpy as np
import os


drug_list = ['INH', 'RIF', 'RFB', 'EMB', 'ETH', 'LEV', 'MXF', 'AMI', 'KAN', 'BDQ', 'CFZ', 'DLM', 'LZD']
# walker_drug_list = ['SM', 'KAN', 'AK', 'CAP', 'EMB', 'CIP', 'OFX', 'MOX', 'INH', 'RIF', 'PZA']
walker_drug_list = ['RIF', 'INH', 'EMB', 'PZA', 'SM', 'OFX', 'CAP', 'AK', 'KAN', 'MOX', 'CIP']

# Careful: PAS only in UKMYC5
mic_5 = {'AMI':['<=0.25','0.5','1','2','4','8','>8'],'BDQ':['<=0.015','0.03','0.06','0.12','0.25','0.5','1','2','>2'],'CFZ':['<=0.06','0.12','0.25','0.5','1','2','4','>4'],'DLM':['<=0.015','0.03','0.06','0.12','0.25','0.5','1','>1']

       ,'EMB':['<=0.06','0.12','0.25','0.5','1','2','4','8','>8'],'ETH':['0.25','0.5','1','2','4','8','>8'],'INH':['<=0.025','0.05','0.1','0.2','0.4','0.8','1.6','>1.6'],

       'KAN':['<=1','2','4','8','16','>16'],'LEV':['<=0.12','0.25','0.5','1','2','4','8','>8'],'LZD':['<=0.03','0.06','0.12','0.25','0.5','1','2','>2'],'MXF':['<=0.06','0.12','0.25','0.5','1','2','4','>4'],'PAS':['<=0.12','0.25','0.5','1','2','4','>4']

       ,'RFB':['<=0.06','0.12','0.25','0.5','1','2','>2'],'RIF':['<=0.06','0.12','0.25','0.5','1','2','4','>4']}

mic_6 = {'AMI':['<=0.25','0.5','1','2','4','8','16','>16'],'BDQ':['<=0.008','0.015','0.03','0.06','0.12','0.25','0.5','1','>1'],'CFZ':['<=0.03','0.06','0.12','0.25','0.5','1','2','>2'],'DLM':['<=0.008','0.015','0.03','0.06','0.12','0.25','0.5','>0.5']

       ,'EMB':['<=0.25','0.5','1','2','4','8','16','32','>32'],'ETH':['<=0.25','0.5','1','2','4','8','>8'],'INH':['<=0.025','0.05','0.1','0.2','0.4','0.8','1.6','3.2','6.4','12.8','>12.8'],

       'KAN':['<=1','2','4','8','16','>16'],'LEV':['<=0.12','0.25','0.5','1','2','4','8','>8'],'LZD':['<=0.06','0.12','0.25','0.5','1','2','4','>4'],'MXF':['<=0.06','0.12','0.25','0.5','1','2','4','>4']

       ,'RIF':['<=0.03','0.06','0.12','0.25','0.5','1','2','4','8','>8'],'RFB':['<=0.06','0.12','0.25','0.5','1','2','>2']}

mic_list = {'AMI':['<=0.25','0.5','1','2','4','8','>8'],'BDQ':['<=0.015','0.03','0.06','0.12','0.25','0.5','1','>1'],'CFZ':['<=0.06','0.12','0.25','0.5','1','2','>2'],'DLM':['<=0.015','0.03','0.06','0.12','0.25','0.5','>0.5']

       ,'EMB':['<=0.25','0.5','1','2','4','8','>8'],'ETH':['<=0.25','0.5','1','2','4','8','>8'],'INH':['<=0.025','0.05','0.1','0.2','0.4','0.8','1.6','>1.6'],

       'KAN':['<=1','2','4','8','16','>16'],'LEV':['<=0.12','0.25','0.5','1','2','4','8','>8'],'LZD':['<=0.06','0.12','0.25','0.5','1','2','>2'],'MXF':['<=0.06','0.12','0.25','0.5','1','2','4','>4'],

       'RFB':['<=0.06','0.12','0.25','0.5','1','2','>2'],'RIF':['<=0.06','0.12','0.25','0.5','1','2','4','>4']}

ecoff = {'INH': '0.1', 'RIF': '0.5', 'RFB': '0.12', 'EMB': '4', 'ETH': '4', 'LEV': '1', 'MXF': '1', 'AMI': '1', 'KAN': '4', 'BDQ': '0.25', 'CFZ': '0.25', 'DLM': '0.12', 'LZD': '1'}


def getDrugList(dataset):
    if dataset == 'Walker':
        return walker_drug_list
    elif dataset == 'CRyPTIC':
        return drug_list
    

def getModelList(label):
    if label == 'binary':
        return ['svm', 'lr', 'rf', 'adaboost', 'gbt', 'nn']
    elif label == 'multi':
        return ['lrit', 'lrat', 'lr', 'rf', 'nn', 'coral', 'corn']
    

# Function to map UKMYC5 and UKMYC6 MIC values to the same scale
def toMixedMIC(drug, value):
    if pd.isnull(value):
        return np.nan
    if value in mic_list[drug]:
        return value
    elif value in mic_5[drug]:
        if mic_5[drug].index(value) < len(mic_list[drug])/2:
            return mic_list[drug][0]
        else:
            return mic_list[drug][-1]
    elif value in mic_6[drug]:
        if mic_6[drug].index(value) < len(mic_list[drug])/2:
            return mic_list[drug][0]
        else:
            return mic_list[drug][-1]
    else:
        print(f'Unknown MIC value {value} for drug {drug}')
        raise ValueError('MIC value not found')


def micToBinary(drug, value):
    if pd.isnull(value):
        return np.nan
    if value not in mic_list[drug]:
        value = toMixedMIC(drug, value)
    if mic_list[drug].index(value) <= mic_list[drug].index(ecoff[drug]):
        return 'S'
    else:
        return 'R'
    

# function to get the labels for Walker dataset from the data splits
def getWalkerLabels(drug):
    geno_pheno = pd.read_pickle('../Data/Walker2015Lancet.pkl')
    # Load the splitted indices for the drug
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}_split.pickle', 'rb') as f:
        splits = pickle.load(f)
    
    y = []
    for i in range(len(splits)):
        train_idx, val_idx, test_idx = splits[i]
        train_pheno = geno_pheno[drug][train_idx]
        y_train = np.array([1 if y == 'R' else 0 for y in train_pheno])
        val_pheno = geno_pheno[drug][val_idx]
        y_val = np.array([1 if y == 'R' else 0 for y in val_pheno])
        test_pheno = geno_pheno[drug][test_idx]
        y_test = np.array([1 if y == 'R' else 0 for y in test_pheno])
        y.append((y_train, y_val, y_test))
    
    return y


def getWalkerLabelsAll(drug):
    geno_pheno = pd.read_pickle('../Data/Walker2015Lancet.pkl')
    with open(f'../Data/idx_splits/Walker_single_binary/{drug}_index.pickle', 'rb') as f:
        indices = pickle.load(f)
    pheno = geno_pheno[drug][indices]
    y = np.array([1 if y == 'R' else 0 for y in pheno])
    return y


# function to get the binary labels for CRyPTIC dataset from the data splits
def getCrypticBinaryLabels(drug):
    geno_pheno = pd.read_pickle('../Data/CRyPTIC_Geno_Pheno_table_v2.pkl')
    # Load the splitted indices for the drug
    with open(f'../Data/idx_splits/CRyPTIC_single_binary/{drug}_split.pickle', 'rb') as f:
        splits = pickle.load(f)
    
    y = []
    for i in range(len(splits)):
        train_idx, val_idx, test_idx = splits[i]
        train_pheno = geno_pheno[drug+'_BINARY'][train_idx]
        y_train = np.array([1 if y == 'R' else 0 for y in train_pheno])
        val_pheno = geno_pheno[drug+'_BINARY'][val_idx]
        y_val = np.array([1 if y == 'R' else 0 for y in val_pheno])
        test_pheno = geno_pheno[drug+'_BINARY'][test_idx]
        y_test = np.array([1 if y == 'R' else 0 for y in test_pheno])
        y.append((y_train, y_val, y_test))
    
    return y


# function to get the MIC labels for CRyPTIC dataset from the data splits
def getCrypticMicLabels(drug):
    geno_pheno = pd.read_pickle('../Data/CRyPTIC_Geno_Pheno_table_v2.pkl')
    # Load the splitted indices for the drug
    with open(f'../Data/idx_splits/CRyPTIC_single_mic/{drug}_split.pickle', 'rb') as f:
        splits = pickle.load(f)
    
    y = []
    for i in range(len(splits)):
        train_idx, val_idx, test_idx = splits[i]
        train_pheno = geno_pheno[drug+'_MIC_UNIVERSAL'][train_idx]
        y_train = np.array([mic_list[drug].index(y) for y in train_pheno])
        val_pheno = geno_pheno[drug+'_MIC_UNIVERSAL'][val_idx]
        y_val = np.array([mic_list[drug].index(y) for y in val_pheno])
        test_pheno = geno_pheno[drug+'_MIC_UNIVERSAL'][test_idx]
        y_test = np.array([mic_list[drug].index(y) for y in test_pheno])
        y.append((y_train, y_val, y_test))
    
    return y


# Function to return the dictionary of mutations from the Dataset 1
# Only the mutations that are present in the training set are included
def getMutDict(drug, split_num):
    # Load the split
    if split_num != None:
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_split.pickle', 'rb') as f:
            train_idx, val_idx, test_idx = pickle.load(f)[split_num]
    else:
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_index.pickle', 'rb') as f:
            train_idx = pickle.load(f)

    # Load the dataset
    geno_pheno = pd.read_pickle('../Data/Walker2015Lancet.pkl')
    mutation = geno_pheno['MUTATIONS'][train_idx]
    
    mut_set = set()
    for i in range(len(mutation)):
        mut_set = mut_set.union(set(mutation[i]))
    mut_list = list(mut_set)
    mut_list.sort()
    
    mut_dict = dict(zip(mut_list, range(0, len(mut_list))))
    return mut_dict


# define a function to delete the interim results
def deleteInterimResult(dataset, drug, model, label, task='single'):
    if label == 'binary' and task == 'single':
        if os.path.exists(f'../Results/{task}_{label}_classification/{dataset}/{drug}_{model}.pkl') == False:
            print(f'../Results/{task}_{label}_classification/{dataset}/{drug}_{model}.pkl does not exist')
            return
        for i in range(20):
            if os.path.exists(f'../Results/{task}_{label}_classification/{dataset}/{drug}_{model}_{i}.pkl'):
                os.remove(f'../Results/{task}_{label}_classification/{dataset}/{drug}_{model}_{i}.pkl')
    else:
        if os.path.exists(f'../Results/{task}_{label}_classification/{drug}_{model}.pkl') == False:
            print(f'../Results/{task}_{label}_classification/{drug}_{model}.pkl does not exist')
            return
        for i in range(20):
            if os.path.exists(f'../Results/{task}_{label}_classification/{drug}_{model}_{i}.pkl'):
                os.remove(f'../Results/{task}_{label}_classification/{drug}_{model}_{i}.pkl')


# define a function to delete the interim results for different datasets, drugs, models and labels
def deleteInterimResults():
    for label in ['binary', 'multi']:
        models = getModelList(label)
        for dataset in ['Walker', 'CRyPTIC']:
            drugs = getDrugList(dataset)
            for drug in drugs:
                for model in models:
                    deleteInterimResult(dataset, drug, model, label)

