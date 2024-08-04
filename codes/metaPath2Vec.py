import numpy as np
import pandas as pd
import torch
import dgl
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec
from bayes_opt import BayesianOptimization
from utils import *

DRUG = 'INH'
SPLIT_NUM = 0
NUM_EPOCHS = 200
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# get the gene, loci and synonymous attributes of mutations
def getMutAttr(mut, attr):
    mut_split = mut.split('_')
    gene = mut_split[0]
    if attr == 'gene':
        return gene

    if len(mut_split) == 3:
        loci = gene + mut_split[1]
    else:
        loci = gene + mut_split[1][1:-1]
    if attr == 'loci':
        return loci
    
    if len(mut_split) == 2 and mut_split[1][0] == mut_split[1][-1]:
        syno = '1'
    else:
        syno = '0'
    if attr == 'syno':
        return syno

    raise ValueError('Invalid attr value: {}'.format(attr))


# function to return a dict of genes given a dict of mutations
def getAttrDict(dict_mut, attr):
    attr_list = [getMutAttr(k, attr) for k in dict_mut.keys()]
    attr_list = np.unique(attr_list)
    np.sort(attr_list)
    return {attr: i for i, attr in enumerate(attr_list)}


# function to return the element of a heterogeneous graph except the co-occurrence edges of mutations
def getEdges(drug, split_num, attr):
    source, target = [], []
    mut_dict = getMutDict(drug, split_num)
    attr_dict = getAttrDict(mut_dict, attr)
    for mut in mut_dict.keys():
        source.append(mut_dict[mut])
        target.append(attr_dict[getMutAttr(mut, attr)])
    return source, target


# function to return the co-occurrence edges of mutations
def getCoEdges(drug, split_num):
    source, target = [], []
    geno_pheno = pd.read_pickle('../Data/Walker2015Lancet.pkl')
    mut_dict = getMutDict(drug, split_num)

    if split_num != None:
        # Load the split
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_split.pickle', 'rb') as f:
            train_idx, _, _ = pickle.load(f)[split_num]
    else:
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_index.pickle', 'rb') as f:
            train_idx = pickle.load(f)

    mutations = geno_pheno['MUTATIONS'][train_idx]
    
    for i, muts in enumerate(mutations):
        for mut in muts:
            source.append(mut_dict[mut])
            target.append(i)
    return source, target


# function to return the constructed heterogeneous graph
def getGraph(drug, split_num):
    graph_data = {}  # the input graph of MetaPath2Vec
    # add the gene, loci and syno information
    for attr in ['gene', 'loci', 'syno']:
        source, target = getEdges(drug, split_num, attr)
        graph_data[('mut', f'mut2{attr}', attr)] = (torch.tensor(source), torch.tensor(target))
        graph_data[attr, f'{attr}2mut', 'mut'] = (torch.tensor(target), torch.tensor(source))
    # add the co-occurrence information
    source, target = getCoEdges(drug, split_num)
    graph_data[('mut', 'mut2iso', 'iso')] = (torch.tensor(source), torch.tensor(target))
    graph_data[('iso', 'iso2mut', 'mut')] = (torch.tensor(target), torch.tensor(source))
    return graph_data


# function to train the model
def trainMetaPath2Vec(drug, split_num, params):
    graph_data = getGraph(drug, split_num)

    g = dgl.heterograph(graph_data, device=device)
    meta_path = ['mut2gene', 'gene2mut', 'mut2loci', 'loci2mut', 'mut2syno', 'syno2mut', 'mut2iso', 'iso2mut']
    model = MetaPath2Vec(g, meta_path, window_size=1, emb_dim=128)
    if torch.cuda.is_available():
        model.cuda()

    dataloader = DataLoader(torch.arange(g.num_nodes('mut')), batch_size = BATCH_SIZE, shuffle=True, collate_fn = model.sample)
    optimizer = SparseAdam(model.parameters(), lr=params['learning_rate'])

    best_loss = np.inf
    best_loss_epoch = 0
    best_model = None

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for (pos_u, pos_v, neg_v) in dataloader:
            loss = model(pos_u.to(device), pos_v.to(device), neg_v.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().to('cpu').item()
        # print(f'Epoch: {epoch}, Loss: {total_loss}')

        if total_loss < best_loss:
            best_loss = total_loss
            best_loss_epoch = epoch
            best_model = model.state_dict()

        # Early stopping
        if epoch - best_loss_epoch > 10:
            break


    # Save the embeddings of all mutations
    model.load_state_dict(best_model)
    model.eval()
    mut_nids = torch.LongTensor(model.local_to_global_nid['mut']).to(device)
    mut_emb = model.node_embed(mut_nids).detach().cpu().numpy()

    return mut_emb, -best_loss  # return the embeddings and the negative loss for hyperparameter optimization


metaPath_params_list = {
    'learning_rate': (1, 50)
}

def metaPath_best_loss(learning_rate, return_res=False):
    params = {}
    params['learning_rate'] = learning_rate * 1e-3
    res, best_loss = trainMetaPath2Vec(DRUG, SPLIT_NUM, params)
    if return_res:
        return res
    else:
        return best_loss


if __name__ == '__main__':
    drug_list = getDrugList('Walker')
    for drug in drug_list:
        if os.path.exists(f'../Data/idx_splits/Walker_single_binary/{drug}_emb.pickle'):
            continue
        DRUG = drug
        res_list = []
        for split_num in range(20):
            SPLIT_NUM = split_num
            print(f'Split {SPLIT_NUM} for {DRUG} using MetaPath2Vec')

            hyper_opt = BayesianOptimization(metaPath_best_loss, metaPath_params_list)
            hyper_opt.maximize(10, 5)
            best_hyper = hyper_opt.max['params']
            res_emb = metaPath_best_loss(**best_hyper, return_res=True)
            res_list.append(res_emb)
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_emb.pickle', 'wb') as f:
            pickle.dump(res_list, f)

    for drug in drug_list:
        if os.path.exists(f'../Data/idx_splits/Walker_single_binary/{drug}_emb_all.pickle'):
            continue
        DRUG = drug
        SPLIT_NUM = None
        print(f'All for {DRUG} using MetaPath2Vec')
        hyper_opt = BayesianOptimization(metaPath_best_loss, metaPath_params_list)
        hyper_opt.maximize(10, 5)
        best_hyper = hyper_opt.max['params']
        res_emb = metaPath_best_loss(**best_hyper, return_res=True)
        with open(f'../Data/idx_splits/Walker_single_binary/{drug}_emb_all.pickle', 'wb') as f:
            pickle.dump(res_emb, f)


