# This file contains the code to pre-train the embeddings of mutations using MetaPath2Vec
import numpy as np
import pandas as pd
import torch
import dgl
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from dgl.nn.pytorch import MetaPath2Vec
import argparse
from codes.util import *


# parse the arguments
def parse_args_metapath2vec():
    parser = argparse.ArgumentParser(description='Pre-train mutation embeddings')
    parser.add_argument('--drug', type=str, default='INH', help='Drug name')
    parser.add_argument('--split_num', type=int, default=0, help='Split number')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='../data/intermediate/', help='Path to save the embeddings')
    return parser.parse_args()


# get the gene, loci and synonymous attributes of mutations
def getMutAttr(mut, attr):
    assert attr in ['gene', 'loci', 'syno']

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


# function to return a dict of genes given a dict of mutations
def getAttrDict(dict_mut, attr):
    attr_list = [getMutAttr(k, attr) for k in dict_mut.keys()]
    attr_list = np.unique(attr_list)
    np.sort(attr_list)
    return {attr: i for i, attr in enumerate(attr_list)}


# function to return the element of a heterogeneous graph except the co-occurrence edges of mutations
def getEdges(drug, split_num, attr):
    source, target = [], []
    mut_dict = get_mut_dict(drug, split_num)
    attr_dict = getAttrDict(mut_dict, attr)
    for mut in mut_dict.keys():
        source.append(mut_dict[mut])
        target.append(attr_dict[getMutAttr(mut, attr)])
    return source, target


# function to return the co-occurrence edges of mutations
def getCoEdges(drug, split_num):
    source, target = [], []
    geno_pheno = get_geno_pheno()
    mut_dict = get_mut_dict(drug, split_num)

    if split_num != -1:
        # Load the split
        train_idx, _, _ = get_data_splits(drug)[split_num]
    else:
        train_idx = get_data_indices(drug)

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
def trainMetaPath2Vec(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_data = getGraph(args.drug, args.split_num)

    g = dgl.heterograph(graph_data, device=device)
    meta_path = ['mut2gene', 'gene2mut', 'mut2loci', 'loci2mut', 'mut2syno', 'syno2mut', 'mut2iso', 'iso2mut']
    model = MetaPath2Vec(g, meta_path, window_size=1, emb_dim=args.emb_dim)
    model = model.to(device)

    dataloader = DataLoader(torch.arange(g.num_nodes('mut')), batch_size = args.batch_size, shuffle=True, collate_fn = model.sample)
    optimizer = SparseAdam(model.parameters(), lr=args.learning_rate)

    best_loss = np.inf
    best_loss_epoch = 0
    best_model = None

    for epoch in range(args.max_epochs):
        total_loss = 0
        for (pos_u, pos_v, neg_v) in dataloader:
            loss = model(pos_u.to(device), pos_v.to(device), neg_v.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().to('cpu').item()

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


def main():
    args = parse_args_metapath2vec()
    mut_emb, _ = trainMetaPath2Vec(args)
    with open(f'{args.save_path}/{args.drug}_{args.split_num}_emb.pickle', 'wb') as f:
        pickle.dump(mut_emb, f)

if __name__ == '__main__':
    main()



