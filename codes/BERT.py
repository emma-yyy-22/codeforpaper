import ast
import datetime
import random
import time
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
"""This file is based on Huggingface Transformer version 4.16.2."""
from transformers import BertPreTrainedModel, BertModel
from transformers import BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
import argparse

from util import *


# parse the arguments
def parse_args_bert():
    parser = argparse.ArgumentParser(description='Bert model for binary classification')
    parser.add_argument('--drug', type=str, default='INH', help='Drug name')
    parser.add_argument('--split_num', type=int, default=0, help='Split number')
    parser.add_argument('--no_pretrained', action='store_true', help='Whether to use pretrained embeddings')
    parser.add_argument('--freeze', action='store_true', help='Whether to freeze the embeddings during training')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model, same as the dimension of embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--num_attention_heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=5, help='Number of epochs for early stopping')
    parser.add_argument('--save_path', type=str, default='../results', help='Path to save the results')
    return parser.parse_args()


# Model that applies the Bert model to do binary classification.
# Updated based on the class BertForSequenceClassification
class BertBinaryClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits = self.dropout(pooled_output)
        logits = self.classifier(logits)
        logits = self.sigmoid(logits)

        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            # (loss), logits, (hidden_states), (attentions)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# The data holder contains all data and labels for a drug.
class BertDataset(Dataset):
    def __init__(self, drug, split_num, mode='train'):
        self.drug = drug
        self.split_num = split_num
        self.mode = mode

        # for all keys in the mutation dictionary, set the value to plus 2
        # because 0 and 1 are reserved for padding and [CLS]
        self.mut_dict = get_mut_dict(drug, split_num)
        for key in self.mut_dict:
            self.mut_dict[key] += 2
        self.mut_dict['[CLS]'] = 1

        self.inputs = []
        # Load the dataset for inputs
        train_idx, val_idx, test_idx = get_data_splits(drug)[split_num]
        geno_pheno = get_geno_pheno()
        indices = train_idx if mode == 'train' else val_idx if mode == 'val' else test_idx
        mutations = geno_pheno['MUTATIONS'][indices]
        for muts in mutations:
            self.inputs.append([1] + [self.mut_dict[mut] for mut in muts if mut in self.mut_dict])

        y_train, y_val, y_test = get_labels(drug, split_num)
        self.labels = y_train if mode == 'train' else y_val if mode == 'val' else y_test

        assert len(self.inputs) == len(self.labels)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx], dtype=torch.float)


# function into the dataloader to make the input length the same for each random batch
# and to make the input mask
def collate_batch(batch):
    input_list = [item[0] for item in batch]

    # Pad the input tensors to a fixed length with 0s.
    # The size of self.data is [batch_size, sequence length].
    input_ids = torch.nn.utils.rnn.pad_sequence(input_list, batch_first=True)

    # The mask input has the same size with input_ids.
    # 0s and 1s means need and no need to ignore(mask) the input ID.
    attention_masks = torch.tensor(np.where(input_ids != 0, 1, 0))

    labels = torch.tensor([item[1] for item in batch]).unsqueeze(1)

    return input_ids, attention_masks, labels


# the function to load the pretrained embeddings
def load_embeddings(drug, split_num, hidden_size):
    mut_dict = get_mut_dict(drug, split_num)
    with open(f'../data/intermediate/{drug}_emb.pickle', 'rb') as f:
        embeddings = pickle.load(f)[split_num]

    assert embeddings.shape[-1] == hidden_size
    assert embeddings.shape[0] == len(mut_dict)
    
    # The embedding for '[CLS]' is defined as the average of all snp embeddings.
    # It should be better than a random one, but may still need to improve.
    emb_cls = np.mean(embeddings, axis=0)
    emb_zero = np.zeros_like(emb_cls)
    embeddings = np.vstack([emb_zero, emb_cls, embeddings])

    return torch.tensor(embeddings, dtype=torch.float)


def train_bert(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the dataset
    train_dataset = BertDataset(args.drug, args.split_num, mode='train')
    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.batch_size, collate_fn=collate_batch)
    val_dataset = BertDataset(args.drug, args.split_num, mode='val')
    val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=args.batch_size, collate_fn=collate_batch)
    test_dataset = BertDataset(args.drug, args.split_num, mode='test')
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=args.batch_size, collate_fn=collate_batch)

    # Set the configuration of Bert model.
    config = BertConfig(hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_encoder_layers,
                        num_attention_heads=args.num_attention_heads,
                        position_embedding_type='Nothing')
    model = BertBinaryClassification(config)

    # Load the pretrained embeddings.
    if not args.no_pretrained:
        embedding_weights = load_embeddings(args.drug, args.split_num, args.hidden_size)
        embedding_layer = torch.nn.Embedding.from_pretrained(embedding_weights, freeze=args.freeze)
        model.set_input_embeddings(embedding_layer)

    model.to(device)

    # Set the optimizer
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    best_val_auroc = 0
    best_val_epoch = 0
    best_model = None

    # Training loop
    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_dataloader:
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()

            # Perform a forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

            # Compute the loss
            loss = outputs['loss']

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_logits, val_labels = [], []
        for batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                logits = outputs['logits']
                val_logits.append(logits.detach().cpu().numpy())
                val_labels.append(b_labels.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_auroc = roc_auc_score(val_labels, val_logits)
        # print(f'Epoch {epoch + 1} validation AUROC: {val_auroc}')

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_val_epoch = epoch
            best_model = model.state_dict()
        
        # Early stopping
        if epoch - best_val_epoch >= args.early_stopping:
            break
    
    # Load the best model and return the results
    model.load_state_dict(best_model)
    model.eval()

    train_logits, train_labels = [], []
    for batch in train_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs['logits']
            train_logits.append(logits.detach().cpu().numpy())
            train_labels.append(b_labels.detach().cpu().numpy())
    train_logits = np.concatenate(train_logits, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    val_logits, val_labels = [], []
    for batch in val_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs['logits']
            val_logits.append(logits.detach().cpu().numpy())
            val_labels.append(b_labels.detach().cpu().numpy())
    val_logits = np.concatenate(val_logits, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)

    test_logits, test_labels = [], []
    for batch in test_dataloader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs['logits']
            test_logits.append(logits.detach().cpu().numpy())
            test_labels.append(b_labels.detach().cpu().numpy())
    test_logits = np.concatenate(test_logits, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    res = (train_logits, val_logits, test_logits, train_labels, val_labels, test_labels)

    return res, best_val_auroc


def main():
    args = parse_args_bert()
    res, _ = train_bert(args)
    np.save(f'{args.save_path}/{args.drug}_{args.split_num}_bert.npy', res) 


if __name__ == '__main__':
    main()
