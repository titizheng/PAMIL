import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch

import os
# import re
# import csv
# import yaml
# import json
# import glob
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,accuracy_score










def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='camelyon16',type=str)  
    parser.add_argument('--mode', default='rlselect',type=str)
    parser.add_argument('--seed', default=2021,type=int)
    parser.add_argument('--epoch', default=300,type=int)
    parser.add_argument('--lr', default=0.00001,type=int)
    

    parser.add_argument('--in_chans', default=1024,type=int)
  
    parser.add_argument('--embed_dim', default=512,type=int)
    parser.add_argument('--attn', default='normal',type=str)
    parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--num_layers', default=2,type=int) 

    parser.add_argument('--instaceclass', default=True,type=bool,help='') 
    parser.add_argument('--CE_CL', default=True,type=bool,help='')
    parser.add_argument('--ape_class', default=False,type=bool,help='') 


    parser.add_argument('--test_h5', default='CAMELYON16/C16-test',type=str)
    parser.add_argument('--train_h5',default='CAMELYON16/C16-train',type=str)
    parser.add_argument('--csv', default='CAMELYON16/camelyon16_test.csv',type=str)
 
    
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--state_dim', type=int, default=512)
    parser.add_argument('--action_size', type=int, default=512) 
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    
    parser.add_argument('--test_total_T', type=int, default=1) 

    parser.add_argument('--reward_rule', type=str, default="cl",help=' ')

    
    
    args = parser.parse_args()
    return args

 

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    schedule_per_epoch = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            value = np.linspace(start_warmup_value, base_value, warmup_epochs)[epoch]
        else:
            iters_passed = epoch * niter_per_ep
            iters_left = epochs * niter_per_ep - iters_passed
            alpha = 0.5 * (1 + np.cos(np.pi * iters_passed / (epochs * niter_per_ep)))
            value = final_value + (base_value - final_value) * alpha
        schedule_per_epoch.append(value)
    return schedule_per_epoch




def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error



def calculate_metrics(targets, probs):
    threshold = 0.5  # You can adjust this threshold as needed
    predictions = (probs[:, 1] >= threshold).astype(int)
 
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions) 
    auc = roc_auc_score(targets, probs[:, 1])  
    accuracy = accuracy_score(targets, predictions)  

    return precision, recall, f1, auc, accuracy




 
 

def cat_msg2cluster_group(x_groups,msg_tokens):
    x_groups_cated = []
    for x in x_groups:  
        x = x.unsqueeze(dim=0)  
        try:
            temp = torch.cat((msg_tokens,x),dim=2)
        except Exception as e:
            print('Error when cat msg tokens to sub-bags')
        x_groups_cated.append(temp)

    return x_groups_cated



def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False)  
    split_indices = np.array_split(indices, m)   

    result = []
    for indices in split_indices:
        result.append(array[indices])

    return result



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.flag = False

    def __call__(self, epoch, val_loss, model, args, ckpt_name = ''):
        ckpt_name = './ckp/{}_checkpoint_{}_{}.pt'.format(str(args.type),str(args.seed),str(epoch))
        score = -val_loss
        self.flag = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
            self.counter = 0
        

    def save_checkpoint(self, val_loss, model, ckpt_name, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose and not args.overfit:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)
        elif self.verbose and args.overfit:
            print(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)           
        torch.save(model.state_dict(), ckpt_name)
        print(ckpt_name)
        self.val_loss_min = val_loss
        self.flag = True

def save_checkpoint(state,best_acc, auc,checkpoint, filename='checkpoint.pth.tar'):
    best_acc = f"{best_acc:.4f}"
    auc = f"{auc:.4f}"
    filepath = os.path.join(checkpoint, best_acc+"_"+auc+"_"+filename)
    torch.save(state, filepath)
 

