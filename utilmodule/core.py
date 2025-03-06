import torchmetrics
import torch.nn as nn
import torch
import copy
from utilmodule.utils import calculate_error,calculate_metrics,f1_score,split_array,save_checkpoint,cosine_scheduler

import numpy as np
from sklearn.cluster import KMeans
 
from tqdm import tqdm
import torch.nn.functional as F
from utilmodule.environment import expand_data


 





def test(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (coords, data, label) in enumerate (test_loader):
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()

           
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data) 
            else:
                update_data = update_data.float() 
            if args.ape: 
                update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim)  

            update_coords, update_data ,total_T = expand_data(update_coords, update_data, action_size = args.action_size, total_steps=args.test_total_T)
            grouping_instance = grouping(action_size=args.action_size)  
            
            for patch_step in range(0, total_T):
                restart = False
                restart_batch = False
                if patch_step == 0:
                    restart = True
                    restart_batch = True
            
                action_index_pro,memory = grouping_instance.rlselectindex_grouping(ppo,memory_space,update_coords,sigma=0.02,restart=restart)  
                features_group , update_coords,update_data ,memory = grouping_instance.action_make_subbags(ppo,memory,action_index_pro,update_coords,update_data,action_size = args.action_size ,restart = restart,delete_begin=True)  
                results_dict,trandata_ppo,memory ,test_cl_logits = basedmodel(FusionHisF,features_group, memory, update_coords,mask_ratio=0)

                _ = ppo.select_action(trandata_ppo,memory,restart_batch=restart_batch,training=True)

            W_results_dict,memory = classifymodel (memory) 
            W_logits, W_Y_prob, W_Y_hat = W_results_dict['logits'], W_results_dict['Y_prob'], W_results_dict['Y_hat']

            memory.clear_memory()
            label_list.append(label)
            Y_prob_list.append(W_Y_prob)  

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)  
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())  
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        print(f'test Accuracy: {accuracy:.4f} " test Precision: {precision:.4f},test Recall: {recall:.4f},test F1 Score: {f1:.4f},test AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy






def interpolate_probs(probs, new_length,action_size):
    b,c = probs.shape
    x = np.linspace(0, 1, c)
    x_new = np.linspace(0, 1, new_length)
    new_probs = np.interp(x_new, x, probs.view(-1).cpu().numpy())
    interpolate_action_probs = new_probs / new_probs.sum()   
    index = np.random.choice(np.arange(new_length), size=action_size, p=interpolate_action_probs)
    return index


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


    
class grouping:

    def __init__(self,action_size = 128):
        self.action_size = action_size 
        self.action_std = 0.1
           
     
    def rlselectindex_grouping(self,ppo,memory,coords,sigma=0.02,restart=False): 
        B, N, C = coords.shape
        if restart  : 
            if self.action_size < N: 
                action = torch.distributions.dirichlet.Dirichlet(torch.ones(self.action_size)).sample((1,)).to(coords.device)
            else: 
                random_values = torch.rand(1, N).to(coords.device)
                indices = torch.randint(0, N, (self.action_size,)).to(coords.device)
                action = random_values[0, indices].unsqueeze(0)
            memory.actions.append(action) 
            memory.logprobs.append(action) 
            return action.detach(), memory
        else:  
            action = memory.actions[-1] 
            
            return action.detach(), memory
          
    
    def action_make_subbags(self,ppo,memory, action_index_pro, update_coords,update_features,action_size= None,restart= False,delete_begin=False): 
        
        B, N, C = update_coords.shape
        idx = interpolate_probs(action_index_pro, new_length = N ,action_size = action_size)
        idx = torch.tensor(idx)
        features_group = update_features[:, idx[:], :]
        action_group = update_coords[:, idx[:], :] 

        if restart and delete_begin:
            memory.coords_actions.append(action_group)
            return features_group, update_coords, update_features ,memory
        else:
            idx = torch.unique(idx)
            mask = torch.ones(update_features.size(1), dtype=torch.bool) 
            mask[idx] = False  
            updated_features = update_features[:, mask, :] 
            updated_coords = update_coords[:, mask, :]
            memory.coords_actions.append(action_group)
            return features_group, updated_coords, updated_features ,memory

    
