from PAMIL.utilmodule.core import *
from PAMIL.utilmodule.utils import *
import torch
import random
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

class Attention(nn.Module): 
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class AttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1,attn_mode='normal'):
        super(AttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.mode = attn_mode
        self.attn = Attention(self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)
    def forward(self,x):
        return x + self.attn(x)

class GroupsAttenLayer(nn.Module):
    def __init__(self,dim,heads=8,dim_head=64,dropout=0.1,attn_mode='normal'):
        super(GroupsAttenLayer, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.AttenLayer = AttenLayer(dim =self.dim,heads=self.heads,dim_head=self.dim_head,dropout=self.dropout)

    def forward(self,x_groups,mask_ratio=0):
        group_after_attn = []
        r = int(len(x_groups) * (1-mask_ratio))
        x_groups_masked = random.sample(x_groups, k=r)
        for x in x_groups_masked:
            x = x.squeeze(dim=0)
            temp = self.AttenLayer(x).unsqueeze(dim=0)
            group_after_attn.append(temp)
        return group_after_attn
      


class basedBasicLayer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.GroupsAttenLayer = GroupsAttenLayer(dim=dim)
    def forward(self,data,mask_ratio=0):
        _, x_groups, msg_tokens_num = data
        x_groups = self.GroupsAttenLayer(x_groups,mask_ratio)  
        data = (_, x_groups, msg_tokens_num)
        return data
    
    
 


class BasedMILTransformer(nn.Module):
    def __init__(self,args):
        super(BasedMILTransformer, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.in_chans, self.args.embed_dim)  
        self.fc2 = nn.Linear(self.args.embed_dim, self.args.n_classes)
        self.msg_tokens_num = self.args.num_msg
        self.msgcls_token = nn.Parameter(torch.randn(1,1,1,self.args.embed_dim))

        self.predictor = nn.Sequential(nn.Linear(self.args.embed_dim, self.args.embed_dim,bias=False),
                                #  nn.BatchNorm1d(self.args.embed_dim),
                                nn.LayerNorm(self.args.embed_dim),
                                nn.ReLU(inplace=True)
                                 )
        
        
        self.msg_tokens = nn.Parameter(torch.zeros(1, 1, 1, self.args.embed_dim))
        self.cat_msg2cluster_group = cat_msg2cluster_group
        if self.args.ape:
                self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 1, self.args.embed_dim))
        

 
        self.basedlayers = nn.ModuleList()
        for i_layer in range(self.args.num_layers):
            layer = basedBasicLayer(dim=self.args.embed_dim)
            self.basedlayers.append(layer)


    def msg_predictor(self,x):
        msg_logits = self.predictor(x)
        return msg_logits

    def head(self,x):
        logits = self.fc2(x)
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat} 

        return results_dict
    

    def forward(self,FusionHisF,x,memory=False,coords=False,mask_ratio=0):
        msg_tokens = self.msg_tokens.expand(1,1,self.msg_tokens_num,-1) 
        msg_cls = self.msgcls_token
        x_groups = []
        x_groups.append(x)
        x_groups = self.cat_msg2cluster_group(x_groups,msg_tokens) 
        trandata_ppo = (msg_cls, x_groups, self.msg_tokens_num) 
        for i in range(len(self.basedlayers)):
            trandata_ppo = self.basedlayers[i](trandata_ppo,mask_ratio=0)
        
        
        msg_cls, x_groups, _ = trandata_ppo 
        msg_token = x_groups[0][:,:,0,:]

        memory.msg_states.append(x_groups[0][:,:,0,:]) 
        memory.merge_msg_states.append(x_groups[0][:,:,0,:]) 
        
        FusionHisF.SFFR(trandata_ppo,memory) 
        msg_token = memory.merge_msg_states[-1].view(1,self.args.embed_dim)

        results_dict = self.head(msg_token)
        
        
        cl_logits = self.msg_predictor(msg_token)
       
        memory.results_dict.append(results_dict)  
       
        return results_dict, trandata_ppo, memory ,cl_logits
        
        
 