
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn

 
        

class Cross_Attention(nn.Module):
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

class FusionHistoryFeatures(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1024):
        super(FusionHistoryFeatures, self).__init__()

        self.hidden_state_dim = hidden_state_dim
        self.feature_dim = feature_dim
        self.merge_catmsg_selfatten = Cross_Attention(dim=state_dim)

    def forward(self):
        raise NotImplementedError
    
    def SFFR(self, state_ini, memory):
        msg_cls, x_groups, msg_tokens_num = state_ini
        msg_state = x_groups[0][:,:,0:1].squeeze(dim=0)
        old_msg_state = torch.stack(memory.msg_states[:], dim=1).view(1,-1,512) 
        msg_state = self.merge_catmsg_selfatten(old_msg_state)
        memory.merge_msg_states[-1] = msg_state[:, -1:, :] 
        
       

       
        
        
class PrototypeNetwork(nn.Module):
    def __init__(self, feature_dim, num_prototypes):
        super(PrototypeNetwork, self).__init__()
        self.prototype_layer = nn.Linear(feature_dim, num_prototypes, bias=False)
    
    def forward(self, x):
        x=  torch.stack(x[:], dim=1).view(-1,512).to(x[1].device)
        proto_c = self.prototype_layer(x)
        return F.normalize(proto_c, p=2, dim=1)



def sinkhorn(out):
    Q = torch.exp(out / 0.05).T
    Q = Q  / torch.sum(Q)

    K, B = Q.shape

    u = torch.zeros(K, dtype=torch.float32).to(out.device)
    r = torch.ones(K, dtype=torch.float32).to(out.device) / K
    c = torch.ones(B, dtype=torch.float32).to(out.device) / B

    for _ in range(3):
        u = torch.sum(Q, dim=1)
        Q = Q * (r / u).unsqueeze(1)
        Q = Q * (c / torch.sum(Q, dim=0)).unsqueeze(0)

    final_quantity = Q / torch.sum(Q, dim=0, keepdims=True)
    final_quantity = final_quantity.T
    return final_quantity



def info_nce_loss(features, window_size=1, temperature=0.1):
    """
    Args:
        features (Tensor): Anchor features, assumed to have a size of [N, D].
        window_size (int): The size of the positive sample window.
        temperature (float): Scaling factor.
    """
    N, D = features.size()
    
    norm_features = F.normalize(features, p=2, dim=1)
    all_similarities = torch.matmul(norm_features, norm_features.T) / temperature  # [N, N]

    total_loss = 0.0
    for index in range(N):
        positive_indices = list(range(max(0, index - window_size), min(N, index + window_size + 1)))
        positive_indices.remove(index)
        negative_indices = list(set(range(N)) - set(positive_indices) - {index})
        positive_sim = all_similarities[index, positive_indices]
        negative_sim = all_similarities[index, negative_indices]
        logits = torch.cat([positive_sim.unsqueeze(0), negative_sim.unsqueeze(0)], dim=1)
        labels = torch.zeros(1, dtype=torch.long, device=features.device)  

        loss = F.cross_entropy(logits, labels)
        total_loss += loss

    return total_loss / N
