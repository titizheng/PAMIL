import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn

class Memory:
    def __init__(self):
        # action 
        self.actions = [] 
        self.coords_actions = [] 
        self.logprobs = [] 
        
        self.rewards = []
        self.is_terminals = []
        self.hidden = []
        
        #state
        # self.origin_states = []  
        self.msg_states = [] 
        self.cls_states = []  
        # self.action_states = []  
        self.merge_msg_states = [] 
        
        self.results_dict = []

        

    def clear_memory(self):
       
        del self.actions[:]
        del self.coords_actions[:]
        del self.logprobs[:]
        
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]
        
        # del self.action_states[:]
        # del self.origin_states[:]
        del self.msg_states[:]
        del self.cls_states[:]
        del self.merge_msg_states[:]
        
        
        del self.results_dict[:]
        
        
 


 

class Cat_Attention(nn.Module):
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

class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1024, policy_conv=False, action_std=0.1, action_size=2):
        super(ActorCritic, self).__init__()

        
        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))
        
        self.merge_catmsg_selfatten = Cat_Attention(dim=state_dim)
        

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)

        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, action_size),
            nn.Sigmoid())

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.action_var = torch.full((action_size,), action_std).cuda()

    def forward(self):
        raise NotImplementedError
    
    def process_state_before_act(self, state_ini, memory, restart_batch=False, training=False):
        msg_cls, x_groups, msg_tokens_num = state_ini

        msg_state = x_groups[0][:,:,0:1].squeeze(dim=0).detach()
        
        
        old_msg_state = torch.stack(memory.msg_states[:], dim=1).view(1,-1,512).detach() 
        msg_state = self.merge_catmsg_selfatten(old_msg_state)
        memory.merge_msg_states[-1] = msg_state[:, -1:, :] 
        return msg_state[:, -1:, :],memory
        
        
        
    def act(self, current_state, memory, restart_batch=False, training=False):
        state_ini = memory.merge_msg_states[-1].detach()
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        msg_state, hidden_output = self.gru(state_ini.view(1, state_ini.size(0), state_ini.size(-1)), memory.hidden[-1])  
        memory.hidden.append(hidden_output) 

        action_mean = self.actor(msg_state[0]) 

        cov_mat = torch.diag(self.action_var).cuda() 
        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)
        action = dist.sample().cuda() 
        # if training:
        action = F.relu(action)
        action = 1 - F.relu(1 - action)
        action_logprob = dist.log_prob(action).cuda()
        
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        # else:
        #     action = action_mean

        return action
    
    


    def evaluate(self, state, action):
        seq_l = state.size(0) 
        batch_size = state.size(1)
        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).cuda()) 
        state = state.view(seq_l * batch_size, -1) 

        action_mean = self.actor(state) 

        cov_mat = torch.diag(self.action_var).cuda()

        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda() #
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2, action_size=2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.bagsize = state_dim 

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std, action_size).cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std, action_size).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, data, memory, restart_batch=False, training=True):
        return self.policy_old.act(data, memory, restart_batch, training)

    
    def select_features(self, idx, features,len_now_coords):
        index = idx
        features_group = []
        for i in range(len(index)):
            member_size = (index[i].size)
            if member_size > self.max_size: 
                index[i] = np.random.choice(index[i],size=self.max_size,replace=False)
            temp = features[index[i]]
            temp = temp.unsqueeze(dim=0) 
            features_group.append(temp)
        return features_group

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_msg_states = torch.stack(memory.merge_msg_states, 0).cuda().detach() 


        old_actions = torch.stack(memory.actions[1:], 0).cuda().detach() 
        old_logprobs = torch.stack(memory.logprobs[1:], 0).cuda().detach() 

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_msg_states, old_actions)
            rewards = rewards.view(-1,1)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


