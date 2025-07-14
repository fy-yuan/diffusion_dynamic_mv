import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import copy

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

#def mlp(sizes, activation, output_activation=nn.Identity):
#    layers = []
#    for j in range(len(sizes)-1):
#        act = activation if j < len(sizes)-2 else output_activation
#        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
#    return nn.Sequential(*layers)

def mlp(sizes, activation, output_activation=nn.Identity, output_activation_kwargs=None):
    layers = []
    output_activation_kwargs = output_activation_kwargs or {}
    for j in range(len(sizes)-1):
        if j < len(sizes)-2:
            act = activation()
        else:
            act = output_activation(**output_activation_kwargs)
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

""" class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [obs_dim + 1] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, nn.Softmax(dim=-1))
        

    def forward(self, time, obs):
        time_expand = torch.full_like(obs[..., :1], time)
        act = self.pi(torch.cat([time_expand, obs], dim=-1))
        return act """


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,output_activation, output_activation_kwargs=None):
        super().__init__()
        pi_sizes = [obs_dim + 1] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation,output_activation=output_activation, output_activation_kwargs=output_activation_kwargs)
    
    def forward(self, time, obs):
        time_expand = torch.full_like(obs[..., :1], time)
        act = self.pi(torch.cat([time_expand, obs], dim=-1))
        return act

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim + 1] + list(hidden_sizes) + [1], activation)

    def forward(self, time, obs, act):
        time_expand = torch.full_like(obs[..., :1], time)
        q = self.q(torch.cat([time_expand, obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.


class MLPActorCritic_TD3(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,128,64)):
        super().__init__()


        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation = nn.ReLU,output_activation=nn.Softmax, output_activation_kwargs={'dim':-1})
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation = nn.ReLU)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation = nn.ReLU)

    def act(self, time, obs):
        with torch.no_grad():
            action = torch.squeeze(self.pi(time, obs),0).numpy()
            return action


class ReplayBuffer:
    """
    The experience buffer D. 
    """

    def __init__(self, act_dim, size):
        self.wealth_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.action_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.multiplier_buf = np.zeros(combined_shape(size, 1), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, wealth, action, multiplier):
        self.wealth_buf[self.ptr] = wealth
        self.action_buf[self.ptr] = action
        self.multiplier_buf[self.ptr] = multiplier
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(wealth=self.wealth_buf[idxs],
                     action=self.action_buf[idxs],
                     multiplier=self.multiplier_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}



class ScenarioPool:
    """
    The scenario pool S.
    """
  
    def __init__(self, price_dim, fea_dim, T, size):
        self.price = np.zeros(combined_shape(size, (T, price_dim)), dtype=np.float32)
        self.feature = np.zeros(combined_shape(size, (T, fea_dim)), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, price, feature):
        self.price[self.ptr] = price
        self.feature[self.ptr] = feature
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(price=self.price[idxs],
                     feature=self.feature[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).transpose_(0,1) for k,v in batch.items()}
    
    def get_scenario(self, idx):
        scenario = dict(price=self.price[[idx]],
                        feature=self.feature[[idx]])
        return {k: torch.as_tensor(v, dtype=torch.float32).transpose_(0,1) for k,v in scenario.items()}

 


class ScenarioGenerator:
    """The scenario generator. 

    Type: 
      simulate: simulate the price by s[t+1] = s[t]*(1+r[t]) where r are iid random variables
                get the feature simply by h[t]=s[t]
      generative: from the generative model (TBD)

    Output: A scenario pool that contains:
    price scenarios s each with size [price_dim,T] 
    and feature scenarios h each with size [fea_dim,T]


"""
    def __init__(self, price_dim, fea_dim, T, pool_size, type='simulate',mean=None,cov=None, inital_feature=None,mode = 'training'):
        self.price_dim = price_dim
        self.fea_dim = fea_dim
        self.T = T
        self.pool_size = pool_size
        self.type = type
        self.mean = mean
        self.cov = cov
        self.mode = mode
    def simulate(self):
        pool = ScenarioPool(self.price_dim, self.fea_dim, self.T, self.pool_size)
        if self.price_dim != self.fea_dim:
            raise NotImplementedError('price_dim should be equal to fea_dim if simulate')
        if self.mean is None or self.cov is None:
            raise NotImplementedError('mean and cov should be provided for simulate')
        s = np.zeros((self.T,self.price_dim))
        h = np.zeros((self.T,self.fea_dim))
        for i in range(self.pool_size):
            r = np.maximum(np.random.multivariate_normal(self.mean,self.cov,self.T), -1)
            
            if  self.mode == 'training':  
               s[0,:] = np.random.uniform(1,200,self.price_dim)
            elif self.mode == 'testing':
               s[0,:] = np.random.uniform(1,200,self.price_dim)
            else:
               raise NotImplementedError('Unknown mode')
            for t in range(1,self.T):
               s[t,:] = np.multiply(s[t-1,:],1+r[t])
            h = copy.copy(s)
            pool.store(s, h)
        return pool

    def generative(self):
        pass # sampling from generative model

    def get_data(self):
        if self.type == 'simulate':
           return self.simulate()
        elif self.type == 'generative':
           raise NotImplementedError('Generative model is not implemented yet')
        else:
           raise NotImplementedError('Unknown type')
        


def save_td3_checkpoint(actor, critic, episode, filename="td3_checkpoint.pth"):
    """
    Save model parameters and training state to a checkpoint file.
    """
    checkpoint = {
        'episode': episode,
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_td3_checkpoint(actor, critic, filename="td3_checkpoint.pth"):
    """
    Load model parameters and training state from a checkpoint file.
    """
    checkpoint = torch.load(filename)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    #actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    #critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    episode = checkpoint.get('episode', 0)
    print(f"Checkpoint loaded from {filename} (episode {episode})")

