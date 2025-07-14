from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import time
import core 
import test_utils
from utils.logx import EpochLogger
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR



def td3(num_assets, fea_dim, time_horizon, initial_feature, simulate_mean, simulate_cov,
         actor_critic=core.MLPActorCritic_TD3, seed=2025, 
         steps_per_epoch=200, epochs=150, pool_size = int(1e5), replay_size=int(1e5),
         polyak=0.99, pi_lr=1e-3, q_lr=1e-3, batch_size=16, start_steps=200, 
         update_after=50, update_every=50, policy_delay = 2, act_noise=0.1, 
         logger_kwargs=dict()):
    """
    Twin Delayed DDPG (TD3)

    """

    logger = EpochLogger(**logger_kwargs)
   #logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)


    obs_dim = fea_dim + 2 #observation = feature + wealth + multiplier(c)
    price_dim = num_assets
    act_dim = num_assets
    T = time_horizon
    
    total_steps = steps_per_epoch * epochs
    # Set up scenario generator and construct scenario pool
    # Sampling of scenarios is has been done here, outside the main loop

   

    generator = core.ScenarioGenerator(price_dim, fea_dim, time_horizon, pool_size, 'simulate', simulate_mean, simulate_cov,  initial_feature, mode = 'training')
    scenario_pool = generator.get_data()
            
    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, act_dim)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Experience buffer
    replay_buffer = core.ReplayBuffer(act_dim=act_dim, size=replay_size)
    
   
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'     %var_counts)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(experience, scenario, time):
        price, feature = scenario['price'], scenario['feature']
        wealth, action,multiplier = experience['wealth'], experience['action'], experience['multiplier']
        multiplier = multiplier.unsqueeze(0).expand(batch_size,-1,-1) # Random multiplier
        obs = torch.cat([feature[time], wealth], dim=-1)
        expanded_action = action.unsqueeze(1).expand(-1,batch_size,-1) # Expand action to match multiplier size
        expanded_obs = torch.cat([obs.unsqueeze(1).expand(-1,batch_size,-1), multiplier], dim=-1)
        expanded_wealth = wealth.unsqueeze(1).expand(-1,batch_size,-1) # Expand wealth to match multiplier size


        q1 = ac.q1(time, expanded_obs, expanded_action)
        q2 = ac.q2(time, expanded_obs, expanded_action)
        #print(f"Shape of q: {q.shape}") #debugging

        if time <= T-2:
           returns = (price[time+1] - price[time]) / price[time]  # returns
           wealth_next = wealth * (1 + torch.sum(action * returns, dim=-1, keepdim=True))
           #wealth_next = wealth + torch.sum(action * (price[time+1] - price[time]), dim = -1, keepdim=True)
           obs_next = torch.cat([feature[time+1], wealth_next], dim=-1)
           expanded_obs_next = torch.cat([obs_next.unsqueeze(1).expand(-1,batch_size,-1), multiplier], dim=-1)

           # Bellman backup for Q function
           with torch.no_grad():
               q1_pi_targ = ac_targ.q1(time+1, expanded_obs_next, ac_targ.pi(time+1, expanded_obs_next))
               q2_pi_targ = ac_targ.q2(time+1, expanded_obs_next, ac_targ.pi(time+1, expanded_obs_next))
        
        else:
           with torch.no_grad():
               q1_pi_targ = (expanded_wealth - multiplier).squeeze(-1)
               q1_pi_targ = q1_pi_targ**2 
               q2_pi_targ = (expanded_wealth - multiplier).squeeze(-1)
               q2_pi_targ = q2_pi_targ**2
               #print(f"Shape of q_pi_targ: {q_pi_targ.shape}") #debugging              

        # MSE loss against Bellman backup
        q_pi_targ = torch.max(q1_pi_targ, q2_pi_targ)
        loss_q1 = ((q1 - q_pi_targ)**2).mean()
        loss_q2 = ((q2 - q_pi_targ)**2).mean()
        loss_q = loss_q1 + loss_q2

        

        return loss_q

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(experience, scenario, time,steps):
        feature = scenario['feature']
        wealth, multiplier = experience['wealth'], experience['multiplier']
        multiplier = multiplier.unsqueeze(0).expand(batch_size,-1,-1) # Random multiplier
        obs = torch.cat([feature[time], wealth], dim=-1)
        expanded_obs = torch.cat([obs.unsqueeze(1).expand(-1,batch_size,-1), multiplier], dim=-1)
        action = ac.pi(time, expanded_obs)
        #q_pi = ac.q1(time, expanded_obs, action) + 0.05*torch.sum(action * action, dim=-1, keepdim=True)
        if steps <= epochs*steps_per_epoch*0.75:
           q_pi = ac.q1(time, expanded_obs, action) + 0.07*torch.sum(action * action, dim=-1, keepdim=True)
        else:
           q_pi = ac.q1(time, expanded_obs, action) + 0.05*torch.sum(action * action, dim=-1, keepdim=True)
        return q_pi.mean()
    

    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)
    

    # Learning rate schedulers 
    pi_scheduler = CosineAnnealingLR(pi_optimizer, T_max=total_steps, eta_min=1e-6)
    q_scheduler = CosineAnnealingLR(q_optimizer, T_max=total_steps, eta_min=1e-6)
  

    def update(experience, scenario, time, update_flag,steps):
        # First run one gradient descent step for Q.
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(experience, scenario, time)
        loss_q.backward()
        q_optimizer.step()
        

         # Record QLoss (only record terminal time step)
        #if time == T-1:
        logger.store(LossQ=loss_q.item())
        
        

        # Delayed update for policy network
        if update_flag % policy_delay == 0 & time<= T-2:
            # Freeze Q-network so you don't waste computational effort 
            # computing gradients for it during the policy learning step.
            for p in q_params:
                p.requires_grad = False
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(experience, scenario, time,steps)
            loss_pi.backward()
            pi_optimizer.step()
            # Unfreeze Q-network so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True
        
            # Record PiLoss
            logger.store(LossPi=loss_pi.item())
        

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    """ def get_action(time, obs, noise_scale, epochs):
         a = ac.act(time, obs)
         noise = np.clip(noise_scale * np.random.randn(act_dim)/epochs, -0.5, 0.5)
         a = np.clip(a + noise, 1e-6, None)  # Avoid 0
         a = a / np.sum(a)  # normalize
         return a """
    
    def get_action(time, obs, noise_scale):
        a = ac.act(time, obs)  # output action
        baseline = 0.01
        alpha =  1.0 / (noise_scale + 1e-6)  # exploration rate
        dirichlet_para = baseline + alpha * a  # dirichlet parameter
        noisy_action = np.random.dirichlet(dirichlet_para)  # noisy action
        return noisy_action

    
    start_time = time.time()
    epoch_list = []
    


    # Main loop 
    for steps in range(total_steps):
        
      #initialization in each step 
      scenario = scenario_pool.sample_batch(1)
      price, feature = scenario['price'], scenario['feature']
      price = torch.squeeze(price, dim=1).numpy()
      wealth = np.random.uniform(0,10)
      #multiplier = np.random.exponential(2)
      multiplier = np.random.uniform(0, 10)
    # The loop for time_horizon
      for t in range(T):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if steps > start_steps:
            wealth_expand = torch.full_like(feature[0][..., :1], wealth)
            multiplier_expand = torch.full_like(feature[0][..., :1], multiplier)
            obs = torch.cat([feature[t], wealth_expand, multiplier_expand], dim=-1)
            a = get_action(t, obs, act_noise)
        else:
            a = np.random.dirichlet(np.ones(act_dim),size=1)
        # Store observations into D
        replay_buffer.store(wealth, a, multiplier)
        if t <= T-2:
           a = wealth * a / price[t]
           wealth = wealth + np.dot(a , (price[t+1] - price[t]))


        # Update handling
        if steps >= update_after and (steps+1) % update_every == 0:
            for j in range(update_every):
                experience_batch = replay_buffer.sample_batch(batch_size)
                scenario_batch = scenario_pool.sample_batch(batch_size)
                update(experience_batch, scenario_batch, t, j,steps)
    
         # Learning rate scheduler step
       

  
     
    # End of epoch handling and logging the test results
    
      if steps >= start_steps and (steps+1) % steps_per_epoch == 0:
            epoch = (steps+1) // steps_per_epoch
            epoch_list.append(epoch)
            
            core.save_td3_checkpoint(ac.pi, ac.q1, epoch, filename=f"checkpoints/td3_ep{epoch}.pth")

            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
            current_pi_lr = pi_optimizer.param_groups[0]['lr']
            current_q_lr = q_optimizer.param_groups[0]['lr']
            print(f"Step {steps}: pi_lr={current_pi_lr:.6f}, q_lr={current_q_lr:.6f}")
      
      pi_scheduler.step()
      q_scheduler.step()

    return ac, epoch_list





