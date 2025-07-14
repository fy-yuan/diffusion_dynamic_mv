
import numpy as np
import torch
from torch.optim import Adam
import copy
import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def update_multiplier_td3(ac, wealth, multiplier, risk_aversion, initial_feature=None):
    
    #print(f"Type of multipier: {type(multiplier)}") #debugging 
          
    wealth =torch.full_like(initial_feature[..., :1], wealth)
    #risk_aversion = torch.full_like(initial_feature[..., :1], risk_aversion)
    multiplier = torch.full_like(initial_feature[..., :1], multiplier, requires_grad=True)
    multiplier_optimizer = Adam([multiplier], lr=3e-2)
    
    obs = torch.cat([initial_feature, wealth, multiplier], dim=-1)
    multiplier_optimizer.zero_grad()
    loss = risk_aversion * ac.q1(0, obs, ac.pi(0, obs))/2 - multiplier
    loss.backward()
    multiplier_optimizer.step()
    #print(f"Shape of multiplier: {multiplier.shape}") #debugging  
    return torch.squeeze(multiplier,dim=-1).item()


def test_agent_td3(test_scenarios,  T, num_test_episodes,num_assets, ac, risk_aversion, test_frequency, multiplier_update_steps=50):
        return_list= np.zeros(num_test_episodes)
        volatility_list= np.zeros(num_test_episodes)
        sharpe_list= np.zeros(num_test_episodes)
        portfolio_array = np.zeros((num_test_episodes, T-1, num_assets))
        wealth_list = np.zeros((num_test_episodes, T))
        years = (T-1) // test_frequency
        #gain_list = np.zeros((years,num_test_episodes))
        return_list_in_episode = np.zeros((years,num_test_episodes)) 
        for j in range(num_test_episodes):
            
            test_multiplier = 2
            
            scenario = test_scenarios.get_scenario(j)
            price, feature = scenario['price'], scenario['feature']
            #initial_feature = feature[0][0]
            price = torch.squeeze(price, dim=1).numpy()
            
    

            wealth = 1 
            wealth_log = 1
            wealth_list[j,0] = wealth

            for time in range(T-1):
                #normalize the feature

                feature_normalize = feature[time] * feature[0] / feature[(time // test_frequency) * test_frequency]
                #print(f"Initial of years: {(time // test_frequency) * test_frequency}, feature[{time}]: {feature[time]}, feature_normalize: {feature_normalize}") #debugging
                wealth_expand = torch.full_like(feature[0][..., :1], wealth)
                if time % test_frequency == 0:  
                    for i in range(multiplier_update_steps):
                       test_multiplier = update_multiplier_td3(ac, wealth, test_multiplier, risk_aversion, feature[0][0])
                       #if j == num_test_episodes-1:
                           #print(f"Update steps: {i}, Test Multiplier: {test_multiplier}") #debugging
                    multiplier_expand = torch.full_like(feature[0][..., :1], test_multiplier)


                obs = torch.cat([feature_normalize, wealth_expand, multiplier_expand], dim=-1)
                action = ac.act(time % test_frequency, obs)
                portfolio_array[j, time,:] = action
                action_shares = wealth * action / price[time]
                wealth = wealth + np.dot(action_shares, (price[time+1] - price[time]))
                wealth_list[j,time+1] = wealth
                if (time+1) % test_frequency == 0:
                    return_list_in_episode[(time+1) // test_frequency - 1, j] = wealth/wealth_log - 1
                    #gain_list[(time+1) // test_frequency - 1, j] = wealth - wealth_log
                    wealth_log = copy.copy(wealth)
            return_list[j] = return_list_in_episode[:,j].mean()
            volatility_list[j] = return_list_in_episode[:,j].std()
            sharpe_list[j] = return_list[j] / volatility_list[j] 
            #if j == num_test_episodes-1:
                #print(f"Result of the last test episode (TD3), Relative Portfolio: {action}")
        mv_list = np.zeros(years)
        for i in range(years):
            mv_list[i] = return_list_in_episode[i,:].mean() - risk_aversion*return_list_in_episode[i,:].std()**2/2
        print(f"The final output action: {action}") #debugging
        print(f"Final wealth:{wealth}, Test Multiplier: {test_multiplier}") #debugging
        print(f"Shape of feature: {feature.shape}") #debugging
        print(f"mv_list: {mv_list}") #debugging
        return return_list, volatility_list, sharpe_list, mv_list.mean(), wealth_list, portfolio_array



def test_agent_equal_weight(test_scenarios, T, num_test_episodes,num_assets, risk_aversion, test_frequency):
        return_list= np.zeros(num_test_episodes)
        volatility_list= np.zeros(num_test_episodes)
        sharpe_list= np.zeros(num_test_episodes)
        wealth_list = np.zeros((num_test_episodes, T))
        years = (T-1) // test_frequency
        #gain_list = np.zeros((years,num_test_episodes))
        return_list_in_episode = np.zeros((years,num_test_episodes))
        for j in range(num_test_episodes):
            
            
            scenario = test_scenarios.get_scenario(j)
            price= scenario['price']

            price = torch.squeeze(price, dim=1).numpy()   
            wealth = 1
            wealth_log = 1
            wealth_list[j,0] = wealth

            for time in range(T-1):                
                action = (wealth/num_assets) / price[time]
                #print(f"Test Portfolio: {action}") #debugging
                wealth = wealth + np.dot(action, (price[time+1] - price[time]))
                wealth_list[j,time+1] = wealth
                if (time+1) % test_frequency == 0:
                    return_list_in_episode[(time+1) // test_frequency - 1,j] = wealth/wealth_log - 1
                    #gain_list[(time+1) // test_frequency - 1, j] = wealth - wealth_log
                    wealth_log = copy.copy(wealth)
            return_list[j] = return_list_in_episode[:,j].mean()
            volatility_list[j] = return_list_in_episode[:,j].std()
            sharpe_list[j] = return_list[j] / volatility_list[j]
            #if j == num_test_episodes-1:
            #    print(f"Result of the last test episode (Equal Weight), Return list: {return_list}, Return list in the last episode: {return_list_in_episode}, Test Wealth: {wealth}")
        mv_list = np.zeros(years)
        for i in range(years):
            mv_list[i] = return_list_in_episode[i,:].mean() - risk_aversion*return_list_in_episode[i,:].std()**2/2
        #print(f"MV in each year: {mv_list}")
        return return_list, volatility_list, sharpe_list, mv_list.mean(), wealth_list


def test_agent_static(test_scenarios, T, num_test_episodes,num_assets, risk_aversion, test_frequency,action):
        return_list= np.zeros(num_test_episodes)
        volatility_list= np.zeros(num_test_episodes)
        sharpe_list= np.zeros(num_test_episodes)
        years = (T-1) // test_frequency
        wealth_list = np.zeros((num_test_episodes, T))
        #gain_list = np.zeros((years,num_test_episodes))  
        # Use QP solver to solve the Markowitz problem with no-short-selling constraints
        return_list_in_episode = np.zeros((years,num_test_episodes))

        for j in range(num_test_episodes):
            
            scenario = test_scenarios.get_scenario(j)
            price, feature = scenario['price'], scenario['feature']
            price = torch.squeeze(price, dim=1).numpy()   
            wealth = 1
            wealth_log = 1
            wealth_list[j,0] = wealth
            action_shares = action*wealth_log/price[0]
            for time in range(T-1):                
                    wealth = wealth + np.dot(action_shares, (price[time+1] - price[time]))
                    wealth_list[j,time+1] = wealth
                    if (time+1) % test_frequency == 0:
                      return_list_in_episode[(time+1) // test_frequency - 1,j] = wealth/wealth_log - 1
                    #gain_list[y, j] = wealth - wealth_log
                      wealth_log = copy.copy(wealth)
                      action_shares = action*wealth_log/price[time+1]
            return_list[j] = return_list_in_episode[:,j].mean()
            volatility_list[j] = return_list_in_episode[:,j].std()
            sharpe_list[j] = return_list[j] / volatility_list[j]
           
           
        mv_list = np.zeros(years)
        for i in range(years):
            mv_list[i] = return_list_in_episode[i,:].mean() - risk_aversion*return_list_in_episode[i,:].std()**2/2
        #print(f"MV in each year: {mv_list}")
        return return_list, volatility_list, sharpe_list, mv_list.mean(),wealth_list

def solve_markowitz(mean, cov, risk_aversion,num_assets):
    x = cp.Variable(num_assets)
    objective = cp.Minimize(-mean @ x + (risk_aversion /2) * cp.quad_form(x, cov))
    constraints = [x >= 0, cp.sum(x) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return x.value


def plot_for_each_type(type_name, algo_list, data_dic):
    plt.figure(figsize=(10, 6))
    #plt.title(f"Performance by Algorithm ({type_name})")
    plt.xlabel("Epoch")
    plt.ylabel(f"{type_name}")
    linestyles = ['-', '--', '-.', ':']
    cycler = itertools.cycle(linestyles)



    for algo in algo_list:
        #stats = calculate_stats(epochs_data)
        num_epochs = len(data_dic[algo, type_name])
        means = [data_dic[algo, type_name][e].mean() for e in range(num_epochs)]
        x = np.arange(num_epochs)  
        
        # Plot mean
        line = plt.plot(x, means, label=algo, linestyle = next(cycler), linewidth=2)
        
        # Fill confidence interval
        if type_name != 'MV':
            num_test = len(data_dic[algo, type_name][0])
            #lower_bounds = [np.percentile(data_dic[algo, type_name][e], 5) for e in range(num_epochs)]
            #upper_bounds = [np.percentile(data_dic[algo, type_name][e], 95) for e in range(num_epochs)]
            lower_bounds = [data_dic[algo, type_name][e].mean() - 1.96*data_dic[algo, type_name][e].std()/np.sqrt(num_test) for e in range(num_epochs)]
            upper_bounds = [data_dic[algo, type_name][e].mean() + 1.96*data_dic[algo, type_name][e].std()/np.sqrt(num_test) for e in range(num_epochs)]
            plt.fill_between(
            x, 
            np.array(lower_bounds),
            np.array(upper_bounds),
            alpha=0.2,
            color=line[0].get_color()
            )
    xticklocations = np.arange(10, num_epochs+1, 10)
    plt.xticks(xticklocations, range(10 , num_epochs+1)[::10])

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def present_training_log(algo_list, type_list, data_dic):   
    for type_name in type_list:
        plot_for_each_type(type_name, algo_list, data_dic)


def plot_portfolio(portfolio_array, T,num_assets):
    #portfolio_array = np.zeros((T-1,num_assets))
    cols = [f"Asset {i+1}" for i in range(num_assets)]
    portfolio_df = pd.DataFrame(portfolio_array, index=range(1, T),
                            columns=cols)
    
    ax = portfolio_df.plot.area(stacked=True,figsize=(10, 6))
    ax.set_xlabel('Time')
    ax.set_ylabel('Proportion')
    #ax.set_title('Portfolio Composition Over Time')

    threshold = 0.01              
    means = portfolio_df.mean(axis=0)
    cumulative = portfolio_df.cumsum(axis=1)

    for asset in cols:
      if means[asset] >= threshold:
        t_star = max(2,portfolio_df[asset].idxmax())                       
        y_mid  = cumulative.loc[t_star, asset] - portfolio_df.loc[t_star, asset] / 2
        ax.text(t_star, y_mid, asset, ha="center", va="center",
                fontsize=8, fontweight="bold")

    if ax.get_legend(): ax.get_legend().remove()



    plt.tight_layout()
    plt.show()



def plot_wealth_trajectory(algo_list, wealth_data):
    plt.figure(figsize=(10, 6))
    #plt.title(f"Performance by Algorithm ({type_name})")
    plt.xlabel("Time")
    plt.ylabel("Wealth")
    linestyles = ['-', '--', '-.', ':']
    cycler = itertools.cycle(linestyles)
  


    for algo in algo_list:
        #stats = calculate_stats(epochs_data)
        T = len(wealth_data[algo][0])
        means = [wealth_data[algo][0,t].mean() for t in range(T)]
        x = np.arange(T)  
        
        # Plot mean
        line = plt.plot(x, means, label=algo, linestyle = next(cycler), linewidth=2)
        
        # Fill confidence interval
        
        #num_test = len(wealth_data[algo][:,0])
            #lower_bounds = [np.percentile(data_dic[algo, type_name][e], 5) for e in range(num_epochs)]
            #upper_bounds = [np.percentile(data_dic[algo, type_name][e], 95) for e in range(num_epochs)]
        #lower_bounds = [wealth_data[algo][:,t].mean() - 1.96*wealth_data[algo][:,t].std()/np.sqrt(num_test) for t in range(T)]
        #upper_bounds = [wealth_data[algo][:,t].mean() + 1.96*wealth_data[algo][:,t].std()/np.sqrt(num_test) for t in range(T)]
        #plt.fill_between(
        #    x, 
        #    np.array(lower_bounds),
        #    np.array(upper_bounds),
        #    alpha=0.2,
        #    color=line[0].get_color()
        #    )
    xticklocations = np.arange(0, T, 10)
    plt.xticks(xticklocations, range(0 , T)[::10])

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()