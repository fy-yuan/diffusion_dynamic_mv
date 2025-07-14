
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import SGD
import copy
import cvxpy as cp
import itertools
import matplotlib.pyplot as plt
import pandas as pd

def update_multiplier_td3(ac, wealth, multiplier, risk_aversion, m_lr = 3e-2, initial_feature=None):
    
    #print(f"Type of multipier: {type(multiplier)}") #debugging 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")      
    wealth =torch.full_like(initial_feature[..., :1], wealth)
    #risk_aversion = torch.full_like(initial_feature[..., :1], risk_aversion)
    multiplier = torch.full_like(initial_feature[..., :1], multiplier, requires_grad=True)
    multiplier_optimizer =  torch.optim.SGD([multiplier], lr=m_lr)
    multiplier_optimizer.zero_grad()
    
    # def closure():
    #   multiplier_optimizer.zero_grad()
    #   obs = torch.cat([initial_feature, wealth, multiplier], dim=-1)
    #   q1_out = ac.q1(0, obs, ac.pi(0, obs))
    #   loss = risk_aversion * q1_out / 2 - multiplier
    #   loss.backward()
    #   return loss
    
    # loss = multiplier_optimizer.step(closure)
    # print(f"  m = {multiplier.item():.4f}, loss = {loss.item():.4f}")
    
    
    obs = torch.cat([initial_feature, wealth, multiplier], dim=-1).to(device)
    q1_out = ac.q1(0, obs, ac.pi(0, obs))            
    loss = risk_aversion * q1_out / 2 - multiplier    

    loss.backward(retain_graph=True)
    
    

    multiplier_optimizer.step()

    return torch.squeeze(multiplier,dim=-1).item()


def test_agent_td3(test_scenarios,  T, num_test_episodes,num_assets, ac_array, risk_aversion, test_frequency,retrain_frequency, scale_t = 1, multiplier_update_steps=50,m_lr = 3e-2):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        portfolio_array = np.zeros((num_test_episodes, T-1, num_assets))
        wealth_list = np.zeros((num_test_episodes, T))
        years = (T-1) // test_frequency
        #gain_list = np.zeros((years,num_test_episodes))
        return_list_in_episode = np.zeros((T-1,num_test_episodes)) 
        for j in range(num_test_episodes):
            test_multiplier = 2
            
            scenario = test_scenarios.get_scenario(np.array([[j]]))
            price, feature = scenario['price'], scenario['feature']
            #initial_feature = feature[0][0]
            price = torch.squeeze(price, dim=1).cpu().numpy()
            
    

            wealth = 1 
            wealth_log = 1
            wealth_list[j,0] = wealth

            for time in range(T-1):
                #normalize the feature

                #feature_normalize = feature[time] * feature[0] / feature[(time // test_frequency) * test_frequency]
                #print(f"Initial of years: {(time // test_frequency) * test_frequency}, feature[{time}]: {feature[time]}, feature_normalize: {feature_normalize}") #debugging
                ac = ac_array[j][time//retrain_frequency]
                ac.pi.eval()
                ac.q1.eval()
                wealth_expand = torch.full_like(feature[0][..., :1], wealth)
                if time % test_frequency == 0:  
                    for i in range(multiplier_update_steps):
                       test_multiplier = update_multiplier_td3(ac, wealth, test_multiplier, risk_aversion, m_lr, feature[time])
                   
            
                       
                    multiplier_expand = torch.full_like(feature[0][..., :1], test_multiplier)
                    print('Time:', time, 'Optimal dual multiplier (hedging target):', test_multiplier)
                
                obs = torch.cat([feature[time], wealth_expand, multiplier_expand], dim=-1).to(device)
                action = ac.act(time%test_frequency, obs, scale_t)
                portfolio_array[j, time,:] = action
                action_shares = wealth * action / price[time]
                wealth = wealth + np.dot(action_shares, (price[time+1] - price[time]))
                wealth_list[j,time+1] = wealth
                return_list_in_episode[time, j] = wealth/wealth_log - 1
                    #gain_list[(time+1) // test_frequency - 1, j] = wealth - wealth_log
                wealth_log = copy.copy(wealth)

        return return_list_in_episode, portfolio_array



def test_agent_baseline(test_scenarios, T, num_test_episodes,num_assets, risk_aversion, test_frequency, action):

        wealth_list = np.zeros((num_test_episodes, T))
        years = (T-1) // test_frequency
        #gain_list = np.zeros((years,num_test_episodes))
        return_list_in_episode = np.zeros((T-1,num_test_episodes))
        for j in range(num_test_episodes):
            
            
            scenario = test_scenarios.get_scenario(np.array([[j]]))
            price= scenario['price']

            price = torch.squeeze(price, dim=1).cpu().numpy()   
            wealth = 1
            wealth_log = 1
            wealth_list[j,0] = wealth

            for time in range(T-1):                
                action_shares = action[j][time]*wealth / price[time]
                #print(f"Test Portfolio: {action}") #debugging
                wealth = wealth + np.dot(action_shares, (price[time+1] - price[time]))
                wealth_list[j,time+1] = wealth
                return_list_in_episode[time,j] = wealth/wealth_log - 1
                wealth_log = copy.copy(wealth)
    
        return return_list_in_episode



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


def plot_portfolio(portfolio_array, T, num_assets, dataset, test_dates):
    cols = dataset.index.get_level_values('Industry').unique().tolist()
    portfolio_df = pd.DataFrame(portfolio_array, index=test_dates[-2-T:-2].tolist(),
                            columns=cols)
    
    ax = portfolio_df.plot.area(stacked=True,figsize=(10, 6))
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion')

    # Remove original legend
    if ax.get_legend(): ax.get_legend().remove()

    # Create new legend on the side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Tickers")
    
    dates_for_plot = test_dates[-2-T:-2].tolist()
    tick_indices = np.arange(0, T, 12)
    tick_dates = [dates_for_plot[i] for i in tick_indices]
    plt.xticks(tick_indices, tick_dates, rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()

def plot_wealth_trajectory(algo_list, wealth_data, test_dates):
    plt.figure(figsize=(10, 6))
    plt.xlabel("Date")
    plt.ylabel("Wealth")
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    cycler = itertools.cycle(linestyles)

    for algo in algo_list:
        T = len(wealth_data[algo][:,0])
        means = [wealth_data[algo][t, :].mean() for t in range(T)]
        x = np.arange(T)

        line = plt.plot(x, means, label=algo, linestyle = next(cycler), linewidth=2)

    # Use the test dates for the x-axis
    dates_for_plot = test_dates[-1-T:-1].tolist()
    # Select a reasonable number of ticks, e.g., every 10 months
    tick_indices = np.arange(0, T, 12)
    tick_dates = [dates_for_plot[i] for i in tick_indices]

    plt.xticks(tick_indices, tick_dates, rotation=45, ha='right')


    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
def compute_metrics(r, test_frequency, years, rf=0.0):
    r = np.asarray(r)
    N = len(r)
    ann_return = (np.prod(1 + r) ** (1 / years)) - 1
    ann_vol    = r.std(ddof=1) * np.sqrt(test_frequency)
    sharpe     = (ann_return - rf) / ann_vol
    downside   = r[r < 0]
    sortino    = (ann_return - rf) / (np.sqrt((downside ** 2).mean()) * np.sqrt(test_frequency))
    wealth     = np.cumprod(1 + r)
    max_dd     = np.min(wealth / np.maximum.accumulate(wealth) - 1)
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else np.nan
    return {
        "Annual Return": ann_return,
        "Annual Vol":    ann_vol,
        "Sharpe":        sharpe,
        "Sortino":       sortino,
        "Max Drawdown":  max_dd,
        "Calmar":        calmar,
    }