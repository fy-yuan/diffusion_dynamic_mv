from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from gluonts.core.component import validated
from gluonts.torch.distributions.distribution_output import DistributionOutput

from DiffusionModel.score_fn import UNet1DSameRes
from DiffusionModel.score_fn_res import EpsilonDilatedTCN


from DiffusionModel.sde_lib import VPSDE, subVPSDE, VESDE
from DiffusionModel.losses import get_loss_fn
from DiffusionModel.sampling import get_sampling_fn



class score_fn_output(DistributionOutput):
    @validated()
    def __init__(self, diffusion, input_size, cond_size):
        self.args_dim = {"cond": cond_size}
        self.diffusion = diffusion
        self.dim = input_size

    @classmethod
    def domain_map(cls, cond):
        return (cond,)

    def distribution(self, distr_args, scale=None):
        (cond,) = distr_args
        if scale is not None:
            self.diffusion.scale = scale
        self.diffusion.cond = cond

        return self.diffusion

    @property
    def event_shape(self) -> Tuple:
        return (self.dim,)
    


class TrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        dropout_rate: float,
        target_dim: int,
        diff_steps: int,
        prediction_length: int,
        beta_min: float,
        beta_end: float,
        scale_c: float,
        scaling: bool = True,
        md_type='vpsde',
        continuous=True,
        reduce_mean=True,
        likelihood_weighting=False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.feature_dim = num_cells
    
        self.scaling = scaling
        self.continuous = continuous
        self.input_size = input_size
        self.reduce_mean = reduce_mean
        self.likelihood_weighting = likelihood_weighting

        self.prediction_length = prediction_length
        self.scale_c = scale_c
     

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        
        #self.model = EpsilonTheta(in_channels=1, cond_channels=1, depth=3, base_channels=16)
        self.model = UNet1DSameRes(in_channels=self.prediction_length, cond_channels=1, depth=3, base_channels=16)

        if md_type == 'vpsde':
            self.sde = VPSDE(beta_min=beta_min, beta_max=beta_end, N=diff_steps)
            self.sampling_eps = 1e-3
        elif md_type == 'subvpsde':
            self.sde = subVPSDE(beta_min=beta_min, beta_max=beta_end, N=diff_steps)
            self.sampling_eps = 1e-3
        elif md_type == 'vesde':
            self.sde = VESDE(sigma_min=beta_min, sigma_max=beta_end, N=diff_steps)
            self.sampling_eps = 1e-5


        self.loss_fn = get_loss_fn(self.sde, reduce_mean=self.reduce_mean, continuous=self.continuous,
                              likelihood_weighting=self.likelihood_weighting)


    def forward(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Computes the loss for training DeepVAR, all inputs tensors representing
        time series have NTC layout.

        Parameters
        ----------
        target_dimension_indicator
            Indices of the target dimension (batch_size, target_dim)
        past_time_feat
            Dynamic features of past time series (batch_size, history_length,
            num_features)
        past_target_cdf
            Past marginal CDF transformed target values (batch_size,
            history_length, target_dim)
        past_observed_values
            Indicator whether or not the values were observed (batch_size,
            history_length, target_dim)
        past_is_pad
            Indicator whether the past target values have been padded
            (batch_size, history_length)
        future_time_feat
            Future time features (batch_size, prediction_length, num_features)
        future_target_cdf
            Future marginal CDF transformed target values (batch_size,
            prediction_length, target_dim)
        future_observed_values
            Indicator whether or not the future values were observed
            (batch_size, prediction_length, target_dim)

        Returns
        -------
        distr
            Loss with shape (batch_size, 1)
        likelihoods
            Likelihoods for each time step
            (batch_size, context + prediction_length, 1)
        distr_args
            Distribution arguments (context + prediction_length,
            number_of_arguments)
        """

        rnn_outputs, _ = self.rnn(data)
        rnn_outputs  = rnn_outputs*self.scale_c

        B, T, _ , k = target.shape

        likelihoods = self.loss_fn(self.model, target.reshape(B*T,self.prediction_length,-1), rnn_outputs.reshape(B*T,1,-1)).unsqueeze(-1)

 

        return (torch.mean(likelihoods), likelihoods)
    

    def sample_offline(self, config, T_forcast, num_samples, X_data, compare = True):

        """
        Generates samples from the model.

        Parameters
        ----------
        sampling_shape
            Shape of the samples to generate
        history
            Historical data for conditioning
        prediction_length
            Length of the prediction horizon

        Returns
        -------
        samples
            Generated samples
        """
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        sampling_shape = [num_samples, self.prediction_length, self.target_dim]
        sampling_fn = get_sampling_fn(config, self.sde, sampling_shape, eps=self.sampling_eps) 
        predictions = torch.zeros((T_forcast+1, num_samples, self.target_dim)).to(device)
        actual = np.zeros((T_forcast+1, self.target_dim))
        features = np.zeros((T_forcast+1, num_samples, self.feature_dim))
        
        


        predictions[0,...] = torch.ones((num_samples,self.target_dim)).to(device)
        actual[0,...] = np.ones(self.target_dim)
        

        with torch.no_grad():
            current_context = X_data[0].unsqueeze(0).repeat(num_samples,1,1).to(device)
            rnn_outputs, _ = self.rnn(current_context)
            features[0,...] = rnn_outputs[:,-1,:].cpu().numpy()*self.scale_c

            for i in range(T_forcast):

                print(f"Sampling timestep {i+1}/{T_forcast}")
                #print("Current context for the first stock", current_context[0,:,0])
                #current_context = X_tensor[start-24+i].unsqueeze(0).repeat(num_samples,1,1).to(device)
                #print(current_context)
                rnn_outputs, _ = self.rnn(current_context)
                #print("Shape of rnn_outputs", rnn_outputs.shape)
                new_samples, _ = sampling_fn(self.model,rnn_outputs[:,-1:,:]*self.scale_c)
                features[i+1,:,:] = rnn_outputs[:,-1,:].cpu().numpy()
                if new_samples.shape[1]>=2:
                   est_next_day_return = torch.mean(new_samples[:, 0:2,:],dim = 1, keepdim = True)  
                else: 
                   est_next_day_return = new_samples[:, -1:,:]
                predictions[i+1,:,:] = est_next_day_return[:,0,:]
                current_context = torch.cat((current_context[:, 1:, :], est_next_day_return), dim=1)
            
            if compare:
                for i in range(T_forcast):
                    actual[i+1,:] = X_data[i+1,-1,:].cpu().numpy()
        


        return predictions.cpu().numpy(), actual, features


    def sample_online(self, config, T_forcast, num_samples, X_data, compare = True):


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval()
        sampling_shape = [num_samples, self.prediction_length, self.target_dim]
        sampling_fn = get_sampling_fn(config, self.sde, sampling_shape, eps=self.sampling_eps) 
        predictions = torch.zeros((T_forcast+1, num_samples, self.target_dim)).to(device)
        actual = np.zeros((T_forcast+1, self.target_dim))
        
        predictions[0,...] = torch.ones((num_samples,self.target_dim)).to(device)
        actual[0,...] = np.ones(self.target_dim)
    

        with torch.no_grad():
            
       
            for i in range(T_forcast):
                current_context = X_data[i].unsqueeze(0).repeat(num_samples,1,1).to(device)
                print(f"Sampling timestep {i+1}/{T_forcast}")
                rnn_outputs, _ = self.rnn(current_context)
                new_samples, _ = sampling_fn(self.model,rnn_outputs[:,-1:,:]*self.scale_c)
                if new_samples.shape[1]>=2:
                   est_next_day_return = torch.mean(new_samples[:, 0:2,:],dim = 1)  
                else: 
                   est_next_day_return = new_samples[:, 0,:]
                predictions[i+1,:,:] = est_next_day_return
                #current_context = torch.cat((current_context[:, 1:, :], new_prices), dim=1)
            
            if compare:
                for i in range(T_forcast):
                    actual[i+1,:] = X_data[i+1,-1,:].cpu().numpy()
        


        return predictions.cpu().numpy(), actual




    

    
