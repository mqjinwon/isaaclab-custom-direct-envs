from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation

class ActorCriticDWAQ(ActorCritic):
    def __init__(
        self, 
        num_actor_obs, 
        num_critic_obs, 
        num_actions, 
        cenet_in_dim, 
        cenet_out_dim, 
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu", 
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs
    ):
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs
        )
        
        # DWAQ 특화 인코더-디코더 네트워크 구성
        activation_fn = resolve_nn_activation(activation)
        
        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            activation_fn,
            nn.Linear(128, 64),
            activation_fn,
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim-3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim-3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            activation_fn,
            nn.Linear(64, 128),
            activation_fn,
            nn.Linear(128, 45)
        )

    def reparameterise(self, mean, logvar):
        var = torch.exp(logvar * 0.5)
        code_temp = torch.randn_like(var)
        code = mean + var * code_temp
        return code
    
    def cenet_forward(self, obs_history):
        distribution = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(distribution)
        logvar_latent = self.encode_logvar_latent(distribution)
        
        mean_vel = self.encode_mean_vel(distribution)
        logvar_vel = self.encode_logvar_vel(distribution)
        
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    def act(self, observations, obs_history, **kwargs):
        code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
        observations = torch.cat((code, observations), dim=-1)
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations, obs_history):
        code, _, decode, _, _, _, _ = self.cenet_forward(obs_history)
        observations = torch.cat((code, observations), dim=-1)
        actions_mean = self.actor(observations)
        return actions_mean
