import os
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

import pytorch_lightning as pl


def get_o6_compact_obs_mask(total_dim=1902):
    keep = torch.cat(
        [
            torch.arange(0, 17),
            torch.arange(51, 86),
            torch.arange(146, 152),
            torch.arange(152, 164),
            torch.arange(176, 183),
            torch.arange(222, total_dim),
        ]
    )
    mask = torch.zeros(total_dim, dtype=torch.bool)
    mask[keep] = True
    return mask


class ActorCriticDexRep(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg, encoder_only=False,**kwargs):
        super(ActorCriticDexRep, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        self.get_obs_process_mask(pro_dim=env_cfg['obs_dim']['prop'])
        self.allow_obs_auto_pad = bool(encoder_cfg.get("allow_obs_auto_pad", False))

        self.encoder_only=encoder_only
        # create BN
        self.bn_type = encoder_cfg["bn_type"]
        if encoder_cfg["bn_type"] == "part":
            self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_pnl'])
        elif encoder_cfg["bn_type"] == "full":
            self.bn_pnl = nn.BatchNorm1d(sum(self.obs_dim[1:]))
        elif encoder_cfg["bn_type"] == "null":
            self.bn_pnl = None
        else:
            raise NotImplementedError(f"bn_type not impleted")
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        self.dexrep_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor'], emb_dim)
        self.dexrep_pointL_enc = nn.Linear(env_cfg['obs_dim']['dexrep_pnl'], emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        if encoder_only==False:
            # Policy
            actor_layers = []
            actor_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, actor_hidden_dim[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dim)):
                if l == len(actor_hidden_dim) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actions_shape))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

            # Value function
            # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
            critic_layers = []

            # critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, critic_hidden_dim[0]))
            critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim*encoder_cfg['n_obs_steps'], critic_hidden_dim[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dim)):
                if l == len(critic_hidden_dim) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

            # Initialize the weights like in stable baselines
            actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
            actor_weights.append(0.01)
            critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
            critic_weights.append(1.0)
            self.init_weights(self.actor, actor_weights)
            self.init_weights(self.critic, critic_weights)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(actions_shape))

        # init link FC layers
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_sensor_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_pointL_enc.weight, gain=np.sqrt(2))

    def get_obs_process_mask(self,pro_dim=100):
        if pro_dim == 77:
            self.obs_mask = get_o6_compact_obs_mask(1902)
            return

        reshape84_149_part = torch.arange(84, 149).reshape(5, 13)  #
        remove_indice_part = reshape84_149_part[:, -6:].reshape(-1)  # fingertips pos&ang vel 30

        remmove28_56 = torch.arange(28, 56)  # shadow vel 28
        remove56_84 = torch.arange(56, 84)  # shadow force 28

        remove149_179 = torch.arange(149, 179)  # finger force 30
        remove216_222 = torch.arange(216, 222)  # obj pos&ang vel 6

        index_to_remove = torch.cat([remove56_84, remove149_179])
        if pro_dim == 134:
            final_indice_remove = torch.cat([ index_to_remove, remove_indice_part])
        elif pro_dim == 128:
            final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222])
        elif pro_dim == 100:
            final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222, remmove28_56])
        elif pro_dim == 222:
            final_indice_remove = torch.empty(0, dtype=torch.long)
        else:
            raise ValueError("unsupported pro_dim for ActorCriticDexRep: {}".format(pro_dim))

        all_indices = torch.arange(2582)
        self.obs_mask = ~torch.isin(all_indices, final_indice_remove)
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1

        total_expected_dim = sum(self.obs_dim)
        if x.size(-1) < total_expected_dim:
            if getattr(self, "allow_obs_auto_pad", False):
                pad_size = total_expected_dim - x.size(-1)
                padding = torch.zeros(*x.shape[:-1], pad_size, device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=-1)
            else:
                raise ValueError(
                    "ActorCriticDexRep expected obs dim {} from {}, got {}. "
                    "This usually means the O6 env/dataset is still using the 21-dim dummy obs "
                    "instead of DexRep features.".format(total_expected_dim, self.obs_dim, x.size(-1))
                )
        elif x.size(-1) > total_expected_dim:
            raise ValueError(
                "ActorCriticDexRep expected obs dim {} from {}, got {}".format(
                    total_expected_dim, self.obs_dim, x.size(-1)
                )
            )

        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]
        return x_list


    def encode(self, observations):

        if self.bn_type == "part":
            self.bn_pnl#eval()
            if observations.size()[-1] == self.obs_mask.numel():
                observations = observations[...,self.obs_mask]

            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl#.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        return joint_emb

    def forward(self, observations):

        joint_emb = self.encode(observations)
        actions_mean = self.actor(joint_emb)

        return actions_mean



    @torch.no_grad()
    def act(self, observations):
        if self.bn_type == "part":
            self.bn_pnl.eval()
            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        if self.bn_type == "part":
            self.bn_pnl.eval()
            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            self.bn_pnl.eval()
            observations[:, self.obs_dim[0]:] = self.bn_pnl(observations[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(observations)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):
        B, T, D = obs_features.size()

        if len(obs_features.size()) == 3:
            obs_features = obs_features.reshape(-1, D)

        if len(state.size()) == 3:
            state = state.reshape(B*T, -1)

        if self.bn_type == "part":
            self.bn_pnl.train()
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[..., :-self.obs_dim[-1]])
            dexrep_pnl = self.bn_pnl(obs_features[..., -self.obs_dim[-1]:])
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            a=1
        elif self.bn_type == "full":
            self.bn_pnl.train()
            obs_features_norm = self.bn_pnl(obs_features)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features_norm[:, :-self.obs_dim[-1]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features_norm[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[:, :-self.obs_dim[-1]])
            dexrep_pnl_emb = self.dexrep_pointL_enc(obs_features[:, -self.obs_dim[-1]:])
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

    def evaluate_(self, obs_features, state):
        B, T, D = obs_features.size()

        if len(obs_features.size()) == 3:
            obs_features = obs_features.reshape(-1, D)

        if len(state.size()) == 3:
            state = state.reshape(B*T, -1)

        if self.bn_type == "part":
            self.bn_pnl.train()
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(obs_features[..., :-self.obs_dim[-1]])
            dexrep_pnl = self.bn_pnl(obs_features[..., -self.obs_dim[-1]:])
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
            a=1

        joint_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        # value = self.critic(joint_emb)


        return joint_emb

        # actions_mean = self.actor(joint_emb)

        # covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        # distribution = MultivariateNormal(actions_mean, scale_tril=covariance)


        # actions_log_prob = distribution.log_prob(actions)
        # entropy = distribution.entropy()


        # return actions_log_prob, entropy, joint_emb, self.log_std.repeat(actions_mean.shape[0], 1)


class ActorCriticDexRepTemporal(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg, encoder_only=False, **kwargs):
        super(ActorCriticDexRepTemporal, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        self.get_obs_process_mask(pro_dim=env_cfg['obs_dim']['prop'])
        self.encoder_only = encoder_only
        self.n_obs_steps = int(encoder_cfg.get("n_obs_steps", 4))
        if self.n_obs_steps < 1:
            raise ValueError("ActorCriticDexRepTemporal expects n_obs_steps >= 1")

        self.bn_type = encoder_cfg["bn_type"]
        if encoder_cfg["bn_type"] == "part":
            self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['dexrep_pnl'])
        elif encoder_cfg["bn_type"] == "full":
            self.bn_pnl = nn.BatchNorm1d(sum(self.obs_dim[1:]))
        elif encoder_cfg["bn_type"] == "null":
            self.bn_pnl = None
        else:
            raise NotImplementedError(f"bn_type not impleted")

        emb_dim = encoder_cfg["emb_dim"]
        self.emb_dim = emb_dim
        self.dexrep_sensor_enc = nn.Linear(env_cfg['obs_dim']['dexrep_sensor'], emb_dim)
        self.dexrep_pointL_enc = nn.Linear(env_cfg['obs_dim']['dexrep_pnl'], emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        temporal_emb_dim = len(self.obs_dim) * emb_dim * self.n_obs_steps
        if encoder_only == False:
            actor_layers = []
            actor_layers.append(nn.Linear(temporal_emb_dim, actor_hidden_dim[0]))
            actor_layers.append(activation)
            for l in range(len(actor_hidden_dim)):
                if l == len(actor_hidden_dim) - 1:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actions_shape))
                else:
                    actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                    actor_layers.append(activation)
            self.actor = nn.Sequential(*actor_layers)

            critic_layers = []
            critic_layers.append(nn.Linear(temporal_emb_dim, critic_hidden_dim[0]))
            critic_layers.append(activation)
            for l in range(len(critic_hidden_dim)):
                if l == len(critic_hidden_dim) - 1:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
                else:
                    critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                    critic_layers.append(activation)
            self.critic = nn.Sequential(*critic_layers)

            actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
            actor_weights.append(0.01)
            critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
            critic_weights.append(1.0)
            self.init_weights(self.actor, actor_weights)
            self.init_weights(self.critic, critic_weights)

        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(actions_shape))

        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_sensor_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.dexrep_pointL_enc.weight, gain=np.sqrt(2))

        self._history_buffer = None
        self._history_valid = None

    def get_obs_process_mask(self, pro_dim=100):
        if pro_dim == 77:
            self.obs_mask = get_o6_compact_obs_mask(1902)
            return

        reshape84_149_part = torch.arange(84, 149).reshape(5, 13)
        remove_indice_part = reshape84_149_part[:, -6:].reshape(-1)

        remmove28_56 = torch.arange(28, 56)
        remove56_84 = torch.arange(56, 84)

        remove149_179 = torch.arange(149, 179)
        remove216_222 = torch.arange(216, 222)

        index_to_remove = torch.cat([remove56_84, remove149_179])
        if pro_dim == 134:
            final_indice_remove = torch.cat([index_to_remove, remove_indice_part])
        elif pro_dim == 128:
            final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222])
        elif pro_dim == 100:
            final_indice_remove = torch.cat([index_to_remove, remove_indice_part, remove216_222, remmove28_56])
        elif pro_dim == 222:
            final_indice_remove = torch.empty(0, dtype=torch.long)
        else:
            raise NotImplementedError(f"pro_dim not implemented: {pro_dim}")

        all_indices = torch.arange(2582)
        self.obs_mask = ~torch.isin(all_indices, final_indice_remove)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1

        total_expected_dim = sum(self.obs_dim)
        if x.size(-1) < total_expected_dim:
            raise ValueError(
                "ActorCriticDexRepTemporal expected obs dim {} from {}, got {}. "
                "This usually means the O6 dataset/model config is still using "
                "dummy obs or stale dexrep dimensions.".format(total_expected_dim, self.obs_dim, x.size(-1))
            )
        elif x.size(-1) > total_expected_dim:
            raise ValueError(
                "ActorCriticDexRepTemporal expected obs dim {} from {}, got {}".format(
                    total_expected_dim, self.obs_dim, x.size(-1)
                )
            )

        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]
        return x_list

    def _ensure_temporal(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(1).repeat(1, self.n_obs_steps, 1)
        elif observations.dim() != 3:
            raise ValueError("temporal DexRep expects obs shape (B, D) or (B, K, D)")

        if observations.size(1) != self.n_obs_steps:
            raise ValueError(
                "temporal DexRep history length mismatch: got {}, expected {}".format(
                    observations.size(1), self.n_obs_steps
                )
            )
        return observations

    def encode_frames(self, observations):
        if observations.size(-1) == self.obs_mask.numel():
            observations = observations[..., self.obs_mask.to(observations.device)]

        observations = self._ensure_temporal(observations)
        batch_size, history_len, obs_dim = observations.size()
        flat_obs = observations.reshape(batch_size * history_len, obs_dim)

        if self.bn_type == "part":
            state, dexrep_sensor, dexrep_pnl_raw = self.obs_division(flat_obs)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_norm = self.bn_pnl(dexrep_pnl_raw)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl_norm)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "full":
            flat_obs[:, self.obs_dim[0]:] = self.bn_pnl(flat_obs[:, self.obs_dim[0]:])
            state, dexrep_sensor, dexrep_pnl = self.obs_division(flat_obs)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        elif self.bn_type == "null":
            state, dexrep_sensor, dexrep_pnl = self.obs_division(flat_obs)
            state_emb = self.state_enc(state)
            dexrep_sensor_emb = self.dexrep_sensor_enc(dexrep_sensor)
            dexrep_pnl_emb = self.dexrep_pointL_enc(dexrep_pnl)
            dexrep_sensor_emb = F.normalize(dexrep_sensor_emb, dim=-1)
            dexrep_pnl_emb = F.normalize(dexrep_pnl_emb, dim=-1)
        else:
            raise NotImplementedError(f"bn_type not impleted")

        frame_emb = torch.cat([state_emb, dexrep_sensor_emb, dexrep_pnl_emb], dim=1)
        return frame_emb.reshape(batch_size, history_len, -1)

    def encode(self, observations):
        frame_emb = self.encode_frames(observations)
        return frame_emb.reshape(frame_emb.size(0), -1)

    def forward(self, observations):
        joint_emb = self.encode(observations)
        actions_mean = self.actor(joint_emb)
        return actions_mean

    def reset_history(self, env_ids=None):
        if env_ids is None or self._history_buffer is None:
            self._history_buffer = None
            self._history_valid = None
            return

        env_ids = torch.as_tensor(env_ids, device=self._history_buffer.device, dtype=torch.long).reshape(-1)
        if env_ids.numel() > 0:
            self._history_valid[env_ids] = False

    def _history_from_current_obs(self, observations):
        batch_size = observations.size(0)
        if (
            self._history_buffer is None
            or self._history_buffer.size(0) != batch_size
            or self._history_buffer.size(-1) != observations.size(-1)
            or self._history_buffer.device != observations.device
        ):
            self._history_buffer = observations.unsqueeze(1).repeat(1, self.n_obs_steps, 1).detach().clone()
            self._history_valid = torch.ones(batch_size, dtype=torch.bool, device=observations.device)
            return self._history_buffer

        invalid = ~self._history_valid
        if invalid.any():
            self._history_buffer[invalid] = observations[invalid].unsqueeze(1).repeat(1, self.n_obs_steps, 1)
            self._history_valid[invalid] = True

        self._history_buffer = torch.cat(
            [self._history_buffer[:, 1:], observations.unsqueeze(1).detach()],
            dim=1,
        )
        return self._history_buffer

    @torch.no_grad()
    def act(self, observations):
        observations = self._ensure_temporal(observations)
        joint_emb = self.encode(observations)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)
        value = self.critic(joint_emb)
        current_state = observations[:, -1, :self.obs_dim[0]]

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               current_state.detach(), \
               observations[:, -1, current_state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        if observations.dim() != 2:
            observations = self._ensure_temporal(observations)
            history = observations
        else:
            history = self._history_from_current_obs(observations)

        joint_emb = self.encode(history)
        actions_mean = self.actor(joint_emb)
        return actions_mean


class CausalConv1dResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, activation_name):
        super(CausalConv1dResidualBlock, self).__init__()
        if kernel_size < 1:
            raise ValueError("TCN kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("TCN dilation must be >= 1")

        causal_pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.ConstantPad1d((causal_pad, 0), 0.0),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            get_activation(activation_name),
            nn.Dropout(dropout),
            nn.ConstantPad1d((causal_pad, 0), 0.0),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            get_activation(activation_name),
            nn.Dropout(dropout),
        )
        self.out_activation = get_activation(activation_name)

    def forward(self, x):
        return self.out_activation(x + self.net(x))


class ActorCriticDexRepTemporalTCN(ActorCriticDexRepTemporal):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg, encoder_only=False, **kwargs):
        super(ActorCriticDexRepTemporalTCN, self).__init__(
            obs_shape,
            actions_shape,
            initial_std,
            model_cfg,
            encoder_cfg,
            env_cfg,
            encoder_only=encoder_only,
            **kwargs,
        )

        frame_emb_dim = len(self.obs_dim) * self.emb_dim
        activation_name = "selu" if model_cfg is None else model_cfg["activation"]
        actor_hidden_dim = [256, 256, 256] if model_cfg is None else model_cfg["pi_hid_sizes"]
        critic_hidden_dim = [256, 256, 256] if model_cfg is None else model_cfg["vf_hid_sizes"]

        self.tcn_kernel_size = int(encoder_cfg.get("tcn_kernel_size", 3))
        self.tcn_dropout = float(encoder_cfg.get("tcn_dropout", 0.05))
        tcn_dilations = encoder_cfg.get("tcn_dilations", [1, 2])
        if isinstance(tcn_dilations, str):
            tcn_dilations = [int(value.strip()) for value in tcn_dilations.split(",") if value.strip()]
        self.tcn_dilations = [int(value) for value in tcn_dilations]
        if not self.tcn_dilations:
            raise ValueError("ActorCriticDexRepTemporalTCN expects at least one TCN dilation")

        self.temporal_tcn = nn.Sequential(
            *[
                CausalConv1dResidualBlock(
                    frame_emb_dim,
                    kernel_size=self.tcn_kernel_size,
                    dilation=dilation,
                    dropout=self.tcn_dropout,
                    activation_name=activation_name,
                )
                for dilation in self.tcn_dilations
            ]
        )

        if encoder_only == False:
            self.actor = self._build_mlp(frame_emb_dim, actor_hidden_dim, actions_shape, activation_name)
            self.critic = self._build_mlp(frame_emb_dim, critic_hidden_dim, 1, activation_name)

            actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
            actor_weights.append(0.01)
            critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
            critic_weights.append(1.0)
            self.init_weights(self.actor, actor_weights)
            self.init_weights(self.critic, critic_weights)
            self._init_tcn_weights()

    @staticmethod
    def _build_mlp(input_dim, hidden_dim, output_dim, activation_name):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(get_activation(activation_name))
        for idx in range(len(hidden_dim)):
            if idx == len(hidden_dim) - 1:
                layers.append(nn.Linear(hidden_dim[idx], output_dim))
            else:
                layers.append(nn.Linear(hidden_dim[idx], hidden_dim[idx + 1]))
                layers.append(get_activation(activation_name))
        return nn.Sequential(*layers)

    def _init_tcn_weights(self):
        for module in self.temporal_tcn.modules():
            if isinstance(module, nn.Conv1d):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def encode(self, observations):
        frame_emb = self.encode_frames(observations)
        tcn_input = frame_emb.transpose(1, 2)
        tcn_output = self.temporal_tcn(tcn_input)
        return tcn_output[:, :, -1]


class ActorCriticPNG(nn.Module):
    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticPNG, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        # self.bn_pnl = nn.BatchNorm1d(env_cfg['obs_dim']['pnG'])
        self.pointG_enc = nn.Linear(env_cfg['obs_dim']['pnG'], emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

        # Initial fn
        torch.nn.init.orthogonal_(self.state_enc.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.pointG_enc.weight, gain=np.sqrt(2))

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list

    def forward(self, observations):
        # self.bn_pnl.eval()
        state, pnG = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(pnG)
        pnG_emb = self.pointG_enc(pnG)
        pnG_emb = F.normalize(pnG_emb, dim=-1)

        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        return actions_mean


    @torch.no_grad()
    def act(self, observations):
        # self.bn_pnl.eval()
        state, pnG = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(pnG)
        pnG_emb = self.pointG_enc(pnG)
        pnG_emb = F.normalize(pnG_emb, dim=-1)

        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        # self.bn_pnl.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        # self.bn_pnl.eval()
        state, pnG = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(pnG)
        pnG_emb = self.pointG_enc(pnG)
        pnG_emb = F.normalize(pnG_emb, dim=-1)

        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        # pnG = self.bn_pnl(obs_features)
        pnG_emb = self.pointG_enc(obs_features)
        pnG_emb = F.normalize(pnG_emb, dim=-1)
        joint_emb = torch.cat([state_emb, pnG_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import pathlib
    args = OmegaConf.load("{}/lhm_bc.yaml".format('../ActionDiffusion/bc/config'))

    model_dexrep = ActorCriticDexRep()
    model_glob = ActorCriticPNG()
