import os.path as osp

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Optional

# maniskill_learn 벤더 프레임워크(getPointNet)는 이 리포에 이식 불가능 -> 대회 baseline이
# 이미 쓰고 있는 dexrep/pointnet_model의 frozen PointNet(ShapeNetAutoEncoder)으로 대체함
# (8/21, dexrep/dexrep.py:62-70의 사용 패턴 그대로 재현: frozen, eval, 1024차원 글로벌 피처).
import dexrep.pointnet_model.model_rec as _model_rec_module
from dexrep.pointnet_model.model_rec import ShapeNetAutoEncoder

_DEXREP_PN_CKPT = osp.join(
    osp.dirname(_model_rec_module.__file__),
    "epo_180_REC_SPnetDenseEncoder_shapenet55_normrot512.pt",
)


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
        raise ValueError("invalid activation function!")


class PointNetBackbone(nn.Module):
    """대회 리포에 이미 들어있는 frozen PointNet(dexrep/pointnet_model)을 재사용한다.

    dexrep/dexrep.py:62-70 의 cv_space.load_pcd_extractor()와 동일한 로딩 패턴
    (load_state_dict -> requires_grad_(False) -> eval). 외부 데이터로 학습한 가중치가
    아니라 대회 baseline(DexRep)이 쓰는 자산이다.

    입력 (B,N,3) -> PointNetEncoder는 채널우선 (B,3,N)을 기대하므로 내부에서 transpose.
    글로벌 피처가 1024차원이라 feature_dim(=pc_emb_dim, 기본 128)으로 Linear 투영한다.
    """

    def __init__(self, pc_dim: int = 3, feature_dim: int = 128,
                 pretrained_model_path: Optional[str] = None, freeze: bool = True):
        super().__init__()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.freeze = freeze

        self.backbone = ShapeNetAutoEncoder()  # in_chan=3, globD=1024
        ckpt_path = pretrained_model_path or _DEXREP_PN_CKPT
        print("Loading pretrained PointNet (dexrep) from:", ckpt_path)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        if len(missing_keys) > 0:
            print("  missing_keys:", len(missing_keys))
        if len(unexpected_keys) > 0:
            print("  unexpected_keys:", len(unexpected_keys))

        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        self.proj = nn.Linear(self.backbone.globD, self.feature_dim)

    def forward(self, input_pc):
        """input_pc: (B, N, 3) -> (B, feature_dim)"""
        pc_chan_first = input_pc.transpose(1, 2)  # (B,3,N)
        if self.freeze:
            with torch.no_grad():
                glob_feat, _ = self.backbone.pcd_encoder(pc_chan_first)  # (B,1024)
            glob_feat = glob_feat.detach()
        else:
            glob_feat, _ = self.backbone.pcd_encoder(pc_chan_first)
        return self.proj(glob_feat)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # BN 통계 고정 유지
        return self


# 원본의 TransPointNetBackbone(getPointNetWithInstanceInfo 사용)은 제거함.
# ActorCritic이 backbone_type != 'pn'이면 ValueError를 내므로 도달 불가능한 죽은 코드였고,
# maniskill_learn 의존이라 이 리포에서 import 자체가 불가능하다.


from torch.distributions import Normal, Independent

def atanh(x, eps=1e-6):
    # clamp to avoid NaNs at |x|=1
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def tanh_squash_and_log_prob(dist_base: Independent, pre_tanh: torch.Tensor, eps: float = 1e-6):
    """
    Given a base (unsquashed) diagonal Gaussian dist and samples pre_tanh ~ N(mu, sigma),
    return squashed actions a = tanh(pre_tanh) and corrected log_prob(a).
    """
    # Base log prob of pre-tanh sample
    log_prob_pre = dist_base.log_prob(pre_tanh)
    # Change-of-variables correction: sum log(1 - tanh(x)^2)
    # Use a numerically stable form: log(1 - tanh(x)^2) = 2*(log(2) - x - softplus(-2x))
    # but the simple version below with clamp is fine for PPO:
    a = torch.tanh(pre_tanh)
    log_det_jacob = torch.sum(torch.log(1 - a.pow(2) + eps), dim=-1)
    log_prob = log_prob_pre - log_det_jacob
    return a, log_prob

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, states_shape, actions_shape, initial_std, model_cfg, 
                 asymmetric=False, use_pcl=False):
        super().__init__()
        self.asymmetric = asymmetric
        self.use_pcl = use_pcl
        self.backbone_type = model_cfg['backbone_type']
        self.freeze_backbone = model_cfg["freeze_backbone"]

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = nn.SELU()
        else:
            actor_hidden_dim = model_cfg["pi_hid_sizes"]
            critic_hidden_dim = model_cfg["vf_hid_sizes"]
            activation = get_activation(model_cfg["activation"])
        
        if self.use_pcl:
            self.pc_shape = model_cfg['pc_shape'] # [512,3]
            self.pc_emb_dim = model_cfg["pc_emb_dim"]
            if self.backbone_type == "pn":
                self.backbone = PointNetBackbone(pc_dim=self.pc_shape[-1], feature_dim=self.pc_emb_dim)
            else:
                raise ValueError(f"Invalid backbone type: {self.backbone_type}")
            #print(self.backbone)
        else:
            self.backbone = None
            self.pc_emb_dim = 0
            self.pc_shape = [0,0]

        self.num_obs = obs_shape[0]
        self.num_state_based_obs = self.num_obs - np.prod(self.pc_shape) + self.pc_emb_dim # replace N*3 pc with pn embedding
        self.pc_start_idx = self.num_obs - np.prod(self.pc_shape)
        self.act_dim = actions_shape[0] if isinstance(actions_shape, (list, tuple)) else int(actions_shape)

        # Actor
        actor_layers = [nn.Linear(self.num_state_based_obs, actor_hidden_dim[0]), activation]
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], self.act_dim))
            else:
                actor_layers += [nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]), activation]
        self.actor_mean = nn.Sequential(*actor_layers)

        # Critic
        critic_layers = [nn.Linear(self.num_state_based_obs, critic_hidden_dim[0]), activation]
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers += [nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]), activation]
        self.critic = nn.Sequential(*critic_layers)

        # Log-std parameter (diagonal)
        init_log_std = float(np.log(initial_std))
        self.log_std = nn.Parameter(torch.full((self.act_dim,), init_log_std))

        # Initialize the weights like in Stable Baselines
        self.init_orthogonal_(self.actor_mean, [np.sqrt(2)] * len(actor_hidden_dim) + [0.01])
        self.init_orthogonal_(self.critic,     [np.sqrt(2)] * len(critic_hidden_dim) + [1.0])

    @staticmethod
    def init_orthogonal_(sequential, gains):
        idx = 0
        for m in sequential:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gains[idx])
                nn.init.zeros_(m.bias)
                idx += 1

    def forward(self, observations, states=None, inference=False):
        """
        Returns (actions, actions_log_prob, value, actions_mean_squashed, log_std_vector)
        - actions in [-1, 1] via tanh-squash
        """
        if self.use_pcl and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            mean = self.actor_mean(observations)
        elif self.use_pcl and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            mean = self.actor_mean(observations) # pre-tanh mean

        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        if inference:
            # Deterministic (mean) action, squashed to bounds
            actions = torch.tanh(mean)
            # if self.asymmetric:
            #     value = self.critic(states)
            # else:
            #     value = self.critic(observations)
            return actions.detach()

        # Reparameterized sample: mean + std * eps ; we can call .rsample() for gradients if needed
        pre_tanh = base.rsample()
        actions, log_prob = tanh_squash_and_log_prob(base, pre_tanh)

        # Critic
        value = self.critic(states) if self.asymmetric else self.critic(observations)

        return (
            actions.detach(),
            log_prob.detach(),
            value.detach(),
            torch.tanh(mean).detach(),                # squashed mean for logging
            self.log_std.expand(mean.shape[0], -1).detach(),
        )

    def evaluate(self, observations, states, actions, eps: float = 1e-6):
        """
        Evaluate log_prob/entropy/value at given (already squashed) actions.
        """
        if self.use_pcl and not self.freeze_backbone:
            if self.backbone_type =="pn":
                pc = observations[:, self.pc_start_idx:].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
            else:
                raise NotImplementedError
            observations = torch.cat([observations[:, :self.pc_start_idx], pc_feature], dim=1)
            mean = self.actor_mean(observations)
        elif self.use_pcl and self.freeze_backbone:
            with torch.no_grad():
                raise NotImplementedError
        else:
            mean = self.actor_mean(observations) # pre-tanh mean
            
        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        # Map actions in [-1,1] back to pre-tanh space
        pre_tanh = atanh(actions, eps=eps)

        # Corrected log prob under the squashed policy
        log_prob_pre = base.log_prob(pre_tanh)
        log_det_jacob = torch.sum(torch.log(1 - actions.pow(2) + eps), dim=-1)
        log_prob = log_prob_pre - log_det_jacob

        # Entropy: use base Gaussian entropy (standard practice in SAC / PPO with squashing)
        entropy = base.entropy()

        value = self.critic(states) if self.asymmetric else self.critic(observations)

        return (
            log_prob,
            entropy,
            value,
            torch.tanh(mean),                          # squashed mean
            self.log_std.expand(mean.shape[0], -1),
        )