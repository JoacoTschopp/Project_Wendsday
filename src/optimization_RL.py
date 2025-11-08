"""
RFEPPO for Hyperparameter Optimization (LightGBM-friendly)
------------------------------------------------------------------
Minimal, practical implementation of the Random-Forest–Enhanced PPO
with curiosity (intrinsic reward) for HPO on tabular ML models.

API (core):
    - class RFEPPOHPO
        .search(evaluate_fn) -> (best_params: dict, best_reward: float)

You provide `evaluate_fn(params: dict) -> float` that trains/evaluates your
model (e.g., LightGBM) and returns a scalar reward (the higher, the better).

Notes:
    * This implementation focuses on numeric/int hyperparameters (typical for LGBM).
    * Categorical spaces can be discretized externally or added similarly if needed.
    * The surrogate (RandomForestRegressor) is trained on normalized param vectors.
    * KL threshold logic follows Ma et al. (2022): if the policy drift is small, use RF; otherwise use the real evaluator.

"""
from __future__ import annotations

import math
import os
import pickle
import random
import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional, Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim

from .config import (
    PARAMETROS_LGB,
    MES_TRAIN,
    MES_VALIDACION,
    SEMILLA,
)
from .gain_function import calcular_ganancia, ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target
from .undersampling import aplicar_undersampling


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ----------------------------
# Search space primitives
# ----------------------------
@dataclass
class ParamDef:
    name: str
    ptype: str  # 'int' or 'float'
    low: float
    high: float
    scale: str = "lin"  # 'lin' | 'log' | 'log2'

    def clip(self, x: float) -> float:
        return float(min(max(x, self.low), self.high))

    def to_unit(self, x: float) -> float:
        # map x in [low, high] to u in [0,1]
        if self.scale == "lin":
            return (x - self.low) / (self.high - self.low)
        elif self.scale == "log":
            lx, llow, lhigh = math.log(x), math.log(self.low), math.log(self.high)
            return (lx - llow) / (lhigh - llow)
        elif self.scale == "log2":
            lx, llow, lhigh = math.log2(x), math.log2(self.low), math.log2(self.high)
            return (lx - llow) / (lhigh - llow)
        else:
            raise ValueError(f"Unknown scale {self.scale}")

    def from_unit(self, u: float) -> float:
        # map u in [0,1] to x in [low, high]
        u = float(min(max(u, 0.0), 1.0))
        if self.scale == "lin":
            x = self.low + u * (self.high - self.low)
        elif self.scale == "log":
            llow, lhigh = math.log(self.low), math.log(self.high)
            x = math.exp(llow + u * (lhigh - llow))
        elif self.scale == "log2":
            llow, lhigh = math.log2(self.low), math.log2(self.high)
            x = 2 ** (llow + u * (lhigh - llow))
        else:
            raise ValueError(f"Unknown scale {self.scale}")
        if self.ptype == "int":
            return float(int(round(x)))
        return float(x)


# ----------------------------
# PPO components
# ----------------------------
class PPOActorCritic(nn.Module):
    """
    LSTM-based actor-critic that outputs a Gaussian policy for a SINGLE scalar action
    at each step (we choose one hyperparameter per step). Which hyperparameter is
    being chosen is indicated via a learned embedding of the step index.
    """
    def __init__(self, num_params: int, hidden_size: int = 128, idx_emb_dim: int = 32):
        super().__init__()
        self.num_params = num_params
        self.idx_emb = nn.Embedding(num_params, idx_emb_dim)
        # State = previous output stats [mu, sigma] (2-dim) + idx embedding
        self.input_size = 2 + idx_emb_dim
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=1, batch_first=True)
        self.actor_mu = nn.Linear(hidden_size, 1)
        self.actor_logstd = nn.Linear(hidden_size, 1)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, prev_mu_sigma: torch.Tensor, idxs: torch.Tensor, hx=None):
        # prev_mu_sigma: [B, T, 2]
        # idxs:          [B, T] (long)
        B, T, _ = prev_mu_sigma.shape
        emb = self.idx_emb(idxs)  # [B, T, idx_emb_dim]
        x = torch.cat([prev_mu_sigma, emb], dim=-1)  # [B, T, input_size]
        out, (h, c) = self.lstm(x, hx)  # out: [B,T,H]
        mu = torch.tanh(self.actor_mu(out))  # keep within ~[-1,1]
        logstd = torch.clamp(self.actor_logstd(out), -3.0, 2.0)  # stabilize
        value = self.critic(out)
        return mu.squeeze(-1), logstd.squeeze(-1), value.squeeze(-1), (h, c)

    @staticmethod
    def gaussian_log_prob(mu: torch.Tensor, logstd: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # action, mu, logstd: same shape
        var = torch.exp(2 * logstd)
        return -0.5 * (math.log(2 * math.pi) + 2 * logstd + (action - mu) ** 2 / var)

    @staticmethod
    def gaussian_kl(mu0, logstd0, mu1, logstd1):
        # KL( N0 || N1 ) elementwise
        var0, var1 = torch.exp(2 * logstd0), torch.exp(2 * logstd1)
        kl = logstd1 - logstd0 + (var0 + (mu0 - mu1) ** 2) / (2 * var1) - 0.5
        return kl


class ForwardModel(nn.Module):
    """Curiosity forward model Φ(s_t), predicts Φ(s_{t+1}) given (Φ(s_t), a_t)."""
    def __init__(self, state_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, state_dim)
        )

    def embed_state(self, mu_sigma: torch.Tensor) -> torch.Tensor:
        return self.encoder(mu_sigma)

    def forward(self, phi_s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([phi_s, a], dim=-1)
        return self.forward_net(x)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95  # GAE lambda
    ppo_clip: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    fm_lr: float = 3e-4  # forward model
    epochs: int = 10
    batch_size: int = 64
    beta_intrinsic: float = 0.1  # weight for intrinsic reward


# ----------------------------
# RFEPPO Orchestrator
# ----------------------------
class RFEPPOHPO:
    def __init__(
        self,
        space: List[ParamDef],
        episodes: int = 200,
        initial_real_episodes: int = 100,
        kl_threshold: float = 0.1,
        ppo_cfg: PPOConfig | None = None,
        seed: int = 42,
        device: str | None = None,
    ):
        self.space = space
        self.dim = len(space)
        self.episodes = episodes
        self.initial_real_episodes = initial_real_episodes
        self.kl_threshold = kl_threshold
        self.ppo_cfg = ppo_cfg or PPOConfig()
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.actor = PPOActorCritic(self.dim).to(self.device)
        self.actor_old = copy.deepcopy(self.actor).to(self.device)  # reference policy π
        self.fm = ForwardModel().to(self.device)
        self.opt_actor = optim.Adam(
            [p for n, p in self.actor.named_parameters() if not n.startswith("critic")],
            lr=self.ppo_cfg.actor_lr,
        )
        self.opt_critic = optim.Adam(self.actor.critic.parameters(), lr=self.ppo_cfg.critic_lr)
        self.opt_fm = optim.Adam(self.fm.parameters(), lr=self.ppo_cfg.fm_lr)

        # Surrogate model data (normalized [0,1] vectors -> reward)
        self.D_X: List[List[float]] = []
        self.D_y: List[float] = []
        self.rf: Optional[RandomForestRegressor] = None

        self.best_reward = -float("inf")
        self.best_params: Dict[str, Any] = {}
        self.episode_done = 0

    # --------- utils: mapping actions <-> params
    def _actions_to_params(self, actions_unit: List[float]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for u, p in zip(actions_unit, self.space):
            val = p.from_unit(u)
            if p.ptype == "int":
                params[p.name] = int(val)
            else:
                params[p.name] = float(val)
        return params

    def _params_to_unit(self, params: Dict[str, Any]) -> List[float]:
        us = []
        for p in self.space:
            us.append(p.to_unit(float(params[p.name])))
        return us

    @staticmethod
    def _squash_to_unit(z: torch.Tensor) -> torch.Tensor:
        # map real line -> (0,1) via sigmoid
        return torch.sigmoid(z)

    # --------- KL between current policy and saved reference π (episode-level)
    def _estimate_policy_kl(self, prev_mu_sigmas: torch.Tensor, idxs: torch.Tensor) -> float:
        self.actor.eval(); self.actor_old.eval()
        with torch.no_grad():
            mu, logstd, _, _ = self.actor(prev_mu_sigmas, idxs)
            mu0, logstd0, _, _ = self.actor_old(prev_mu_sigmas, idxs)
            kl = PPOActorCritic.gaussian_kl(mu0, logstd0, mu, logstd)
            # average across steps then mean
            kl_mean = kl.mean().item()
        return float(kl_mean)

    # --------- One full episode rollout + PPO update
    def _rollout_and_update(self, evaluate_fn: Callable[[Dict[str, Any]], float], use_surrogate: bool) -> Tuple[float, Dict[str, Any]]:
        self.actor.train(); self.fm.train()
        device = self.device
        T = self.dim
        # containers
        states_prev = []      # [T, 2]
        idxs = []             # [T]
        actions = []          # [T, 1]
        logps = []            # [T]
        values = []           # [T]
        rewards = []          # [T] (intrinsic for steps 0..T-2, intrinsic+env at T-1)

        # initial previous output distribution (standard normal)
        prev = torch.zeros(1, 1, 2, device=device)  # [B=1, t=1, 2]
        hx = None
        chosen_unit = []

        for t in range(T):
            step_idx = torch.tensor([[t]], dtype=torch.long, device=device)  # [B=1,T=1]
            mu, logstd, val, hx = self.actor(prev, step_idx, hx)  # shapes [1,1]
            dist_mu = mu.squeeze(0).squeeze(0)
            dist_logstd = logstd.squeeze(0).squeeze(0)
            value_t = val.squeeze(0).squeeze(0)

            # sample action (scalar), then squash to (0,1)
            eps = torch.randn_like(dist_mu)
            a = dist_mu + torch.exp(dist_logstd) * eps
            u = self._squash_to_unit(a)

            # log prob under Gaussian BEFORE squash (treat as reparam; we still use this approx)
            logp = PPOActorCritic.gaussian_log_prob(dist_mu, dist_logstd, a)

            # curiosity: compute intrinsic reward r_i = MSE(Φ(s_{t+1}) - 
            #  forward(Φ(s_t), a_t)) after we form next state (which uses current mu/logstd)
            mu_sigma = torch.stack([dist_mu.detach(), torch.exp(dist_logstd.detach())], dim=-1)  # [2]
            states_prev.append(mu_sigma.cpu().numpy())
            idxs.append(t)
            actions.append(a.detach().cpu().numpy())
            logps.append(logp.item())
            values.append(value_t.item())

            # next state's prev is current [mu, sigma]
            prev = torch.stack([dist_mu, torch.exp(dist_logstd)], dim=-1).view(1, 1, 2)

            # store the unit choice
            chosen_unit.append(float(u.item()))

        # Build params dict
        params = self._actions_to_params(chosen_unit)

        # Evaluate environment reward at final step
        if use_surrogate and self.rf is not None and len(self.D_X) >= 20:
            rT = float(self.rf.predict(np.array([chosen_unit]))[0])
        else:
            rT = float(evaluate_fn(params))
            # accumulate to dataset for surrogate
            self.D_X.append(chosen_unit)
            self.D_y.append(rT)

        # Compute intrinsic rewards using forward model (requires Φ(s_t), Φ(s_{t+1}))
        # Recompute the forward pass to get embeddings for curiosity
        with torch.no_grad():
            states_prev_t = torch.tensor(np.stack(states_prev, axis=0), dtype=torch.float32, device=device)  # [T,2]
        phi = self.fm.embed_state(states_prev_t)  # [T,state_dim]
        # For actions, we reuse the sampled a values (pre-sigmoid), shape [T]
        a_tensor = torch.tensor(np.array(actions).reshape(-1, 1), dtype=torch.float32, device=device)
        phi_pred_next = self.fm(phi[:-1], a_tensor[:-1])  # predict for steps 0..T-2
        phi_true_next = phi[1:]
        intrinsic = torch.mean((phi_pred_next - phi_true_next).pow(2), dim=-1).detach().cpu().numpy()  # [T-1]

        # Compose rewards per step
        rewards = list(self.ppo_cfg.beta_intrinsic * intrinsic)
        rewards.append(self.ppo_cfg.beta_intrinsic * (intrinsic[-1] if len(intrinsic) > 0 else 0.0) + rT)

        # Train forward model on curiosity signal
        fm_loss = nn.functional.mse_loss(phi_pred_next, phi_true_next)
        self.opt_fm.zero_grad(); fm_loss.backward(); self.opt_fm.step()

        # ===== PPO update (on this single-episode trajectory) =====
        # Compute returns/advantages (GAE)
        gamma, lam = self.ppo_cfg.gamma, self.ppo_cfg.lam
        values_t = np.array(values + [0.0], dtype=np.float32)  # V(s_T)≈0 terminal
        rewards_t = np.array(rewards, dtype=np.float32)
        Tlen = len(rewards_t)
        adv = np.zeros(Tlen, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(Tlen)):
            delta = rewards_t[t] + gamma * values_t[t+1] - values_t[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values_t[:-1]

        # Pack tensors
        prev_mu_sigmas = torch.tensor(np.stack(states_prev, axis=0), dtype=torch.float32, device=device).unsqueeze(0)  # [1,T,2]
        idxs_t = torch.tensor(np.array(idxs)[None, :], dtype=torch.long, device=device)
        actions_t = torch.tensor(np.array(actions).reshape(1, -1), dtype=torch.float32, device=device)
        old_logps_t = torch.tensor(np.array(logps).reshape(1, -1), dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv.reshape(1, -1), dtype=torch.float32, device=device)
        returns_t = torch.tensor(returns.reshape(1, -1), dtype=torch.float32, device=device)

        # Multiple epochs PPO
        for _ in range(self.ppo_cfg.epochs):
            mu, logstd, value_pred, _ = self.actor(prev_mu_sigmas, idxs_t)
            logp = PPOActorCritic.gaussian_log_prob(mu, logstd, actions_t)
            ratio = torch.exp(logp - old_logps_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_cfg.ppo_clip, 1.0 + self.ppo_cfg.ppo_clip) * adv_t
            actor_loss = -torch.mean(torch.min(surr1, surr2))
            critic_loss = nn.functional.mse_loss(value_pred, returns_t)
            loss = actor_loss + 0.5 * critic_loss

            self.opt_actor.zero_grad(); self.opt_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_actor.step(); self.opt_critic.step()

        return rT, params

    ###
    #
    #Solo por las dudas se corte y vuelva a empezarlo utilizaria este
    #Por ahroa solo usao search()
    #
    ###

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "version": 1,
            "episodes_total": self.episodes,
            "episode_done": self.episode_done,
            "ppo_cfg": self.ppo_cfg.__dict__,
            "space": [p.__dict__ for p in self.space],
            "D_X": self.D_X,
            "D_y": self.D_y,
            "best_reward": self.best_reward,
            "best_params": self.best_params,
            "rf_bytes": pickle.dumps(self.rf) if self.rf is not None else None,
            "actor_state": self.actor.state_dict(),
            "actor_old_state": self.actor_old.state_dict(),
            "fm_state": self.fm.state_dict(),
            "opt_actor_state": self.opt_actor.state_dict(),
            "opt_critic_state": self.opt_critic.state_dict(),
            "opt_fm_state": self.opt_fm.state_dict(),
            # RNG states
            "py_random_state": random.getstate(),
            "np_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        # basic
        self.episodes = int(state.get("episodes_total", self.episodes))
        self.episode_done = int(state.get("episode_done", 0))
        # data
        self.D_X = list(state.get("D_X", []))
        self.D_y = list(state.get("D_y", []))
        self.best_reward = float(state.get("best_reward", -float("inf")))
        self.best_params = dict(state.get("best_params", {}))
        rf_bytes = state.get("rf_bytes", None)
        self.rf = pickle.loads(rf_bytes) if rf_bytes is not None else None
        # models
        self.actor.load_state_dict(state["actor_state"])
        self.actor_old.load_state_dict(state["actor_old_state"])
        self.fm.load_state_dict(state["fm_state"])
        self.opt_actor.load_state_dict(state["opt_actor_state"])
        self.opt_critic.load_state_dict(state["opt_critic_state"])
        self.opt_fm.load_state_dict(state["opt_fm_state"])
        self.actor_old.eval()
        # RNG
        try:
            random.setstate(state["py_random_state"]) # Python RNG
        except Exception:
            pass
        try:
            # Recreate a Generator and set its bit_generator state
            self.rng = np.random.default_rng()
            self.rng.bit_generator.state = state["np_rng_state"]
        except Exception:
            pass
        try:
            torch.set_rng_state(state["torch_rng_state"]) # CPU RNG
        except Exception:
            pass

    def search_ckpt(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 1,
        resume: bool = False,
    ) -> Tuple[Dict[str, Any], float]:
        """Run RFEPPO search with optional checkpointing and resume support."""

        start_ep = 1
        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        start_ep = self.episode_done + 1
        if start_ep < 1:
            start_ep = 1


        for ep in range(start_ep, self.episodes + 1):
            # Decide whether to use surrogate based on KL to reference policy
            use_surrogate = False
            if ep <= self.initial_real_episodes:
                use_surrogate = False
            else:
                # compute KL between actor and reference actor_old on a mock rollout (states come from actor)
                with torch.no_grad():
                    device = self.device
                    prev = torch.zeros(1, 1, 2, device=device)
                    traj_prev = []
                    idxs = []
                    hx = None
                    for t in range(self.dim):
                        step_idx = torch.tensor([[t]], dtype=torch.long, device=device)
                        mu, logstd, _, hx = self.actor(prev, step_idx, hx)
                        prev = torch.stack([mu.squeeze(0), torch.exp(logstd.squeeze(0))], dim=-1).view(1, 1, 2)
                        traj_prev.append(prev.squeeze(0).squeeze(0).cpu().numpy())
                        idxs.append(t)
                    prev_mu_sigmas = torch.tensor(
                        np.stack(traj_prev, axis=0),
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)
                    idxs_t = torch.tensor(np.array(idxs)[None, :], dtype=torch.long, device=device)
                    kl = self._estimate_policy_kl(prev_mu_sigmas, idxs_t)
                    use_surrogate = (self.rf is not None) and (kl <= self.kl_threshold)

            rT, params = self._rollout_and_update(evaluate_fn, use_surrogate)

            if rT > self.best_reward:
                self.best_reward = rT
                self.best_params = params


            # Train/update surrogate after initial_real_episodes, whenever we have new real data
            if ep == self.initial_real_episodes or (ep > self.initial_real_episodes and (not use_surrogate)):
                if len(self.D_X) >= 20:
                    self.rf = RandomForestRegressor(n_estimators=150, random_state=0)
                    self.rf.fit(np.array(self.D_X), np.array(self.D_y))
                    # Save current policy as reference π
                    self.actor_old = copy.deepcopy(self.actor).to(self.device).eval()


            # update episode counter, checkpoint if requested
            self.episode_done = ep
            if checkpoint_path and (ep % max(1, checkpoint_every) == 0):
                self.save_checkpoint(checkpoint_path)


        # final checkpoint
        if checkpoint_path:
            self.save_checkpoint(checkpoint_path)

        return self.best_params, float(self.best_reward)

##########################################################################################################################

    def search(self, evaluate_fn: Callable[[Dict[str, Any]], float]) -> Tuple[Dict[str, Any], float]:
        """Run RFEPPO search. Returns (best_params, best_reward)."""
        # warmup: real evaluations
        for ep in range(1, self.episodes + 1):
            # Decide whether to use surrogate based on KL to reference policy
            use_surrogate = False
            if ep <= self.initial_real_episodes:
                use_surrogate = False
            else:
                # compute KL between actor and reference actor_old on a mock rollout (states come from actor)
                # Construct a mock prev_mu_sigmas trajectory from the actor itself
                with torch.no_grad():
                    device = self.device
                    prev = torch.zeros(1, 1, 2, device=device)
                    traj_prev = []
                    idxs = []
                    hx = None
                    for t in range(self.dim):
                        step_idx = torch.tensor([[t]], dtype=torch.long, device=device)
                        mu, logstd, _, hx = self.actor(prev, step_idx, hx)
                        prev = torch.stack([mu.squeeze(0), torch.exp(logstd.squeeze(0))], dim=-1).view(1, 1, 2)
                        traj_prev.append(prev.squeeze(0).squeeze(0).cpu().numpy())
                        idxs.append(t)
                    prev_mu_sigmas = torch.tensor(np.stack(traj_prev, axis=0), dtype=torch.float32, device=device).unsqueeze(0)
                    idxs_t = torch.tensor(np.array(idxs)[None, :], dtype=torch.long, device=device)

                kl = self._estimate_policy_kl(prev_mu_sigmas, idxs_t)
                use_surrogate = (self.rf is not None) and (kl <= self.kl_threshold)

            rT, params = self._rollout_and_update(evaluate_fn, use_surrogate)

            if rT > self.best_reward:
                self.best_reward = rT
                self.best_params = params

            # Train/update surrogate after initial_real_episodes, whenever we have new real data
            if ep == self.initial_real_episodes or (ep > self.initial_real_episodes and (not use_surrogate)):
                if len(self.D_X) >= 20:
                    self.rf = RandomForestRegressor(n_estimators=150, random_state=0)
                    self.rf.fit(np.array(self.D_X), np.array(self.D_y))
                    # Save current policy as reference π
                    self.actor_old = copy.deepcopy(self.actor).to(self.device).eval()

        return self.best_params, float(self.best_reward)


# ----------------------------
# Convenience: default LGBM search space
# ----------------------------
def default_lgbm_space() -> List[ParamDef]:
    scale_overrides = {
        "num_leaves": "log2",
        "learning_rate": "log",
        "min_child_samples": "log",
        "min_data_in_leaf": "log",
        "num_iterations": "log",
    }

    space: List[ParamDef] = []
    for name, bounds in PARAMETROS_LGB.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            continue

        low, high = bounds
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            continue

        ptype = "int" if all(float(v).is_integer() for v in (low, high)) else "float"
        scale = scale_overrides.get(name, "lin")
        if scale in {"log", "log2"} and (low <= 0 or high <= 0):
            scale = "lin"

        space.append(
            ParamDef(name, ptype, float(low), float(high), scale)
        )

    return space


# ----------------------------
# RFEPPO optimization helper
# ----------------------------
def optimizar_rfppo_hpo(
    df: pd.DataFrame,
    undersampling: float = 1.0,
    episodes: int = 200,
    initial_real_episodes: int = 100,
    kl_threshold: float = 0.1,
) -> Tuple[Dict[str, Any], float]:
    """
    Ejecuta RFEPPO sobre LightGBM utilizando los períodos configurados.

    Args:
        df: DataFrame completo con feature engineering aplicado y columna ``clase_ternaria``.
        undersampling: Ratio de undersampling para la clase negativa (1.0 lo desactiva).
        episodes: Episodios totales para RFEPPO.
        initial_real_episodes: Episodios iniciales con evaluaciones reales antes de usar el surrogate.
        kl_threshold: Umbral de divergencia KL para habilitar el uso del surrogate.

    Returns:
        Tuple[Dict[str, Any], float]: mejores parámetros completos de LightGBM y ganancia en validación.
    """

    if isinstance(MES_TRAIN, list):
        train_periods = list(MES_TRAIN)
    else:
        train_periods = [MES_TRAIN]

    logger.info(
        "RFEPPO - preparando datos (TRAIN=%s | VALID=%s)",
        train_periods,
        MES_VALIDACION,
    )

    df_train = df[df["foto_mes"].isin(train_periods)].copy()
    df_val = df[df["foto_mes"] == MES_VALIDACION].copy()

    if df_train.empty or df_val.empty:
        raise ValueError(
            "Los períodos configurados no generan datos suficientes para entrenamiento/validación."
        )

    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)

    df_train["clase_ternaria"] = df_train["clase_ternaria"].astype(np.int8)
    df_val["clase_ternaria"] = df_val["clase_ternaria"].astype(np.int8)

    if undersampling < 1.0:
        logger.info("RFEPPO - aplicando undersampling (ratio=%.3f)", undersampling)
        base_seed = SEMILLA[0] if isinstance(SEMILLA, list) else int(SEMILLA)
        df_train = aplicar_undersampling(df_train, undersampling, random_state=base_seed)

    feature_columns = [col for col in df_train.columns if col != "clase_ternaria"]
    X_train = df_train[feature_columns].astype(np.float32)
    y_train = df_train["clase_ternaria"].to_numpy(dtype=np.int32)

    # Asegurar mismas columnas ordenadas en validación
    X_val = df_val[feature_columns].astype(np.float32)
    y_val = df_val["clase_ternaria"].to_numpy(dtype=np.int32)

    space = default_lgbm_space()

    fixed_params = {
        "objective": PARAMETROS_LGB.get("objective", "binary"),
        "metric": PARAMETROS_LGB.get("metric", "None"),
        "min_gain_to_split": PARAMETROS_LGB.get("min_gain_to_split", 0.0),
        "verbosity": PARAMETROS_LGB.get("verbosity", -1),
        "max_bin": PARAMETROS_LGB.get("max_bin", 255),
        "feature_pre_filter": False,
        "force_col_wise": True,
    }

    base_seed = SEMILLA[0] if isinstance(SEMILLA, list) else int(SEMILLA)

    def _evaluate(params: Dict[str, Any]) -> float:
        params_train = {**fixed_params, **params}
        params_train["seed"] = base_seed
        params_train.setdefault("bagging_freq", 0)
        params_train.setdefault("min_data_in_leaf", params_train.get("min_child_samples", 20))

        num_boost_round = int(params_train.pop("num_iterations", 3000))

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        valid_data = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

        model = lgb.train(
            params_train,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[valid_data],
            feval=ganancia_evaluator,
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping_rounds(300)],
        )

        best_iter = model.best_iteration or num_boost_round
        y_pred_val = model.predict(X_val, num_iteration=best_iter)
        ganancia_val, _ = calcular_ganancia(y_true=y_val, y_pred=y_pred_val)
        return float(ganancia_val)

    optimizer = RFEPPOHPO(
        space=space,
        episodes=episodes,
        initial_real_episodes=initial_real_episodes,
        kl_threshold=kl_threshold,
        seed=base_seed,
    )

    best_params_candidate, best_reward = optimizer.search(_evaluate)
    best_params = {**fixed_params, **best_params_candidate}
    best_params["seed"] = base_seed

    if "num_iterations" not in best_params:
        best_params["num_iterations"] = int(PARAMETROS_LGB.get("num_iterations", [300, 300])[1])

    logger.info(
        "RFEPPO - mejor ganancia VALID=%s",
        f"{best_reward:,.0f}",
    )

    return best_params, float(best_reward)


# ----------------------------
# Example usage (pseudo-code) – keep minimal
# ----------------------------

    # 1) Define your evaluator using your own data split & gain function.
    # def evaluate_fn(params: Dict[str, Any]) -> float:
    #     import lightgbm as lgb
    #     # Build/train model using your X_train, y_train, X_valid, y_valid
    #     # Return a scalar reward (e.g., custom gain or validation AUC)
    #     return gain
    #
    # 2) Instantiate and run search
    # space = default_lgbm_space()
    # hpo = RFEPPOHPO(space, episodes=200, initial_real_episodes=100, kl_threshold=0.1)
    # best_params, best_reward = hpo.search(evaluate_fn)
    # print(best_params, best_reward)
