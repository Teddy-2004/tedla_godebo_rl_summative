"""Shared TorchRL utilities: env construction, networks, evaluation, logging."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import GymWrapper, RewardSum, StepCounter, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import WolaitaCorpusEnv

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
MODEL_DIR = ROOT / "models"

EVAL_SEEDS_TRAIN = list(range(10_000, 10_040))     # held-out seeds, training distribution
EVAL_SEEDS_UNSEEN = list(range(20_000, 20_040))    # unseen (shifted) initial states


def make_env(profile: str = "train", seed: int | None = None, render_mode=None):
    """The single environment used by every algorithm (fair comparison)."""
    base = GymWrapper(WolaitaCorpusEnv(profile=profile, render_mode=render_mode),
                      categorical_action_encoding=True)
    env = TransformedEnv(base, StepCounter(max_steps=60))
    env.append_transform(RewardSum())
    if seed is not None:
        env.set_seed(seed)
    return env


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, ...], act=nn.Tanh) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(policy, seeds, profile: str = "train", deterministic: bool = True) -> dict:
    """Run one episode per seed with the raw gym env; greedy actions by default."""
    rets, wers, succ, days, hours = [], [], [], [], []
    etype = ExplorationType.DETERMINISTIC if deterministic else ExplorationType.RANDOM
    env = make_env(profile)
    rng_state = torch.get_rng_state()   # do not disturb the training RNG stream
    for s in seeds:
        env.set_seed(s)
        torch.manual_seed(s)          # reproducible action sampling
        with set_exploration_type(etype):
            td = env.rollout(60, policy, break_when_any_done=True)
        rets.append(float(td["next", "episode_reward"][-1]))
        gym_env = env.base_env._env
        while hasattr(gym_env, "env") and not isinstance(gym_env, WolaitaCorpusEnv):
            gym_env = gym_env.env
        info = gym_env._info()
        wers.append(info["checkpoint_wer"])
        succ.append(float(info["success"]))
        days.append(gym_env.day)
        hours.append(info["annotated_hours"])
    env.close()
    torch.set_rng_state(rng_state)
    return {"mean_return": float(np.mean(rets)), "std_return": float(np.std(rets)),
            "success_rate": float(np.mean(succ)), "mean_final_wer": float(np.mean(wers)),
            "mean_days": float(np.mean(days)), "mean_hours": float(np.mean(hours))}


class RunLogger:
    def __init__(self, algo: str, run_name: str, config: dict):
        self.algo, self.run_name, self.config = algo, run_name, config
        self.ep_returns, self.ep_frames = [], []
        self.loss, self.entropy, self.eval_curve = [], [], []
        self.t0 = time.time()

    def log_episodes(self, returns, frame):
        for r in returns:
            self.ep_returns.append(float(r))
            self.ep_frames.append(int(frame))

    def save(self, final_eval: dict, unseen_eval: dict) -> dict:
        out = {"algo": self.algo, "run": self.run_name, "config": self.config,
               "episode_returns": self.ep_returns, "episode_frames": self.ep_frames,
               "loss": self.loss, "entropy": self.entropy, "eval_curve": self.eval_curve,
               "final_eval": final_eval, "unseen_eval": unseen_eval,
               "episodes_to_converge": episodes_to_converge(self.ep_returns),
               "wall_time_s": time.time() - self.t0}
        d = LOG_DIR / self.algo
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.run_name}.json").write_text(json.dumps(out))
        return out


def episodes_to_converge(returns, window: int = 50, frac: float = 0.9):
    """First episode at which the 50-episode moving average reaches 90 % of
    the final 100-episode mean (measured from the first-window baseline)."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 * window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    start, final = ma[0], r[-100:].mean()
    if final <= start:
        return None
    thresh = start + frac * (final - start)
    idx = int(np.argmax(ma >= thresh))
    return idx + window


def save_policy(module: nn.Module, algo: str, run_name: str, config: dict, sub: str):
    d = MODEL_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": module.state_dict(), "config": config, "algo": algo},
               d / f"{run_name}.pt")


def obs_act_dims():
    e = WolaitaCorpusEnv()
    return e.observation_space.shape[0], e.action_space.n


def td_module(net, in_keys, out_keys):
    return TensorDictModule(net, in_keys=in_keys, out_keys=out_keys)
