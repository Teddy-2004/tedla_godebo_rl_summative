"""DQN on WolaitaCorpusEnv with TorchRL (QValueActor + EGreedy + replay + target net).

Usage:
    uv run python -m training.dqn_training --sweep          # 10+ configs
    uv run python -m training.dqn_training --run dqn_01     # a single config
"""
from __future__ import annotations

import argparse
import json

import torch
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data import LazyTensorStorage, PrioritizedReplayBuffer, ReplayBuffer
from torchrl.modules import EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from training.common import (EVAL_SEEDS_TRAIN, EVAL_SEEDS_UNSEEN, LOG_DIR, RunLogger,
                             evaluate, make_env, mlp, obs_act_dims, save_policy)

TOTAL_FRAMES = 120_000     # identical interaction budget for every algorithm

BASE = dict(lr=1e-3, gamma=0.99, buffer_size=50_000, batch_size=128,
            eps_start=1.0, eps_end=0.05, eps_decay_frac=0.4, target_update="hard",
            target_interval=500, tau=0.01, double_dqn=False, prioritized=False,
            hidden=(128, 128), frames_per_batch=100, updates_per_batch=25, warmup=2_000)

SWEEP = {
    "dqn_01": {},                                                         # baseline
    "dqn_02": dict(lr=3e-4),
    "dqn_03": dict(lr=3e-3),
    "dqn_04": dict(gamma=0.95),
    "dqn_05": dict(buffer_size=5_000),
    "dqn_06": dict(batch_size=32),
    "dqn_07": dict(eps_decay_frac=0.1, eps_end=0.01),                     # fast greedy
    "dqn_08": dict(eps_decay_frac=0.8, eps_end=0.10),                     # long exploration
    "dqn_09": dict(target_update="soft", tau=0.005),
    "dqn_10": dict(target_interval=2_000),
    "dqn_11": dict(double_dqn=True),
    "dqn_12": dict(double_dqn=True, prioritized=True, lr=5e-4, hidden=(256, 256)),
}


def build(cfg):
    n_obs, n_act = obs_act_dims()
    env = make_env()
    qnet = QValueActor(mlp(n_obs, n_act, tuple(cfg["hidden"]), act=nn.ReLU),
                       in_keys=["observation"], spec=env.action_spec)
    return env, qnet


def train(run_name: str, overrides: dict, total_frames: int = TOTAL_FRAMES, seed: int = 0):
    cfg = {**BASE, **overrides}
    torch.manual_seed(seed)
    env, qnet = build(cfg)
    env.set_seed(seed)
    greedy = EGreedyModule(env.action_spec, eps_init=cfg["eps_start"], eps_end=cfg["eps_end"],
                           annealing_num_steps=int(cfg["eps_decay_frac"] * total_frames))
    explore = TensorDictSequential(qnet, greedy)
    collector = SyncDataCollector(env, explore, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=total_frames, init_random_frames=cfg["warmup"])
    storage = LazyTensorStorage(cfg["buffer_size"])
    rb = (PrioritizedReplayBuffer(alpha=0.6, beta=0.4, storage=storage,
                                  batch_size=cfg["batch_size"])
          if cfg["prioritized"] else ReplayBuffer(storage=storage, batch_size=cfg["batch_size"]))
    loss_fn = DQNLoss(qnet, action_space=env.action_spec, delay_value=True,
                      double_dqn=cfg["double_dqn"], loss_function="smooth_l1")
    loss_fn.make_value_estimator(gamma=cfg["gamma"])
    updater = (SoftUpdate(loss_fn, tau=cfg["tau"]) if cfg["target_update"] == "soft"
               else HardUpdate(loss_fn, value_network_update_interval=cfg["target_interval"]))
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    log = RunLogger("dqn", run_name, cfg)

    frames, n_updates = 0, 0
    for data in collector:
        frames += data.numel()
        done = data["next", "done"].squeeze(-1)
        log.log_episodes(data["next", "episode_reward"][done].flatten().tolist(), frames)
        rb.extend(data.reshape(-1))
        greedy.step(data.numel())
        if frames < cfg["warmup"]:
            continue
        losses = []
        for _ in range(cfg["updates_per_batch"]):
            batch = rb.sample()
            out = loss_fn(batch)
            optim.zero_grad()
            out["loss"].backward()
            nn.utils.clip_grad_norm_(loss_fn.parameters(), 10.0)
            optim.step()
            updater.step()
            if cfg["prioritized"]:
                rb.update_tensordict_priority(batch)
            losses.append(out["loss"].item())
            n_updates += 1
        log.loss.append([frames, sum(losses) / len(losses)])
        if frames % 10_000 < cfg["frames_per_batch"]:
            ev = evaluate(qnet, EVAL_SEEDS_TRAIN[:10])
            log.eval_curve.append([frames, ev["mean_return"], ev["success_rate"]])
            print(f"[{run_name}] frames {frames:6d} eps {greedy.eps.item():.2f} "
                  f"loss {log.loss[-1][1]:.3f} eval_return {ev['mean_return']:.1f} "
                  f"success {ev['success_rate']:.0%}", flush=True)
    collector.shutdown()
    final, unseen = evaluate(qnet, EVAL_SEEDS_TRAIN), evaluate(qnet, EVAL_SEEDS_UNSEEN, "unseen")
    save_policy(qnet, "dqn", run_name, cfg, "dqn")
    res = log.save(final, unseen)
    print(f"[{run_name}] FINAL {json.dumps(final)} UNSEEN {json.dumps(unseen)}", flush=True)
    return res


def load_policy(path):
    ckpt = torch.load(path, weights_only=False)
    _, qnet = build(ckpt["config"])
    qnet.load_state_dict(ckpt["state_dict"])
    return qnet


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    args = ap.parse_args()
    names = list(SWEEP) if args.sweep or not args.run else args.run
    LOG_DIR.mkdir(exist_ok=True)
    for n in names:
        train(n, SWEEP[n], args.frames)
