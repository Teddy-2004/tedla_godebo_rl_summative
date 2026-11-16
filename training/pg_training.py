"""Policy-gradient methods on WolaitaCorpusEnv with TorchRL.

* REINFORCE  - Monte-Carlo returns on complete episodes (env.rollout), TorchRL
               ReinforceLoss, optional learned value baseline.
* A2C        - TorchRL A2CLoss + GAE, one synchronous update per rollout batch.
* PPO        - TorchRL ClipPPOLoss + GAE, several epochs of minibatch updates.

Usage:
    uv run python -m training.pg_training --algo ppo --sweep
    uv run python -m training.pg_training --algo a2c --run a2c_03
"""
from __future__ import annotations

import argparse
import json

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector as SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives.value import GAE

from training.common import (EVAL_SEEDS_TRAIN, EVAL_SEEDS_UNSEEN, LOG_DIR, RunLogger,
                             evaluate, make_env, mlp, obs_act_dims, save_policy)

TOTAL_FRAMES = 120_000

BASE = {
    "reinforce": dict(lr=1e-3, gamma=0.99, episodes_per_update=8, baseline=True,
                      entropy_coef=0.01, normalize_adv=True, hidden=(128, 128)),
    "a2c": dict(lr=7e-4, gamma=0.99, gae_lambda=0.95, frames_per_batch=240,
                entropy_coef=0.01, critic_coef=0.5, hidden=(128, 128)),
    "ppo": dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, frames_per_batch=2_400,
                epochs=8, minibatch=240, clip_eps=0.2, entropy_coef=0.01,
                critic_coef=0.5, normalize_adv=True, hidden=(128, 128)),
}

SWEEP = {
    "reinforce": {
        "reinforce_01": {},
        "reinforce_02": dict(lr=3e-4),
        "reinforce_03": dict(lr=3e-3),
        "reinforce_04": dict(gamma=0.95),
        "reinforce_05": dict(baseline=False),
        "reinforce_06": dict(episodes_per_update=2),
        "reinforce_07": dict(episodes_per_update=32, lr=2e-3),
        "reinforce_08": dict(entropy_coef=0.0),
        "reinforce_09": dict(entropy_coef=0.05),
        "reinforce_10": dict(normalize_adv=False),
        "reinforce_11": dict(hidden=(64, 64)),
    },
    "a2c": {
        "a2c_01": {},
        "a2c_02": dict(lr=2e-4),
        "a2c_03": dict(lr=2e-3),
        "a2c_04": dict(gamma=0.95),
        "a2c_05": dict(gae_lambda=1.0),
        "a2c_06": dict(gae_lambda=0.8),
        "a2c_07": dict(frames_per_batch=60),
        "a2c_08": dict(frames_per_batch=960, lr=1.5e-3),
        "a2c_09": dict(entropy_coef=0.0),
        "a2c_10": dict(entropy_coef=0.05),
        "a2c_11": dict(critic_coef=1.0),
    },
    "ppo": {
        "ppo_01": {},
        "ppo_02": dict(lr=1e-4),
        "ppo_03": dict(lr=1e-3),
        "ppo_04": dict(gamma=0.95),
        "ppo_05": dict(gae_lambda=0.8),
        "ppo_06": dict(clip_eps=0.1),
        "ppo_07": dict(clip_eps=0.3),
        "ppo_08": dict(epochs=3),
        "ppo_09": dict(epochs=15),
        "ppo_10": dict(entropy_coef=0.0),
        "ppo_11": dict(entropy_coef=0.05),
        "ppo_12": dict(frames_per_batch=600, minibatch=120),
    },
}


def build(cfg, env):
    n_obs, n_act = obs_act_dims()
    h = tuple(cfg["hidden"])
    actor = ProbabilisticActor(
        TensorDictModule(mlp(n_obs, n_act, h), in_keys=["observation"], out_keys=["logits"]),
        in_keys=["logits"], distribution_class=Categorical, spec=env.action_spec,
        return_log_prob=True)
    critic = ValueOperator(mlp(n_obs, 1, h), in_keys=["observation"])
    return actor, critic


def _entropy(actor, td):
    with torch.no_grad():
        return actor.get_dist(td).entropy().mean().item()


def _periodic_eval(log, actor, frames, last_eval, run_name, extra=""):
    if frames - last_eval >= 10_000:
        ev = evaluate(actor, EVAL_SEEDS_TRAIN[:10], deterministic=False)
        log.eval_curve.append([frames, ev["mean_return"], ev["success_rate"]])
        print(f"[{run_name}] frames {frames:6d} {extra} eval_return {ev['mean_return']:.1f} "
              f"success {ev['success_rate']:.0%}", flush=True)
        return frames
    return last_eval


def train_reinforce(run_name, cfg, total_frames, env, actor, critic, log):
    loss_fn = ReinforceLoss(actor, critic)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    frames, last_eval = 0, 0
    while frames < total_frames:
        with set_exploration_type(ExplorationType.RANDOM):   # sample from pi(a|s)
            eps = [env.rollout(60, actor, break_when_any_done=True)
                   for _ in range(cfg["episodes_per_update"])]
        for ep in eps:   # Monte-Carlo discounted returns G_t
            r = ep["next", "reward"].squeeze(-1)
            g, G = torch.zeros_like(r), 0.0
            for t in reversed(range(len(r))):
                G = r[t] + cfg["gamma"] * G
                g[t] = G
            ep["value_target"] = g.unsqueeze(-1)
        batch = torch.cat(eps, 0)
        frames += batch.numel()
        log.log_episodes([float(e["next", "episode_reward"][-1]) for e in eps], frames)
        with torch.no_grad():
            v = critic(batch.clone())["state_value"]
        adv = batch["value_target"] - (v if cfg["baseline"] else 0.0)
        if cfg["normalize_adv"]:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch["advantage"] = adv
        out = loss_fn(batch)
        ent = actor.get_dist(batch).entropy().mean()
        loss = out["loss_actor"] - cfg["entropy_coef"] * ent
        if cfg["baseline"]:
            loss = loss + 0.5 * out["loss_value"]
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
        optim.step()
        log.loss.append([frames, out["loss_actor"].item()])
        log.entropy.append([frames, ent.item()])
        last_eval = _periodic_eval(log, actor, frames, last_eval, run_name,
                                   f"entropy {ent.item():.3f}")


def train_ac(algo, run_name, cfg, total_frames, env, actor, critic, log):
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["gae_lambda"], value_network=critic,
              average_gae=False)
    if algo == "ppo":
        loss_fn = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip_eps"],
                              entropy_coeff=cfg["entropy_coef"], critic_coeff=cfg["critic_coef"],
                              normalize_advantage=cfg["normalize_adv"])
    else:
        loss_fn = A2CLoss(actor, critic, entropy_coeff=cfg["entropy_coef"],
                          critic_coeff=cfg["critic_coef"])
    optim = torch.optim.Adam(loss_fn.parameters(), lr=cfg["lr"])
    collector = SyncDataCollector(env, actor, frames_per_batch=cfg["frames_per_batch"],
                                  total_frames=total_frames)
    frames, last_eval = 0, 0
    for data in collector:
        frames += data.numel()
        done = data["next", "done"].squeeze(-1)
        log.log_episodes(data["next", "episode_reward"][done].flatten().tolist(), frames)
        if algo == "ppo":
            rb = ReplayBuffer(storage=LazyTensorStorage(data.numel()),
                              sampler=SamplerWithoutReplacement(), batch_size=cfg["minibatch"])
            ents, lsum = [], []
            for _ in range(cfg["epochs"]):
                with torch.no_grad():
                    gae(data)
                rb.extend(data.reshape(-1))
                for mb in rb:
                    out = loss_fn(mb)
                    loss = out["loss_objective"] + out["loss_critic"] + out["loss_entropy"]
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(loss_fn.parameters(), 0.5)
                    optim.step()
                    ents.append(out["entropy"].item())
                    lsum.append(out["loss_objective"].item())
                rb.empty()
            ent, lval = sum(ents) / len(ents), sum(lsum) / len(lsum)
        else:
            with torch.no_grad():
                gae(data)
            flat = data.reshape(-1)
            out = loss_fn(flat)
            loss = out["loss_objective"] + out["loss_critic"]
            if "loss_entropy" in out.keys():      # absent when entropy_coef == 0
                loss = loss + out["loss_entropy"]
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(loss_fn.parameters(), 0.5)
            optim.step()
            ent, lval = _entropy(actor, flat), out["loss_objective"].item()
        log.loss.append([frames, lval])
        log.entropy.append([frames, ent])
        last_eval = _periodic_eval(log, actor, frames, last_eval, run_name, f"entropy {ent:.3f}")
    collector.shutdown()


def train(algo, run_name, overrides, total_frames=TOTAL_FRAMES, seed=0):
    cfg = {**BASE[algo], **overrides}
    torch.manual_seed(seed)
    env = make_env()
    env.set_seed(seed)
    actor, critic = build(cfg, env)
    log = RunLogger(algo, run_name, cfg)
    if algo == "reinforce":
        train_reinforce(run_name, cfg, total_frames, env, actor, critic, log)
    else:
        train_ac(algo, run_name, cfg, total_frames, env, actor, critic, log)
    # policy-gradient agents learn a stochastic policy -> evaluate by sampling from it
    final = evaluate(actor, EVAL_SEEDS_TRAIN, deterministic=False)
    unseen = evaluate(actor, EVAL_SEEDS_UNSEEN, "unseen", deterministic=False)
    final["greedy"] = evaluate(actor, EVAL_SEEDS_TRAIN)
    save_policy(actor, algo, run_name, cfg, f"pg/{algo}")
    res = log.save(final, unseen)
    print(f"[{run_name}] FINAL {json.dumps(final)} UNSEEN {json.dumps(unseen)}", flush=True)
    return res


def load_policy(algo, path):
    ckpt = torch.load(path, weights_only=False)
    actor, _ = build(ckpt["config"], make_env())
    actor.load_state_dict(ckpt["state_dict"])
    return actor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=list(SWEEP), required=True)
    ap.add_argument("--run", nargs="*", default=None)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--frames", type=int, default=TOTAL_FRAMES)
    args = ap.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    names = list(SWEEP[args.algo]) if args.sweep or not args.run else args.run
    for n in names:
        train(args.algo, n, SWEEP[args.algo][n], args.frames)
