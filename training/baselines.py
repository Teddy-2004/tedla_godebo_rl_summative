"""Reference points: a random agent and a hand-written rule-based planner,
evaluated on the same seeds as the trained agents.

    uv run python -m training.baselines
"""
from __future__ import annotations

import json

import numpy as np

from environment.custom_env import N_ACTIONS, WolaitaCorpusEnv
from training.common import EVAL_SEEDS_TRAIN, EVAL_SEEDS_UNSEEN, LOG_DIR


def rule_based(e: WolaitaCorpusEnv) -> int:
    """Rest when tired, fine-tune every 3 new hours, keep a mixed 6/3/3/3 h corpus plan."""
    if e.fatigue > 0.55:
        return 9
    if e.new_since_ft >= 3.0 or (e.day >= 55 and e.new_since_ft >= 1.0):
        return 10
    if e.label_err > 0.07 and e.annotated.sum() > 3 and e.day % 7 == 0:
        return 8
    target = np.array([6.0, 3.0, 3.0, 3.0])
    need = target - (e.annotated + e.raw)
    for d in np.argsort(-need):
        if need[d] > 0 and e.avail[d] > 0.5 and e.raw[d] < 2:
            return int(d)
    for d in np.argsort(-(target - e.annotated)):
        if e.raw[d] > 0.05:
            return 4 + int(d)
    for d in (0, 1, 3, 2):
        if e.avail[d] > 0.5:
            return d
    return 9


def run(policy, seeds, profile):
    rets, succ = [], []
    for s in seeds:
        env = WolaitaCorpusEnv(profile=profile)
        env.reset(seed=s)
        rng = np.random.default_rng(s)
        total, done = 0.0, False
        while not done:
            a = int(rng.integers(N_ACTIONS)) if policy is None else policy(env)
            _, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
        succ.append(info["success"])
    return {"mean_return": float(np.mean(rets)), "success_rate": float(np.mean(succ))}


if __name__ == "__main__":
    out = {}
    for name, pol in (("random", None), ("rule_based", rule_based)):
        out[name] = {"seen": run(pol, EVAL_SEEDS_TRAIN, "train"),
                     "unseen": run(pol, EVAL_SEEDS_UNSEEN, "unseen")}
        print(name, json.dumps(out[name]))
    (LOG_DIR / "baselines.json").write_text(json.dumps(out, indent=1))
