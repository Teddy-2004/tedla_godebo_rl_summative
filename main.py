"""Entry point: watch the best trained agent run the Wolaita corpus campaign.

    uv sync
    uv run main.py                       # best agent, 3D GUI + verbose terminal log
    uv run main.py --algo ppo            # best PPO / dqn / a2c / reinforce model
    uv run main.py --profile unseen      # start from unseen (harder) initial states
    uv run main.py --random              # untrained random agent, for contrast
    uv run main.py --no-gui --episodes 5 # terminal only

"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from environment.custom_env import ACTIONS, WolaitaCorpusEnv

ROOT = Path(__file__).parent
BEST = ROOT / "models" / "best.json"


def load_agent(algo: str | None):
    """Returns (act_fn, description). act_fn maps a numpy observation to an action id."""
    if not BEST.exists():
        return None, "no trained models found (run the training scripts first)"
    best = json.loads(BEST.read_text())
    algo = algo or best["overall"]
    entry = best["per_algo"][algo]
    path = ROOT / entry["model_path"]
    if algo == "dqn":
        from training.dqn_training import load_policy
        policy, mode = load_policy(path), ExplorationType.DETERMINISTIC
    else:
        from training.pg_training import load_policy
        policy, mode = load_policy(algo, path), ExplorationType.RANDOM   # stochastic policy

    @torch.no_grad()
    def act(obs) -> int:
        td = TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, [])
        with set_exploration_type(mode):
            return int(policy(td)["action"])

    return act, f"{algo.upper()} run {entry['run']} ({entry['model_path']})"


def choose_model() -> str | None:
    """Interactive model picker. Enter (or a non-interactive run) selects the best model."""
    if not BEST.exists():
        return None
    best = json.loads(BEST.read_text())
    algos = list(best["per_algo"])
    algos.sort(key=lambda a: -best["per_algo"][a]["final_eval"]["mean_return"])
    if not sys.stdin.isatty():
        return best["overall"]
    print("\nTrained models (best run of each algorithm, 40 held-out test campaigns):")
    for i, a in enumerate(algos, 1):
        e = best["per_algo"][a]
        tag = "  <- best" if a == best["overall"] else ""
        print(f"  [{i}] {a.upper():10s} {e['run']:13s} mean return {e['final_eval']['mean_return']:5.1f} | "
              f"success {e['final_eval']['success_rate']:.0%} | unseen-start success "
              f"{e['unseen_eval']['success_rate']:.0%}{tag}")
    choice = input(f"Select a model [1-{len(algos)}] (Enter = best): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(algos):
        return algos[int(choice) - 1]
    return best["overall"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--algo", choices=["dqn", "reinforce", "a2c", "ppo"], default=None)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--profile", choices=["train", "unseen"], default="train")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--delay", type=float, default=0.7, help="seconds per simulated day")
    ap.add_argument("--random", action="store_true", help="use an untrained random agent")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--no-browser", action="store_true", help="start viewer but do not open a tab")
    args = ap.parse_args()

    if not args.random and args.algo is None:
        args.algo = choose_model()
    act, desc = (None, "random agent") if args.random else load_agent(args.algo)
    if act is None:
        rng = np.random.default_rng(args.seed)
        act = lambda obs: int(rng.integers(len(ACTIONS)))   # noqa: E731
        desc = f"RANDOM agent - {desc}" if not args.random else "RANDOM agent"
    torch.manual_seed(args.seed)

    print("=" * 100)
    print(" Voice of Wolaita - RL agent managing the ASR corpus campaign")
    print(f" Agent  : {desc}")
    print(f" Goal   : Whisper WER < 20% within 60 working days and the budget | start states: {args.profile}")
    print("=" * 100)

    env = WolaitaCorpusEnv(profile=args.profile)
    if not args.no_gui:
        from environment.rendering import ThreeJSViewer
        env._viewer = ThreeJSViewer(open_browser=not args.no_browser)
        env._viewer.set_policy(lambda o: (lambda a: {"action": a, "name": ACTIONS[a]})(act(o)))
        env.render_mode = "human"
        time.sleep(2.5)            # let the browser tab load

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        print(f"\n--- Episode {ep + 1}/{args.episodes} | budget {info['budget']:.1f}k ETB | "
              f"start WER {info['checkpoint_wer']:.1%} ---")
        print(f"{'day':>3} | {'action':<19}| {'reward':>7} | {'return':>7} | {'WER':>6} | "
              f"{'budget':>6} | {'hours':>5} | {'fatigue':>7} | event")
        total, done = 0.0, False
        while not done:
            a = act(obs)
            obs, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
            print(f"{info['day']:>3} | {ACTIONS[a]:<19}| {r:>+7.2f} | {total:>7.1f} | "
                  f"{info['checkpoint_wer']:>6.1%} | {info['budget']:>6.1f} | "
                  f"{info['annotated_hours']:>5.1f} | {info['fatigue']:>7.0%} | {info['event']}",
                  flush=True)
            if not args.no_gui:
                time.sleep(args.delay)
        h = info["annotated_by_domain"]
        print(f">>> Episode {ep + 1}: return {total:.1f} | final WER {info['checkpoint_wer']:.1%} | "
              f"{'SUCCESS' if info['success'] else 'target missed'} on day {info['day']} | "
              f"corpus {info['annotated_hours']:.1f} h (radio {h[0]:.1f}, tv {h[1]:.1f}, "
              f"read {h[2]:.1f}, community {h[3]:.1f}) | budget left {info['budget']:.1f}")
        results.append((total, info["success"], info["checkpoint_wer"], info["day"]))
        if not args.no_gui:
            time.sleep(2.0)

    r = np.array(results, dtype=float)
    print("\n" + "=" * 100)
    print(f" Summary over {len(r)} episodes: mean return {r[:, 0].mean():.1f} | "
          f"success rate {r[:, 1].mean():.0%} | mean final WER {r[:, 2].mean():.1%} | "
          f"mean days used {r[:, 3].mean():.1f}")
    print("=" * 100)
    if not args.no_gui:
        print("Viewer still open - press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    env.close()


if __name__ == "__main__":
    main()
