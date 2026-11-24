"""Aggregate sweep logs -> result tables, best-model index and report figures.

    uv run python -m training.analysis
"""
from __future__ import annotations

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.common import LOG_DIR, MODEL_DIR, ROOT

ALGOS = ["dqn", "reinforce", "a2c", "ppo"]
LABEL = {"dqn": "DQN", "reinforce": "REINFORCE", "a2c": "A2C", "ppo": "PPO"}
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "a2c": "#1baf7a", "ppo": "#eda100"}
ASSETS = ROOT / "assets"
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})

COLUMNS = {
    "dqn": ["lr", "gamma", "buffer_size", "batch_size", "eps_decay_frac", "eps_end",
            "target_update", "target_interval", "tau", "double_dqn", "prioritized", "hidden"],
    "reinforce": ["lr", "gamma", "episodes_per_update", "baseline", "entropy_coef",
                  "normalize_adv", "hidden"],
    "a2c": ["lr", "gamma", "gae_lambda", "frames_per_batch", "entropy_coef", "critic_coef"],
    "ppo": ["lr", "gamma", "gae_lambda", "clip_eps", "epochs", "frames_per_batch",
            "minibatch", "entropy_coef"],
}


def load():
    runs = {a: [] for a in ALGOS}
    for a in ALGOS:
        for f in sorted((LOG_DIR / a).glob("*.json")):
            runs[a].append(json.loads(f.read_text()))
    return runs


def ma(x, w=50):
    x = np.asarray(x, float)
    if len(x) < w:
        return np.arange(len(x)), x
    return np.arange(w - 1, len(x)), np.convolve(x, np.ones(w) / w, mode="valid")


def stability(run):
    """Std of the 50-episode moving average over the last 30 % of training."""
    _, m = ma(run["episode_returns"])
    return float(np.std(m[int(0.7 * len(m)):]))


def tables(runs):
    best = {"per_algo": {}}
    for a in ALGOS:
        rows = []
        for r in runs[a]:
            c, fe, ue = r["config"], r["final_eval"], r["unseen_eval"]
            row = {"run": r["run"], **{k: c[k] for k in COLUMNS[a]},
                   "mean_reward": round(fe["mean_return"], 1), "std": round(fe["std_return"], 1),
                   "success_rate": round(fe["success_rate"], 2),
                   "final_wer": round(fe["mean_final_wer"], 3),
                   "unseen_reward": round(ue["mean_return"], 1),
                   "unseen_success": round(ue["success_rate"], 2),
                   "episodes_to_converge": r["episodes_to_converge"],
                   "late_stability_std": round(stability(r), 1)}
            rows.append(row)
        if not rows:
            continue
        with open(LOG_DIR / f"results_{a}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        top = max(runs[a], key=lambda r: r["final_eval"]["mean_return"])
        sub = "dqn" if a == "dqn" else f"pg/{a}"
        best["per_algo"][a] = {
            "run": top["run"], "model_path": f"models/{sub}/{top['run']}.pt",
            "config": top["config"], "final_eval": top["final_eval"],
            "unseen_eval": top["unseen_eval"],
            "episodes_to_converge": top["episodes_to_converge"]}
    best["overall"] = max(best["per_algo"],
                          key=lambda a: best["per_algo"][a]["final_eval"]["mean_return"])
    (MODEL_DIR / "best.json").write_text(json.dumps(best, indent=2))
    return best


def best_runs(runs, best):
    return {a: next(r for r in runs[a] if r["run"] == best["per_algo"][a]["run"])
            for a in best["per_algo"]}


def fig_rewards(br):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharey=True)
    for ax, a in zip(axes.flat, br):
        r = br[a]["episode_returns"]
        ax.plot(r, color=COLOR[a], alpha=0.22, lw=0.6)
        x, m = ma(r)
        ax.plot(x, m, color=COLOR[a], lw=2)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(f"{LABEL[a]} - best run ({br[a]['run']})", color=INK)
        ax.annotate(f"last-100 mean {np.mean(r[-100:]):.1f}", (0.98, 0.06),
                    xycoords="axes fraction", ha="right", color=INK)
    for ax in axes[1]:
        ax.set_xlabel("training episode")
    for ax in axes[:, 0]:
        ax.set_ylabel("episode return")
    fig.suptitle("Cumulative reward per episode (faint = raw, bold = 50-episode mean)",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig1_cumulative_rewards.png")
    plt.close(fig)


def fig_stability(br):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    ax = axes[0]
    l = np.array(br["dqn"]["loss"])
    ax.plot(l[:, 0] / 1000, l[:, 1], color=COLOR["dqn"], alpha=0.3, lw=0.7)
    x, m = ma(l[:, 1], 25)
    ax.plot(l[x, 0] / 1000, m, color=COLOR["dqn"], lw=2)
    ax.set_title("DQN objective: smooth-L1 TD loss", color=INK)
    ax.set_xlabel("environment frames (thousands)")
    ax.set_ylabel("TD loss")
    ax = axes[1]
    for a in ["reinforce", "a2c", "ppo"]:
        e = np.array(br[a]["entropy"])
        x, m = ma(e[:, 1], max(1, len(e) // 60))
        ax.plot(e[x, 0] / 1000, m, color=COLOR[a], lw=2, label=LABEL[a])
    ax.axhline(np.log(11), color=MUTED, lw=0.8, ls="--")
    ax.annotate("uniform policy (ln 11)", (2, np.log(11) - 0.02), va="top", color=MUTED,
                fontsize=8)
    ax.set_title("Policy entropy (policy-gradient methods)", color=INK)
    ax.set_xlabel("environment frames (thousands)")
    ax.set_ylabel("entropy (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "fig2_training_stability.png")
    plt.close(fig)


def fig_convergence(br):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    for a in br:
        x, m = ma(br[a]["episode_returns"])
        ax.plot(x, m, color=COLOR[a], lw=2, label=LABEL[a])
        c = br[a]["episodes_to_converge"]
        if c:
            ax.plot([c], [m[min(c - 50, len(m) - 1)]], "o", color=COLOR[a], ms=7,
                    mec="white", mew=1.5)
    ax.set_title("Training return (50-ep mean); dot = 90% of final level", color=INK)
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode return")
    ax.legend()
    ax = axes[1]
    for a in br:
        e = np.array(br[a]["eval_curve"])
        ax.plot(e[:, 0] / 1000, e[:, 1], color=COLOR[a], lw=2, marker="o", ms=3.5,
                label=LABEL[a])
    ax.set_title("Periodic evaluation on 10 held-out seeds", color=INK)
    ax.set_xlabel("environment frames (thousands)")
    ax.set_ylabel("mean evaluation return")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig3_convergence.png")
    plt.close(fig)


def fig_generalization(best):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    algos = list(best["per_algo"])
    x = np.arange(len(algos))
    for ax, key, title, fmt in [(axes[0], "mean_return", "Mean return", "{:.0f}"),
                                (axes[1], "success_rate", "Success rate (WER < 20%)", "{:.0%}")]:
        seen = [best["per_algo"][a]["final_eval"][key] for a in algos]
        uns = [best["per_algo"][a]["unseen_eval"][key] for a in algos]
        b1 = ax.bar(x - 0.2, seen, 0.36, color=[COLOR[a] for a in algos], label="seen start distribution")
        b2 = ax.bar(x + 0.2, uns, 0.36, color=[COLOR[a] for a in algos], alpha=0.45,
                    hatch="///", edgecolor="white", label="unseen start states")
        for bars, vals in ((b1, seen), (b2, uns)):
            for b, v in zip(bars, vals):
                ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, max(v, 0)),
                            ha="center", va="bottom", fontsize=8, color=INK,
                            xytext=(0, 2), textcoords="offset points")
        ax.set_xticks(x, [LABEL[a] for a in algos])
        ax.tick_params(axis="x", colors=INK)
        ax.set_title(title, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=MUTED),
                        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.45, hatch="///",
                                      edgecolor="white")],
               labels=["held-out seeds, training distribution", "unseen initial states"],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Generalisation of each algorithm's best model (40 episodes each)",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig4_generalization.png")
    plt.close(fig)


def fig_sweep(runs, best):
    fig, ax = plt.subplots(figsize=(9, 3.0))
    rng = np.random.default_rng(0)
    for i, a in enumerate(ALGOS):
        v = [r["final_eval"]["mean_return"] for r in runs[a]]
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v, s=42, color=COLOR[a],
                   edgecolor="white", linewidth=1.2, zorder=3)
        ax.hlines(np.median(v), i - 0.28, i + 0.28, color=INK, lw=1.5, zorder=4)
        ax.annotate(f"median {np.median(v):.0f}\nbest {max(v):.0f}", (i + 0.32, np.median(v)),
                    va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(ALGOS)), [f"{LABEL[a]} ({len(runs[a])} runs)" for a in ALGOS])
    ax.tick_params(axis="x", colors=INK)
    ax.set_xlim(-0.5, len(ALGOS) - 0.3)
    ax.set_ylabel("final mean return")
    ax.grid(axis="x", visible=False)
    ax.set_title("Hyperparameter sensitivity: final evaluation return of every sweep run",
                 color=INK)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig5_sweep_sensitivity.png")
    plt.close(fig)


def main():
    ASSETS.mkdir(exist_ok=True)
    runs = load()
    best = tables(runs)
    br = best_runs(runs, best)
    fig_rewards(br)
    fig_stability(br)
    fig_convergence(br)
    fig_generalization(best)
    fig_sweep(runs, best)
    for a, b in best["per_algo"].items():
        fe, ue = b["final_eval"], b["unseen_eval"]
        print(f"{LABEL[a]:10s} best={b['run']:13s} return {fe['mean_return']:6.1f} "
              f"success {fe['success_rate']:.0%} | unseen {ue['mean_return']:6.1f} "
              f"success {ue['success_rate']:.0%} | converge @ ep {b['episodes_to_converge']}")
    print("overall best:", best["overall"])


if __name__ == "__main__":
    main()
