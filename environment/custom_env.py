"""
WolaitaCorpusEnv - a sequential decision environment for the
"Voice of Wolaita" ASR capstone (Tedla Tesfaye Godebo, ALU).

Mission link
------------
Capstone objectives 1-2: gather >=15 h of authentic Wolaita speech and fine-tune
Whisper to WER < 20 % before the 01/03/2027 deadline. In practice this is a
resource-allocation campaign run day by day by the data lead:

* where to collect raw audio (partner radio, partner TV, read-speech sessions,
  community field recordings);
* which backlog to send to the transcription team;
* when to run a quality-assurance (QA) pass on transcripts;
* when to rest the annotators;
* when to spend GPU budget fine-tuning + evaluating a Whisper checkpoint.

Why RL (not a one-shot predictor)
---------------------------------
Every action changes persistent state (budget, backlogs, annotator fatigue,
label noise, partner goodwill, source availability) and the payoff of most
actions is *delayed*: collected audio is worthless until transcribed, and
transcripts only lower WER after a fine-tune. The agent also only *observes*
the model's WER when it pays for a fine-tune/eval and only observes label noise
when it pays for QA -> partial observability and an explore/exploit trade-off.

The WER model is a calibrated simulator (diminishing returns per domain,
domain relevance to the broadcast-heavy test set, label-noise penalty) that
mirrors the capstone research question RQ3 (impact of data type on WER).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ----------------------------------------------------------------------------
# Domains of Wolaita speech
# ----------------------------------------------------------------------------
DOMAINS = ("radio", "tv", "read", "community")
RADIO, TV, READ, COMMUNITY = range(4)

# Action ids (Discrete(11))
ACTIONS = (
    "collect_radio",        # 0  request broadcast archive from partner radio station
    "collect_tv",           # 1  request programme audio from partner TV station
    "collect_read",         # 2  run a read-speech recording session with volunteers
    "collect_community",    # 3  field trip: record spontaneous community speech
    "annotate_radio",       # 4  transcription team works on radio backlog
    "annotate_tv",          # 5  ... TV backlog
    "annotate_read",        # 6  ... read-speech backlog
    "annotate_community",   # 7  ... community backlog
    "quality_review",       # 8  QA pass: second annotator re-checks transcripts
    "rest_team",            # 9  give annotators a rest / training day
    "finetune_evaluate",    # 10 fine-tune Whisper-small on corpus + eval on test set
)
N_ACTIONS = len(ACTIONS)


@dataclass(frozen=True)
class Params:
    max_days: int = 60                       # working days until fine-tune deadline
    budget_scale: float = 140.0              # obs normaliser (1 unit = 1,000 ETB)
    # per-domain constants                    radio  tv    read  community
    collect_rate: tuple = (3.0, 2.0, 1.5, 1.0)      # raw hours per collection day
    collect_cost: tuple = (1.0, 1.5, 2.0, 2.5)      # budget units per collection day
    avail_cap: tuple = (12.0, 8.0, 6.0, 5.0)        # max un-collected hours on offer
    avail_regen: tuple = (1.0, 0.6, 0.5, 0.4)       # new hours produced per day
    annot_rate: tuple = (0.8, 0.7, 1.2, 0.5)        # transcribed hours / day (fresh team)
    annot_base_err: tuple = (0.06, 0.07, 0.03, 0.12)  # label error of fresh team
    relevance: tuple = (1.0, 0.9, 0.45, 0.8)        # usefulness for broadcast test set
    annot_cost_per_hour: float = 3.0
    qa_cost: float = 1.5
    rest_cost: float = 0.2
    ft_cost: float = 4.0
    ft_min_new_hours: float = 1.0
    wer_zero: float = 0.62                   # Whisper-small zero-shot-ish WER
    wer_floor: float = 0.11
    wer_k: float = 5.3
    target_wer: float = 0.20                 # capstone objective 2
    hours_norm: float = 20.0


P = Params()


def simulate_wer(annotated: np.ndarray, label_err: float, p: Params = P) -> float:
    """Deterministic WER of a checkpoint trained on the given corpus."""
    eff = annotated * max(0.0, 1.0 - 1.5 * label_err)
    per_domain = np.array(p.relevance) * 6.0 * np.log1p(eff / 6.0)
    coverage = float(np.sum(annotated >= 1.0))
    e = per_domain.sum() * (1.0 + 0.05 * coverage)
    return p.wer_floor + (p.wer_zero - p.wer_floor) * math.exp(-e / p.wer_k)


class WolaitaCorpusEnv(gym.Env):
    """Gymnasium environment. One step = one working day."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    OBS_NAMES = (
        "day_fraction", "budget_remaining",
        "raw_backlog_radio", "raw_backlog_tv", "raw_backlog_read", "raw_backlog_community",
        "annotated_radio", "annotated_tv", "annotated_read", "annotated_community",
        "last_checkpoint_wer", "new_hours_since_finetune", "annotator_fatigue",
        "label_error_estimate",
        "available_radio", "available_tv", "available_read", "available_community",
        "partner_goodwill",
    )

    def __init__(self, render_mode: str | None = None, profile: str = "train",
                 params: Params = P):
        super().__init__()
        self.p = params
        self.profile = profile
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(len(self.OBS_NAMES),),
                                            dtype=np.float32)
        self._viewer = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_event = ""

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        profile = (options or {}).get("profile", self.profile)
        r = self.np_random
        p = self.p
        caps = np.array(p.avail_cap)
        if profile == "unseen":
            # harder / shifted start states never seen during training
            budget = float(r.choice([r.uniform(70, 84), r.uniform(116, 130)]))
            self.avail = caps * r.uniform(0.10, 0.35, size=4)
            self.fatigue = float(r.uniform(0.30, 0.50))
            self.goodwill = float(r.uniform(0.50, 0.70))
            seed_read = float(r.uniform(0.0, 0.5))
            self.label_err = float(r.uniform(0.10, 0.14))
        else:
            budget = float(r.uniform(85, 115))
            self.avail = caps * r.uniform(0.30, 0.70, size=4)
            self.fatigue = float(r.uniform(0.0, 0.20))
            self.goodwill = float(r.uniform(0.80, 1.00))
            seed_read = float(r.uniform(0.5, 1.5))
            self.label_err = float(r.uniform(0.06, 0.10))
        self.budget0 = budget
        self.budget = budget
        self.day = 0
        self.raw = np.zeros(4)
        self.annotated = np.zeros(4)
        self.annotated[READ] = seed_read          # pilot corpus already transcribed
        self.wer_ckpt = simulate_wer(self.annotated, self.label_err, p)
        self.err_estimate = self.label_err        # pilot QA gave a measurement
        self.new_since_ft = 0.0
        self.n_finetunes = 0
        self.history = []
        self.last_action, self.last_reward, self.last_event = None, 0.0, "campaign start"
        if self.render_mode == "human":
            self.render()
        return self._obs(), self._info()

    # ------------------------------------------------------------------ helpers
    def _obs(self) -> np.ndarray:
        p = self.p
        o = np.concatenate([
            [self.day / p.max_days, self.budget / p.budget_scale],
            self.raw / p.hours_norm,
            self.annotated / p.hours_norm,
            [self.wer_ckpt, self.new_since_ft / 10.0, self.fatigue, self.err_estimate / 0.5],
            self.avail / np.array(p.avail_cap),
            [self.goodwill],
        ])
        return np.clip(o, 0.0, 1.0).astype(np.float32)

    def _info(self) -> dict:
        return {
            "day": self.day, "budget": self.budget, "checkpoint_wer": self.wer_ckpt,
            "true_wer_if_trained_now": simulate_wer(self.annotated, self.label_err, self.p),
            "annotated_hours": float(self.annotated.sum()),
            "annotated_by_domain": self.annotated.copy(), "raw_backlog": self.raw.copy(),
            "label_error": self.label_err, "fatigue": self.fatigue,
            "goodwill": self.goodwill, "finetunes": self.n_finetunes,
            "success": self.wer_ckpt < self.p.target_wer,
        }

    def action_cost(self, a: int) -> float:
        p = self.p
        if a <= 3:
            return p.collect_cost[a]
        if a <= 7:
            return p.annot_cost_per_hour * self._annot_capacity(a - 4)
        return {8: p.qa_cost, 9: p.rest_cost, 10: p.ft_cost}[a]

    def _annot_capacity(self, d: int) -> float:
        cap = self.p.annot_rate[d] * (1.0 - 0.6 * self.fatigue)
        return float(min(cap, self.raw[d]))

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = int(action)
        p = self.p
        reward = -0.05                 # time pressure: each day costs something
        invalid = False
        event = ""
        cost = self.action_cost(a)

        if cost > self.budget + 1e-9:
            invalid, event = True, "cannot afford action"
        elif a <= 3:                                             # ---- collect
            d = a
            rate = p.collect_rate[d] * (self.goodwill if d in (RADIO, TV) else 1.0)
            got = min(self.avail[d], rate)
            if got < 0.1:
                invalid, event = True, f"no new {DOMAINS[d]} audio available"
            else:
                self.avail[d] -= got
                self.raw[d] += got
                self.budget -= cost
                if d in (RADIO, TV):
                    self.goodwill = max(0.3, self.goodwill - 0.12)
                event = f"collected {got:.2f} h of {DOMAINS[d]} audio"
        elif a <= 7:                                             # ---- annotate
            d = a - 4
            h = self._annot_capacity(d)
            if h < 0.05:
                invalid, event = True, f"{DOMAINS[d]} backlog is empty"
            else:
                batch_err = p.annot_base_err[d] + 0.25 * self.fatigue
                tot = self.annotated.sum()
                self.label_err = (self.label_err * tot + batch_err * h) / (tot + h)
                self.raw[d] -= h
                self.annotated[d] += h
                self.new_since_ft += h
                self.budget -= cost
                self.fatigue = min(1.0, self.fatigue + 0.15)
                reward += 0.3 * h * p.relevance[d]
                if self.fatigue >= 0.9:
                    reward -= 1.0                     # burnout risk
                event = f"transcribed {h:.2f} h of {DOMAINS[d]} (batch err {batch_err:.0%})"
        elif a == 8:                                             # ---- QA
            if self.annotated.sum() < 0.5:
                invalid, event = True, "nothing to review"
            else:
                self.label_err = max(0.02, self.label_err * 0.65)
                self.err_estimate = self.label_err
                self.budget -= cost
                self.fatigue = min(1.0, self.fatigue + 0.05)
                event = f"QA pass: label error now {self.label_err:.1%}"
        elif a == 9:                                             # ---- rest
            self.fatigue = max(0.0, self.fatigue - 0.35)
            self.budget -= cost
            event = "team rest / training day"
        else:                                                    # ---- fine-tune
            if self.new_since_ft < p.ft_min_new_hours:
                invalid, event = True, "not enough new data to fine-tune"
            else:
                self.budget -= cost
                new_wer = simulate_wer(self.annotated, self.label_err, p)
                new_wer += float(self.np_random.normal(0, 0.004))
                reward += 100.0 * (self.wer_ckpt - new_wer)
                event = f"fine-tuned Whisper: WER {self.wer_ckpt:.1%} -> {new_wer:.1%}"
                self.wer_ckpt = float(np.clip(new_wer, 0.0, 1.0))
                self.new_since_ft = 0.0
                self.n_finetunes += 1

        if invalid:
            reward -= 1.0

        # ---- world dynamics (every day)
        self.avail = np.minimum(np.array(p.avail_cap), self.avail + np.array(p.avail_regen))
        self.goodwill = min(1.0, self.goodwill + 0.05)
        if a != 9:
            self.fatigue = max(0.0, self.fatigue - 0.03)   # small overnight recovery
        self.day += 1

        terminated, truncated = False, False
        if self.wer_ckpt < p.target_wer:
            terminated = True
            reward += 30.0 + 0.5 * (p.max_days - self.day)
            event += " | TARGET REACHED"
        elif self.budget < p.rest_cost:
            terminated = True
            reward -= 40.0 * (self.wer_ckpt - p.target_wer)
            event += " | budget exhausted"
        elif self.day >= p.max_days:
            truncated = True
            reward -= 40.0 * (self.wer_ckpt - p.target_wer)
            event += " | deadline reached"

        self.last_action, self.last_reward, self.last_event = a, reward, event
        self.history.append((self.day, a, reward, self.wer_ckpt))
        info = self._info()
        info.update(invalid=invalid, event=event, action_name=ACTIONS[a])
        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ render
    def state_dict_for_render(self) -> dict:
        return {
            "day": self.day, "max_days": self.p.max_days,
            "budget": round(self.budget, 2), "budget0": round(self.budget0, 2),
            "raw": [round(x, 2) for x in self.raw],
            "annotated": [round(x, 2) for x in self.annotated],
            "avail": [round(x, 2) for x in self.avail],
            "avail_cap": list(self.p.avail_cap),
            "wer": round(self.wer_ckpt, 4), "target": self.p.target_wer,
            "fatigue": round(self.fatigue, 3), "label_err": round(self.label_err, 4),
            "goodwill": round(self.goodwill, 3), "finetunes": self.n_finetunes,
            "action": None if self.last_action is None else ACTIONS[self.last_action],
            "action_id": self.last_action, "reward": round(self.last_reward, 3),
            "event": self.last_event,
        }

    def render(self):
        if self.render_mode == "ansi":
            s = self.state_dict_for_render()
            return (f"day {s['day']:2d} | {s['action'] or '-':18s} | r={s['reward']:+7.2f} | "
                    f"WER {s['wer']:.1%} | budget {s['budget']:6.1f} | "
                    f"hours {sum(s['annotated']):5.1f} | {s['event']}")
        if self.render_mode == "human":
            if self._viewer is None:
                from environment.rendering import ThreeJSViewer
                self._viewer = ThreeJSViewer()
            self._viewer.push(self.state_dict_for_render())

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
