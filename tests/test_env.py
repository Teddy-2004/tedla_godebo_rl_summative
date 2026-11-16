"""Environment contract tests:  uv run pytest"""
import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTIONS, P, WolaitaCorpusEnv, simulate_wer


def test_gymnasium_api():
    check_env(WolaitaCorpusEnv(), skip_render_check=True)


def test_reset_is_seeded_and_bounded():
    a, _ = WolaitaCorpusEnv().reset(seed=7)
    b, _ = WolaitaCorpusEnv().reset(seed=7)
    assert np.array_equal(a, b) and a.shape == (19,)
    assert (a >= 0).all() and (a <= 1).all()


def test_invalid_action_is_penalised_and_changes_nothing():
    env = WolaitaCorpusEnv()
    env.reset(seed=0)
    hours, budget = env.annotated.sum(), env.budget
    _, r, *_, info = env.step(ACTIONS.index("annotate_radio"))   # backlog is empty
    assert info["invalid"] and r <= -1.0
    assert env.annotated.sum() == hours and env.budget == budget


def test_actions_change_persistent_state_and_reward_is_delayed():
    env = WolaitaCorpusEnv()
    env.reset(seed=0)
    wer0 = env.wer_ckpt
    env.step(ACTIONS.index("collect_radio"))
    assert env.raw[0] > 0
    for _ in range(3):
        env.step(ACTIONS.index("annotate_radio"))
    assert env.annotated[0] > 0 and env.wer_ckpt == wer0      # no gain before fine-tune
    _, r, *_ = env.step(ACTIONS.index("finetune_evaluate"))
    assert env.wer_ckpt < wer0 and r > 0


def test_episode_ends_at_deadline():
    env = WolaitaCorpusEnv()
    env.reset(seed=1)
    for t in range(P.max_days):
        _, _, term, trunc, _ = env.step(ACTIONS.index("rest_team"))
    assert trunc and not term and env.day == P.max_days


def test_mixed_corpus_beats_read_speech_only():
    assert simulate_wer(np.array([6, 3, 3, 3.0]), 0.05) < P.target_wer
    assert simulate_wer(np.array([0, 0, 15, 0.0]), 0.05) > 0.30


def test_unseen_profile_is_outside_training_range():
    env = WolaitaCorpusEnv(profile="unseen")
    for s in range(20):
        env.reset(seed=s)
        assert not (85 <= env.budget0 <= 115)
