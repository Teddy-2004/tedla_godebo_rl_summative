from gymnasium.envs.registration import register

from environment.custom_env import ACTIONS, DOMAINS, WolaitaCorpusEnv, simulate_wer

register(id="WolaitaCorpus-v0", entry_point="environment.custom_env:WolaitaCorpusEnv",
         max_episode_steps=60)

__all__ = ["WolaitaCorpusEnv", "ACTIONS", "DOMAINS", "simulate_wer"]
