"""Procedural seed separation and performance-gated rehearsal curriculum."""

from collections import deque

import numpy as np

from .activation import require_ready
from .config import Config


def maze_seed(master: int, split: str, level: int, counter: int) -> int:
    require_ready()
    domains = {"train": 1, "validation": 2, "evaluation": 3}
    if split not in domains or not 0 <= master < 2**31 or not 1 <= level <= 35 or not 0 <= counter < 2**48:
        raise ValueError("invalid procedural seed namespace")
    return (domains[split] << 96) | (master << 64) | (level << 48) | counter


class Curriculum:
    def __init__(self, config: Config):
        require_ready()
        self.config = config
        self.frontier = config.start_level
        self.counter = 0
        self.rng = np.random.default_rng(config.seed + 721)
        self.history = {level: deque(maxlen=config.curriculum_window) for level in range(1, 36)}
        self.counts = {level: 0 for level in range(1, 36)}

    def next_episode(self) -> tuple[int, int]:
        require_ready()
        level = self.frontier
        if self.frontier > self.config.start_level and self.rng.random() < self.config.rehearsal_probability:
            levels = list(range(self.config.start_level, self.frontier))
            # Weak earlier levels get more rehearsal; all retain nonzero weight.
            weights = np.array([1.1 - np.mean(self.history[x]) if self.history[x] else 1.1 for x in levels])
            level = int(self.rng.choice(levels, p=weights / weights.sum()))
        seed = maze_seed(self.config.seed, "train", level, self.counter)
        self.counter += 1
        return level, seed

    def record(self, level: int, success: bool) -> None:
        require_ready()
        self.history[level].append(float(success))
        self.counts[level] += 1

    def promote(self, validation_success: float) -> bool:
        require_ready()
        history = self.history[self.frontier]
        if (
            self.frontier < self.config.max_level
            and len(history) >= self.config.curriculum_window
            and self.counts[self.frontier] >= self.config.curriculum_min_episodes
            and np.mean(history) >= self.config.curriculum_threshold
            and validation_success >= self.config.promotion_validation_threshold
        ):
            self.frontier += 1
            return True
        return False

    def state_dict(self) -> dict:
        require_ready()
        return {
            "frontier": self.frontier,
            "counter": self.counter,
            "rng": self.rng.bit_generator.state,
            "history": {str(k): list(v) for k, v in self.history.items()},
            "counts": self.counts,
        }

    def load_state_dict(self, state: dict) -> None:
        require_ready()
        self.frontier, self.counter = state["frontier"], state["counter"]
        self.rng.bit_generator.state = state["rng"]
        self.history = {
            int(k): deque(v, maxlen=self.config.curriculum_window) for k, v in state["history"].items()
        }
        self.counts = {int(k): v for k, v in state["counts"].items()}
