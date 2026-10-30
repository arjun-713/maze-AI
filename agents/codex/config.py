"""Strict, serializable configuration; reading configuration never runs an agent."""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 15
    global_size: int = 21
    max_grid: int = 127
    hidden_size: int = 128

    def validate(self) -> None:
        for key in ("patch_size", "global_size", "max_grid", "hidden_size"):
            value = getattr(self, key)
            if type(value) is not int or value < 5:
                raise ValueError(f"model.{key} must be an integer >= 5")
        if self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        if self.max_grid < 83:
            raise ValueError("max_grid must support the 83-block level-35 board")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str = "cpu"
    threads: int = 2
    deterministic: bool = True
    total_steps: int = 5_000_000
    num_envs: int = 8
    rollout_steps: int = 256
    sequence_length: int = 32
    burn_in: int = 16
    sequences_per_batch: int = 8
    epochs: int = 4
    learning_rate: float = 0.0003
    final_lr_fraction: float = 0.1
    gamma: float = 0.999
    intrinsic_gamma: float = 0.99
    gae_lambda: float = 0.97
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_start: float = 0.02
    entropy_end: float = 0.003
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    reward_scale: float = 100.0
    step_cost: float = 0.002
    first_visit_bonus: float = 0.05
    discovery_bonus: float = 0.002
    intrinsic_start: float = 1.0
    intrinsic_end: float = 0.05
    intrinsic_anneal_fraction: float = 0.8
    start_level: int = 1
    max_level: int = 35
    curriculum_window: int = 50
    curriculum_threshold: float = 0.8
    curriculum_min_episodes: int = 50
    rehearsal_probability: float = 0.3
    validation_every: int = 50
    validation_episodes: int = 10
    promotion_validation_threshold: float = 0.7
    checkpoint_every: int = 25
    record_every: int = 25
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        self.model.validate()
        for key in (
            "threads",
            "total_steps",
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "sequences_per_batch",
            "epochs",
            "curriculum_window",
            "curriculum_min_episodes",
            "validation_every",
            "validation_episodes",
            "checkpoint_every",
            "record_every",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.burn_in) is not int or not 0 <= self.burn_in <= self.sequence_length:
            raise ValueError("burn_in must be between zero and sequence_length")
        if self.rollout_steps % self.sequence_length:
            raise ValueError("rollout_steps must be divisible by sequence_length")
        if self.total_steps < self.num_envs * self.sequence_length:
            raise ValueError("total_steps must allow at least one recurrent sequence batch")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if (
            type(self.start_level) is not int
            or type(self.max_level) is not int
            or not 1 <= self.start_level <= self.max_level <= 35
        ):
            raise ValueError("levels must satisfy 1 <= start_level <= max_level <= 35")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a JSON boolean")
        integer_keys = {
            "seed",
            "threads",
            "total_steps",
            "num_envs",
            "rollout_steps",
            "sequence_length",
            "burn_in",
            "sequences_per_batch",
            "epochs",
            "start_level",
            "max_level",
            "curriculum_window",
            "curriculum_min_episodes",
            "validation_every",
            "validation_episodes",
            "checkpoint_every",
            "record_every",
        }
        for key, value in asdict(self).items():
            if key not in integer_keys | {"device", "deterministic", "model"} and type(value) not in (
                int,
                float,
            ):
                raise ValueError(f"{key} must be a number")
        if self.device not in ("cpu", "cuda", "mps", "auto"):
            raise ValueError("device must be cpu, cuda, mps, or auto")
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        for key in (
            "gamma",
            "intrinsic_gamma",
            "gae_lambda",
            "curriculum_threshold",
            "rehearsal_probability",
            "promotion_validation_threshold",
            "final_lr_fraction",
            "intrinsic_anneal_fraction",
        ):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        for key in (
            "learning_rate",
            "clip_ratio",
            "value_clip",
            "value_coefficient",
            "max_grad_norm",
            "target_kl",
            "reward_scale",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "entropy_start",
            "entropy_end",
            "step_cost",
            "first_visit_bonus",
            "discovery_bonus",
            "intrinsic_start",
            "intrinsic_end",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} must be nonnegative")


def from_dict(data: dict) -> Config:
    data = dict(data)
    data["model"] = ModelConfig(**data.get("model", {}))
    result = Config(**data)
    result.validate()
    return result


def load_config(path: str | Path | None = None) -> Config:
    return from_dict(json.loads(Path(path).read_text()) if path else {})
