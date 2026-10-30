"""Checkpoint-only competition agent and shared inference state. Never learns online."""

import time
from pathlib import Path

from .activation import require_ready


class Policy:
    """Per-episode observation memory with a private reproducible action generator."""

    def __init__(self, model, config, device, seed: int, greedy: bool = False):
        require_ready()
        import torch

        from .memory import SpatialMemory

        self.model, self.config, self.device, self.greedy = model, config, device, greedy
        self.memory = SpatialMemory(config.model)
        self.hidden = torch.zeros((1, config.model.hidden_size), device=device)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.started = False
        self.last_score = 0
        self.last_action = -1

    def reset(self, observation: dict) -> None:
        require_ready()
        import torch

        self.memory.reset(observation)
        self.hidden = torch.zeros((1, self.config.model.hidden_size), device=self.device)
        self.started = True
        self.last_score = observation["score"]
        self.last_action = -1
        self.last_moves = -1

    def act(self, observation: dict):
        require_ready()
        import numpy as np
        import torch
        from maze_game.training import AgentDecision

        from .memory import ACTIONS
        from .network import sample_action, tensor_inputs

        if not self.started or (observation["moves"] == 0 and self.last_moves >= 0):
            self.reset(observation)
        if observation["moves"] <= self.last_moves:
            raise ValueError("one act call is allowed per observation; reset between episodes")
        if self.last_action >= 0:
            self.memory.feedback(self.last_action, observation["score"] - self.last_score)
        self.memory.observe(observation)
        inputs = tensor_inputs(
            {k: np.expand_dims(v, 0) for k, v in self.memory.encode(observation).items()}, self.device
        )
        start = time.perf_counter()
        with torch.inference_mode():
            logits, values, self.hidden = self.model(
                inputs, self.hidden, torch.tensor([False], device=self.device)
            )
            probabilities = logits.softmax(-1)
            index = int(sample_action(logits, self.generator, self.greedy).item())
            probs = probabilities[0].cpu().tolist()  # Synchronizes CUDA before timing ends.
            value = values[0].cpu().tolist()
        self.last_score, self.last_action, self.last_moves = observation["score"], index, observation["moves"]
        debug = {
            "action_probabilities": dict(zip(ACTIONS, probs)),
            "extrinsic_value": value[0],
            "intrinsic_value": value[1],
            "seen_cells": int(self.memory.flags[0].sum()),
            "inference_seconds": time.perf_counter() - start,
            "learning": False,
        }
        return AgentDecision(ACTIONS[index], debug)


class CodexAgent:
    """Existing studio interface: act(observation) -> AgentDecision. Constructor is gated."""

    name = "Codex Memory PPO"

    @property
    def ready(self) -> bool:
        from . import activation

        return activation.READY is True

    def __init__(
        self, seed: int = 42, checkpoint: str | Path | None = None, device: str = "cpu", greedy: bool = False
    ):
        require_ready()
        import torch

        from .network import Navigator
        from .runtime import read_checkpoint, select_device

        chosen = Path(checkpoint) if checkpoint else Path(__file__).parent / "artifacts" / "champion.pt"
        payload, config = read_checkpoint(chosen)
        torch.set_num_threads(config.threads)
        target = select_device(device)
        # Keep random initialization from changing a training caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            model = Navigator(config.model)
        model.load_state_dict(payload["model"], strict=True)
        model.to(target).eval().requires_grad_(False)
        self.policy = Policy(model, config, target, seed, greedy)

    def on_episode_start(self, observation: dict) -> None:
        require_ready()
        self.policy.reset(observation)

    def act(self, observation: dict):
        require_ready()
        return self.policy.act(observation)

    def on_transition(self, transition) -> None:
        require_ready()
        # Feedback is derived from the next permitted observation in act(). There
        # is no optimizer, parameter update, or hidden-state access in this adapter.
