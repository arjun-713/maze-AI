"""Synchronous batched recurrent PPO with public-API resume and a 35-level curriculum."""

import json
import signal
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .activation import require_ready
from .config import Config
from .curriculum import Curriculum, maze_seed
from .memory import ACTIONS, SpatialMemory
from .network import Navigator, sample_action, tensor_inputs
from .ppo import Rollout, optimize
from .recording import StudioRecorder
from .runtime import append_json, configure, read_checkpoint, restore_rng, save_checkpoint


class Slot:
    """An actor owns only a normal MazeEnv and its publicly accumulated observations."""

    def __init__(self, config: Config, level: int, seed: int, record: bool):
        require_ready()
        from maze_game import MazeEnv
        from maze_game.training import EpisodeRecord

        self.config, self.level, self.seed = config, level, seed
        self.env = MazeEnv(level=level, seed=seed)
        self.observation = self.env.observation
        self.env.pause_timer()
        self.memory = SpatialMemory(config.model)
        self.memory.reset(self.observation)
        self.actions = []
        self.traps = self.walls = 0
        self.intrinsic_return = 0.0
        self.time_offset = 0.0
        self.record = (
            EpisodeRecord((seed & (2**48 - 1)) + 1, level, seed, tuple(self.observation["position"]))
            if record
            else None
        )

    @property
    def seconds(self) -> float:
        return self.time_offset + self.env.elapsed_seconds

    def step(self, action: int, debug: dict) -> tuple[np.ndarray, bool]:
        require_ready()
        from maze_game.training import StepRecord

        self.env.resume_timer()
        observation, raw, done, truncated, info = self.env.step(action)
        self.actions.append(action)
        self.traps += int(info["entered_trap"])
        self.walls += int(info["hit_wall"])
        self.memory.feedback(action, raw)
        new_cells, first = self.memory.observe(observation)
        intrinsic = self.config.first_visit_bonus * float(first) + self.config.discovery_bonus * new_cells
        self.intrinsic_return += intrinsic
        self.observation = observation
        if self.record is not None:
            self.record.steps.append(
                StepRecord(
                    ACTIONS[action],
                    raw,
                    tuple(observation["position"]),
                    info["hit_wall"],
                    info["entered_trap"],
                    info["extra_move"],
                    done,
                    truncated,
                    self.seconds,
                    debug,
                )
            )
        # Both done and move-budget exhaustion are true episodic terminals here.
        return np.array(
            [raw / self.config.reward_scale - self.config.step_cost, intrinsic], dtype=np.float32
        ), bool(done or truncated)

    def summary(self) -> dict:
        require_ready()
        return {
            "level": self.level,
            "tier": (self.level - 1) // 5 + 1,
            "seed": self.seed,
            "score": self.observation["score"],
            "moves": self.observation["moves"],
            "traps": self.traps,
            "walls": self.walls,
            "success": bool(self.env.done),
            "truncated": bool(self.env.truncated),
            "seconds": self.seconds,
            "intrinsic_return": self.intrinsic_return,
        }

    def state_dict(self) -> dict:
        require_ready()
        return {
            "level": self.level,
            "seed": self.seed,
            "actions": self.actions,
            "observation": self.observation,
            "seconds": self.seconds,
            "record": asdict(self.record) if self.record is not None else None,
        }

    @classmethod
    def restore(cls, config: Config, state: dict):
        require_ready()
        from maze_game.training import EpisodeRecord, StepRecord

        slot = cls(config, state["level"], state["seed"], False)
        # The env has no public snapshot API. Recreate through reset/step only;
        # no policy inference or learning occurs during this explicit resume.
        for action in state["actions"]:
            slot.step(action, {})
        if slot.observation != state["observation"]:
            raise ValueError("environment changed: resumed public trajectory does not match checkpoint")
        slot.env.pause_timer()
        slot.time_offset = state["seconds"] - slot.env.elapsed_seconds
        saved = state["record"]
        if saved is not None:
            slot.record = EpisodeRecord(
                saved["number"],
                saved["level"],
                saved["seed"],
                tuple(saved["start"]),
                [StepRecord(**{**step, "position": tuple(step["position"])}) for step in saved["steps"]],
            )
        return slot


class Trainer:
    def __init__(self, config: Config, directory: Path, resume: Path | None = None):
        require_ready()
        self.config = config
        self.directory = directory
        payload = None
        if resume:
            payload, saved_config = read_checkpoint(resume)
            # Only total budget and runtime placement can change on resume. This
            # prevents silent optimizer, observation, or curriculum incompatibility.
            if (
                replace(
                    config,
                    total_steps=saved_config.total_steps,
                    device=saved_config.device,
                    threads=saved_config.threads,
                )
                != saved_config
            ):
                raise ValueError("resume may change only total_steps, device, and threads")
            if payload["optimizer"] is None or payload["state"] is None:
                raise ValueError("checkpoint has no resumable training state")
        elif (directory / "config.json").exists():
            raise FileExistsError("run already exists; choose a new directory or pass --resume")
        self.device = configure(config)
        self.model = Navigator(config.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.curriculum = Curriculum(config)
        self.rng = np.random.default_rng(config.seed + 1729)
        self.steps = self.updates = self.completed = 0
        self.best_key = (-1.0, -1.0, -float("inf"))
        self.hidden = torch.zeros((config.num_envs, config.model.hidden_size), device=self.device)
        self.starts = np.ones(config.num_envs, dtype=bool)
        run_ids = {}
        if payload:
            self.model.load_state_dict(payload["model"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer"])
            state = payload["state"]
            self.steps, self.updates, self.completed = state["steps"], state["updates"], state["completed"]
            self.best_key = tuple(state["best_key"])
            self.curriculum.load_state_dict(state["curriculum"])
            self.rng.bit_generator.state = state["batch_rng"]
            self.slots = [Slot.restore(config, item) for item in state["slots"]]
            self.hidden = state["hidden"].to(self.device)
            self.starts = state["starts"].cpu().numpy().astype(bool)
            if len(self.slots) != config.num_envs or self.hidden.shape != (
                config.num_envs,
                config.model.hidden_size,
            ):
                raise ValueError("checkpoint actor dimensions do not match configuration")
            run_ids = state["run_ids"]
            restore_rng(payload, self.device)
        else:
            self.slots = [self.new_slot() for _ in range(config.num_envs)]
        directory.mkdir(parents=True, exist_ok=True)
        if payload:
            offsets = payload["state"]["log_offsets"]
            if set(offsets) != {"episodes.jsonl", "updates.jsonl", "validation.jsonl"}:
                raise ValueError("checkpoint has invalid telemetry offsets")
            for name, offset in offsets.items():
                if type(offset) is not int or offset < 0:
                    raise ValueError("checkpoint has an invalid log byte offset")
                path = directory / name
                if path.exists() and path.stat().st_size < offset:
                    raise ValueError(
                        f"{path} is shorter than its checkpoint offset; resume in a new directory"
                    )
            for name, offset in offsets.items():
                path = directory / name
                if path.exists():
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
        (directory / "config.json").write_text(json.dumps(asdict(config), indent=2))
        self.recorder = StudioRecorder(directory / "recordings", directory.name, run_ids)
        if payload:
            self.recorder.rollback(self.curriculum.counter, [slot.record for slot in self.slots], config.seed)
        self.started = time.perf_counter()
        self.stop_requested = False
        self.last_metrics = {}
        for slot in self.slots:
            slot.env.pause_timer()

    def new_slot(self) -> Slot:
        require_ready()
        level, seed = self.curriculum.next_episode()
        return Slot(self.config, level, seed, (self.curriculum.counter - 1) % self.config.record_every == 0)

    def collect(self, length: int) -> Rollout:
        require_ready()
        self.model.eval()
        frames, hidden, starts, actions, logs, values, rewards, ended = [], [], [], [], [], [], [], []
        for slot in self.slots:
            slot.env.resume_timer()
        for _ in range(length):
            require_ready()
            encoded = [slot.memory.encode(slot.observation) for slot in self.slots]
            frame = {
                key: np.stack([item[key] for item in encoded])
                for key in ("patch", "global", "scalars", "legal")
            }
            # Store the hidden state BEFORE this observation, not its output state.
            frames.append(frame)
            hidden.append(self.hidden.detach().cpu().numpy())
            starts.append(self.starts.copy())
            with torch.no_grad():
                logits, predicted, self.hidden = self.model(
                    tensor_inputs(frame, self.device),
                    self.hidden,
                    torch.as_tensor(self.starts, device=self.device),
                )
                distribution = torch.distributions.Categorical(logits=logits)
                chosen = sample_action(logits)
                log_probs = distribution.log_prob(chosen)
                probabilities = logits.softmax(-1).cpu().numpy()
            selected, predictions = chosen.cpu().numpy(), predicted.cpu().numpy()
            actions.append(selected)
            logs.append(log_probs.cpu().numpy())
            values.append(predictions)
            step_rewards, step_ended = [], []
            for index, slot in enumerate(self.slots):
                debug = {
                    "action_probabilities": dict(zip(ACTIONS, probabilities[index].tolist())),
                    "extrinsic_value": float(predictions[index, 0]),
                    "intrinsic_value": float(predictions[index, 1]),
                    "learning": True,
                }
                reward, terminal = slot.step(int(selected[index]), debug)
                step_rewards.append(reward)
                step_ended.append(terminal)
                if terminal:
                    summary = slot.summary()
                    self.curriculum.record(slot.level, summary["success"])
                    self.completed += 1
                    append_json(
                        self.directory / "episodes.jsonl",
                        {
                            **summary,
                            "episode": self.completed,
                            "steps": self.steps + self.config.num_envs,
                            "frontier": self.curriculum.frontier,
                        },
                    )
                    if slot.record is not None:
                        self.recorder.save(slot.record, self.config.seed)
                    self.slots[index] = self.new_slot()
            rewards.append(np.stack(step_rewards))
            ended.append(np.array(step_ended, dtype=bool))
            self.starts = np.array(step_ended, dtype=bool)
            self.hidden = self.hidden * torch.as_tensor(~self.starts, device=self.device)[:, None]
            self.steps += self.config.num_envs
        for slot in self.slots:
            slot.env.pause_timer()
        encoded = [slot.memory.encode(slot.observation) for slot in self.slots]
        last = {key: np.stack([item[key] for item in encoded]) for key in encoded[0]}
        with torch.no_grad():
            _, bootstrap, _ = self.model(
                tensor_inputs(last, self.device),
                self.hidden,
                torch.as_tensor(self.starts, device=self.device),
            )
        return Rollout(
            {key: np.stack([item[key] for item in frames]) for key in frames[0]},
            np.stack(hidden),
            np.stack(starts),
            np.stack(actions),
            np.stack(logs),
            np.stack(values),
            np.stack(rewards),
            np.stack(ended),
            bootstrap.cpu().numpy(),
        )

    def checkpoint(self, name: str = "latest.pt") -> None:
        require_ready()
        state = {
            "steps": self.steps,
            "updates": self.updates,
            "completed": self.completed,
            "best_key": list(self.best_key),
            "curriculum": self.curriculum.state_dict(),
            "batch_rng": self.rng.bit_generator.state,
            "slots": [slot.state_dict() for slot in self.slots],
            "hidden": self.hidden.detach().cpu(),
            "starts": torch.as_tensor(self.starts),
            "run_ids": self.recorder.run_ids,
            "log_offsets": {
                name: (self.directory / name).stat().st_size if (self.directory / name).exists() else 0
                for name in ("episodes.jsonl", "updates.jsonl", "validation.jsonl")
            },
        }
        save_checkpoint(self.directory / name, self.model, self.config, optimizer=self.optimizer, state=state)

    def validate(self) -> dict:
        require_ready()
        from .evaluation import evaluate_policy

        frontier = self.curriculum.frontier
        levels = sorted({frontier, *range(5, frontier, 5)})
        cases = [
            (level, maze_seed(self.config.seed, "validation", level, index))
            for level in levels
            for index in range(self.config.validation_episodes)
        ]
        report = evaluate_policy(self.model, self.config, self.device, cases, seed=self.config.seed + 9001)
        append_json(
            self.directory / "validation.jsonl", {"steps": self.steps, "frontier": frontier, **report}
        )
        validation_success = report["groups"][f"level_{frontier:02d}"]["success_rate"]
        promoted = self.curriculum.promote(validation_success)
        key = (frontier, report["aggregate"]["success_rate"], report["aggregate"]["mean_score"])
        if key > self.best_key:
            self.best_key = key
            self.checkpoint("best.pt")
        return {
            "validation_success": report["aggregate"]["success_rate"],
            "frontier_validation_success": validation_success,
            "promoted": promoted,
        }

    def run(self) -> dict:
        require_ready()
        original_handlers = {}

        def request_stop(signum, frame):
            self.stop_requested = True  # Finish the current update for a coherent checkpoint.

        for sig in (signal.SIGINT, signal.SIGTERM):
            original_handlers[sig] = signal.signal(sig, request_stop)
        status = "paused"
        try:
            while not self.stop_requested and not (self.directory / "STOP").exists():
                require_ready()
                remaining = (self.config.total_steps - self.steps) // self.config.num_envs
                length = min(
                    self.config.rollout_steps,
                    (remaining // self.config.sequence_length) * self.config.sequence_length,
                )
                if length < self.config.sequence_length:
                    status = "completed"
                    break
                begun = time.perf_counter()
                rollout = self.collect(length)
                metrics = optimize(
                    self.model,
                    self.optimizer,
                    rollout,
                    self.config,
                    self.device,
                    self.rng,
                    min(1.0, self.steps / self.config.total_steps),
                )
                self.updates += 1
                if self.updates % self.config.validation_every == 0:
                    metrics.update(self.validate())
                metrics.update(
                    steps=self.steps,
                    update=self.updates,
                    episodes=self.completed,
                    frontier=self.curriculum.frontier,
                    seconds=time.perf_counter() - self.started,
                    update_seconds=time.perf_counter() - begun,
                    mean_task_reward=float(rollout.rewards[:, :, 0].mean()),
                    mean_intrinsic_reward=float(rollout.rewards[:, :, 1].mean()),
                )
                self.last_metrics = metrics
                append_json(self.directory / "updates.jsonl", metrics)
                print(json.dumps(metrics, allow_nan=False), flush=True)
                if self.updates % self.config.checkpoint_every == 0:
                    self.checkpoint()
            # The final checkpoint always follows a complete optimization update.
            self.checkpoint()
            return {
                "status": status,
                "steps": self.steps,
                "updates": self.updates,
                "episodes": self.completed,
                "frontier": self.curriculum.frontier,
            }
        except BaseException:
            status = "error"
            # Do not overwrite the last coherent checkpoint on a mid-update error.
            raise
        finally:
            for slot in self.slots:
                slot.env.pause_timer()
                if slot.record is not None and status != "error":
                    self.recorder.save(slot.record, self.config.seed)
            self.recorder.close(status)
            for sig, handler in original_handlers.items():
                signal.signal(sig, handler)


def train(config: Config, directory: Path, resume: Path | None = None) -> dict:
    require_ready()
    return Trainer(config, directory, resume).run()
