"""Rendering-independent experiment recording and deterministic replay for user agents."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import random
import sqlite3
from typing import Any, Protocol
from uuid import uuid4

from .engine import MazeEnv

FORMAT_VERSION = 1


@dataclass(frozen=True)
class RunConfig:
    level: int = 1
    seed: int = 42
    episodes: int = 100

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 35:
            raise ValueError("level must be between 1 and 35")
        if not 1 <= self.episodes <= 100_000:
            raise ValueError("episodes must be between 1 and 100000")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class AgentDecision:
    action: str
    debug: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    name: str

    def act(self, observation: dict[str, Any]) -> AgentDecision: ...


def load_agent(spec: str = "Redbull.agent:Redbull", seed: int = 42) -> Agent:
    """Instantiate user code; there is deliberately no built-in policy or fallback."""
    module, separator, symbol = spec.partition(":")
    if not separator or not module or not symbol or ":" in symbol:
        raise ValueError("agent must be a Python module:Class, for example Redbull.agent:Redbull")
    factory = getattr(import_module(module), symbol)
    agent = factory(seed=seed)
    if not isinstance(getattr(agent, "name", None), str) or not agent.name.strip():
        raise TypeError("agent must expose a nonempty name")
    if not callable(getattr(agent, "act", None)):
        raise TypeError("agent must implement act(observation)")
    return agent


@dataclass(frozen=True)
class Transition:
    observation: dict[str, Any]
    action: str
    reward: int
    next_observation: dict[str, Any]
    done: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class StepRecord:
    action: str
    reward: int
    position: tuple[int, int]
    hit_wall: bool
    entered_trap: bool
    extra_move: bool
    done: bool
    truncated: bool
    seconds: float
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeSummary:
    number: int
    level: int
    seed: int
    reward: int
    success: bool
    truncated: bool
    moves: int
    traps: int
    walls: int
    seconds: float
    complete: bool


@dataclass
class EpisodeRecord:
    number: int
    level: int
    seed: int
    start: tuple[int, int]
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def summary(self) -> EpisodeSummary:
        last = self.steps[-1] if self.steps else None
        return EpisodeSummary(
            self.number, self.level, self.seed, sum(step.reward for step in self.steps),
            bool(last and last.done), bool(last and last.truncated), len(self.steps),
            sum(step.entered_trap for step in self.steps), sum(step.hit_wall for step in self.steps),
            last.seconds if last else 0.0, bool(last and (last.done or last.truncated)),
        )


def discovery(observation: dict[str, Any]) -> set[tuple[int, int]]:
    row, col = observation["position"]
    radius = observation["local_view_radius"]
    return {(row + r - radius, col + c - radius)
            for r, cells in enumerate(observation["local_view"])
            for c, cell in enumerate(cells) if cell is not None}


class RunStore:
    """Append episode payloads to SQLite, without rewriting an entire run."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, agent TEXT NOT NULL,
                created TEXT NOT NULL, config TEXT NOT NULL, status TEXT NOT NULL,
                version INTEGER NOT NULL, error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS episodes (
                run_id TEXT NOT NULL REFERENCES runs(id), number INTEGER NOT NULL,
                summary TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (run_id, number)
            );
        """)
        self.connection.commit()

    def create_run(self, name: str, agent: str, config: RunConfig) -> str:
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:10]
        with self.connection:
            self.connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (run_id, name, agent, now.isoformat(), json.dumps(asdict(config)),
                                     "paused", FORMAT_VERSION, ""))
        return run_id

    def set_status(self, run_id: str, status: str, error: str = "") -> None:
        with self.connection:
            self.connection.execute("UPDATE runs SET status=?, error=? WHERE id=?", (status, error, run_id))

    def save_episode(self, run_id: str, record: EpisodeRecord) -> None:
        summary = json.dumps(asdict(record.summary), allow_nan=False)
        payload = json.dumps(asdict(record), allow_nan=False, separators=(",", ":"))
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO episodes VALUES (?, ?, ?, ?)",
                                    (run_id, record.number, summary, payload))

    def runs(self) -> list[dict[str, Any]]:
        cursor = self.connection.execute("SELECT id,name,agent,created,config,status,version,error FROM runs ORDER BY created DESC")
        return [dict(zip(("id", "name", "agent", "created", "config", "status", "version", "error"),
                         (*row[:4], json.loads(row[4]), *row[5:]))) for row in cursor.fetchall()]

    def episodes(self, run_id: str, *, complete_only: bool = False) -> list[EpisodeSummary]:
        rows = self.connection.execute("SELECT summary FROM episodes WHERE run_id=? ORDER BY number", (run_id,)).fetchall()
        summaries = [EpisodeSummary(**json.loads(row[0])) for row in rows]
        return [item for item in summaries if item.complete] if complete_only else summaries

    def load_episode(self, run_id: str, number: int) -> EpisodeRecord:
        version = self.connection.execute("SELECT version FROM runs WHERE id=?", (run_id,)).fetchone()
        if not version or version[0] != FORMAT_VERSION:
            raise ValueError("This run uses an unsupported recording version")
        row = self.connection.execute("SELECT payload FROM episodes WHERE run_id=? AND number=?", (run_id, number)).fetchone()
        if row is None:
            raise ValueError("Episode was not found")
        data = json.loads(row[0])
        data["start"] = tuple(data["start"])
        data["steps"] = [StepRecord(**{**step, "position": tuple(step["position"])}) for step in data["steps"]]
        return EpisodeRecord(**data)

    def close(self) -> None:
        self.connection.close()


class TrainingSession:
    """One action per step, with renderer-independent pause and episode budgets.

    Agents receive only public observations. An optional on_transition(Transition)
    hook receives public feedback; architecture-specific debug is never required.
    """

    def __init__(self, config: RunConfig, store: RunStore, agent: Agent, name: str | None = None) -> None:
        self.config = config
        self.store = store
        self.agent = agent
        self.run_id = store.create_run(name or f"{agent.name} {len(store.runs()) + 1:02d}", self.agent.name, config)
        self.status = "paused"
        self.rng = random.Random(config.seed)
        self.episodes: list[EpisodeSummary] = []
        self.env = MazeEnv(config.level, config.seed)
        self.pending_episode = False
        self.last_observation: dict[str, Any] | None = None
        self.last_step: StepRecord | None = None
        self._begin_episode()

    def _begin_episode(self) -> None:
        seed = self.rng.getrandbits(32)
        observation = self.env.reset(self.config.level, seed)
        self.current = EpisodeRecord(len(self.episodes) + 1, self.config.level, seed, self.env.position)
        self.seen = discovery(observation)
        self.path = [self.env.position]
        self.wall_events: list[tuple[tuple[int, int], str]] = []
        self.last_step = None
        self.last_observation = observation
        self.pending_episode = False
        if self.status != "running":
            self.env.pause_timer()
        hook = getattr(self.agent, "on_episode_start", None)
        if hook is not None:
            hook(deepcopy(observation))

    def resume(self) -> None:
        if self.status == "paused":
            if not getattr(self.agent, "ready", True):
                raise ValueError("Implement Redbull/agent.py, set Redbull.ready = True, then restart the studio")
            self.status = "running"
            self.env.resume_timer()
            self.store.set_status(self.run_id, self.status)

    def pause(self) -> None:
        if self.status == "running":
            self.env.pause_timer()
            self.status = "paused"
            self.store.save_episode(self.run_id, self.current)
            self.store.set_status(self.run_id, self.status)

    def stop(self) -> None:
        if self.status in ("paused", "running"):
            self.env.pause_timer()
            self.store.save_episode(self.run_id, self.current)
            self.status = "stopped"
            self.store.set_status(self.run_id, self.status)

    def step(self) -> StepRecord | None:
        if self.status != "running":
            return None
        try:
            if self.pending_episode:
                self._begin_episode()
            observation = self.env.observation
            decision = self.agent.act(deepcopy(observation))
            if not isinstance(decision, AgentDecision):
                raise TypeError("agent.act must return an AgentDecision")
            # Freeze debug snapshots and reject non-JSON/NaN values at the boundary.
            debug = json.loads(json.dumps(decision.debug, allow_nan=False))
            if not isinstance(debug, dict):
                raise TypeError("agent debug must be a JSON object")
            previous = self.env.position
            next_observation, reward, done, truncated, info = self.env.step(decision.action)
            record = StepRecord(decision.action, reward, self.env.position, info["hit_wall"],
                                info["entered_trap"], info["extra_move"], done, truncated,
                                round(self.env.elapsed_seconds, 6), debug)
            self.current.steps.append(record)
            self.last_observation = observation
            self.last_step = record
            self.path.append(self.env.position)
            self.seen.update(discovery(next_observation))
            if record.hit_wall:
                self.wall_events.append((previous, record.action))
            hook = getattr(self.agent, "on_transition", None)
            if hook is not None:
                hook(Transition(deepcopy(observation), record.action, reward, deepcopy(next_observation),
                                done, truncated, deepcopy(info)))
            if done or truncated:
                self.store.save_episode(self.run_id, self.current)
                self.episodes.append(self.current.summary)
                self.pending_episode = True
                if len(self.episodes) >= self.config.episodes:
                    self.status = "completed"
                    self.store.set_status(self.run_id, self.status)
            return record
        except Exception as exc:
            self.env.pause_timer()
            self.status = "error"
            self.store.save_episode(self.run_id, self.current)
            self.store.set_status(self.run_id, self.status, str(exc))
            raise


class EpisodeReplay:
    """Re-simulate recorded actions; validate every outcome against the recording."""

    def __init__(self, record: EpisodeRecord) -> None:
        self.record = record
        self.env = MazeEnv(record.level, record.seed)
        self.index = 0
        self.last_step: StepRecord | None = None
        self.last_observation = self.env.observation
        self.path = [self.env.position]
        self.seen = discovery(self.env.observation)
        self.wall_events: list[tuple[tuple[int, int], str]] = []
        self.env.pause_timer()
        if self.env.position != record.start:
            raise ValueError("Replay start does not match the recording")

    def seek(self, index: int) -> None:
        if not 0 <= index <= len(self.record.steps):
            raise ValueError("replay index is outside this episode")
        if index < self.index:
            self.env.reset(self.record.level, self.record.seed)
            self.env.pause_timer()
            self.index = 0
            self.path = [self.env.position]
            self.seen = discovery(self.env.observation)
            self.wall_events = []
            self.last_step = None
            self.last_observation = self.env.observation
        while self.index < index:
            step = self.record.steps[self.index]
            before = self.env.observation
            previous = self.env.position
            after, reward, done, truncated, info = self.env.step(step.action)
            if (self.env.position, reward, done, truncated, info["hit_wall"], info["entered_trap"], info["extra_move"]) != (
                step.position, step.reward, step.done, step.truncated, step.hit_wall, step.entered_trap, step.extra_move
            ):
                raise ValueError(f"Replay differs from recording at step {self.index + 1}")
            self.last_observation = before
            self.last_step = step
            self.path.append(self.env.position)
            self.seen.update(discovery(after))
            if step.hit_wall:
                self.wall_events.append((previous, step.action))
            self.index += 1

    @property
    def seconds(self) -> float:
        return self.last_step.seconds if self.last_step else 0.0


def aggregate(episodes: list[EpisodeSummary]) -> dict[str, float | int | None]:
    finished = [episode for episode in episodes if episode.complete]
    successes = [episode for episode in finished if episode.success]
    return {
        "episodes": len(finished),
        "success_rate": len(successes) / len(finished) if finished else 0.0,
        "mean_reward": sum(e.reward for e in finished) / len(finished) if finished else None,
        "mean_moves": sum(e.moves for e in finished) / len(finished) if finished else None,
        "mean_completion_seconds": sum(e.seconds for e in successes) / len(successes) if successes else None,
        "traps": sum(e.traps for e in finished),
        "walls": sum(e.walls for e in finished),
    }
