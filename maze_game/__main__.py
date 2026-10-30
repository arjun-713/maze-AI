"""Desktop launcher and a JSON-lines headless environment adapter."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .engine import ACTIONS, LEVEL_COUNT, MazeEnv


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _headless(level: int, seed: int) -> int:
    env = MazeEnv(level=level, seed=seed)
    _emit({"observation": env.observation})
    for raw in sys.stdin:
        command = raw.strip()
        if not command:
            continue
        if command.lower() in {"quit", "exit"}:
            break
        try:
            decoded = json.loads(command)
            action = decoded.get("action") if isinstance(decoded, dict) else decoded
        except json.JSONDecodeError:
            action = command
        if action not in ACTIONS:
            _emit({"error": f"action must be one of {ACTIONS}"})
            continue
        observation, reward, done, truncated, info = env.step(action)
        _emit({
            "observation": observation,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info,
        })
        if done or truncated:
            break
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Play or control the 35-level procedural maze game.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--headless", action="store_true", help="use the JSON-lines action protocol without opening Pygame")
    modes.add_argument("--dashboard", action="store_true", help="open Training Studio for your agent")
    modes.add_argument("--train-headless", action="store_true", help="run your agent without Pygame")
    parser.add_argument("--agent", default="Redbull.agent:Redbull", help="your agent's Python module:Class (constructor accepts seed=)")
    parser.add_argument("--level", type=int, default=1, choices=range(1, LEVEL_COUNT + 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs-dir", default=".maze-runs", help="directory for the training SQLite database")
    args = parser.parse_args()
    if args.headless:
        raise SystemExit(_headless(args.level, args.seed))
    if args.dashboard or args.train_headless:
        from pathlib import Path
        from .training import RunConfig, RunStore, TrainingSession, aggregate, load_agent

        try:
            config = RunConfig(args.level, args.seed, args.episodes)
        except ValueError as exc:
            parser.error(str(exc))
        if args.dashboard:
            from .training_ui import launch_dashboard

            launch_dashboard(level=config.level, seed=config.seed, episodes=config.episodes, runs_dir=args.runs_dir,
                             agent_spec=args.agent)
        else:
            try:
                agent = load_agent(args.agent, config.seed)
            except Exception as exc:
                parser.error(f"Could not load agent {args.agent}: {exc}")
            if not getattr(agent, "ready", True):
                parser.error("Implement Redbull/agent.py, set Redbull.ready = True, then run again")
            store = RunStore(Path(args.runs_dir) / "training.sqlite3")
            session = TrainingSession(config, store, agent)
            try:
                session.resume()
                while session.status == "running":
                    session.step()
                _emit({"run_id": session.run_id, "status": session.status, **aggregate(session.episodes)})
            finally:
                store.close()
        return
    from .ui import launch

    launch()


if __name__ == "__main__":
    main()
