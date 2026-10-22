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
    parser.add_argument("--headless", action="store_true", help="use the JSON-lines action protocol without opening Pygame")
    parser.add_argument("--level", type=int, default=1, choices=range(1, LEVEL_COUNT + 1))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.headless:
        raise SystemExit(_headless(args.level, args.seed))
    from .ui import launch

    launch()


if __name__ == "__main__":
    main()
