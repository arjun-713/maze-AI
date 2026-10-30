"""Explicit contestant commands. Parsing/help/status do not import ML or create environments."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import activation


def parse_levels(text: str) -> list[int]:
    levels = []
    for part in text.split(","):
        first, separator, last = part.partition("-")
        if separator:
            levels.extend(range(int(first), int(last) + 1))
        else:
            levels.append(int(first))
    if not levels or any(not 1 <= level <= 35 for level in levels):
        raise ValueError("levels must be 1..35, for example 1-35 or 1,5,10-15")
    return sorted(set(levels))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Codex Memory PPO contestant. Dormant until activation.py READY is True.",
        epilog="All operational commands require activation. Status and --help start nothing.",
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show the switch and workspace paths without initializing an agent")
    training = commands.add_parser("train", help="explicitly train or resume recurrent PPO")
    training.add_argument(
        "--config", type=Path, help="JSON config; omitted on resume uses checkpoint settings"
    )
    training.add_argument(
        "--run", type=Path, required=True, help="output directory; reuse only with --resume"
    )
    training.add_argument("--resume", type=Path, help="a coherent latest.pt or best.pt training checkpoint")
    training.add_argument("--steps", type=int, help="total lifetime step budget, including completed steps")
    training.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"))
    evaluation = commands.add_parser("evaluate", help="evaluate a frozen checkpoint on held-out mazes")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--levels", default="1-35")
    evaluation.add_argument("--episodes", type=int, default=20, help="episodes per selected level")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    evaluation.add_argument("--output", type=Path, required=True, help="new directory for report and replays")
    evaluation.add_argument("--manifest", type=Path, help="exactly five championship {level, seed} entries")
    evaluation.add_argument("--greedy", action="store_true", help="argmax policy; default seeded stochastic")
    inference = commands.add_parser(
        "infer", help="observation JSON lines in, action JSON lines out; no learning"
    )
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--seed", type=int, default=42)
    inference.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    inference.add_argument("--greedy", action="store_true")
    dashboard = commands.add_parser(
        "dashboard", help="open the existing studio with frozen checkpoint inference"
    )
    dashboard.add_argument("--checkpoint", type=Path, required=True)
    dashboard.add_argument(
        "--recordings", type=Path, required=True, help="directory containing training.sqlite3"
    )
    dashboard.add_argument("--level", type=int, default=1, choices=range(1, 36))
    dashboard.add_argument("--seed", type=int, default=42)
    dashboard.add_argument("--episodes", type=int, default=100)
    dashboard.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    monitoring = commands.add_parser(
        "monitor", help="read the latest training metrics; does not run a policy"
    )
    monitoring.add_argument("--run", type=Path, required=True)
    return result


def inference_stream(checkpoint: Path, seed: int, device: str, greedy: bool) -> None:
    activation.require_ready()
    from .agent import CodexAgent

    agent = CodexAgent(seed, checkpoint, device, greedy)
    for line in sys.stdin:
        activation.require_ready()
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("done") or item.get("truncated"):
            agent.policy.started = False
            print(json.dumps({"episode_done": True}), flush=True)
            continue
        observation = item.get("observation", item)
        decision = agent.act(observation)
        print(json.dumps({"action": decision.action, "debug": decision.debug}, allow_nan=False), flush=True)


def main() -> int:
    arguments = parser().parse_args()
    if arguments.command == "status":
        print(
            json.dumps(
                {
                    "ready": activation.READY,
                    "agent": "Codex Memory PPO",
                    "switch": "agents/codex/activation.py",
                    "default_checkpoint": "agents/codex/artifacts/champion.pt",
                }
            )
        )
        return 0
    try:
        # This runs BEFORE importing torch, constructing a model, touching a
        # checkpoint, opening a DB/window, or creating a maze environment.
        activation.require_ready()
        if arguments.command == "train":
            from .config import load_config
            from .runtime import read_checkpoint
            from .trainer import train

            config = (
                load_config(arguments.config)
                if arguments.config or not arguments.resume
                else read_checkpoint(arguments.resume)[1]
            )
            changes = {
                key: getattr(arguments, key) for key in ("device",) if getattr(arguments, key) is not None
            }
            if arguments.steps is not None:
                changes["total_steps"] = arguments.steps
            config = replace(config, **changes)
            config.validate()
            print(json.dumps(train(config, arguments.run, arguments.resume)), flush=True)
        elif arguments.command == "evaluate":
            from .evaluation import run_evaluation

            result = run_evaluation(
                arguments.checkpoint,
                levels=parse_levels(arguments.levels),
                episodes=arguments.episodes,
                seed=arguments.seed,
                device=arguments.device,
                output=arguments.output,
                manifest=arguments.manifest,
                greedy=arguments.greedy,
            )
            print(json.dumps(result["aggregate"], allow_nan=False))
        elif arguments.command == "infer":
            inference_stream(arguments.checkpoint, arguments.seed, arguments.device, arguments.greedy)
        elif arguments.command == "dashboard":
            from .agent import CodexAgent

            # Validate the checkpoint before importing Pygame/initializing the studio.
            first = CodexAgent(arguments.seed, arguments.checkpoint, arguments.device)
            import pygame
            from maze_game.training_ui import TrainingStudio

            available = [first]

            def factory(seed):
                activation.require_ready()
                if available:
                    return available.pop()
                return CodexAgent(seed, arguments.checkpoint, arguments.device)

            TrainingStudio(
                pygame,
                level=arguments.level,
                seed=arguments.seed,
                episodes=arguments.episodes,
                runs_dir=arguments.recordings,
                agent_factory=factory,
            ).run()
        elif arguments.command == "monitor":
            activation.require_ready()
            path = arguments.run / "updates.jsonl"
            last = None
            with path.open() as stream:
                for line in stream:
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A writer may currently be appending the final line.
            print(json.dumps(last if last is not None else {"status": "no completed updates yet"}, indent=2))
        return 0
    except activation.DormantError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, KeyError, ArithmeticError) as error:
        print(f"Codex contestant: {error}", file=sys.stderr)
        print(
            "See agents/codex/README.md for setup and recovery. No automatic retry was started.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("Codex contestant interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
