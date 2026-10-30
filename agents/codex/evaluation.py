"""Frozen-policy validation and held-out evaluation; no learning or reward shaping."""

import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .activation import require_ready


def evaluate_policy(
    model, config, device, cases: list[tuple[int, int]], *, seed: int, greedy: bool = False, recorder=None
) -> dict:
    require_ready()
    from maze_game import MazeEnv
    from maze_game.training import EpisodeRecord, StepRecord

    from .agent import Policy
    from .memory import ACTIONS

    was_training = model.training
    model.eval()
    policy = Policy(model, config, device, seed, greedy)
    results = []
    try:
        for number, (level, maze_seed) in enumerate(cases, 1):
            require_ready()
            env = MazeEnv(level=level, seed=maze_seed)
            observation = env.observation
            policy.reset(observation)
            record = EpisodeRecord(number, level, maze_seed, tuple(observation["position"]))
            started = time.perf_counter()
            action_seconds = []
            traps = walls = 0
            while True:
                require_ready()
                before = time.perf_counter()
                decision = policy.act(observation)
                action_seconds.append(time.perf_counter() - before)
                observation, reward, done, truncated, info = env.step(ACTIONS.index(decision.action))
                traps += int(info["entered_trap"])
                walls += int(info["hit_wall"])
                if recorder is not None:
                    record.steps.append(
                        StepRecord(
                            decision.action,
                            reward,
                            tuple(observation["position"]),
                            info["hit_wall"],
                            info["entered_trap"],
                            info["extra_move"],
                            done,
                            truncated,
                            env.elapsed_seconds,
                            decision.debug,
                        )
                    )
                if done or truncated:
                    break
            summary = {
                "level": level,
                "tier": (level - 1) // 5 + 1,
                "seed": maze_seed,
                "success": bool(done),
                "truncated": bool(truncated),
                "moves": observation["moves"],
                "move_budget_fraction": observation["moves"] / observation["move_limit"],
                "score": observation["score"],
                "traps": traps,
                "walls": walls,
                "seconds": time.perf_counter() - started,
                "action_seconds_mean": float(np.mean(action_seconds)),
                "action_seconds_p95": float(np.percentile(action_seconds, 95)),
            }
            results.append(summary)
            if recorder is not None:
                recorder.save(record, config.seed)
    finally:
        model.train(was_training)
    groups = defaultdict(list)
    for result in results:
        groups[f"level_{result['level']:02d}"].append(result)
        groups[f"tier_{result['tier']}"].append(result)
    return {
        "episodes": results,
        "aggregate": summarize(results),
        "groups": {key: summarize(items) for key, items in groups.items()},
        "policy_mode": "greedy" if greedy else "seeded stochastic",
        "config": asdict(config),
    }


def summarize(rows: list[dict]) -> dict:
    require_ready()
    if not rows:
        raise ValueError("evaluation requires at least one episode")
    successes = [row for row in rows if row["success"]]
    return {
        "episodes": len(rows),
        "success_rate": len(successes) / len(rows),
        "mean_score": float(np.mean([row["score"] for row in rows])),
        "mean_moves": float(np.mean([row["moves"] for row in rows])),
        "successful_mean_moves": float(np.mean([row["moves"] for row in successes])) if successes else None,
        "successful_mean_seconds": float(np.mean([row["seconds"] for row in successes]))
        if successes
        else None,
        "traps": sum(row["traps"] for row in rows),
        "walls": sum(row["walls"] for row in rows),
        "mean_action_seconds": float(np.mean([row["action_seconds_mean"] for row in rows])),
    }


def run_evaluation(
    checkpoint: Path,
    *,
    levels: list[int],
    episodes: int,
    seed: int,
    device: str,
    output: Path,
    manifest: Path | None = None,
    greedy: bool = False,
) -> dict:
    require_ready()
    import json

    from .agent import CodexAgent
    from .curriculum import maze_seed
    from .recording import StudioRecorder

    if episodes < 1 or not levels or any(not 1 <= level <= 35 for level in levels):
        raise ValueError("select valid levels and a positive episode count")
    if manifest:
        entries = json.loads(manifest.read_text())
        if not isinstance(entries, list) or len(entries) != 5:
            raise ValueError("a championship manifest must contain exactly five {level, seed} entries")
        cases = [(item["level"], item["seed"]) for item in entries]
        if any(
            type(level) is not int or not 1 <= level <= 35 or type(value) is not int for level, value in cases
        ):
            raise ValueError("manifest level/seed must be integers with levels 1..35")
        if len(set(cases)) != 5:
            raise ValueError("championship manifest cases must be distinct")
    else:
        cases = [
            (level, maze_seed(seed, "evaluation", level, index))
            for level in levels
            for index in range(episodes)
        ]
    # Refuse accidental overwrites of an earlier championship report.
    if output.exists():
        raise FileExistsError(f"choose a new evaluation output directory: {output}")
    agent = CodexAgent(seed, checkpoint, device, greedy)
    output.mkdir(parents=True)
    recorder = StudioRecorder(output / "recordings", "Codex evaluation")
    status = "error"
    try:
        report = evaluate_policy(
            agent.policy.model,
            agent.policy.config,
            agent.policy.device,
            cases,
            seed=seed,
            greedy=greedy,
            recorder=recorder,
        )
        report["checkpoint"] = str(checkpoint.resolve())
        report["manifest"] = str(manifest) if manifest else None
        (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        status = "completed"
        return report
    finally:
        recorder.close(status)
