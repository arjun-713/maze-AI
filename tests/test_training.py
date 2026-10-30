"""Training telemetry, storage, replay, and studio integration checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from maze_game._generator import DELTAS, _shortest_path
from maze_game.engine import MazeEnv
from maze_game.training import (AgentDecision, EpisodeRecord, EpisodeReplay, EpisodeSummary,
                                RunConfig, RunStore, TrainingSession, aggregate, load_agent)


class WallAgent:
    name = "Wall probe"

    def __init__(self, seed=42) -> None:
        self.observations = []
        self.transitions = []

    def act(self, observation):
        self.observations.append(deepcopy(observation))
        observation["position"][0] = -999  # The runner must pass an isolated observation.
        return AgentDecision("up")

    def on_transition(self, transition):
        self.transitions.append(transition)


def finish(session):
    session.resume()
    while session.status == "running":
        session.step()


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.store = RunStore()

    def tearDown(self):
        self.store.close()

    def test_pause_resume_and_exact_episode_budget(self):
        agent = WallAgent()
        session = TrainingSession(RunConfig(1, 23, 2), self.store, agent)
        self.assertIsNone(session.step())
        self.assertEqual(session.env.moves, 0)
        session.resume()
        session.step()
        session.pause()
        self.assertIsNone(session.step())
        self.assertEqual(session.env.moves, 1)
        self.assertEqual(self.store.episodes(session.run_id)[0].moves, 1)
        finish(session)
        self.assertEqual(session.status, "completed")
        self.assertEqual(len(session.episodes), 2)
        self.assertIsNone(session.step())
        self.assertTrue(all(e.truncated and e.walls == e.moves and e.reward < 0 for e in session.episodes))
        self.assertEqual(len(agent.transitions), sum(e.moves for e in session.episodes))
        self.assertEqual(agent.observations[0]["position"], [1, 1])
        self.assertNotEqual(session.last_observation["position"][0], -999)
        for observation in agent.observations:
            for hidden in ("maze", "layout", "seed", "optimal_path", "optimal_moves"):
                self.assertNotIn(hidden, observation)
        self.assertTrue(all(not e.success for e in session.episodes))

    def test_seed_reproducibility_and_different_episode_layouts(self):
        config = RunConfig(1, 77, 3)
        first = TrainingSession(config, self.store, WallAgent())
        second = TrainingSession(config, self.store, WallAgent())
        finish(first)
        finish(second)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(len({e.seed for e in first.episodes}), 3)
        for number in range(1, 4):
            a = self.store.load_episode(first.run_id, number)
            b = self.store.load_episode(second.run_id, number)
            self.assertEqual(a.seed, b.seed)
            self.assertEqual([(s.action, s.position, s.reward, s.debug) for s in a.steps],
                             [(s.action, s.position, s.reward, s.debug) for s in b.steps])

    def test_success_metrics_and_optional_agent_specific_debug(self):
        config = RunConfig(1, 9, 1)
        maze_seed = random.Random(config.seed).getrandbits(32)
        env = MazeEnv(config.level, maze_seed)
        path = _shortest_path(env._maze.passages, env.position, env._maze.goal)
        inverse = {delta: action for action, delta in DELTAS.items()}
        actions = iter(inverse[(b[0] - a[0], b[1] - a[1])] for a, b in zip(path, path[1:]))

        class ScriptedAgent:
            name = "Trusted test fixture"

            def act(self, observation):
                return AgentDecision(next(actions), {"q_values": {"up": -2, "right": 3}, "custom": [1, {"feature": "value"}]})

        session = TrainingSession(config, self.store, ScriptedAgent())
        finish(session)
        summary = session.episodes[0]
        self.assertTrue(summary.success)
        self.assertEqual(summary.moves, len(path) - 1)
        self.assertEqual(summary.reward, session.env.score)
        self.assertEqual(summary.traps, sum(point in env._maze.traps for point in path[1:]))
        metrics = aggregate(session.episodes)
        self.assertEqual(metrics["success_rate"], 1)
        self.assertIsNotNone(metrics["mean_completion_seconds"])
        record = self.store.load_episode(session.run_id, 1)
        self.assertEqual(record.steps[0].debug["custom"], [1, {"feature": "value"}])

    def test_user_agent_loader_and_unfinished_headless_run(self):
        # User code is loaded from its own module, with no policy injected by the runner.
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            module = Path(folder) / "probe_agent.py"
            module.write_text("from maze_game.training import AgentDecision\n"
                              "class Probe:\n"
                              "    name = 'User policy probe'\n"
                              "    def __init__(self, seed): self.seed = seed\n"
                              "    def act(self, observation): return AgentDecision('left', {'seed': self.seed})\n")
            sys.path.insert(0, folder)
            try:
                agent = load_agent("probe_agent:Probe", 789)
                self.assertEqual(agent.act({}), AgentDecision("left", {"seed": 789}))
            finally:
                sys.path.remove(folder)
                sys.modules.pop("probe_agent", None)
            process = subprocess.run([sys.executable, "-m", "maze_game", "--train-headless", "--runs-dir", folder],
                                     cwd=root, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("Implement Redbull/agent.py", process.stderr)
            self.assertFalse((Path(folder) / "training.sqlite3").exists())

    def test_persisted_replay_matches_every_outcome_and_seeks_both_directions(self):
        with TemporaryDirectory() as folder:
            store = RunStore(Path(folder) / "runs.sqlite3")
            session = TrainingSession(RunConfig(1, 26, 1), store, WallAgent())
            finish(session)
            run_id = session.run_id
            store.close()
            store = RunStore(Path(folder) / "runs.sqlite3")
            try:
                self.assertEqual(store.runs()[0]["status"], "completed")
                record = store.load_episode(run_id, 1)
                replay = EpisodeReplay(record)
                middle = len(record.steps) // 2
                replay.seek(middle)
                self.assertEqual(replay.env.position, record.steps[middle - 1].position)
                self.assertEqual(replay.last_step.debug, record.steps[middle - 1].debug)
                replay.seek(len(record.steps))
                self.assertEqual(replay.env.score, record.summary.reward)
                self.assertEqual(replay.env.moves, record.summary.moves)
                self.assertEqual(replay.seconds, record.summary.seconds)
                replay.seek(0)
                self.assertEqual(replay.env.position, record.start)
                self.assertEqual(replay.env.moves, 0)
                self.assertIsNone(replay.last_step)
                replay.seek(len(record.steps))
                self.assertEqual(replay.env.score, record.summary.reward)
            finally:
                store.close()

    def test_partial_episode_replay_and_drift_detection(self):
        session = TrainingSession(RunConfig(1, 42, 1), self.store, WallAgent())
        session.resume()
        for _ in range(8):
            session.step()
        session.pause()
        record = self.store.load_episode(session.run_id, 1)
        self.assertFalse(record.summary.complete)
        replay = EpisodeReplay(record)
        replay.seek(8)
        self.assertEqual(replay.env.position, session.env.position)
        with self.assertRaises(ValueError):
            replay.seek(9)
        corrupt = EpisodeRecord(record.number, record.level, record.seed, record.start, list(record.steps))
        corrupt.steps[2] = replace(corrupt.steps[2], reward=99999)
        with self.assertRaisesRegex(ValueError, "step 3"):
            EpisodeReplay(corrupt).seek(8)

    def test_invalid_agent_debug_and_actions_stop_the_run_with_saved_error(self):
        class InvalidAgent:
            name = "Invalid fixture"

            def __init__(self, decision):
                self.decision = decision

            def act(self, observation):
                return self.decision

        for decision in (AgentDecision("diagonal"), AgentDecision("up", {"value": float("nan")})):
            session = TrainingSession(RunConfig(1, 42, 1), self.store, InvalidAgent(decision))
            session.resume()
            with self.assertRaises(ValueError):
                session.step()
            self.assertEqual(session.status, "error")
            self.assertEqual(session.env.moves, 0)
            saved = next(run for run in self.store.runs() if run["id"] == session.run_id)
            self.assertEqual(saved["status"], "error")
            self.assertTrue(saved["error"])

    def test_aggregation_excludes_partial_episodes_and_counts_real_successes(self):
        episodes = [EpisodeSummary(1, 1, 10, 950, True, False, 30, 1, 0, 3.5, True),
                    EpisodeSummary(2, 1, 11, -100, False, True, 100, 0, 20, 8, True),
                    EpisodeSummary(3, 1, 12, -5, False, False, 1, 0, 1, .1, False)]
        stats = aggregate(episodes)
        self.assertEqual(stats["episodes"], 2)
        self.assertEqual(stats["success_rate"], .5)
        self.assertEqual(stats["mean_reward"], 425)
        self.assertEqual(stats["mean_moves"], 65)
        self.assertEqual(stats["mean_completion_seconds"], 3.5)
        self.assertEqual(stats["walls"], 20)
        self.assertIsNone(aggregate([])["mean_completion_seconds"])

    def test_headless_training_cli_and_core_without_pygame(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as folder:
            Path(folder, "cli_probe.py").write_text("from maze_game.training import AgentDecision\n"
                                                  "class Probe:\n"
                                                  "    name = 'CLI probe'\n"
                                                  "    def __init__(self, seed): pass\n"
                                                  "    def act(self, observation): return AgentDecision('up')\n")
            environment = {**os.environ, "PYTHONPATH": os.pathsep.join((folder, str(root), os.environ.get("PYTHONPATH", "")))}
            process = subprocess.run([sys.executable, "-m", "maze_game", "--train-headless", "--episodes", "2", "--runs-dir", folder, "--agent", "cli_probe:Probe"],
                                     cwd=root, env=environment, text=True, capture_output=True, check=True, timeout=20)
            report = json.loads(process.stdout)
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["status"], "completed")
            store = RunStore(Path(folder) / "training.sqlite3")
            self.assertEqual(len(store.episodes(report["run_id"])), 2)
            store.close()
        program = """
from maze_game.training import AgentDecision, RunStore, RunConfig, TrainingSession
import sys
class Probe:
    name = "Headless probe"
    def act(self, observation):
        return AgentDecision("up")
session = TrainingSession(RunConfig(1, 42, 1), RunStore(), Probe())
session.resume()
session.step()
assert "pygame" not in sys.modules
"""
        subprocess.run([sys.executable, "-c", program], cwd=root, check=True, capture_output=True, timeout=15)

    def test_config_validation_and_unfinished_user_agent(self):
        for args in ((0, 42, 1), (36, 42, 1), (1, 42, 0), (1, 42, 100001)):
            with self.assertRaises(ValueError):
                RunConfig(*args)
        agent = load_agent(seed=123)
        self.assertEqual(agent.seed, 123)
        self.assertFalse(agent.ready)
        with self.assertRaises(NotImplementedError):
            agent.act({})
        session = TrainingSession(RunConfig(), self.store, agent)
        with self.assertRaisesRegex(ValueError, "Implement Redbull/agent.py"):
            session.resume()
        self.assertEqual(session.status, "paused")
        self.assertIsNone(session.step())
        self.assertEqual(session.env.moves, 0)
        self.assertEqual(session.episodes, [])
        with self.assertRaises(TypeError):
            TrainingSession(RunConfig(), self.store)
        with self.assertRaises(ValueError):
            load_agent("missing-colon")


try:
    import pygame
except ImportError:
    pygame = None


@unittest.skipIf(pygame is None, "install pygame-ce for studio integration checks")
class StudioTests(unittest.TestCase):
    def setUp(self):
        from maze_game.training_ui import TrainingStudio
        self.directory = TemporaryDirectory()
        self.app = TrainingStudio(pygame, episodes=2, runs_dir=self.directory.name, agent_factory=WallAgent)

    def tearDown(self):
        self.app.close()
        self.directory.cleanup()

    def test_pause_speed_new_run_comparison_and_replay_controls(self):
        app = self.app
        app.draw()
        app._click(app.buttons["toggle"].center)
        app.update(.1)
        self.assertGreater(app.env.moves, 0)
        app.draw()
        app._click(app.buttons["toggle"].center)
        moves = app.env.moves
        app.update(.1)
        self.assertEqual(app.env.moves, moves)
        self.assertEqual(app.session.status, "paused")

        old_id = app.session.run_id
        app._click(app.buttons["speed:500"].center)
        self.assertEqual(app.speed, 500)
        app._click(app.buttons["new"].center)
        self.assertNotEqual(app.session.run_id, old_id)
        app.view = "Dashboard"
        app.draw()
        app._click(app.buttons["compare"].center)
        self.assertEqual(app.compare_run, old_id)
        app._click(app.buttons["run:" + old_id].center)
        app.draw()
        app._click(app.buttons["episode:1"].center)
        self.assertIsNotNone(app.replay)
        self.assertEqual(app.session.status, "paused")
        app.draw()
        app._click(app.buttons["forward"].center)
        self.assertEqual(app.replay.index, 1)
        app._seek(len(app.replay.record.steps))
        self.assertEqual(app.replay.env.score, app.replay.record.summary.reward)
        app.draw()
        app._click(app.buttons["live"].center)
        self.assertIsNone(app.replay)
        self.assertEqual(app.session.status, "paused")

    def test_default_studio_waits_for_the_users_implementation(self):
        from maze_game.training_ui import TrainingStudio

        self.app.close()
        self.app = TrainingStudio(pygame, runs_dir=self.directory.name)
        app = self.app
        for size in ((1100, 760), (1560, 980)):
            app.screen = pygame.display.set_mode(size)
            for view in ("Game", "Learning", "Dashboard"):
                app.view = view
                app.draw()
                self.assertIn("toggle", app._disabled_buttons)
                app._click(app.buttons["toggle"].center)
                app.update(.1)
                self.assertEqual(app.env.moves, 0)
                self.assertEqual(app.session.status, "paused")
        app._toggle_running()  # Keyboard shortcuts must not bypass the readiness guard.
        self.assertEqual(app.session.status, "paused")
        self.assertIn("Implement Redbull/agent.py", app.error)

    def test_all_views_scale_and_debug_is_optional(self):
        app = self.app
        finish(app.session)
        app._refresh(True)
        for size in ((1100, 760), (1560, 980), (2400, 1350)):
            app.screen = pygame.display.set_mode(size)
            for view in ("Game", "Learning", "Dashboard"):
                app.view = view
                app.draw()
                self.assertNotIn("motion", app.buttons)
                for rect in app.buttons.values():
                    self.assertTrue(app.screen.get_rect().contains(rect), (size, view, rect))
        app.view = "Learning"
        for debug in ({}, {"q_values": {"up": -5, "down": 2, "left": 0, "right": 7}},
                      {"custom": {"hidden_state": [1, 2, 3]}}, {"action_values": {"up": "not numeric"}}):
            app.session.last_step = replace(app.session.last_step, debug=debug)
            app.draw()
        app.view = "Dashboard"
        app.draw()
        app._load_replay(1)
        app.screen = pygame.display.set_mode((1100, 760))
        for view in ("Game", "Learning", "Dashboard"):
            app.view = view
            app.draw()

    def test_manual_game_can_request_the_studio_without_an_extra_move(self):
        from maze_game.ui import MazeGame

        game = MazeGame(pygame)
        side = game._draw_sidebar(880, 1380)
        game._draw_maze(side, 1380, 880)
        game._click(game.buttons["training"].center)
        self.assertEqual(game._next_mode, "training")
        self.assertFalse(game.running)
        self.assertEqual(game.env.moves, 0)


if __name__ == "__main__":
    unittest.main()
