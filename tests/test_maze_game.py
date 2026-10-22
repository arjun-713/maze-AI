"""Acceptance tests for the Phase 1 maze game."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from maze_game._generator import (  # noqa: E402
    DELTAS,
    DIRECTIONS,
    Maze,
    _shortest_path,
    generate_maze,
    preset_for_level,
)
from maze_game.engine import MazeEnv  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTION_FOR_DELTA = {delta: direction for direction, delta in DELTAS.items()}


def maze_signature(maze: Maze) -> tuple[tuple[tuple[str, ...], ...], ...]:
    return tuple(tuple(tuple(sorted(cell)) for cell in row) for row in maze.passages)


def path_actions(path: list[tuple[int, int]]) -> list[str]:
    return [
        ACTION_FOR_DELTA[(end[0] - start[0], end[1] - start[1])]
        for start, end in zip(path, path[1:])
    ]


class MazeGenerationTests(unittest.TestCase):
    def test_all_levels_meet_their_measured_difficulty_gates(self) -> None:
        expected_sizes = (9, 13, 17, 21, 27, 33, 41)
        for level in range(1, 36):
            with self.subTest(level=level):
                preset = preset_for_level(level)
                maze = generate_maze(level, seed=20261022)
                metrics = maze.metrics

                self.assertEqual(maze.size, expected_sizes[(level - 1) // 5])
                self.assertEqual((maze.start, maze.goal), ((0, 0), (maze.size - 1, maze.size - 1)))
                self.assertNotIn(maze.start, maze.traps)
                self.assertNotIn(maze.goal, maze.traps)
                self.assertEqual(len(maze.traps), preset.trap_count)
                self.assertGreaterEqual(metrics.shortest_path, preset.min_shortest_path)
                self.assertLessEqual(metrics.shortest_path, preset.max_shortest_path)
                self.assertGreaterEqual(metrics.decision_points, preset.min_decision_points)
                self.assertGreaterEqual(metrics.dead_ends, preset.min_dead_ends)
                self.assertTrue(metrics.has_alternative_route)
                self.assertGreaterEqual(metrics.alternate_routes, 1)
                self.assertTrue(_shortest_path(maze.passages, maze.start, maze.goal))

    def test_seeds_are_reproducible_and_produce_layout_variation(self) -> None:
        for level in range(1, 36):
            with self.subTest(level=level):
                first = generate_maze(level, seed=71)
                again = generate_maze(level, seed=71)
                different = generate_maze(level, seed=72)
                self.assertEqual(maze_signature(first), maze_signature(again))
                self.assertNotEqual(maze_signature(first), maze_signature(different))

    def test_level_validation(self) -> None:
        for level in (0, 36, -1):
            with self.subTest(level=level), self.assertRaises(ValueError):
                generate_maze(level, seed=1)


class EnvironmentTests(unittest.TestCase):
    def test_limited_observation_and_seeded_reset(self) -> None:
        env = MazeEnv(level=12, seed=31415)
        first = env.observation
        env.reset(level=12, seed=31415)
        self.assertEqual(first, env.observation)
        self.assertEqual(len(first["local_view"]), 5)
        self.assertTrue(all(len(row) == 5 for row in first["local_view"]))
        self.assertEqual(first["remaining_moves"], first["move_limit"])
        for hidden_key in ("maze", "layout", "seed", "traps", "optimal_path", "optimal_moves"):
            self.assertNotIn(hidden_key, first)

    def test_all_35_levels_complete_through_the_shared_environment(self) -> None:
        for level in range(1, 36):
            with self.subTest(level=level):
                env = MazeEnv(level=level, seed=9000 + level)
                route = _shortest_path(env._maze.passages, env._maze.start, env._maze.goal)
                rewards: list[int] = []
                for action in path_actions(route):
                    observation, reward, done, truncated, _ = env.step(action)
                    rewards.append(reward)
                    self.assertFalse(truncated)
                    if done:
                        self.assertEqual(observation["position"], list(env._maze.goal))

                self.assertTrue(env.done)
                self.assertEqual(env.moves, len(route) - 1)
                self.assertEqual(env.score, 1000 - 50 * sum(point in env._maze.traps for point in route[1:]))
                self.assertEqual(sum(rewards), env.score)
                with self.assertRaises(RuntimeError):
                    env.step("up")

    def test_wall_and_repeated_trap_penalties(self) -> None:
        env = MazeEnv(level=4, seed=123)
        closed_direction = next(
            direction for direction in DIRECTIONS if not env._maze.can_move(env.position, direction)
        )
        _, reward, done, truncated, info = env.step(closed_direction)
        self.assertEqual(reward, -5)
        self.assertTrue(info["hit_wall"])
        self.assertFalse(done)
        self.assertFalse(truncated)

        trap = next(iter(env._maze.traps))
        route = _shortest_path(env._maze.passages, env.position, trap)
        actions = path_actions(route)
        for action in actions:
            _, _, _, _, info = env.step(action)
        self.assertTrue(info["entered_trap"])
        self.assertEqual(info["score_delta"], -50 - (3 if env.moves > env._optimal_moves else 0))

        back_to_neighbor = path_actions([route[-1], route[-2]])[0]
        env.step(back_to_neighbor)
        return_to_trap = path_actions([route[-2], route[-1]])[0]
        _, reward, _, _, info = env.step(return_to_trap)
        self.assertTrue(info["entered_trap"])
        self.assertEqual(reward, -50 - (3 if env.moves > env._optimal_moves else 0))

    def test_move_limit_truncates_without_completion_points(self) -> None:
        env = MazeEnv(level=1, seed=42)
        wall = next(direction for direction in DIRECTIONS if not env._maze.can_move(env.position, direction))
        limit = env.observation["move_limit"]
        last_info: dict[str, object] = {}
        last_reward = 0
        for move in range(1, limit + 1):
            _, last_reward, done, truncated, last_info = env.step(wall)
            if move <= env._optimal_moves:
                self.assertFalse(truncated)
            if move == env._optimal_moves + 1:
                self.assertEqual(last_reward, -8)  # wall impact and first extra-move penalty
            self.assertFalse(done)
        self.assertTrue(env.truncated)
        self.assertFalse(env.done)
        self.assertTrue(last_info["hit_wall"])
        self.assertLess(env.score, 0)
        with self.assertRaises(RuntimeError):
            env.step(wall)

    def test_headless_json_lines_protocol(self) -> None:
        process = subprocess.run(
            [sys.executable, "-m", "maze_game", "--headless", "--level", "1", "--seed", "42"],
            input='"up"\nquit\n',
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            timeout=15,
            check=True,
        )
        records = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["observation"]["level"], 1)
        self.assertNotIn("maze", records[0]["observation"])
        self.assertEqual(records[1]["reward"], -5)
        self.assertTrue(records[1]["info"]["hit_wall"])

    def test_engine_imports_and_runs_when_pygame_is_unavailable(self) -> None:
        program = "\n".join(
            (
                "import importlib.abc, sys",
                "class NoPygame(importlib.abc.MetaPathFinder):",
                "    def find_spec(self, fullname, path=None, target=None):",
                "        if fullname == 'pygame' or fullname.startswith('pygame.'):",
                "            raise ModuleNotFoundError('Pygame intentionally blocked')",
                "sys.meta_path.insert(0, NoPygame())",
                "from maze_game.engine import MazeEnv",
                "assert MazeEnv(level=1, seed=3).observation['level'] == 1",
                "assert 'pygame' not in sys.modules",
            )
        )
        subprocess.run(
            [sys.executable, "-c", program],
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            timeout=15,
            check=True,
        )


try:
    import pygame
except ImportError:  # The core environment intentionally has no Pygame dependency.
    pygame = None


@unittest.skipIf(pygame is None, "install the optional pygame-ce extra to run UI integration tests")
class PygameIntegrationTests(unittest.TestCase):
    def test_controls_and_responsive_rendering(self) -> None:
        from maze_game.ui import MazeGame

        pygame.init()
        game = MazeGame(pygame)
        try:
            game._reset(level=35, seed=42)
            for width, height in ((900, 680), (1024, 768), (1920, 1000)):
                game.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                side_width = game._draw_sidebar(height, width)
                game._draw_maze(side_width, width, height)
                pygame.display.flip()
                self.assertLess(side_width, width)
                self.assertGreaterEqual(game._tile_size, 24)
                for control in ("fit", "zoom_in", "zoom_out"):
                    self.assertFalse(game.buttons[control].colliderect(game.viewport))

            game._reset(level=1, seed=42)
            open_direction = next(direction for direction in DIRECTIONS if game.env._maze.can_move(game.env.position, direction))
            key_by_direction = {"up": pygame.K_w, "right": pygame.K_d, "down": pygame.K_s, "left": pygame.K_a}
            game._handle_key(pygame.event.Event(pygame.KEYDOWN, key=key_by_direction[open_direction], unicode=""))
            self.assertEqual(game.env.moves, 1)

            game._toggle_pause()
            paused_moves = game.env.moves
            game._move(open_direction)
            self.assertEqual(game.env.moves, paused_moves)
            game._toggle_pause()
            game._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r, unicode="r"))
            self.assertEqual(game.env.moves, 0)
            self.assertEqual(game.env.position, (1, 1))

            before_rebuild = maze_signature(game.env._maze)
            game.seed_text = "43"
            game._rebuild_from_text()
            self.assertEqual(game.seed, 43)
            self.assertNotEqual(before_rebuild, maze_signature(game.env._maze))
        finally:
            pygame.quit()

    def test_block_board_matches_collisions_and_camera_controls_work(self) -> None:
        from maze_game.ui import MazeGame

        game = MazeGame(pygame)
        try:
            for level in (1, 17, 35):
                game._reset(level=level, seed=42)
                maze = game.env._maze
                surface = game._paint_world(32)
                floor_colors = {(244, 236, 215), (239, 228, 202), (210, 229, 249), (250, 220, 144), (246, 204, 198)}
                for row in range(maze.size):
                    for col in range(maze.size):
                        if not maze.passages[row][col]:
                            continue
                        for direction, (dr, dc) in DELTAS.items():
                            target = ((col + dc) * 32, (row + dr) * 32 + 31)
                            visible_opening = tuple(surface.get_at(target)[:3]) in floor_colors
                            self.assertEqual(visible_opening, maze.can_move((row, col), direction))

            game.screen = pygame.display.set_mode((900, 680))
            route = _shortest_path(game.env._maze.passages, game.env.position, game.env._maze.goal)
            for action in path_actions(route)[:-1]:
                game._move(action)
            game._moved_at = 0.0
            side = game._draw_sidebar(680, 900)
            game._draw_maze(side, 900, 680)
            player_center = game._cell_center(game.env.position, game._tile_size)
            self.assertTrue(game._camera_world.collidepoint(player_center))

            game._click(game.buttons["fit"].center)
            side = game._draw_sidebar(680, 900)
            game._draw_maze(side, 900, 680)
            self.assertTrue(game.fit_map)
            self.assertLessEqual(game._world.get_width(), game.viewport.width)
            self.assertLessEqual(game._world.get_height(), game.viewport.height)
            fit_tile_size = game._tile_size
            game._click(game.buttons["zoom_in"].center)
            side = game._draw_sidebar(680, 900)
            game._draw_maze(side, 900, 680)
            self.assertFalse(game.fit_map)
            self.assertGreater(game._tile_size, fit_tile_size)

            self.assertNotIn("motion", game.buttons)
        finally:
            pygame.quit()

    def test_one_key_press_moves_one_visible_cell_and_ignores_repeats(self) -> None:
        from maze_game.ui import MazeGame

        game = MazeGame(pygame)
        try:
            for use_wasd in (False, True):
                game._reset(level=1, seed=42)
                direction = next(d for d in DIRECTIONS if game.env._maze.can_move(game.env.position, d))
                keys = (
                    {"up": pygame.K_w, "right": pygame.K_d, "down": pygame.K_s, "left": pygame.K_a}
                    if use_wasd else
                    {"up": pygame.K_UP, "right": pygame.K_RIGHT, "down": pygame.K_DOWN, "left": pygame.K_LEFT}
                )
                key = keys[direction]
                before = game._cell_center(game.env.position, 48)
                game._handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=""))
                after = game._cell_center(game.env.position, 48)
                self.assertEqual(abs(after[0] - before[0]) + abs(after[1] - before[1]), 48)
                self.assertEqual(game.env.moves, 1)
                for _ in range(5):
                    game._handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode="", repeat=True))
                    game._handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=""))
                self.assertEqual(game.env.moves, 1)
                game._handle_event(pygame.event.Event(pygame.KEYUP, key=key))
                game._handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=""))
                self.assertEqual(game.env.moves, 2)
                game._handle_event(pygame.event.Event(pygame.KEYUP, key=key))
        finally:
            pygame.quit()

    def test_reported_seed_moves_one_actual_wall_block_in_the_rendered_frame(self) -> None:
        from maze_game.ui import MazeGame

        game = MazeGame(pygame)
        try:
            game.screen = pygame.display.set_mode((2560, 1372))
            game._reset(level=1, seed=953391530)

            def press(key: int) -> None:
                game._handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, unicode=""))
                game._handle_event(pygame.event.Event(pygame.KEYUP, key=key))

            # Reproduce the user's first screenshot without teleporting the player.
            for key in ([pygame.K_DOWN] * 2 + [pygame.K_RIGHT] * 4 +
                        [pygame.K_DOWN] * 2 + [pygame.K_LEFT] * 4):
                press(key)
            self.assertEqual(game.env.position, (5, 1))
            self.assertEqual(game.env.moves, 12)
            game._moved_at = 0.0
            side = game._draw_sidebar(1372, 2560)
            game._draw_maze(side, 2560, 1372)
            before = game._player_rect.center

            # Measure the visible wall artwork's repeated pixel boundary, rather
            # than assuming that the renderer's cell-size variable means a block.
            wall_pixels = [game._world.get_at((x, 3)) for x in range(256)]
            block_width = next(
                period for period in range(3, len(wall_pixels) // 3)
                if all(wall_pixels[x] == wall_pixels[x + period]
                       for x in range(len(wall_pixels) - period))
            )
            press(pygame.K_DOWN)
            game._moved_at = 0.0
            side = game._draw_sidebar(1372, 2560)
            game._draw_maze(side, 2560, 1372)
            after = game._player_rect.center
            self.assertEqual(game.env.position, (6, 1))
            self.assertEqual(game.env.observation["position"], [6, 1])
            self.assertEqual(game.env.moves, 13)
            self.assertEqual(after[0], before[0])
            self.assertEqual(after[1] - before[1], block_width)

            press(pygame.K_DOWN)
            self.assertEqual(game.env.position, (7, 1))
            self.assertEqual(game.env.moves, 14)
        finally:
            pygame.quit()

    def test_motion_is_always_enabled_and_has_no_toggle(self) -> None:
        from maze_game.ui import MazeGame

        game = MazeGame(pygame)
        try:
            game._reset(level=1, seed=42)
            before = game.env.position
            direction = next(d for d in DIRECTIONS if game.env._maze.can_move(before, d))
            with patch("maze_game.ui.time.monotonic", return_value=100.0):
                game._move(direction)
            with patch("maze_game.ui.time.monotonic", return_value=100.05):
                halfway = game._visual_position()
                self.assertNotEqual(halfway, before)
                self.assertNotEqual(halfway, game.env.position)
                game._handle_key(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v, unicode="v"))
                self.assertEqual(game._visual_position(), halfway)
            side = game._draw_sidebar(880, 1380)
            game._draw_maze(side, 1380, 880)
            self.assertNotIn("motion", game.buttons)
            self.assertFalse(hasattr(game, "animations"))
        finally:
            pygame.quit()

    def test_keyboard_can_finish_and_advance_through_all_levels(self) -> None:
        from maze_game.ui import MazeGame

        pygame.init()
        game = MazeGame(pygame)
        key_by_direction = {
            "up": pygame.K_UP,
            "right": pygame.K_RIGHT,
            "down": pygame.K_DOWN,
            "left": pygame.K_LEFT,
        }
        try:
            game._reset(level=1, seed=8080)
            for level in range(1, 36):
                with self.subTest(level=level):
                    if level > 1:
                        game._handle_event(
                            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, unicode="\r")
                        )
                        game._handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_RETURN))
                    self.assertEqual(game.level, level)
                    route = _shortest_path(
                        game.env._maze.passages, game.env._maze.start, game.env._maze.goal
                    )
                    for direction in path_actions(route):
                        game._handle_event(
                            pygame.event.Event(
                                pygame.KEYDOWN,
                                key=key_by_direction[direction],
                                unicode="",
                            )
                        )
                        game._handle_event(pygame.event.Event(pygame.KEYUP, key=key_by_direction[direction]))
                    expected_score = 1000 - 50 * sum(
                        point in game.env._maze.traps for point in route[1:]
                    )
                    self.assertTrue(game.env.done)
                    self.assertEqual(game.env.score, expected_score)
                    self.assertIn(level, game.completed)
        finally:
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
