"""UI-independent rules for the 35-level maze game."""

from __future__ import annotations

from enum import Enum
import secrets
import time
from typing import Any

from ._generator import DELTAS, DIRECTIONS, Maze, generate_maze, preset_for_level
from ._tiles import tile_maze

ACTIONS = DIRECTIONS
LEVEL_COUNT = 35
OBSERVATION_RADIUS = 2


class Action(str, Enum):
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


class MazeEnv:
    """A headless environment shared by manual play and programmatic agents.

    Public observations contain a five-by-five local view and goal bearing.
    The generated map and trusted shortest-path metrics stay inside the engine.
    """

    def __init__(self, level: int = 1, seed: int | None = None) -> None:
        self._maze: Maze
        self._layout: Maze
        self._optimal_moves = 0
        self._start_time = 0.0
        self._paused_at: float | None = None
        self.level = 1
        self._seed = 0
        self.position = (0, 0)
        self.moves = 0
        self.score = 0
        self.done = False
        self.truncated = False
        self._move_limit = 0
        self.reset(level=level, seed=seed)

    def reset(self, level: int = 1, seed: int | None = None) -> dict[str, Any]:
        """Start a new episode and return its initial observation."""
        preset_for_level(level)  # Validate before changing the current episode.
        chosen_seed = secrets.randbits(32) if seed is None else int(seed)
        self.level = level
        self._seed = chosen_seed
        self._layout = generate_maze(level, chosen_seed)
        self._maze = tile_maze(self._layout)
        self._optimal_moves = self._maze.metrics.shortest_path
        self.position = self._maze.start
        self.moves = 0
        self.score = 0
        self.done = False
        self.truncated = False
        floor_blocks = sum(bool(cell) for row in self._maze.passages for cell in row)
        self._move_limit = max(self._optimal_moves * 3, floor_blocks)
        self._start_time = time.monotonic()
        self._paused_at = None
        return self._observation()

    @property
    def move_limit(self) -> int:
        return self._move_limit

    @property
    def observation(self) -> dict[str, Any]:
        """Return the current limited-view observation."""
        return self._observation()

    @property
    def elapsed_seconds(self) -> float:
        end = self._paused_at if self._paused_at is not None else time.monotonic()
        return max(0.0, end - self._start_time)

    def pause_timer(self) -> None:
        if self._paused_at is None and not (self.done or self.truncated):
            self._paused_at = time.monotonic()

    def resume_timer(self) -> None:
        if self._paused_at is not None and not (self.done or self.truncated):
            self._start_time += time.monotonic() - self._paused_at
            self._paused_at = None

    def step(self, action: str | Action | int) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        """Apply one cardinal move and return observation, reward, flags, info."""
        if self.done or self.truncated:
            raise RuntimeError("episode has ended; call reset() before stepping again")
        direction = self._normalize_action(action)
        self.moves += 1
        score_before = self.score
        entered_trap = False
        hit_wall = not self._maze.can_move(self.position, direction)
        if hit_wall:
            self.score -= 5
        else:
            row, col = self.position
            dr, dc = DELTAS[direction]
            self.position = (row + dr, col + dc)
            if self.position in self._maze.traps:
                entered_trap = True
                self.score -= 50

        extra_move = self.moves > self._optimal_moves
        if extra_move:
            self.score -= 3

        if self.position == self._maze.goal:
            self.done = True
            self.score += 1000
        elif self.moves >= self._move_limit:
            self.truncated = True
        if self.done or self.truncated:
            self._paused_at = time.monotonic()

        info = {
            "level": self.level,
            "moves": self.moves,
            "score": self.score,
            "score_delta": self.score - score_before,
            "entered_trap": entered_trap,
            "hit_wall": hit_wall,
            "extra_move": extra_move,
            "is_success": self.done,
            "move_limit_reached": self.truncated,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        return self._observation(), self.score - score_before, self.done, self.truncated, info

    def _normalize_action(self, action: str | Action | int) -> str:
        if isinstance(action, int):
            if not 0 <= action < len(ACTIONS):
                raise ValueError("integer actions must be 0=up, 1=right, 2=down, 3=left")
            return ACTIONS[action]
        value = action.value if isinstance(action, Action) else str(action).lower()
        if value not in DIRECTIONS:
            raise ValueError(f"unknown action {action!r}; choose one of {DIRECTIONS}")
        return value

    def _observation(self) -> dict[str, Any]:
        row, col = self.position
        goal_row, goal_col = self._maze.goal
        local_view: list[list[dict[str, Any] | None]] = []
        for dr in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS + 1):
            view_row: list[dict[str, Any] | None] = []
            for dc in range(-OBSERVATION_RADIUS, OBSERVATION_RADIUS + 1):
                point = (row + dr, col + dc)
                if not (0 <= point[0] < self._maze.size and 0 <= point[1] < self._maze.size):
                    view_row.append(None)
                    continue
                view_row.append(
                    {
                        "open": [direction for direction in DIRECTIONS if direction in self._maze.passages[point[0]][point[1]]],
                        "wall": not bool(self._maze.passages[point[0]][point[1]]),
                        "trap": point in self._maze.traps,
                        "start": point == self._maze.start,
                        "goal": point == self._maze.goal,
                    }
                )
            local_view.append(view_row)
        return {
            "level": self.level,
            "grid_size": self._maze.size,
            "curriculum_grid_size": self._layout.size,
            "position": [row, col],
            "goal_direction": {
                "vertical": "south" if goal_row > row else "north" if goal_row < row else "none",
                "horizontal": "east" if goal_col > col else "west" if goal_col < col else "none",
            },
            "remaining_moves": self._move_limit - self.moves,
            "move_limit": self._move_limit,
            "moves": self.moves,
            "score": self.score,
            "local_view_radius": OBSERVATION_RADIUS,
            "local_view": local_view,
        }
