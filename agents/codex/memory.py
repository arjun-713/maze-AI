"""Observation-only episodic memory. No graph search, solver, or goal-coordinate guess."""

import numpy as np

from .activation import require_ready
from .config import ModelConfig

ACTIONS = ("up", "right", "down", "left")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CHANNELS = 13
SCALARS = 20


class SpatialMemory:
    def __init__(self, config: ModelConfig):
        require_ready()
        self.config = config
        self.size = 0
        self.last_move = -1
        self.previous_action = -1
        self.previous_reward = 0.0
        self.last_new = 0

    def reset(self, observation: dict) -> None:
        require_ready()
        size = observation["grid_size"]
        if type(size) is not int or not 5 <= size <= self.config.max_grid:
            raise ValueError("observation grid_size is unsupported")
        self.size = size
        self.pool_starts = np.arange(self.config.global_size) * size // self.config.global_size
        self.flags = np.zeros((9, size, size), dtype=np.uint8)
        self.visits = np.zeros((size, size), dtype=np.int32)
        self.last_visit = np.full((size, size), -1, dtype=np.int32)
        self.last_move = -1
        self.previous_action = -1
        self.previous_reward = 0.0
        self.observe(observation)

    def observe(self, observation: dict) -> tuple[int, bool]:
        require_ready()
        if not self.size:
            self.reset(observation)
            return self.last_new, True
        if observation["grid_size"] != self.size:
            raise ValueError("reset episode memory when grid_size changes")
        move = int(observation["moves"])
        if move == self.last_move:
            return 0, False  # Idempotent when act follows on_transition.
        if move < self.last_move:
            raise ValueError("episode memory must be reset before a new maze")
        row, col = observation["position"]
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError("observation position is out of bounds")
        radius = observation["local_view_radius"]
        local = observation["local_view"]
        if radius != 2 or len(local) != 5 or any(len(cells) != 5 for cells in local):
            raise ValueError("expected the competition's 5x5 observation")
        new_floor = 0
        for r, cells in enumerate(local):
            for c, cell in enumerate(cells):
                rr, cc = row + r - radius, col + c - radius
                if cell is None or not (0 <= rr < self.size and 0 <= cc < self.size):
                    continue
                new_floor += int(not self.flags[0, rr, cc] and not cell["wall"])
                self.flags[:5, rr, cc] = [
                    1,
                    bool(cell["wall"]),
                    bool(cell["trap"]),
                    bool(cell["goal"]),
                    bool(cell["start"]),
                ]
                self.flags[5:, rr, cc] = [action in cell["open"] for action in ACTIONS]
        first_visit = self.visits[row, col] == 0
        self.visits[row, col] += 1
        self.last_visit[row, col] = move
        self.position = (row, col)
        self.last_move = move
        self.last_new = new_floor
        return new_floor, bool(first_visit)

    def feedback(self, action: int, reward: float) -> None:
        require_ready()
        self.previous_action = action
        self.previous_reward = float(np.clip(reward / 100.0, -1.0, 1.0))

    def encode(self, observation: dict) -> dict[str, np.ndarray]:
        require_ready()
        row, col = self.position
        size = self.size
        floor = (self.flags[0] != 0) & (self.flags[1] == 0)
        frontier = np.zeros((size, size), dtype=bool)
        for dr, dc in DELTAS:
            r0, r1 = max(0, -dr), min(size, size - dr)
            c0, c1 = max(0, -dc), min(size, size - dc)
            frontier[r0:r1, c0:c1] |= self.flags[0, r0 + dr : r1 + dr, c0 + dc : c1 + dc] == 0
        grid = np.zeros((CHANNELS, size, size), dtype=np.float32)
        grid[:5] = self.flags[:5]
        grid[5] = np.minimum(np.log1p(self.visits) / np.log(33.0), 1.0)
        grid[6] = np.where(
            self.last_visit >= 0, np.exp(-np.maximum(0, self.last_move - self.last_visit) / 64.0), 0.0
        )
        grid[7] = frontier & floor
        grid[8, row, col] = 1.0
        grid[9:] = self.flags[5:]
        patch_size = self.config.patch_size
        radius = patch_size // 2
        padded = np.pad(grid, ((0, 0), (radius, radius), (radius, radius)))
        patch = padded[:, row : row + patch_size, col : col + patch_size]
        # Fixed-size max bins keep even a single seen goal/frontier visible. These
        # are remembered observations, never a generated full-map observation.
        coarse = np.maximum.reduceat(
            np.maximum.reduceat(grid, self.pool_starts, axis=1), self.pool_starts, axis=2
        )
        center = observation["local_view"][2][2]
        legal = np.array([action in center["open"] for action in ACTIONS], dtype=bool)
        if not legal.any():
            raise ValueError("the public observation contains no legal action")
        bearing = observation["goal_direction"]
        vertical = {"north": -1.0, "none": 0.0, "south": 1.0}[bearing["vertical"]]
        horizontal = {"west": -1.0, "none": 0.0, "east": 1.0}[bearing["horizontal"]]
        previous = [float(self.previous_action == i) for i in range(4)]
        adjacent_visits = [
            min(1.0, self.visits[row + dr, col + dc] / 8.0)
            if 0 <= row + dr < size and 0 <= col + dc < size
            else 1.0
            for dr, dc in DELTAS
        ]
        scalars = np.array(
            [
                row / max(1, size - 1),
                col / max(1, size - 1),
                vertical,
                horizontal,
                observation["remaining_moves"] / max(1, observation["move_limit"]),
                size / self.config.max_grid,
                float(floor.mean()),
                min(1.0, self.visits[row, col] / 16.0),
                self.previous_reward,
                *previous,
                *adjacent_visits,
                float(center["trap"]),
                float((self.visits > 0).mean()),
                min(1.0, self.last_new / 10.0),
            ],
            dtype=np.float32,
        )
        return {
            "patch": np.rint(patch * 255).astype(np.uint8),
            "global": np.rint(coarse * 255).astype(np.uint8),
            "scalars": scalars,
            "legal": legal,
        }
