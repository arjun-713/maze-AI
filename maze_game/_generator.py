"""Trusted maze construction and shortest-path helpers.

This module belongs to the game engine. Agent code should use MazeEnv's
observation and step API and should not receive a Maze instance.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
import math
import random

Point = tuple[int, int]
UP, RIGHT, DOWN, LEFT = "up", "right", "down", "left"
DIRECTIONS = (UP, RIGHT, DOWN, LEFT)
DELTAS: dict[str, Point] = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}
LEVEL_SIZES = (9, 13, 17, 21, 27, 33, 41)


@dataclass(frozen=True, slots=True)
class LevelPreset:
    level: int
    size: int
    tier: int
    min_shortest_path: int
    max_shortest_path: int
    min_decision_points: int
    min_dead_ends: int
    loop_density: float
    trap_count: int


@dataclass(frozen=True, slots=True)
class MazeMetrics:
    shortest_path: int
    decision_points: int
    dead_ends: int
    alternate_routes: int
    has_alternative_route: bool


@dataclass(slots=True)
class Maze:
    size: int
    passages: tuple[tuple[frozenset[str], ...], ...]
    traps: frozenset[Point]
    start: Point
    goal: Point
    metrics: MazeMetrics

    def can_move(self, position: Point, direction: str) -> bool:
        row, col = position
        dr, dc = DELTAS[direction]
        next_row, next_col = row + dr, col + dc
        return (
            0 <= next_row < self.size
            and 0 <= next_col < self.size
            and direction in self.passages[row][col]
        )

    def neighbors(self, position: Point) -> list[Point]:
        row, col = position
        return [
            (row + DELTAS[direction][0], col + DELTAS[direction][1])
            for direction in DIRECTIONS
            if direction in self.passages[row][col]
        ]


def preset_for_level(level: int) -> LevelPreset:
    if not 1 <= level <= 35:
        raise ValueError("level must be between 1 and 35")
    tier = (level - 1) // 5 + 1
    size = LEVEL_SIZES[tier - 1]
    cells = size * size
    min_path = max(2 * (size - 1), round(size * (1.40 + 0.12 * tier)))
    max_path = min_path + 2 * size
    min_decisions = max(2, math.ceil(cells * (0.060 + 0.012 * tier)))
    min_dead_ends = max(2, math.ceil(cells * (0.045 + 0.005 * tier)))
    density = 0.012 + 0.004 * tier
    traps = max(tier, round(cells * (0.0015 + 0.001 * tier)))
    return LevelPreset(
        level, size, tier, min_path, max_path, min_decisions, min_dead_ends, density, traps
    )


def _open(passages: list[list[set[str]]], point: Point, direction: str) -> Point:
    row, col = point
    dr, dc = DELTAS[direction]
    target = (row + dr, col + dc)
    passages[row][col].add(direction)
    passages[target[0]][target[1]].add(OPPOSITE[direction])
    return target


def _carve_depth_first(size: int, rng: random.Random) -> list[list[set[str]]]:
    passages = [[set() for _ in range(size)] for _ in range(size)]
    visited = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        row, col = stack[-1]
        choices = [
            direction
            for direction in DIRECTIONS
            if (row + DELTAS[direction][0], col + DELTAS[direction][1])
            not in visited
            and 0 <= row + DELTAS[direction][0] < size
            and 0 <= col + DELTAS[direction][1] < size
        ]
        if not choices:
            stack.pop()
            continue
        direction = rng.choice(choices)
        target = _open(passages, (row, col), direction)
        visited.add(target)
        stack.append(target)
    return passages


def _carve_prim(size: int, rng: random.Random) -> list[list[set[str]]]:
    passages = [[set() for _ in range(size)] for _ in range(size)]
    visited = {(0, 0)}
    frontier: list[tuple[Point, str]] = []

    def add_frontier(point: Point) -> None:
        row, col = point
        for direction in DIRECTIONS:
            dr, dc = DELTAS[direction]
            target = (row + dr, col + dc)
            if 0 <= target[0] < size and 0 <= target[1] < size and target not in visited:
                frontier.append((point, direction))

    add_frontier((0, 0))
    while frontier:
        index = rng.randrange(len(frontier))
        frontier[index], frontier[-1] = frontier[-1], frontier[index]
        point, direction = frontier.pop()
        row, col = point
        dr, dc = DELTAS[direction]
        target = (row + dr, col + dc)
        if target in visited:
            continue
        _open(passages, point, direction)
        visited.add(target)
        add_frontier(target)
    return passages


def _shortest_path(passages: list[list[set[str]]] | tuple[tuple[frozenset[str], ...], ...], start: Point, goal: Point) -> list[Point]:
    queue = deque([start])
    previous: dict[Point, Point | None] = {start: None}
    while queue:
        point = queue.popleft()
        if point == goal:
            break
        row, col = point
        for direction in DIRECTIONS:
            if direction not in passages[row][col]:
                continue
            dr, dc = DELTAS[direction]
            neighbor = (row + dr, col + dc)
            if neighbor not in previous:
                previous[neighbor] = point
                queue.append(neighbor)
    if goal not in previous:
        return []
    route: list[Point] = []
    cursor: Point | None = goal
    while cursor is not None:
        route.append(cursor)
        cursor = previous[cursor]
    route.reverse()
    return route


def _has_alternative_route(passages: list[list[set[str]]], start: Point, goal: Point) -> bool:
    """Return whether at least one start-to-goal route edge belongs to a cycle."""
    discovery: dict[Point, int] = {start: 0}
    low: dict[Point, int] = {start: 0}
    parent: dict[Point, Point] = {}
    bridges: set[tuple[Point, Point]] = set()
    counter = 1

    def edge_key(left: Point, right: Point) -> tuple[Point, Point]:
        return (left, right) if left < right else (right, left)

    stack: list[tuple[Point, Iterator[Point]]] = []

    def push(point: Point) -> None:
        row, col = point
        neighbors = []
        for direction in DIRECTIONS:
            if direction in passages[row][col]:
                dr, dc = DELTAS[direction]
                neighbors.append((row + dr, col + dc))
        stack.append((point, iter(neighbors)))

    push(start)
    while stack:
        point, neighbor_iter = stack[-1]
        try:
            neighbor = next(neighbor_iter)
        except StopIteration:
            stack.pop()
            if point in parent:
                previous = parent[point]
                low[previous] = min(low[previous], low[point])
                if low[point] > discovery[previous]:
                    bridges.add(edge_key(previous, point))
            continue
        if parent.get(point) == neighbor:
            continue
        if neighbor not in discovery:
            parent[neighbor] = point
            discovery[neighbor] = counter
            low[neighbor] = counter
            counter += 1
            push(neighbor)
        else:
            low[point] = min(low[point], discovery[neighbor])

    route = _shortest_path(passages, start, goal)
    return any(edge_key(left, right) not in bridges for left, right in zip(route, route[1:]))


def _measure(passages: list[list[set[str]]], start: Point, goal: Point) -> MazeMetrics:
    decisions = 0
    dead_ends = 0
    edges = 0
    for row in passages:
        for openings in row:
            degree = len(openings)
            decisions += degree >= 3
            dead_ends += degree == 1
            edges += degree
    edges //= 2
    route = _shortest_path(passages, start, goal)
    return MazeMetrics(
        len(route) - 1,
        decisions,
        dead_ends,
        edges - len(passages) ** 2 + 1,
        _has_alternative_route(passages, start, goal),
    )


def _candidate(preset: LevelPreset, rng: random.Random) -> tuple[list[list[set[str]]], MazeMetrics]:
    size = preset.size
    passages = (_carve_depth_first if rng.random() < 0.5 else _carve_prim)(size, rng)
    candidates: list[tuple[Point, str]] = []
    for row in range(size):
        for col in range(size):
            for direction in (RIGHT, DOWN):
                dr, dc = DELTAS[direction]
                nr, nc = row + dr, col + dc
                if nr < size and nc < size and direction not in passages[row][col]:
                    candidates.append(((row, col), direction))
    target_loops = max(1, round(size * size * preset.loop_density * rng.uniform(0.85, 1.15)))
    rng.shuffle(candidates)
    for point, direction in candidates[:target_loops]:
        _open(passages, point, direction)
    metrics = _measure(passages, (0, 0), (size - 1, size - 1))
    return passages, metrics


def generate_maze(level: int, seed: int) -> Maze:
    """Generate a reproducible layout after enforcing that tier's metric gates."""
    preset = preset_for_level(level)
    # The level participates in the stream so adjacent curriculum slots do not
    # repeat the same layout when a caller keeps one seed across levels.
    rng = random.Random((int(seed) << 6) + level)
    best: tuple[int, list[list[set[str]]], MazeMetrics] | None = None
    for _ in range(600):
        passages, metrics = _candidate(preset, rng)
        shortfall = (
            max(0, preset.min_shortest_path - metrics.shortest_path) * 8
            + max(0, metrics.shortest_path - preset.max_shortest_path) * 4
            + max(0, preset.min_decision_points - metrics.decision_points) * 3
            + max(0, preset.min_dead_ends - metrics.dead_ends) * 3
            + (0 if metrics.has_alternative_route else 1_000_000)
        )
        if best is None or shortfall < best[0]:
            best = (shortfall, passages, metrics)
        if shortfall == 0:
            break
    if best is None or best[0] != 0:
        raise RuntimeError(
            f"Could not meet difficulty gates for level {level}; "
            f"best metrics were {best[2] if best else 'unavailable'}"
        )

    _, passages, metrics = best
    size = preset.size
    reserved = {(0, 0), (size - 1, size - 1)}
    trap_candidates = [
        (row, col)
        for row in range(size)
        for col in range(size)
        if (row, col) not in reserved
    ]
    rng.shuffle(trap_candidates)
    traps = frozenset(trap_candidates[: preset.trap_count])
    frozen = tuple(tuple(frozenset(cell) for cell in row) for row in passages)
    return Maze(
        size,
        frozen,
        traps,
        (0, 0),
        (size - 1, size - 1),
        metrics,
    )
