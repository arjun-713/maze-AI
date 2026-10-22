"""Compile generated corridors into the actual one-block movement grid."""

from __future__ import annotations

from dataclasses import replace

from ._generator import DIRECTIONS, Maze, _open


def tile_maze(layout: Maze) -> Maze:
    """Make every visible floor block a position in the shared simulation.

    Generated cells sit at odd coordinates. The floor block between adjacent
    cells is also walkable, so crossing a generated passage takes two actions.
    Walls occupy the remaining blocks and have no outgoing passages.
    """
    size = layout.size * 2 + 1
    passages: list[list[set[str]]] = [[set() for _ in range(size)] for _ in range(size)]
    for row in range(layout.size):
        for col in range(layout.size):
            center = (row * 2 + 1, col * 2 + 1)
            for direction in DIRECTIONS:
                if direction in layout.passages[row][col]:
                    _open(passages, center, direction)

    def point(position: tuple[int, int]) -> tuple[int, int]:
        return position[0] * 2 + 1, position[1] * 2 + 1

    return Maze(
        size=size,
        passages=tuple(tuple(frozenset(cell) for cell in row) for row in passages),
        traps=frozenset(point(trap) for trap in layout.traps),
        start=point(layout.start),
        goal=point(layout.goal),
        metrics=replace(layout.metrics, shortest_path=layout.metrics.shortest_path * 2),
    )
