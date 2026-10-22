"""Authored pixel sprites and block tiles; Pygame is supplied by the UI."""

from __future__ import annotations

from typing import Any

INK = (36, 54, 75)


class SpriteAtlas:
    """Cache crisp artwork at the board or legend's requested size."""

    def __init__(self, pygame: Any) -> None:
        self.pg = pygame
        self._originals = {name: self._make(name) for name in ("bunny", "enemy", "flag", "home")}
        self._scaled: dict[tuple[str, int], Any] = {}
        self._walls: dict[int, Any] = {}

    def get(self, name: str, size: int) -> Any:
        key = (name, size)
        if key not in self._scaled:
            self._scaled[key] = self.pg.transform.scale(self._originals[name], (size, size))
        return self._scaled[key]

    def _make(self, name: str) -> Any:
        pg = self.pg
        image = pg.Surface((32, 32), pg.SRCALPHA)

        def rect(color: tuple[int, int, int], box: tuple[int, int, int, int], radius: int = 0) -> None:
            pg.draw.rect(image, color, box, border_radius=radius)

        if name == "bunny":
            cream, white, pink = (239, 225, 202), (255, 248, 232), (233, 149, 155)
            for x in (8, 20):
                rect(INK, (x, 1, 5, 15), 2)
                rect(white, (x + 1, 2, 3, 12), 1)
                rect(pink, (x + 2, 3, 1, 8))
            rect(INK, (8, 21, 17, 10), 3)
            rect(cream, (9, 22, 15, 8), 2)
            rect(INK, (4, 10, 25, 16), 5)
            rect(white, (5, 11, 23, 13), 4)
            rect(cream, (6, 22, 21, 3), 1)
            rect(INK, (10, 16, 2, 4), 1)
            rect(INK, (21, 16, 2, 4), 1)
            rect((255, 255, 255), (10, 16, 1, 1))
            rect((255, 255, 255), (21, 16, 1, 1))
            rect(pink, (7, 20, 4, 2), 1)
            rect(pink, (22, 20, 4, 2), 1)
            rect((187, 108, 119), (16, 20, 2, 1))
            rect(INK, (16, 21, 1, 1))
            rect((25, 127, 133), (8, 25, 18, 3), 1)
            rect((17, 91, 107), (20, 27, 4, 4))
            rect(white, (8, 29, 5, 2), 1)
            rect(white, (21, 29, 5, 2), 1)
        elif name == "enemy":
            dark, red = (105, 39, 57), (217, 54, 69)
            pg.draw.polygon(image, dark, [(5, 13), (4, 2), (14, 11)])
            pg.draw.polygon(image, dark, [(18, 11), (28, 2), (27, 14)])
            pg.draw.polygon(image, (255, 235, 188), [(6, 10), (6, 5), (11, 10)])
            pg.draw.polygon(image, (255, 235, 188), [(21, 10), (26, 5), (25, 11)])
            rect(dark, (3, 10, 27, 18), 6)
            rect(red, (4, 10, 25, 16), 5)
            rect((239, 89, 93), (7, 11, 18, 4), 2)
            rect(dark, (5, 25, 7, 5), 2)
            rect(dark, (22, 25, 7, 5), 2)
            rect(red, (6, 25, 5, 3), 1)
            rect(red, (23, 25, 5, 3), 1)
            rect((255, 250, 231), (8, 16, 6, 5), 1)
            rect((255, 250, 231), (20, 16, 6, 5), 1)
            rect(dark, (11, 17, 2, 4))
            rect(dark, (20, 17, 2, 4))
            pg.draw.line(image, dark, (7, 14), (14, 17), 2)
            pg.draw.line(image, dark, (19, 17), (26, 14), 2)
            rect(dark, (13, 23, 9, 3), 1)
            rect((255, 250, 231), (14, 23, 2, 2))
            rect((255, 250, 231), (19, 23, 2, 2))
        elif name == "flag":
            pg.draw.ellipse(image, (187, 133, 39), (3, 26, 17, 5))
            pg.draw.ellipse(image, (243, 197, 91), (3, 24, 17, 5))
            rect(INK, (8, 2, 3, 25), 1)
            rect((244, 200, 102), (8, 3, 1, 23))
            rect(INK, (10, 3, 21, 17))
            for row in range(4):
                for col in range(5):
                    color = (255, 254, 246) if (row + col) % 2 == 0 else INK
                    rect(color, (10 + col * 4, 3 + row * 4, 4, 4))
        else:
            rect((30, 83, 153), (2, 5, 28, 26), 5)
            rect((52, 123, 210), (2, 2, 28, 26), 5)
            pg.draw.polygon(image, (246, 252, 255), [(7, 15), (16, 7), (25, 15)])
            rect((246, 252, 255), (10, 14, 13, 10), 1)
            rect((52, 123, 210), (15, 17, 4, 7))
        return image

    def wall(self, size: int) -> Any:
        if size not in self._walls:
            pg = self.pg
            tile = pg.Surface((size, size))
            tile.fill((46, 66, 88))
            if size >= 5:
                edge = max(1, size // 9)
                pg.draw.rect(tile, (82, 107, 130), (1, 1, size - 2, size - 2))
                pg.draw.rect(tile, (113, 137, 158), (1, 1, size - 2, size - edge - 2))
                pg.draw.line(tile, (143, 163, 178), (2, 2), (size - 3, 2), max(1, size // 18))
                pg.draw.rect(tile, (62, 84, 106), (size - edge, edge, edge, size - edge))
            self._walls[size] = tile
        return self._walls[size]
