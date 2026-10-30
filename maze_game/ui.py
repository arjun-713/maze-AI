"""A tactile block-maze desktop game. Rendering stays outside the engine."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import Any

from ._art import SpriteAtlas
from .engine import LEVEL_COUNT, MazeEnv

BG = (245, 244, 237)
PANEL = (255, 254, 250)
TEXT = (36, 54, 75)
MUTED = (92, 107, 119)
DIVIDER = (221, 220, 211)
ACCENT = (188, 78, 38)
SOFT = (239, 237, 227)
TEAL = (25, 127, 133)
RED = (186, 41, 57)
GOLD = (233, 183, 78)
BLUE = (52, 123, 210)
FONTS = Path(__file__).parent / "assets" / "fonts"


class MazeGame:
    def __init__(self, pygame: Any) -> None:
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("Bunny Maze")
        self.screen = pygame.display.set_mode((1380, 880), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(str(FONTS / "Nunito-Medium.ttf"), 16)
        self.small = pygame.font.Font(str(FONTS / "Nunito-Medium.ttf"), 14)
        self.tiny = pygame.font.Font(str(FONTS / "Nunito-Medium.ttf"), 12)
        self.bold = pygame.font.Font(str(FONTS / "Nunito-Bold.ttf"), 14)
        self.title = pygame.font.Font(str(FONTS / "Fredoka-SemiBold.ttf"), 32)
        self.heading = pygame.font.Font(str(FONTS / "Fredoka-SemiBold.ttf"), 24)
        self.number = pygame.font.Font(str(FONTS / "Fredoka-SemiBold.ttf"), 28)
        self.art = SpriteAtlas(pygame)
        self.level = 1
        self.seed = secrets.randbits(31)
        self.seed_text = str(self.seed)
        self.env = MazeEnv(self.level, self.seed)
        self.completed: set[int] = set()
        self.buttons: dict[str, Any] = {}
        self.level_buttons: dict[int, Any] = {}
        self._disabled_buttons: set[str] = set()
        self.seed_rect = pygame.Rect(0, 0, 0, 0)
        self.overview_rect = pygame.Rect(0, 0, 0, 0)
        self.viewport = pygame.Rect(0, 0, 0, 0)
        self.running = True
        self._held_keys: set[int] = set()
        self._reset()

    def _reset(self, level: int | None = None, seed: int | None = None) -> None:
        if level is not None:
            self.level = level
        if seed is not None:
            self.seed = seed
        self.seed_text = str(self.seed)
        self.env.reset(level=self.level, seed=self.seed)
        self.visited = {self.env.position}
        self.paused = False
        self.seed_active = False
        self.feedback = ""
        self.feedback_until = 0.0
        self.trap_hits = 0
        self.wall_hits = 0
        self.fit_map = False
        self._zoom_override: int | None = None
        self._world = None
        self._world_key = None
        self._overview = None
        self._camera_world = None
        self._previous_position = self.env.position
        self._moved_at = 0.0
        self.buttons.clear()
        self._disabled_buttons.clear()

    def _text(self, content: str, x: int, y: int, color: tuple[int, int, int] = TEXT, font: Any = None) -> Any:
        image = (font or self.font).render(content, True, color)
        self.screen.blit(image, (x, y))
        return image.get_rect(topleft=(x, y))

    def _button(self, key: str, label: str, rect: Any, *, primary: bool = False, disabled: bool = False) -> None:
        self.buttons[key] = rect
        if disabled:
            self._disabled_buttons.add(key)
        hovered = rect.collidepoint(self.pg.mouse.get_pos()) and not disabled
        pressed = hovered and self.pg.mouse.get_pressed()[0]
        fill = ACCENT if primary else SOFT
        foreground = (255, 255, 255) if primary else TEXT
        if hovered:
            fill = (166, 66, 30) if primary else (225, 228, 218)
        if disabled:
            fill, foreground = (233, 232, 224), (115, 119, 115)
        self.pg.draw.rect(self.screen, (214, 212, 201), rect.move(0, 2), border_radius=9)
        face = rect.move(0, 2 if pressed else 0)
        self.pg.draw.rect(self.screen, fill, face, border_radius=9)
        image = self.bold.render(label, True, foreground)
        self.screen.blit(image, image.get_rect(center=face.center))

    def _draw_header(self, width: int) -> None:
        self.screen.blit(self.art.get("bunny", 54), (24, 15))
        self._text("Bunny Maze", 88, 17, TEXT, self.title)
        self._text("Find the flag. Dodge the trouble.", 90, 56, MUTED, self.small)
        stats = (
            ("SCORE", f"{self.env.score:,}", 128),
            ("MOVES", f"{self.env.moves} / {self.env.move_limit}", 164),
            ("TIME", self._format_time(self.env.elapsed_seconds), 104),
        )
        x = width - 436
        for label, value, span in stats:
            self.pg.draw.line(self.screen, DIVIDER, (x - 17, 25), (x - 17, 72))
            self._text(label, x, 21, MUTED, self.tiny)
            self._text(value, x, 42, TEXT, self.number)
            x += span

    def _draw_sidebar(self, height: int, width: int) -> int:
        self.screen.fill(BG)
        self.buttons = {}
        self._disabled_buttons = set()
        self._draw_header(width)
        side_width = 294 if width >= 1100 else 254
        panel = self.pg.Rect(20, 108, side_width - 28, height - 128)
        self.pg.draw.rect(self.screen, PANEL, panel, border_radius=18)
        x = panel.x + 16
        inner = panel.width - 32
        self._text("Pick a level", x, 126, TEXT, self.heading)
        progress = self.small.render(f"{len(self.completed)} / 35", True, MUTED)
        self.screen.blit(progress, (panel.right - 16 - progress.get_width(), 133))
        gap = 6 if height >= 800 else 5
        button_h = 32 if height >= 800 else 29
        button_w = (inner - 4 * gap) // 5
        self.level_buttons = {}
        grid_y = 166
        for level in range(1, LEVEL_COUNT + 1):
            row, col = divmod(level - 1, 5)
            rect = self.pg.Rect(x + col * (button_w + gap), grid_y + row * (button_h + gap), button_w, button_h)
            self.level_buttons[level] = rect
            selected = level == self.level
            hovered = rect.collidepoint(self.pg.mouse.get_pos())
            fill = ACCENT if selected else (229, 234, 226) if hovered else SOFT
            self.pg.draw.rect(self.screen, fill, rect, border_radius=7)
            label = self.bold.render(str(level), True, (255, 255, 255) if selected else TEXT)
            self.screen.blit(label, label.get_rect(center=rect.center))
            if level in self.completed and not selected:
                self.pg.draw.circle(self.screen, TEAL, (rect.right - 5, rect.y + 5), 2)
        grid_bottom = grid_y + 7 * (button_h + gap)
        self.pg.draw.line(self.screen, DIVIDER, (x, grid_bottom + 13), (panel.right - 16, grid_bottom + 13))
        self._text("Layout seed", x, grid_bottom + 28, MUTED, self.small)
        self.seed_rect = self.pg.Rect(x, grid_bottom + 51, inner - 65, 38)
        self.pg.draw.rect(self.screen, SOFT, self.seed_rect, border_radius=8)
        if self.seed_active:
            self.pg.draw.rect(self.screen, TEAL, self.seed_rect, 2, border_radius=8)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(self.seed_rect.inflate(-12, -6))
        shown = self.seed_text + ("|" if self.seed_active and int(time.monotonic() * 2) % 2 == 0 else "")
        image = self.small.render(shown or "Type a number", True, TEXT)
        self.screen.blit(image, (max(self.seed_rect.x + 9, self.seed_rect.right - image.get_width() - 9), self.seed_rect.y + 9))
        self.screen.set_clip(old_clip)
        self._button("rebuild", "Apply", self.pg.Rect(self.seed_rect.right + 7, self.seed_rect.y, 58, 38))
        self._button("shuffle", "Shuffle maze", self.pg.Rect(x, grid_bottom + 102, inner, 38))
        overview_y = grid_bottom + 160
        available = min(inner, height - overview_y - 77)
        self.overview_rect = self.pg.Rect(0, 0, 0, 0)
        if available >= 92:
            self._text("Overview", x, overview_y - 22, MUTED, self.small)
            self.overview_rect = self.pg.Rect(x + (inner - available) // 2, overview_y, available, available)
            self._draw_overview(self.overview_rect)
        self._text("Move with arrows or WASD", x, height - 61, MUTED, self.tiny)
        self._text("Space: pause   R: restart", x, height - 42, MUTED, self.tiny)
        return side_width

    def _draw_overview(self, rect: Any) -> None:
        if self._overview is None:
            self._overview = self._paint_world(8)
        self.screen.blit(self.pg.transform.scale(self._overview, rect.size), rect)
        if self._camera_world is not None:
            world_size = self._world.get_width()
            camera = self._camera_world
            bounds = self.pg.Rect(rect.x + camera.x / world_size * rect.width,
                                  rect.y + camera.y / world_size * rect.height,
                                  camera.width / world_size * rect.width,
                                  camera.height / world_size * rect.height).clip(rect)
            self.pg.draw.rect(self.screen, (49, 124, 141), bounds, 2)
        for point, color in ((self.env._maze.start, BLUE), (self.env._maze.goal, GOLD), (self.env.position, (38, 213, 187))):
            x, y = self._cell_center(point, 8)
            extent = self._overview.get_width()
            center = (round(rect.x + x / extent * rect.width), round(rect.y + y / extent * rect.height))
            self.pg.draw.circle(self.screen, TEXT, center, 4)
            self.pg.draw.circle(self.screen, color, center, 3)

    def _cell_center(self, point: tuple[float, float], tile: int) -> tuple[float, float]:
        """Floor, wall, and movement blocks all use this exact pixel pitch."""
        row, col = point
        return (col + 0.5) * tile, (row + 0.5) * tile

    def _paint_world(self, tile: int) -> Any:
        maze = self.env._maze
        extent = maze.size * tile
        surface = self.pg.Surface((extent, extent)).convert()
        wall = self.art.wall(tile)
        for row in range(maze.size):
            for col in range(maze.size):
                rect = self.pg.Rect(col * tile, row * tile, tile, tile)
                if maze.passages[row][col]:
                    color = (244, 236, 215) if (row + col) % 2 else (239, 228, 202)
                    self.pg.draw.rect(surface, color, rect)
                else:
                    surface.blit(wall, rect)
        for point, name, color in (
            (maze.start, "home", (210, 229, 249)),
            (maze.goal, "flag", (250, 220, 144)),
            *((point, "enemy", (246, 204, 198)) for point in maze.traps),
        ):
            row, col = point
            rect = self.pg.Rect(col * tile, row * tile, tile, tile)
            self.pg.draw.rect(surface, color, rect)
            surface.blit(self.art.get(name, max(2, tile - 2)), (rect.x + 1, rect.y + 1))
        return surface

    def _build_world(self, tile: int) -> None:
        key = (id(self.env._maze), tile)
        if self._world_key != key:
            self._world = self._paint_world(tile)
            self._world_key = key

    def _visual_position(self) -> tuple[float, float]:
        row, col = self.env.position
        t = min(1.0, (time.monotonic() - self._moved_at) / 0.10)
        eased = 1 - (1 - t) ** 3
        pr, pc = self._previous_position
        return pr + (row - pr) * eased, pc + (col - pc) * eased

    def _map_label(self, label: str, center: tuple[int, int], color: tuple[int, int, int], below: bool = False) -> None:
        if not self.viewport.collidepoint(center):
            return
        image = self.tiny.render(label, True, (255, 255, 255) if color == BLUE else TEXT)
        box = image.get_rect().inflate(12, 5)
        box.centerx = center[0]
        box.y = center[1] + self._tile_size // 2 + 2 if below else center[1] - self._tile_size // 2 - box.height - 2
        box.clamp_ip(self.viewport.inflate(-8, -8))
        self.pg.draw.rect(self.screen, color, box, border_radius=4)
        self.screen.blit(image, image.get_rect(center=box.center))

    def _draw_maze(self, side_width: int, width: int, height: int) -> None:
        stage = self.pg.Rect(side_width + 12, 108, width - side_width - 32, height - 128)
        self.pg.draw.rect(self.screen, PANEL, stage, border_radius=18)
        self._text(f"Level {self.level:02d}", stage.x + 20, stage.y + 13, TEXT, self.heading)
        self._text(f"{self.env._layout.size} x {self.env._layout.size}   /   Tier {(self.level - 1) // 5 + 1}", stage.x + 21, stage.y + 43, MUTED, self.small)
        self._button("restart", "Restart", self.pg.Rect(stage.right - 186, stage.y + 20, 82, 36))
        self._button("pause", "Resume" if self.paused else "Pause", self.pg.Rect(stage.right - 96, stage.y + 20, 76, 36), disabled=self.env.done or self.env.truncated)
        legend_height = 104 if stage.width < 920 else 72
        self.viewport = self.pg.Rect(stage.x + 16, stage.y + 76, stage.width - 32, stage.height - 132 - legend_height)
        vp = self.viewport
        self.pg.draw.rect(self.screen, (217, 230, 229), vp, border_radius=12)
        span = self.env._maze.size
        fit = max(2, min((vp.width - 28) // span, (vp.height - 28) // span))
        minimum = 18 if self.env._layout.size <= 13 else 24
        tile = fit if self.fit_map else self._zoom_override or min(36, max(minimum, fit))
        self._tile_size = tile
        self._build_world(tile)
        world_width = self._world.get_width()
        mode = "Full map" if self.fit_map else "Follow view" if world_width > min(vp.width, vp.height) else "Whole maze"
        self._text(mode, stage.x + 180, stage.y + 43, MUTED, self.small)
        visual_row, visual_col = self._visual_position()
        px, py = self._cell_center((visual_row, visual_col), tile)

        def origin(start: int, length: int, point: float) -> int:
            if world_width <= length - 28:
                return start + (length - world_width) // 2
            return round(max(start + length - world_width - 14, min(start + 14, start + length / 2 - point)))

        ox, oy = origin(vp.x, vp.width, px), origin(vp.y, vp.height, py)
        self._camera_world = self.pg.Rect(vp.x - ox, vp.y - oy, vp.width, vp.height).clip(self._world.get_rect())
        clip = self.screen.get_clip()
        self.screen.set_clip(vp.inflate(-2, -2))
        frame = self.pg.Rect(ox - 5, oy - 5, world_width + 10, world_width + 10)
        self.pg.draw.rect(self.screen, (166, 187, 187), frame.move(0, 5), border_radius=5)
        self.pg.draw.rect(self.screen, (46, 66, 88), frame, border_radius=5)
        self.screen.blit(self._world, (ox, oy))
        for point in self.visited:
            if point in self.env._maze.traps or point in (self.env._maze.start, self.env._maze.goal):
                continue
            cx, cy = self._cell_center(point, tile)
            center = (round(ox + cx), round(oy + cy))
            if vp.collidepoint(center):
                self.pg.draw.circle(self.screen, (161, 191, 171), center, max(1, tile // 10))
        player_size = max(14, tile)
        player = (round(ox + px), round(oy + py))
        self.pg.draw.ellipse(self.screen, (114, 147, 132), (player[0] - player_size // 3, player[1] + player_size // 4, player_size * 2 // 3, max(2, player_size // 5)))
        bunny = self.art.get("bunny", player_size)
        self._player_rect = bunny.get_rect(center=player)
        self.screen.blit(bunny, self._player_rect)
        for point, label, color, below in ((self.env._maze.start, "START", BLUE, False), (self.env._maze.goal, "FINISH", GOLD, True)):
            cx, cy = self._cell_center(point, tile)
            center = (round(ox + cx), round(oy + cy))
            if label == "FINISH" and tile < 18:
                flag = self.art.get("flag", 18)
                self.screen.blit(flag, flag.get_rect(center=center))
            self._map_label(label, center, color, below)
        self.screen.set_clip(clip)
        toolbar_y = vp.bottom + 8
        self._button("fit", "Follow bunny" if self.fit_map else "Fit map", self.pg.Rect(vp.x + 10, toolbar_y, 108, 31))
        self._button("zoom_out", "-", self.pg.Rect(vp.x + 126, toolbar_y, 31, 31))
        self._button("zoom_in", "+", self.pg.Rect(vp.x + 164, toolbar_y, 31, 31))
        self._button("training", "Training Studio", self.pg.Rect(vp.right - 150, toolbar_y, 140, 31))
        if not vp.collidepoint(center) and vp.width >= 550:
            row, col = self.env.position
            goal_row, goal_col = self.env._maze.goal
            direction = ("south" if goal_row > row else "north" if goal_row < row else "")
            direction += ("east" if goal_col > col else "west" if goal_col < col else "")
            self.screen.blit(self.art.get("flag", 22), (vp.x + 208, toolbar_y + 4))
            self._text(f"Finish: {direction}", vp.x + 235, toolbar_y + 7, MUTED, self.small)
        self._draw_legend(stage, legend_height)
        if self.feedback and time.monotonic() < self.feedback_until:
            image = self.bold.render(self.feedback, True, RED if self.feedback_kind == "penalty" else TEXT)
            box = image.get_rect().inflate(28, 18)
            box.midtop = (vp.centerx, vp.y + 32)
            self.pg.draw.rect(self.screen, PANEL, box, border_radius=9)
            self.screen.blit(image, image.get_rect(center=box.center))
        if self.paused or self.env.done or self.env.truncated:
            self._draw_result(vp)

    def _draw_legend(self, stage: Any, height: int) -> None:
        entries = (("bunny", "You", "Pip the bunny"), ("home", "Start", "Blue home pad"),
                   ("flag", "Finish", "Checkered flag"), ("enemy", "Troublemaker", "-50 each visit"),
                   ("wall", "Wall", "-5 if bumped"))
        columns = 3 if height > 80 else 5
        y = stage.bottom - height + 10
        for i, (sprite, title, detail) in enumerate(entries):
            x = stage.x + 20 + (i % columns) * ((stage.width - 32) // columns)
            row_y = y + (i // columns) * 44
            if sprite == "wall":
                self.screen.blit(self.art.wall(28), (x + 3, row_y + 3))
            else:
                self.screen.blit(self.art.get(sprite, 34), (x, row_y))
            self._text(title, x + 42, row_y, TEXT, self.bold)
            self._text(detail, x + 42, row_y + 19, MUTED, self.tiny)

    def _draw_result(self, board: Any) -> None:
        shade = self.pg.Surface(board.size, self.pg.SRCALPHA)
        shade.fill((31, 47, 60, 165))
        self.screen.blit(shade, board.topleft)
        panel = self.pg.Rect(0, 0, min(440, board.width - 28), 260)
        panel.center = board.center
        self.pg.draw.rect(self.screen, (43, 63, 75), panel.move(0, 6), border_radius=18)
        self.pg.draw.rect(self.screen, PANEL, panel, border_radius=18)
        sprite = "bunny" if self.paused or self.env.truncated else "flag"
        self.screen.blit(self.art.get(sprite, 56), (panel.x + 22, panel.y + 22))
        title = "Taking a breather" if self.paused else "Level cleared!" if self.env.done else "Let's try again"
        self._text(title, panel.x + 90, panel.y + 27, TEXT, self.heading)
        subtitle = "Your next hop can wait." if self.paused else "Pip made it to the finish." if self.env.done else "You reached the move limit."
        self._text(subtitle, panel.x + 90, panel.y + 59, MUTED, self.small)
        if self.paused:
            self._text("Space to jump back in", panel.x + 25, panel.y + 116, MUTED, self.font)
            self._button("resume", "Resume playing", self.pg.Rect(panel.x + 24, panel.bottom - 70, panel.width - 48, 44), primary=True)
            return
        self._text(f"{self.env.score:,}", panel.x + 25, panel.y + 97, ACCENT, self.title)
        self._text("POINTS", panel.x + 25, panel.y + 136, MUTED, self.tiny)
        quality = min(100, round(self.env._optimal_moves / max(1, self.env.moves) * 100))
        detail = f"{self.env.moves} moves  /  {self.env._optimal_moves} optimal  /  {quality}% efficiency"
        self._text(detail, panel.x + 25, panel.y + 163, TEXT, self.small)
        if self.env.done and self.level < LEVEL_COUNT:
            self._button("retry", "Play again", self.pg.Rect(panel.x + 24, panel.bottom - 59, 125, 38))
            self._button("next", "Next level", self.pg.Rect(panel.x + 160, panel.bottom - 59, panel.width - 184, 38), primary=True)
        else:
            self._button("retry", "Play again" if self.env.done else "Try again", self.pg.Rect(panel.x + 24, panel.bottom - 59, panel.width - 48, 38), primary=True)

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _move(self, direction: str) -> None:
        if self.paused or self.env.done or self.env.truncated:
            return
        previous = self.env.position
        _, _, done, _, info = self.env.step(direction)
        self._previous_position = previous
        self._moved_at = time.monotonic()
        self.visited.add(self.env.position)
        self.feedback_kind = "penalty"
        if info["hit_wall"]:
            self.wall_hits += 1
            self.feedback = "Blocked by a wall!  -5"
        elif info["entered_trap"]:
            self.trap_hits += 1
            self.feedback = "Troublemaker!  -50"
        elif info["extra_move"] and not done:
            self.feedback = "Extra move  -3"
        else:
            self.feedback = ""
        self.feedback_until = time.monotonic() + 0.9
        if done:
            self.completed.add(self.level)

    def _toggle_pause(self) -> None:
        if self.env.done or self.env.truncated:
            return
        self.paused = not self.paused
        if self.paused:
            self.env.pause_timer()
        else:
            self.env.resume_timer()

    def _toggle_map(self) -> None:
        self.fit_map = not self.fit_map
        self._zoom_override = None

    def _zoom(self, amount: int) -> None:
        current = getattr(self, "_tile_size", 24)
        self._zoom_override = max(10, min(48, current + amount))
        self.fit_map = False

    def _rebuild_from_text(self) -> None:
        try:
            chosen_seed = int(self.seed_text)
        except ValueError:
            self.feedback = "Enter a whole number for the seed."
            self.feedback_kind = "penalty"
            self.feedback_until = time.monotonic() + 3
            return
        self._reset(seed=chosen_seed)

    def _click(self, point: tuple[int, int]) -> None:
        for level, rect in self.level_buttons.items():
            if rect.collidepoint(point):
                self._reset(level=level)
                return
        for key, rect in self.buttons.items():
            if not rect.collidepoint(point) or key in self._disabled_buttons:
                continue
            if key in ("restart", "retry"):
                self._reset()
            elif key in ("pause", "resume"):
                self._toggle_pause()
            elif key == "rebuild":
                self._rebuild_from_text()
            elif key == "shuffle":
                self._reset(seed=secrets.randbits(31))
            elif key == "next" and self.level < LEVEL_COUNT:
                self._reset(level=self.level + 1)
            elif key == "fit":
                self._toggle_map()
            elif key == "training":
                self._next_mode = "training"
                self.running = False
            elif key in ("zoom_in", "zoom_out"):
                self._zoom(4 if key == "zoom_in" else -4)
            return
        if self.overview_rect.collidepoint(point):
            self._toggle_map()
            return
        self.seed_active = self.seed_rect.collidepoint(point)

    def _handle_key(self, event: Any) -> None:
        pg = self.pg
        if event.key == pg.K_ESCAPE:
            self.seed_active = False
            if self.paused:
                self._toggle_pause()
            return
        if self.seed_active:
            if event.key == pg.K_RETURN:
                self._rebuild_from_text()
            elif event.key == pg.K_BACKSPACE:
                self.seed_text = self.seed_text[:-1]
            else:
                char = getattr(event, "unicode", "")
                if len(self.seed_text) < 20 and (char.isdigit() or (char == "-" and not self.seed_text)):
                    self.seed_text += char
            return
        if event.key in (pg.K_SPACE, pg.K_p):
            self._toggle_pause()
        elif event.key == pg.K_r:
            self._reset()
        elif event.key == pg.K_m:
            self._toggle_map()
        elif event.key == pg.K_t:
            self._next_mode = "training"
            self.running = False
        elif event.key in (pg.K_EQUALS, pg.K_PLUS, pg.K_KP_PLUS):
            self._zoom(4)
        elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
            self._zoom(-4)
        elif event.key == pg.K_a:
            self._move("left")
        elif self.env.done or self.env.truncated:
            if event.key in (pg.K_RETURN, pg.K_RIGHT) and self.env.done and self.level < LEVEL_COUNT:
                self._reset(level=self.level + 1)
        else:
            keymap = {pg.K_UP: "up", pg.K_w: "up", pg.K_RIGHT: "right", pg.K_d: "right", pg.K_DOWN: "down", pg.K_s: "down", pg.K_LEFT: "left"}
            if event.key in keymap:
                self._move(keymap[event.key])

    def _handle_event(self, event: Any) -> None:
        pg = self.pg
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.VIDEORESIZE:
            size = (max(900, event.w), max(680, event.h))
            self.screen = pg.display.set_mode(size, pg.RESIZABLE)
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self._click(event.pos)
        elif event.type == pg.MOUSEWHEEL and self.viewport.collidepoint(pg.mouse.get_pos()):
            self._zoom(4 * event.y)
        elif event.type == pg.KEYUP:
            self._held_keys.discard(event.key)
        elif event.type == pg.WINDOWFOCUSLOST:
            self._held_keys.clear()
        elif event.type == pg.KEYDOWN:
            if event.key in self._held_keys or getattr(event, "repeat", False):
                return
            self._held_keys.add(event.key)
            self._handle_key(event)

    def run(self) -> None:
        self.pg.key.set_repeat()
        while self.running:
            for event in self.pg.event.get():
                self._handle_event(event)
            width, height = self.screen.get_size()
            side = self._draw_sidebar(height, width)
            self._draw_maze(side, width, height)
            self.pg.display.flip()
            self.clock.tick(60)
        self.pg.quit()


def launch() -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install the desktop dependency with: python -m pip install -e '.[pygame]'") from exc
    game = MazeGame(pygame)
    game.run()
    if getattr(game, "_next_mode", None) == "training":
        from .training_ui import launch_dashboard

        launch_dashboard(level=game.level, seed=game.seed)
