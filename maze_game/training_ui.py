"""Pygame Training Studio: live maze, agent observations, metrics, and replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
import secrets
import time
from typing import Any, Callable

from ._generator import DELTAS
from ._window import initial_size
from .training import Agent, EpisodeReplay, RunConfig, RunStore, TrainingSession, aggregate, load_agent
from .ui import ACCENT, BG, BLUE, DIVIDER, GOLD, MUTED, PANEL, RED, SOFT, TEAL, TEXT, MazeGame


class TrainingStudio(MazeGame):
    SPEEDS = (1, 5, 20, 100, 500)

    def __init__(self, pygame: Any, *, level: int = 1, seed: int = 42, episodes: int = 100,
                 runs_dir: str | Path = ".maze-runs", agent_spec: str = "Redbull.agent:Redbull",
                 agent_factory: Callable[[int], Agent] | None = None) -> None:
        super().__init__(pygame)
        self.screen = pygame.display.set_mode(initial_size(pygame, (1560, 980), (1100, 760)), pygame.RESIZABLE)
        pygame.display.set_caption("Bunny Maze / Training Studio")
        self.store = RunStore(Path(runs_dir) / "training.sqlite3")
        self.agent_factory = agent_factory or (lambda run_seed: load_agent(agent_spec, run_seed))
        self.session = TrainingSession(RunConfig(level, seed, episodes), self.store,
                                       self.agent_factory(seed))
        self.next_level, self.next_episodes = level, episodes
        self.run_seed_text = str(seed)
        self.run_seed_active = False
        self.speed = 20
        self.credit = 0.0
        self.view = "Game"
        self.selected_run = self.session.run_id
        self.compare_run: str | None = None
        self.replay: EpisodeReplay | None = None
        self.replay_run: str | None = None
        self.replay_playing = False
        self.replay_drag = False
        self.full_map = False
        self.fit_map = False
        self._zoom_override = None
        self.error = ""
        self.table_scroll = 0
        self.run_scroll = 0
        self.debug_scroll = 0
        self._cache_at = 0.0
        self.table_rect = pygame.Rect(0, 0, 0, 0)
        self.history_rect = pygame.Rect(0, 0, 0, 0)
        self.debug_rect = pygame.Rect(0, 0, 0, 0)
        self.replay_slider = pygame.Rect(0, 0, 0, 0)
        self._sync_source()
        self._refresh(True)

    @property
    def source(self) -> Any:
        return self.replay if self.replay is not None else self.session

    @property
    def agent_ready(self) -> bool:
        return bool(getattr(self.session.agent, "ready", True))

    def _sync_source(self) -> None:
        self.env = self.source.env
        self.level = self.env.level

    def _refresh(self, force: bool = False) -> None:
        if not force and time.monotonic() - self._cache_at < 0.5:
            return
        self.runs = self.store.runs()
        self.run_names = {run["id"]: run["name"] for run in self.runs}
        self.run_info = {run["id"]: run for run in self.runs}
        self.records = self.store.episodes(self.selected_run)
        self.comparison = self.store.episodes(self.compare_run, complete_only=True) if self.compare_run else []
        self._cache_at = time.monotonic()

    def _new_run(self) -> None:
        try:
            config = RunConfig(self.next_level, int(self.run_seed_text), self.next_episodes)
            agent = self.agent_factory(config.seed)
        except Exception as exc:
            self.error = str(exc)
            return
        self.session.stop()
        self.session = TrainingSession(config, self.store, agent)
        self.replay = None
        self.replay_playing = False
        self.selected_run = self.session.run_id
        self.compare_run = None
        self.credit = 0.0
        self.error = ""
        self.table_scroll = 0
        self._previous_position = self.session.env.position
        self._moved_at = 0.0
        self._sync_source()
        self._refresh(True)

    def _toggle_running(self) -> None:
        self.credit = 0.0
        if self.replay is not None:
            if self.replay.index == len(self.replay.record.steps):
                self._seek(0)
            self.replay_playing = not self.replay_playing
        elif self.session.status == "running":
            self.session.pause()
        else:
            try:
                self.session.resume()
            except ValueError as exc:
                self.error = str(exc)
        self._refresh(True)

    def _load_replay(self, number: int) -> None:
        self.session.pause()
        try:
            replay = EpisodeReplay(self.store.load_episode(self.selected_run, number))
        except (ValueError, TypeError, KeyError) as exc:
            self.error = str(exc)
            return
        self.replay = replay
        self.replay_run = self.selected_run
        self.replay_playing = False
        self.view = "Game"
        self.credit = 0.0
        self._previous_position = replay.env.position
        self._moved_at = 0.0
        self._sync_source()

    def _seek(self, index: int) -> None:
        if self.replay is None:
            return
        try:
            self.replay.seek(max(0, min(len(self.replay.record.steps), index)))
            self._sync_source()
            self._previous_position = self.env.position
            self._moved_at = 0.0
        except (ValueError, RuntimeError) as exc:
            self.error = str(exc)
            self.replay_playing = False

    def update(self, delta_seconds: float) -> None:
        active = self.replay_playing if self.replay is not None else self.session.status == "running"
        if not active:
            self.credit = 0.0
            return
        self.credit = min(250.0, self.credit + min(0.1, max(0.0, delta_seconds)) * self.speed)
        deadline = time.perf_counter() + 0.008
        while self.credit >= 1 and time.perf_counter() < deadline:
            try:
                if self.replay is not None:
                    if self.replay.index >= len(self.replay.record.steps):
                        self.replay_playing = False
                        break
                    self.replay.seek(self.replay.index + 1)
                else:
                    if self.session.status != "running":
                        break
                    self.session.step()
                self._sync_source()
                self._previous_position = self.source.path[-2] if len(self.source.path) >= 2 else self.env.position
                self._moved_at = time.monotonic()
            except Exception as exc:
                self.error = str(exc)
                self.replay_playing = False
                break
            self.credit -= 1
        self._refresh()

    def _panel(self, rect: Any) -> None:
        self.pg.draw.rect(self.screen, PANEL, rect, border_radius=18)

    def _short(self, text: str, width: int, font: Any = None) -> str:
        font = font or self.small
        while text and font.size(text)[0] > width:
            text = text[:-2]
        return text

    def _header(self, width: int) -> None:
        self.screen.blit(self.art.get("bunny", 50), (22, 17))
        self._text("Training Studio", 84, 15, TEXT, self.title)
        agent_name = self.run_info.get(self.replay_run, {}).get("agent", self.session.agent.name) if self.replay else self.session.agent.name
        description = f"{agent_name} / recorded replay" if self.replay else (
            f"{agent_name} / agent telemetry" if self.agent_ready else "Your AI workspace / waiting for your implementation")
        self._text(self._short(description, width - 590), 86, 56, MUTED, self.small)
        state = f"Step {self.replay.index}/{len(self.replay.record.steps)}" if self.replay else f"{len(self.session.episodes)} / {self.session.config.episodes} episodes"
        self._text(state, width - 437, 39, MUTED, self.small)
        label = ("Pause replay" if self.replay_playing else "Play replay") if self.replay else (
            "Pause" if self.session.status == "running" else "Start agent" if not self.session.current.steps else "Resume")
        disabled = self.replay is None and (not self.agent_ready or self.session.status not in ("paused", "running"))
        if disabled:
            label = self.session.status.capitalize() if self.agent_ready else "Not ready"
        self._button("toggle", label, self.pg.Rect(width - 290, 27, 130, 40), primary=True, disabled=disabled)
        self._button("new", "New run", self.pg.Rect(width - 148, 27, 126, 40))

    def _sidebar(self, height: int) -> None:
        panel = self.pg.Rect(20, 102, 272, height - 122)
        self._panel(panel)
        x = panel.x + 16
        self._text("Run setup", x, 120, TEXT, self.heading)
        self._text("Changes apply to the next run", x, 151, MUTED, self.tiny)
        self._text("Level", x, 184, MUTED, self.small)
        self._button("level-", "-", self.pg.Rect(x + 92, 176, 34, 34), disabled=self.next_level == 1)
        self._text(f"{self.next_level:02d}", x + 142, 181, TEXT, self.bold)
        self._button("level+", "+", self.pg.Rect(x + 204, 176, 34, 34), disabled=self.next_level == 35)
        self._text("Episodes", x, 229, MUTED, self.small)
        self._button("episodes-", "-", self.pg.Rect(x + 92, 221, 34, 34), disabled=self.next_episodes <= 1)
        self._text(str(self.next_episodes), x + 137, 228, TEXT, self.bold)
        self._button("episodes+", "+", self.pg.Rect(x + 204, 221, 34, 34), disabled=self.next_episodes >= 10000)
        self._text("Run seed", x, 275, MUTED, self.small)
        self.seed_rect = self.pg.Rect(x, 300, 158, 36)
        self.pg.draw.rect(self.screen, SOFT, self.seed_rect, border_radius=8)
        if self.run_seed_active:
            self.pg.draw.rect(self.screen, TEAL, self.seed_rect, 2, border_radius=8)
        self._text(self._short(self.run_seed_text or "Type a number", 140), x + 9, 309, TEXT, self.small)
        self._button("seed", "Shuffle", self.pg.Rect(x + 166, 300, 72, 36))
        self._text("Simulation speed", x, 359, TEXT, self.bold)
        for i, speed in enumerate(self.SPEEDS):
            self._button(f"speed:{speed}", str(speed), self.pg.Rect(x + i * 49, 384, 43, 34), primary=speed == self.speed)
        self._text(f"{self.speed} actions / second", x, 428, MUTED, self.small)
        self.pg.draw.line(self.screen, DIVIDER, (x, 460), (panel.right - 16, 460))
        self._text("Saved runs", x, 477, TEXT, self.heading)
        self._text("Select a run to inspect its metrics", x, 510, MUTED, self.tiny)
        history_height = max(84, panel.bottom - 575)
        self.history_rect = self.pg.Rect(x, 537, 240, history_height)
        visible = max(1, history_height // 61)
        self.run_scroll = min(self.run_scroll, max(0, len(self.runs) - visible))
        for i, run in enumerate(self.runs[self.run_scroll:self.run_scroll + visible]):
            rect = self.pg.Rect(x, 537 + i * 61, 240, 52)
            self.buttons["run:" + run["id"]] = rect
            selected = run["id"] == self.selected_run
            self.pg.draw.rect(self.screen, (221, 237, 229) if selected else SOFT, rect, border_radius=8)
            self._text(self._short(run["name"], 220, self.bold), x + 10, rect.y + 6, TEXT, self.bold)
            detail = f"L{run['config']['level']:02d}  /  {run['status']}"
            self._text(detail, x + 10, rect.y + 29, MUTED, self.tiny)
        self._text("Space: pause / resume", x, height - 56, MUTED, self.tiny)
        self._text("1: Game   2: Learning   3: Dashboard", x, height - 37, MUTED, self.tiny)

    def _replay_strip(self, rect: Any) -> None:
        if self.replay is None:
            return
        self._panel(rect)
        count = len(self.replay.record.steps)
        replay_name = self.run_names.get(self.replay_run, "Replay")
        self._text(self._short(f"Ep {self.replay.record.number} / {replay_name}", 137, self.bold), rect.x + 12, rect.y + 8, TEXT, self.bold)
        self._text(f"Step {self.replay.index} / {count}", rect.x + 12, rect.y + 27, MUTED, self.tiny)
        self._button("back", "-1", self.pg.Rect(rect.x + 156, rect.y + 9, 35, 30))
        self._button("forward", "+1", self.pg.Rect(rect.x + 198, rect.y + 9, 35, 30))
        self.replay_slider = self.pg.Rect(rect.x + 253, rect.y + 15, rect.width - 405, 20)
        track = self.pg.Rect(self.replay_slider.x, rect.y + 22, self.replay_slider.width, 5)
        self.pg.draw.rect(self.screen, DIVIDER, track, border_radius=2)
        ratio = self.replay.index / max(1, count)
        self.pg.draw.circle(self.screen, TEAL, (round(track.x + ratio * track.width), track.centery), 7)
        self._button("live", "Return live", self.pg.Rect(rect.right - 134, rect.y + 9, 122, 30))

    def draw(self) -> None:
        self._sync_source()
        self.screen.fill(BG)
        self.buttons = {}
        self._disabled_buttons = set()
        width, height = self.screen.get_size()
        self._header(width)
        self._sidebar(height)
        for i, view in enumerate(("Game", "Learning", "Dashboard")):
            self._button("tab:" + view, view, self.pg.Rect(312 + i * 154, 103, 144, 36), primary=self.view == view)
        caption = "Recorded actions / deterministic replay" if self.replay else "New maze each episode"
        self._text(caption, width - 330, 113, MUTED, self.small)
        top = 154
        if self.replay:
            self._replay_strip(self.pg.Rect(312, top, width - 332, 48))
            top += 60
        elif not self.agent_ready:
            self._panel(self.pg.Rect(312, top, width - 332, 76))
            self._text("Build Redbull in Redbull/agent.py", 328, top + 12, TEXT, self.bold)
            self._text("Set ready = True, then restart. No actions are generated for you.", 328, top + 40, MUTED, self.small)
            top += 88
        body = self.pg.Rect(312, top, width - 332, height - top - 20)
        if self.view == "Game":
            self._draw_game(body)
        elif self.view == "Learning":
            self._draw_learning(body)
        else:
            self._draw_dashboard(body)
        if self.error:
            rect = self.pg.Rect(312, height - 64, width - 332, 44)
            self.pg.draw.rect(self.screen, (252, 226, 217), rect, border_radius=8)
            self._text(self._short(self.error, rect.width - 24), rect.x + 12, rect.y + 12, RED, self.small)

    def _draw_game(self, body: Any) -> None:
        right_width = 250 if body.width >= 950 else 208
        board = self.pg.Rect(body.x, body.y, body.width - right_width - 12, body.height)
        info = self.pg.Rect(board.right + 12, body.y, right_width, body.height)
        self._panel(board)
        self._panel(info)
        self._text("Agent in the maze", board.x + 16, board.y + 14, TEXT, self.heading)
        current = self.replay.record if self.replay else self.session.current
        self._text(f"Episode {current.number} / L{current.level:02d} / seed {current.seed}", board.x + 17, board.y + 47, MUTED, self.tiny)
        self._button("fog", "Full map" if not self.full_map else "Discovery", self.pg.Rect(board.right - 116, board.y + 16, 100, 32))
        self.viewport = self.pg.Rect(board.x + 12, board.y + 78, board.width - 24, board.height - 164)
        vp = self.viewport
        self.pg.draw.rect(self.screen, (217, 230, 229), vp, border_radius=10)
        fit = max(2, min((vp.width - 24) // self.env._maze.size, (vp.height - 24) // self.env._maze.size))
        tile = fit if self.fit_map else self._zoom_override or max(24, min(36, fit))
        self._tile_size = tile
        self._build_world(tile)
        extent = self._world.get_width()
        row, col = self._visual_position()
        px, py = self._cell_center((row, col), tile)

        def origin(start: int, length: int, center: float) -> int:
            if extent <= length - 24:
                return start + (length - extent) // 2
            return round(max(start + length - extent - 12, min(start + 12, start + length / 2 - center)))

        ox, oy = origin(vp.x, vp.width, px), origin(vp.y, vp.height, py)
        clip = self.screen.get_clip()
        self.screen.set_clip(vp)
        self.screen.blit(self._world, (ox, oy))
        if not self.full_map:
            veil = self.pg.Surface((tile, tile), self.pg.SRCALPHA)
            veil.fill((36, 54, 75, 225))
            r0, c0 = max(0, (vp.y - oy) // tile), max(0, (vp.x - ox) // tile)
            r1, c1 = min(self.env._maze.size, (vp.bottom - oy) // tile + 1), min(self.env._maze.size, (vp.right - ox) // tile + 1)
            for r in range(r0, r1):
                for c in range(c0, c1):
                    if (r, c) not in self.source.seen:
                        self.screen.blit(veil, (ox + c * tile, oy + r * tile))
        path = [tuple(round(v) for v in self._cell_center(point, tile)) for point in self.source.path]
        if len(path) >= 2:
            self.pg.draw.lines(self.screen, (47, 145, 143), False, [(ox + x, oy + y) for x, y in path], max(2, tile // 9))
        for previous, action in self.source.wall_events[-30:]:
            dr, dc = DELTAS[action]
            cx, cy = self._cell_center((previous[0] + dr, previous[1] + dc), tile)
            cx, cy = round(ox + cx), round(oy + cy)
            radius = max(2, tile // 5)
            self.pg.draw.line(self.screen, RED, (cx - radius, cy - radius), (cx + radius, cy + radius), 2)
            self.pg.draw.line(self.screen, RED, (cx - radius, cy + radius), (cx + radius, cy - radius), 2)
        sprite = self.art.get("bunny", max(14, tile))
        self.screen.blit(sprite, sprite.get_rect(center=(round(ox + px), round(oy + py))))
        self.screen.set_clip(clip)
        self._button("fit", "Follow" if self.fit_map else "Fit map", self.pg.Rect(vp.x, vp.bottom + 10, 88, 30))
        self._button("zoom_out", "-", self.pg.Rect(vp.x + 98, vp.bottom + 10, 32, 30))
        self._button("zoom_in", "+", self.pg.Rect(vp.x + 138, vp.bottom + 10, 32, 30))
        self._text("Teal: path / red X: collision", vp.x, board.bottom - 29, MUTED, self.tiny)
        x = info.x + 16
        self._text("This episode", x, info.y + 16, TEXT, self.heading)
        steps = current.steps[:self.replay.index] if self.replay else current.steps
        seconds = self.replay.seconds if self.replay else self.env.elapsed_seconds
        metrics = (("Return", f"{self.env.score:+,}"), ("Moves", f"{self.env.moves} / {self.env.move_limit}"),
                   ("Trap entries", str(sum(s.entered_trap for s in steps))), ("Wall collisions", str(sum(s.hit_wall for s in steps))),
                   ("Seen cells", str(len(self.source.seen))), ("Active time", f"{seconds:.1f}s"))
        for i, (label, value) in enumerate(metrics):
            y = info.y + 65 + i * 46
            self._text(label, x, y, MUTED, self.small)
            image = self.bold.render(value, True, TEXT)
            self.screen.blit(image, (info.right - 16 - image.get_width(), y))
        y = info.y + 354
        self.pg.draw.line(self.screen, DIVIDER, (x, y - 14), (info.right - 16, y - 14))
        self._text("Recent actions", x, y, TEXT, self.bold)
        count = max(1, (info.bottom - y - 50) // 35)
        for i, step in enumerate(reversed(steps[-count:])):
            yy = y + 31 + i * 35
            color = RED if step.hit_wall or step.entered_trap else TEXT
            self._text(step.action.capitalize(), x, yy, color, self.small)
            event = "trap" if step.entered_trap else "wall" if step.hit_wall else "step"
            value = f"{event}  {step.reward:+d}"
            image = self.tiny.render(value, True, color)
            self.screen.blit(image, (info.right - 16 - image.get_width(), yy + 2))

    def _draw_learning(self, body: Any) -> None:
        triple = body.width >= 1000
        obs_width = int(body.width * (0.34 if triple else 0.46))
        obs_rect = self.pg.Rect(body.x, body.y, obs_width, body.height)
        rest_x = obs_rect.right + 12
        rest_width = body.right - rest_x
        if triple:
            decision = self.pg.Rect(rest_x, body.y, int(rest_width * 0.49), body.height)
            debug_rect = self.pg.Rect(decision.right + 12, body.y, body.right - decision.right - 12, body.height)
        else:
            decision = self.pg.Rect(rest_x, body.y, rest_width, min(400, max(360, round(body.height * 0.67))))
            debug_rect = self.pg.Rect(rest_x, decision.bottom + 12, rest_width, body.bottom - decision.bottom - 12)
        for rect in (obs_rect, decision, debug_rect):
            self._panel(rect)
        observation = self.source.last_observation or self.env.observation
        step = self.source.last_step
        x, y = obs_rect.x + 16, obs_rect.y + 16
        self._text("Agent observation", x, y, TEXT, self.heading)
        self._text("Input to the last selected action", x, y + 34, MUTED, self.tiny)
        tile = min(58, (obs_rect.width - 40) // 5, max(20, (obs_rect.height - 300) // 5))
        gx, gy = obs_rect.centerx - tile * 5 // 2, y + 76
        for r, row in enumerate(observation["local_view"]):
            for c, cell in enumerate(row):
                rect = self.pg.Rect(gx + c * tile, gy + r * tile, tile - 1, tile - 1)
                if cell is None:
                    self.pg.draw.rect(self.screen, (219, 223, 219), rect)
                    self.pg.draw.line(self.screen, (177, 185, 182), rect.bottomleft, rect.topright)
                elif cell["wall"]:
                    self.screen.blit(self.art.wall(tile - 1), rect)
                else:
                    self.pg.draw.rect(self.screen, (244, 236, 215), rect)
                    name = "enemy" if cell["trap"] else "flag" if cell["goal"] else "home" if cell["start"] else None
                    if name:
                        self.screen.blit(self.art.get(name, tile - 2), (rect.x, rect.y))
        self.screen.blit(self.art.get("bunny", tile - 1), (gx + 2 * tile, gy + 2 * tile))
        y = gy + tile * 5 + 25
        for label, value in (("Position", str(observation["position"])),
                             ("Goal bearing", "/".join(v for v in observation["goal_direction"].values() if v != "none") or "here"),
                             ("Moves remaining", str(observation["remaining_moves"]))):
            self._text(label, x, y, MUTED, self.small)
            self._text(value, x, y + 23, TEXT, self.bold)
            y += 65
        if y + 52 < obs_rect.bottom:
            self._text("Only this 5 x 5 window is sent", x, y, MUTED, self.small)
            self._text("to the agent. No full maze or solver.", x, y + 22, MUTED, self.small)
        x, y = decision.x + 16, decision.y + 16
        self._text("Selected action", x, y, TEXT, self.heading)
        action = step.action if step else None
        self._draw_arrow(decision.centerx, y + 77, action)
        self._text(action.capitalize() if action else "Waiting for a step", x, y + 112, TEXT, self.bold)
        self._text(f"Reward: {step.reward:+d}" if step else "Your agent supplies the decisions", x, y + 142, MUTED, self.small)
        if step:
            outcomes = []
            if step.hit_wall:
                outcomes.append("wall")
            if step.entered_trap:
                outcomes.append("trap")
            if step.done:
                outcomes.append("goal")
            if step.truncated:
                outcomes.append("move limit")
            self._text("Outcome: " + (" / ".join(outcomes) or "step"), x, y + 166, MUTED, self.tiny)
        debug = step.debug if step else {}
        known = next((key for key in ("q_values", "action_values", "action_probabilities") if isinstance(debug.get(key), dict)), None)
        self._draw_values(decision, debug, known, y + 198)
        self.debug_rect = debug_rect
        self._text("Agent debug", debug_rect.x + 16, debug_rect.y + 16, TEXT, self.heading)
        top = debug_rect.y + 54
        caption = "Optional / architecture-specific"
        if top + 24 < debug_rect.bottom:
            self._text(caption, debug_rect.x + 16, top, MUTED, self.tiny)
        text = json.dumps(debug, indent=2, ensure_ascii=False) if debug else "No debug data exposed yet.\nAgents may omit this information."
        lines: list[str] = []
        for line in text.splitlines():
            while self.small.size(line)[0] > debug_rect.width - 32:
                length = max(1, len(line) - 1)
                while self.small.size(line[:length])[0] > debug_rect.width - 32:
                    length -= 1
                lines.append(line[:length])
                line = line[length:]
            lines.append(line)
        available = max(0, (debug_rect.bottom - top - 39) // 21)
        self.debug_scroll = min(self.debug_scroll, max(0, len(lines) - available))
        for i, line in enumerate(lines[self.debug_scroll:self.debug_scroll + available]):
            self._text(line, debug_rect.x + 16, top + 28 + i * 21, TEXT, self.small)
        if len(lines) > available:
            self._text("Scroll for more", debug_rect.right - 100, debug_rect.bottom - 20, MUTED, self.tiny)

    def _draw_arrow(self, x: int, y: int, action: str | None) -> None:
        if action is None:
            self.pg.draw.circle(self.screen, DIVIDER, (x, y), 21, 3)
            return
        dr, dc = DELTAS[action]
        end = (x + dc * 24, y + dr * 24)
        start = (x - dc * 20, y - dr * 20)
        self.pg.draw.line(self.screen, TEAL, start, end, 6)
        base = (x + dc * 9, y + dr * 9)
        self.pg.draw.polygon(self.screen, TEAL, [end, (base[0] + dr * 14, base[1] - dc * 14), (base[0] - dr * 14, base[1] + dc * 14)])

    def _draw_values(self, rect: Any, debug: dict[str, Any], key: str | None, y: int) -> None:
        x = rect.x + 16
        if key is None:
            self._text("No action values exposed", x, y, MUTED, self.small)
            return
        labels = {"q_values": "Q-values", "action_values": "Predicted action values", "action_probabilities": "Action probabilities"}
        self._text(labels[key], x, y, TEXT, self.bold)
        values = {a: float(debug[key].get(a, 0)) for a in DELTAS
                  if isinstance(debug[key].get(a, 0), (int, float)) and math.isfinite(float(debug[key].get(a, 0)))}
        low, high = min([0.0, *values.values()]), max([1e-6, *values.values()])
        if key == "action_probabilities":
            low, high = 0.0, 1.0
        for i, action in enumerate(DELTAS):
            yy = y + 32 + i * 31
            self._text(action.capitalize(), x, yy, MUTED, self.tiny)
            bar = self.pg.Rect(x + 48, yy + 3, max(20, rect.width - 120), 10)
            self.pg.draw.rect(self.screen, SOFT, bar, border_radius=3)
            value = values.get(action, 0.0)
            zero = bar.x + (0 - low) / (high - low) * bar.width
            point = bar.x + (value - low) / (high - low) * bar.width
            fill = self.pg.Rect(round(min(zero, point)), bar.y, max(1, round(abs(point - zero))), bar.height)
            self.pg.draw.rect(self.screen, TEAL, fill, border_radius=3)
            text = f"{value:.0%}" if key == "action_probabilities" else f"{value:.2g}"
            self._text(text, bar.right + 6, yy, TEXT, self.tiny)

    def _draw_dashboard(self, body: Any) -> None:
        complete = [record for record in self.records if record.complete]
        stats = aggregate(complete)
        self._panel(self.pg.Rect(body.x, body.y, body.width, 76))
        def number(key: str, precision: int = 0) -> str:
            value = stats[key]
            return "--" if value is None else f"{value:,.{precision}f}"
        values = (("EPISODES", number("episodes")), ("SUCCESS", f"{stats['success_rate']:.0%}"),
                  ("MEAN RETURN", number("mean_reward")), ("MEAN MOVES", number("mean_moves", 1)),
                  ("MEAN FINISH", number("mean_completion_seconds", 2) + "s" if stats['mean_completion_seconds'] is not None else "--"),
                  ("TRAP ENTRIES", number("traps")), ("WALL HITS", number("walls")))
        span = body.width / len(values)
        for i, (label, value) in enumerate(values):
            x = round(body.x + 15 + i * span)
            self._text(label, x, body.y + 12, MUTED, self.tiny)
            self._text(value, x, body.y + 34, TEXT, self.heading)
        y = body.y + 90
        name = self.run_names.get(self.selected_run, "Current run")
        selected = self.run_info.get(self.selected_run, {})
        config = selected.get("config", {})
        caption = f"{name} / L{config.get('level', 1):02d} / seed {config.get('seed', 42)}"
        self._text(self._short(caption, body.width - 296, self.bold), body.x + 3, y + 5, TEXT, self.bold)
        compare_name = self.run_names.get(self.compare_run, "none")
        self._button("compare", "Compare: " + self._short(compare_name, 155, self.bold), self.pg.Rect(body.right - 278, y - 3, 278, 34))
        plot_top = y + 45
        plots_height = max(248, min(408, round(body.height * 0.47)))
        plot_w, plot_h = (body.width - 12) // 2, (plots_height - 12) // 2
        curves = [(name, complete, TEAL)]
        if self.compare_run:
            curves.append((compare_name, self.comparison, ACCENT))
        for i, (title, key) in enumerate((("Reward / episode", "reward"), ("Success / last 20 episodes", "success"),
                                           ("Moves / episode", "moves"), ("Active seconds / episode", "seconds"))):
            rect = self.pg.Rect(body.x + (i % 2) * (plot_w + 12), plot_top + (i // 2) * (plot_h + 12), plot_w, plot_h)
            self._chart(rect, title, key, curves)
        table_top = plot_top + plots_height + 14
        self.table_rect = self.pg.Rect(body.x, table_top, body.width, max(0, body.bottom - table_top))
        self._panel(self.table_rect)
        self._text("Episodes / click a row to replay", body.x + 14, table_top + 10, TEXT, self.bold)
        columns = (("Ep", 0.0), ("Level", .07), ("Return", .15), ("Moves", .28), ("Traps", .39),
                   ("Walls", .49), ("Time", .60), ("Outcome", .74))
        inner = body.width - 28
        for label, ratio in columns:
            self._text(label, body.x + 14 + round(inner * ratio), table_top + 40, MUTED, self.tiny)
        visible = max(0, (self.table_rect.height - 75) // 29)
        records = list(reversed(self.records))
        self.table_scroll = min(self.table_scroll, max(0, len(records) - visible))
        for i, record in enumerate(records[self.table_scroll:self.table_scroll + visible]):
            yy = table_top + 62 + i * 29
            rect = self.pg.Rect(body.x + 8, yy, body.width - 16, 28)
            self.buttons[f"episode:{record.number}"] = rect
            if rect.collidepoint(self.pg.mouse.get_pos()):
                self.pg.draw.rect(self.screen, SOFT, rect, border_radius=4)
            outcome = "Goal" if record.success else "Move limit" if record.complete else "Partial"
            vals = (str(record.number), str(record.level), f"{record.reward:+d}", str(record.moves), str(record.traps), str(record.walls), f"{record.seconds:.2f}s", outcome)
            for (label, ratio), value in zip(columns, vals):
                self._text(value, body.x + 14 + round(inner * ratio), yy + 4, TEAL if outcome == "Goal" and label == "Outcome" else TEXT, self.tiny)
        if not records:
            self._text("Completed or paused episodes will appear here.", body.x + 14, table_top + 69, MUTED, self.small)

    def _chart(self, rect: Any, title: str, key: str, curves: list[Any]) -> None:
        self._panel(rect)
        self._text(title, rect.x + 13, rect.y + 10, TEXT, self.bold)
        plot = self.pg.Rect(rect.x + 48, rect.y + 42, rect.width - 64, rect.height - 68)
        data: list[Any] = []
        for name, records, color in curves:
            values = []
            for i, record in enumerate(records):
                value = (sum(e.success for e in records[max(0, i - 19):i + 1]) / min(20, i + 1) * 100) if key == "success" else float(getattr(record, key))
                values.append((record.number, value))
            data.append((name, values, color))
        all_values = [value for _, values, _ in data for _, value in values]
        if not all_values:
            self._text("No recorded episodes yet", rect.x + 16, rect.centery, MUTED, self.small)
            return
        low, high = (0.0, 100.0) if key == "success" else (min(0.0, min(all_values)), max(all_values))
        if high <= low:
            high = low + 1
        if key != "success":
            padding = (high - low) * 0.08
            low, high = low - padding, high + padding
            if key in ("moves", "seconds"):
                low = max(0.0, low)
        max_episode = max((n for _, values, _ in data for n, _ in values), default=1)
        for i in range(3):
            yy = plot.y + i * plot.height // 2
            self.pg.draw.line(self.screen, DIVIDER, (plot.x, yy), (plot.right, yy))
            value = high - (high - low) * i / 2
            label = f"{value:.0f}" if abs(value) >= 10 or key == "success" else f"{value:.2g}"
            self._text(label, rect.x + 5, yy - 7, MUTED, self.tiny)
        for name, values, color in data:
            stride = max(1, len(values) // max(1, plot.width))
            sampled = values[::stride]
            if values and sampled[-1] != values[-1]:
                sampled.append(values[-1])
            points = [(round(plot.x + (n - 1) / max(1, max_episode - 1) * plot.width), round(plot.bottom - (value - low) / (high - low) * plot.height)) for n, value in sampled]
            if len(points) >= 2:
                self.pg.draw.lines(self.screen, color, False, points, 2)
            if points:
                self.pg.draw.circle(self.screen, color, points[-1], 3)
        self._text("1", plot.x, plot.bottom + 5, MUTED, self.tiny)
        self._text(f"episode {max_episode}", plot.right - 78, plot.bottom + 5, MUTED, self.tiny)
        if len(data) > 1:
            for i, (name, _, color) in enumerate(data):
                self.pg.draw.line(self.screen, color, (rect.right - 155, rect.y + 13 + i * 13), (rect.right - 142, rect.y + 13 + i * 13), 2)
                self._text(self._short(name, 125, self.tiny), rect.right - 136, rect.y + 6 + i * 13, color, self.tiny)
        if plot.collidepoint(self.pg.mouse.get_pos()):
            number = round((self.pg.mouse.get_pos()[0] - plot.x) / plot.width * max(1, max_episode - 1)) + 1
            text = f"Ep {number}: " + " / ".join(f"{values[number - 1][1]:.2f}" if 0 < number <= len(values) else "--" for _, values, _ in data)
            image = self.tiny.render(text, True, TEXT)
            box = image.get_rect().inflate(12, 8)
            box.midtop = (plot.centerx, plot.y + 3)
            self.pg.draw.rect(self.screen, SOFT, box, border_radius=5)
            self.screen.blit(image, image.get_rect(center=box.center))

    def _click(self, point: tuple[int, int]) -> None:
        for key, rect in list(self.buttons.items()):
            if key in self._disabled_buttons or not rect.collidepoint(point):
                continue
            self.run_seed_active = False
            if key == "toggle":
                self._toggle_running()
            elif key == "new":
                self._new_run()
            elif key == "level-":
                self.next_level = max(1, self.next_level - 1)
            elif key == "level+":
                self.next_level = min(35, self.next_level + 1)
            elif key == "episodes-":
                self.next_episodes = max(1, self.next_episodes - 25)
            elif key == "episodes+":
                self.next_episodes = min(10000, self.next_episodes + 25)
            elif key == "seed":
                self.run_seed_text = str(secrets.randbits(32))
            elif key.startswith("speed:"):
                self.speed = int(key.split(":")[1])
                self.credit = 0.0
            elif key.startswith("tab:"):
                self.view = key.split(":")[1]
            elif key.startswith("run:"):
                self.selected_run = key.split(":", 1)[1]
                if self.compare_run == self.selected_run:
                    self.compare_run = None
                self.table_scroll = 0
                self.view = "Dashboard"
                self._refresh(True)
            elif key == "compare":
                options = [None] + [run["id"] for run in self.runs if run["id"] != self.selected_run]
                current = options.index(self.compare_run) if self.compare_run in options else 0
                self.compare_run = options[(current + 1) % len(options)]
                self._refresh(True)
            elif key.startswith("episode:"):
                self._load_replay(int(key.split(":")[1]))
            elif key == "live":
                self.replay = None
                self.replay_playing = False
                self.credit = 0.0
                self._sync_source()
                self._previous_position = self.env.position
                self._moved_at = 0.0
            elif key in ("back", "forward") and self.replay:
                self.replay_playing = False
                self._seek(self.replay.index + (1 if key == "forward" else -1))
            elif key == "fog":
                self.full_map = not self.full_map
            elif key == "fit":
                self._toggle_map()
            elif key in ("zoom_in", "zoom_out"):
                self._zoom(4 if key == "zoom_in" else -4)
            return
        self.run_seed_active = self.seed_rect.collidepoint(point)

    def _event(self, event: Any) -> None:
        pg = self.pg
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.VIDEORESIZE:
            self.screen = pg.display.set_mode((max(1100, event.w), max(760, event.h)), pg.RESIZABLE)
        elif event.type == pg.MOUSEBUTTONUP:
            self.replay_drag = False
        elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.replay and self.replay_slider.collidepoint(event.pos):
                self.replay_drag = True
                self.replay_playing = False
                self._seek(round((event.pos[0] - self.replay_slider.x) / self.replay_slider.width * len(self.replay.record.steps)))
            else:
                self._click(event.pos)
        elif event.type == pg.MOUSEMOTION and self.replay_drag and self.replay:
            self._seek(round((event.pos[0] - self.replay_slider.x) / self.replay_slider.width * len(self.replay.record.steps)))
        elif event.type == pg.MOUSEWHEEL:
            point = pg.mouse.get_pos()
            if self.history_rect.collidepoint(point):
                self.run_scroll = max(0, self.run_scroll - event.y)
            elif self.view == "Dashboard" and self.table_rect.collidepoint(point):
                self.table_scroll = max(0, self.table_scroll - event.y * 3)
            elif self.view == "Learning" and self.debug_rect.collidepoint(point):
                self.debug_scroll = max(0, self.debug_scroll - event.y * 3)
            elif self.view == "Game" and self.viewport.collidepoint(point):
                self._zoom(4 * event.y)
        elif event.type == pg.KEYUP:
            self._held_keys.discard(event.key)
        elif event.type == pg.WINDOWFOCUSLOST:
            self._held_keys.clear()
        elif event.type == pg.KEYDOWN:
            if event.key in self._held_keys or getattr(event, "repeat", False):
                return
            self._held_keys.add(event.key)
            if self.run_seed_active:
                if event.key in (pg.K_ESCAPE, pg.K_RETURN):
                    self.run_seed_active = False
                elif event.key == pg.K_BACKSPACE:
                    self.run_seed_text = self.run_seed_text[:-1]
                else:
                    char = getattr(event, "unicode", "")
                    if len(self.run_seed_text) < 16 and (char.isdigit() or (char == "-" and not self.run_seed_text)):
                        self.run_seed_text += char
            elif event.key == pg.K_SPACE:
                self._toggle_running()
            elif event.key in (pg.K_1, pg.K_2, pg.K_3):
                self.view = {pg.K_1: "Game", pg.K_2: "Learning", pg.K_3: "Dashboard"}[event.key]
            elif event.key == pg.K_m:
                self._toggle_map()
            elif event.key in (pg.K_PLUS, pg.K_EQUALS, pg.K_KP_PLUS):
                self._zoom(4)
            elif event.key in (pg.K_MINUS, pg.K_KP_MINUS):
                self._zoom(-4)

    def close(self) -> None:
        self.session.pause()
        self.store.close()
        self.pg.quit()

    def run(self) -> None:
        self.pg.key.set_repeat()
        try:
            while self.running:
                delta = self.clock.tick(60) / 1000
                for event in self.pg.event.get():
                    self._event(event)
                self.update(delta)
                self.draw()
                self.pg.display.flip()
        finally:
            self.close()


def launch_dashboard(*, level: int = 1, seed: int = 42, episodes: int = 100,
                     runs_dir: str | Path = ".maze-runs", agent_spec: str = "Redbull.agent:Redbull") -> None:
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("Install the desktop dependency with: python -m pip install -e '.[pygame]'") from exc
    TrainingStudio(pygame, level=level, seed=seed, episodes=episodes, runs_dir=runs_dir,
                   agent_spec=agent_spec).run()
