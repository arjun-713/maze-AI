# Bunny Maze

A desktop Pygame maze game for manual play, with 35 procedural difficulty presets in seven tiers and a separate headless simulation engine. Players move in four directions from start to goal. Traps cost 50 points per entry, walls cost 5 points per attempt, and moves beyond the shortest route cost 3 points. Completion awards 1,000 points; the move limit ends an unfinished episode.

The redesign follows the user's request for recognizable block walls, a cute player character, a checkered finish flag, red villain penalty characters, and a clear distinction between start, finish, obstacles, and player. Generation, scoring, seeds, level selection, restart, pause, and headless behavior are preserved. There is no machine learning in Phase 1.

Phase 2 adds a native Training Studio with live exploration, agent observations and optional debug telemetry, recorded metrics, pause/resume, adjustable action speed, deterministic episode replay, and comparisons between persisted runs. The default Redbull agent is an unfinished user-owned scaffold; the studio waits until it is implemented. The independent Codex contestant remains dormant behind READY = False. The recording runner works without Pygame and uses the existing limited-visibility environment API.
