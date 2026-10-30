# Bunny Maze

A standalone Pygame maze game with a UI-independent simulation engine. The curriculum has 35 levels with generation grids of 9, 13, 17, 21, 27, 33, and 41 cells per side. Layouts are generated from a seed with randomized depth-first search or randomized Prim, then braided with extra passages. Candidates are accepted only after meeting that tier's shortest-path, intersection, and dead-end thresholds. The engine compiles each layout into a `(2n + 1)` block grid, including the floor squares between generated cells. Every visible floor square is a real movement position, so one action moves exactly one wall-block width.

## Install and play

Python 3.10 or newer is required. Install the desktop dependency and launch the game:

```sh
python -m pip install -e '.[pygame]'
python -m maze_game
```

Guide Pip, the cream bunny with a teal scarf, from the blue home pad to the checkered finish flag. Slate-blue blocks are walls; red horned troublemakers cost 50 points per visit. A permanent illustrated legend identifies each marker.

Use arrow keys or WASD to move one visible cell per key press. Release the key before pressing it again for the next step; holding a direction does not repeat movement. Click a level to load it, enter a seed and choose **Apply** for a reproducible layout, or choose **Shuffle maze** for a new seed. **Restart** / R replays the same layout; **Pause** / Space stops the clock and movement. Reaching the flag enables **Next level**.

Small mazes fit the board automatically. Large mazes use a readable camera that follows Pip, with an overview in the sidebar when space allows. Choose **Fit map** / M to see the whole maze, or use **+ / -** and the mouse wheel to zoom. Movement animation is always enabled. The desktop window supports sizes from 900 x 680 upward.

## Scoring

- Finish a level: +1,000.
- Each move after the trusted shortest-path length: −3.
- Enter a trap: −50.
- Attempt to cross a wall: −5.
- Run out of moves: no completion points.

Every attempted move, including a blocked move, consumes one move. Optimal routes and scoring count individual floor-block steps. The move limit is three times the shortest route, or the number of walkable floor blocks when that is larger. Traps trigger each time they are entered. The result panel reports route quality as shortest-path length divided by moves used.

## Headless environment

The simulation package does not import or require Pygame:

```python
from maze_game import MazeEnv

env = MazeEnv(level=12, seed=42)
observation = env.reset(level=12, seed=42)
observation, reward, done, truncated, info = env.step("right")
```

Actions are `"up"`, `"right"`, `"down"`, and `"left"` (or integer actions 0–3 in that order). `reset()` returns one observation. `step()` returns `(observation, reward, done, truncated, info)`. Reward is the score change for the step. Positions and `grid_size` use the actual rendered block grid; `curriculum_grid_size` identifies the generation preset. The start is `[1, 1]`, inside the outer wall. The observation includes approximate goal bearing, remaining moves, and a radius-two local view with explicit wall flags. It does not contain the full maze or trusted shortest route.

For a line-oriented controller without a window, run:

```sh
python -m maze_game --headless --level 12 --seed 42
```

The process prints one JSON observation, then accepts one action per input line (`up`, `right`, `down`, `left`, or `{"action":"right"}`) and prints a JSON step result. Send `quit` to end the session.

For an untrusted competition agent, run the environment as a separate process and pass only these JSON observations and actions across the pipe. Do not give agent code the `MazeEnv` Python object: its implementation contains trusted game state and the scoring solver, while the JSON protocol omits the full map, seed, and optimal route.

## Phase 2: Training Studio

Launch the three-view desktop studio with the same Pygame extra:

```sh
python -m maze_game --dashboard --level 1 --seed 42 --episodes 100
```

The manual game also has a **Training Studio** button and a T shortcut to open the studio with the current level and seed.

Choose **Start demo** to run a uniformly random agent. It chooses among all four actions, including walls; it does not learn and does not need a trained model. **Pause/Resume** or Space freezes and resumes the simulation. The speed controls select actions per second. Level, episode budget, and run seed changes take effect when **New run** is selected.

- **Game** shows the bunny moving one block per agent action, discovered cells, the teal route, recent actions, traps, and red crosses at wall collisions. **Full map** is an administrator inspection view; it does not change what the agent receives.
- **Learning** shows the exact 5 x 5 observation used for the selected action, its reward, and optional agent debug information. The demo exposes uniform action probabilities. Q-values, predicted action values, and other JSON debug data are optional, with no requirement that agents have identical internals. Debug text scrolls with the mouse wheel.
- **Dashboard** plots episode returns, rolling success over the last 20 completed episodes, moves, and active episode duration. It also shows successful completion time, trap entries, wall collisions, and episode counts. Unfinished partial episodes are excluded from aggregate metrics. A missing successful completion time is shown as `--`, not invented.

Use 1, 2, and 3 to switch views. Select a saved run from the left rail to inspect it. **Compare** cycles through other saved runs and overlays their curves; selecting `none` clears comparison. Click an episode row to pause the current run and open its replay. The replay supports play/pause, single stepping, a seek slider, and **Return live**. Returning live leaves the current run paused until resumed.

Runs are saved in `.maze-runs/training.sqlite3`, including configurations, episode seeds, actions, rewards, collision flags, timing, and optional debug snapshots. Completed episodes and paused partial episodes can be replayed after restarting the studio. Replay uses recorded actions and verifies every outcome against the environment; it never invokes the agent again. The desktop studio supports window sizes of 1100 x 760 and larger. Movement animation remains always enabled.

Times measure active wall-clock time, excluding pauses. They vary with simulation speed; moves and rewards are independent of playback/rendering speed. Replays display recorded times. The initial agent is random, so low success rates and non-improving reward curves are expected.

The runner also works without importing Pygame:

```sh
python -m maze_game --train-headless --level 1 --seed 42 --episodes 100
```

Use `--runs-dir PATH` in either mode to select a different recording directory. The headless command saves the same run format and prints a JSON summary that can be reviewed in the studio by using the same directory.

The programmatic agent boundary is `act(observation) -> AgentDecision(action, debug={})`. `TrainingSession` accepts an agent with a `name` and this method. Optional `on_episode_start(observation)` and `on_transition(Transition)` hooks support future learning integrations. Agents receive copied public observations and public transition feedback; the game engine has no dependency on an agent or the studio. The recorder requires debug values to be JSON-serializable and finite. No model, Q-learning implementation, or neural training is introduced in this phase.

## Tests

Run the acceptance suite with the standard library:

```sh
python -m unittest discover -s tests -v
```

The generator, scoring, observation, and headless tests need no third-party packages. Pygame control and rendering checks run when the optional `pygame-ce` extra is installed; they use SDL's dummy video driver and do not open a desktop window.

## Layout

- `maze_game/_generator.py` — trusted generation, tier gates, graph metrics, and solver.
- `maze_game/_tiles.py` — compiles generated corridors into individual playable floor blocks.
- `maze_game/engine.py` — public observations, shared movement, timing, scoring, and truncation.
- `maze_game/ui.py` — Pygame rendering and manual controls.
- `maze_game/_art.py` — authored bunny, troublemaker, home, flag, and block artwork.
- `maze_game/__main__.py` — desktop launcher and headless JSON-lines adapter.
- `maze_game/training.py` — demo agent, recording runner, SQLite storage, metrics, and verified replay.
- `maze_game/training_ui.py` — Game, Learning, and Dashboard views with run and replay controls.
- `tests/test_maze_game.py` — procedural-generation, engine, headless, and optional Pygame acceptance checks.
- `tests/test_training.py` — run budgets, pause/resume, privacy, debug data, persistence, replay, comparisons, and studio integration.
- `DESIGN.md` — visual tokens and camera behavior for the toy-board interface.
- `maze_game/assets/fonts/` — bundled Fredoka and Nunito fonts and their Open Font Licenses.
