# Codex Memory PPO — dormant MazeMind contestant

**`READY = False` in `activation.py`. Nothing has been trained or evaluated.**
This directory contains a complete learning contestant and its operating tools,
independent of Redbull. The shared game, scoring, maze generator, observation API,
and Redbull files are not modified. All runtime outputs belong under this
directory's ignored `artifacts/` folder unless you explicitly choose another path.

The policy is a two-scale CNN and a GRU, trained by our recurrent PPO implementation,
with an observation-built episodic map and separate task/exploration critics.
There is no pretrained model, solver policy, hardcoded route, or external AI service.
See [RESEARCH.md](RESEARCH.md) for source links, alternatives, and limitations and
[STATIC_REVIEW.md](STATIC_REVIEW.md) for what was checked without executing the agent.

## 1. Install independently

On Apple Silicon, follow [the macOS setup guide](../../MACOS.md) for native ARM64 Python, Finder launchers, and CPU/MPS options. Setup does not activate this agent.

Use Python 3.10 or newer, preferably a supported Python/PyTorch wheel combination
(3.11 or 3.12 is a conservative choice). From the project root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# The game and optional existing Pygame dashboard:
python -m pip install -e '.[pygame]'
# Independent contestant package and its own CLI:
python -m pip install -e ./agents/codex
```

For a headless-only machine install the game with `python -m pip install -e .`
instead of the Pygame extra. On Linux/Windows, if you specifically need CPU-only
PyTorch wheels, use the official CPU index before installing the contestant. On
macOS use the normal PyPI installation above; no CUDA-specific index is needed:

```sh
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA acceleration use the wheel command appropriate to your driver from
[PyTorch's installer](https://pytorch.org/get-started/locally/); do not assume an
arbitrary CUDA toolkit matches. NumPy and PyTorch are the only learning dependencies.
No SB3, RLlib, Gym wrapper, training service, API key, or internet connection is
needed for runtime. `requirements.txt` supplies dependency ranges for manual installs.
The nested `pyproject.toml` installs `agents.codex` without changing the shared
project's package selection. Commands below work from the repository root; after
installation `codex-maze` is equivalent to `python -m agents.codex`.

## 2. Activation, with no automatic launch

The single switch is:

```python
# agents/codex/activation.py
READY = False
```

Leave it false until competition training is authorized. To activate later, edit
only that assignment to `READY = True` and start a new process. Changing it starts
nothing. There are no startup jobs, import hooks, downloads, background trainers,
environment-variable overrides, or flags that enable a dormant agent.

```sh
python -m agents.codex status
python -m agents.codex --help
```

Those are informational and do not load a model or initialize an environment.
Every operational command checks readiness before loading ML modules, opening
checkpoints/databases, making a window, or creating an environment. The model,
memory, training/evaluation functions, checkpoint I/O, and inference adapter also
have direct-call guards. Disabled commands print a dormant explanation to stderr
and exit with code 2; user/runtime errors use code 1. There is no silent random
fallback if the switch is false or a checkpoint is missing.

Do **not** run the agent or the project acceptance suite as part of dormant
verification: the shared suite deliberately solves mazes. The delivered review
used source analysis, Ruff, parsing, and compilation of source strings only.

Source-only checks, which do not import or run the agent, can be repeated with:

```sh
python agents/codex/static_check.py
ruff check agents/codex
ruff format --check agents/codex
```

The delivery audit intentionally requires `READY = False`; after authorized
activation it reports that the package is no longer a dormant delivery. That is
not an instruction to bypass a training guard.

## 3. Start training (after activation)

```sh
python -m agents.codex train \
  --config agents/codex/configs/cpu.json \
  --run agents/codex/artifacts/cpu01
```

Optional NVIDIA CUDA preset:

```sh
python -m agents.codex train \
  --config agents/codex/configs/gpu.json \
  --run agents/codex/artifacts/gpu01
```

These are actual training commands, not tests; do not invoke them while training
is prohibited. Presets merge with the defaults in `config.py`. Unknown JSON keys
are rejected. `--steps` overrides the **total lifetime** environment-action budget;
`--device cpu|cuda|mps|auto` overrides runtime placement. Defaults start at level 1
and permit progression through 35. Each episode uses a fresh procedural seed.
The runner is synchronous with batched policy inference, not a multiprocessing
system; more actors do not imply more Python environment threads.

The budget stops at the last complete recurrent sequence batch below the requested
total, possibly leaving fewer than `num_envs * sequence_length` actions unused.
Evaluation/validation actions are additional and not included in the training
budget. Reaching a budget does **not** imply completion of the curriculum. If a
stage is not mastered, the curriculum deliberately stays there.

## 4. How the learner works

Each episode begins with zero GRU state and empty observation memory. A 5x5 window
is copied into the remembered map at the **observed** current position. Unknown
cells stay unknown. There is no inferred goal coordinate from a known board corner.
The actor and both critics see exactly the same permitted features.

The 13 map channels are known, wall, trap, goal, start, logarithmic visit count,
visit recency, observed floor frontier, player, and four observed outgoing-direction
channels. Inputs are a player-centered 15x15 crop and a pooled 21x21 remembered map.
Twenty scalars encode normalized position, goal bearing, budget, board scale,
observed-floor coverage, current visits, previous reward/action, adjacent visits,
current trap, visited coverage, and recent discovery count. Features do not include
maze seeds, optimal path lengths, privileged generation metrics, or unseen cells.

Each CNN uses 16/32/32 filters and a 96-dimensional embedding. Scalars embed to
64 dimensions. Their concatenation feeds a 128-unit GRU with a four-action actor
and two value outputs. The default architecture has approximately **222,000**
parameters by static shape arithmetic, not a runtime model measurement.

The current cell's public `open` directions mask blocked actions. Traps remain
available because they may be required. Every decision still comes from learned
logits. No branch selects a shortest path, fallback route, or scripted exploration.
The default policy samples a seeded categorical distribution; evaluation also
offers `--greedy` for argmax. Episode memory changes during inference, but network
parameters do not.

Training task reward is `score_delta / reward_scale - step_cost`. Exploration
reward is `first_visit_bonus * first_visit + discovery_bonus * newly_seen_floor`.
Wall attempts cannot produce first-visit bonuses. The exact raw score remains the
reported competition score and the reward in dashboard replays. The task action
cost and exploration terms are training objectives only. Two GAE streams train
separate value heads; the actor uses normalized `A_task + beta * A_exploration`.
`beta`, entropy, and learning rate decay with the total configured step budget.

Contiguous 32-step training chunks preserve temporal order. Up to 16 preceding
steps recompute detached hidden state before gradients. Starts reset GRU state;
both goal and exhausted move budget end returns. A rollout cut while the episode
continues bootstraps from the next observation. Clipped actor/value losses,
gradient norm clipping, and a KL threshold control update size.

## 5. Curriculum and generalization protocol

The active frontier initially equals `start_level`. Seventy percent of new episodes
target it by default; the remainder rehearse earlier levels, weighted toward weak
ones. A new level unlocks only after the frontier has at least 50 completed episodes,
at least 80% success in the last 50, and at least 70% held-out frontier validation
success. Validation also checks earlier tier boundaries. No unsupervised time-based
unlock silently pushes a failing policy into larger mazes.

Seed namespaces are integer-disjoint: training, validation, and evaluation use
different high bits, with separate master seed, level, and per-split counter fields.
Counter width is 48 bits; the master seed is an integer in `[0, 2**31)`. Reset/step
accept these large Python integer seeds. The policy never receives a maze's seed.
Validation uses a fixed bank for repeatability and model selection; repeated tuning
can overfit that bank. Final evaluation and five championship cases must remain
unused for selection. Distinct seeds prevent direct seed overlap, but two seeds
can theoretically still produce an identical layout; no hidden-map comparison is
used to remove such collisions.

## 6. Monitoring and the existing dashboard

Training prints one JSON object per completed update and saves:

| File under the run directory | Contents |
| --- | --- |
| `config.json` | Resolved parameters, including defaults |
| `episodes.jsonl` | Every completed training episode: level/tier, score, success, moves, traps, wall hits, time, intrinsic return |
| `updates.jsonl` | Actor/value loss, entropy, KL, clipping, learning rate, intrinsic weight, update time, frontier, completed episodes |
| `validation.jsonl` | Per-case, level/tier, and aggregate frozen-policy validation results |
| `latest.pt` | Latest periodic/coherent final training state |
| `best.pt` | Best validation key: evaluated frontier, macro success, mean raw score |
| `recordings/training.sqlite3` | Sampled trajectories in the existing studio's replay schema |

```sh
python -m agents.codex monitor --run agents/codex/artifacts/cpu01
tail -f agents/codex/artifacts/cpu01/updates.jsonl
```

`monitor` reads metrics only and is also guarded. JSON output has no ANSI styling
or prompts and works in pipes and with `NO_COLOR`. Diagnostics go to stderr.

After there is a real checkpoint, open the **existing** Training Studio:

```sh
python -m agents.codex dashboard \
  --checkpoint agents/codex/artifacts/cpu01/latest.pt \
  --recordings agents/codex/artifacts/cpu01/recordings
```

This opens a **paused, frozen-policy inference session**, not PPO training. Select
recorded level runs in the sidebar to inspect curves and replay episodes. Clicking
Start agent runs the checkpoint without weight updates. Loss metrics remain in
`updates.jsonl`; the shared UI's existing charts display episode rewards, success,
moves, and times. One trajectory in 25 is saved by default for storage efficiency.
Dashboard training aggregates therefore describe that recorded sample, while
`episodes.jsonl` and `updates.jsonl` contain full training counts. Evaluation saves
every episode. Increasing `record_every` reduces replay storage. Existing shared
recordings and Redbull data are not overwritten.

## 7. Stop, checkpoint, and resume

Press Ctrl-C once or send SIGTERM. The trainer completes its current collection /
optimization update, writes `latest.pt`, saves sampled partial trajectories, and
closes the database. This may take time on large mazes or during validation. To
request a stop without a signal, create a file named `STOP` inside the run directory;
remove it before resuming. `kill -9` cannot save state; use the previous atomic
checkpoint. Checkpoint writes use a temporary file, fsync, and atomic replacement.

```sh
python -m agents.codex train \
  --resume agents/codex/artifacts/cpu01/latest.pt \
  --run agents/codex/artifacts/cpu01
```

Extend the total lifetime budget, rather than adding an implicit extra budget:

```sh
python -m agents.codex train \
  --resume agents/codex/artifacts/cpu01/latest.pt \
  --run agents/codex/artifacts/cpu01 --steps 10000000 --device cpu
```

The checkpoint includes weights, Adam moments, config/format versions, actor GRU
states, public action histories, curriculum histories/counters/RNG, batching RNG,
PyTorch CPU/CUDA/MPS RNG, Python RNG, recorder IDs, source digest, and library versions.
Active episodes are reconstructed through **public reset/step calls** using their
recorded actions, then their permitted observations are compared with the saved
observations. No internal environment state is serialized or read. Original
episode timing and sampled trajectory snapshots are retained. Resume may replay
thousands of actions and is itself gated; it must not be used before activation.

Only total budget, device, and thread count may differ from the saved config.
Use checkpoint defaults when resuming; do not pass a different preset casually.
Increasing the total budget changes the annealing schedule and training course.
Same machine/library/device/settings are the target for deterministic resume;
cross-device, version changes, and timing are not promised bit-identical. If a
mid-update error occurs, the last coherent checkpoint is preserved rather than
replaced with a partially optimized state. Fix the cause and resume that checkpoint.

Resuming into the **same** run directory rolls JSONL files and sampled SQLite
episodes back to the selected checkpoint so post-checkpoint crash data is not
duplicated or mixed with the resumed branch. Active episode recordings are replaced
with checkpoint snapshots. Use a **new** `--run` directory to fork a checkpoint
without rolling back an existing run; old logs are not copied into that new branch.

Checkpoint loading uses `torch.load(..., weights_only=True)` and strict state-dict
matching. Only load files you trust; restricted loading is not a general defense
against malicious file sizes or every possible deserialization issue. No pretrained
or downloaded checkpoint is shipped.

## 8. Frozen evaluation and five championship mazes

After activation and training, evaluate held-out seeds:

```sh
python -m agents.codex evaluate \
  --checkpoint agents/codex/artifacts/cpu01/best.pt \
  --levels 1-35 --episodes 20 --seed 42 \
  --output agents/codex/artifacts/eval01
```

Select levels with `1,5,10-15,31-35`. Each case resets episodic memory. No gradient,
optimizer, or exploration reward is used. The report contains completion rate,
raw score, moves, traps, wall hits, successful episode time, and inference latency,
with per-level and per-tier groups. `move_budget_fraction` is moves divided by the
public move limit, **not** shortest-path efficiency. The policy does not request
the optimal path to manufacture an efficiency metric. Latency includes observation
preprocessing in the reported full action timing; internal debug inference timing
measures the neural decision portion. Measurements are meaningful only after you run it.

For championship cases create your own private JSON array of **exactly five**
distinct objects with integer `level` (1..35) and `seed` fields. No secret or
preselected championship seeds are supplied here. Pass that file explicitly:

```sh
python -m agents.codex evaluate \
  --checkpoint agents/codex/artifacts/cpu01/best.pt \
  --manifest /path/to/private-five-mazes.json \
  --output agents/codex/artifacts/championship01
```

The manifest overrides generated evaluation cases; keep it out of training and
model selection. Output directories must be new, preventing accidental report
overwrite. The default seeded stochastic policy is reproducible under the same
runtime; choose and freeze stochastic versus `--greedy` before the championship.
Changing evaluation seeds or policy mode after viewing results is model selection.

## 9. Championship inference interfaces

Load any trained checkpoint explicitly in Python:

```python
from agents.codex.agent import CodexAgent

agent = CodexAgent(seed=42, checkpoint="agents/codex/artifacts/cpu01/best.pt")
agent.on_episode_start(observation)  # Every new maze
decision = agent.act(observation)  # decision.action, decision.debug
```

Call `act()` once per new permitted observation. It returns the existing game's
`AgentDecision`; `on_transition(Transition)` is supported and never learns.
Only episode memory changes. The adapter has no environment/solver reference.
For the shared loader's `seed=`-only constructor, copy the chosen checkpoint to
`agents/codex/artifacts/champion.pt` after activation/training; then the import
path is `agents.codex.agent:CodexAgent`. An absent file fails instead of falling
back to untrained inference.

For strict process isolation use JSON lines:

```sh
python -m agents.codex infer --checkpoint agents/codex/artifacts/cpu01/best.pt
```

Each stdin line is a bare observation or `{"observation": ...}`. Each stdout line
is `{"action": "up|right|down|left", "debug": {...}}`. Engine responses with
`done` or `truncated` return `{"episode_done": true}` without choosing an action.
The next reset observation begins a fresh episode. The process does not create
mazes or train itself. The championship harness supplies observations and applies
the returned actions through the shared movement API. Do not expose private game
objects to any contestant. In-process compatibility is convenience, not a security
sandbox against arbitrary Python submitted by other contestants.

## 10. Configuration reference

All keys below are top-level JSON keys except those under `model`. Partial JSON
files use the remaining defaults. Edit `config.py` only when changing the schema;
record new configs as separate files for experiments.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `seed` | 42 | Experiment, sampling, and procedural master seed |
| `device` | cpu | cpu, cuda, mps, or explicit auto selection (auto chooses CUDA or CPU) |
| `threads` | 2 | PyTorch CPU thread count |
| `deterministic` | true | Request deterministic PyTorch kernels; may reject unsupported kernels |
| `total_steps` | 5,000,000 | Total lifetime training actions, across actors |
| `num_envs` | 8 | Synchronous independent actors |
| `rollout_steps` | 256 | Time steps per actor per full collection |
| `sequence_length` | 32 | Consecutive steps per gradient sequence; divides rollout_steps |
| `burn_in` | 16 | Preceding detached recurrent steps, 0..sequence_length |
| `sequences_per_batch` | 8 | Recurrent sequences in each gradient minibatch |
| `epochs` | 4 | PPO passes per rollout, subject to KL stop |
| `learning_rate` | 0.0003 | Adam starting learning rate |
| `final_lr_fraction` | 0.1 | Fraction of initial LR at full configured budget |
| `gamma` | 0.999 | Task critic discount |
| `intrinsic_gamma` | 0.99 | Exploration critic discount |
| `gae_lambda` | 0.97 | Both GAE trace factors |
| `clip_ratio` | 0.2 | Actor probability ratio clipping |
| `value_clip` | 0.2 | Absolute value prediction change limit in clipped loss |
| `value_coefficient` | 0.5 | Weight of mean dual-value loss |
| `entropy_start`, `entropy_end` | 0.02, 0.003 | Linear entropy coefficient schedule |
| `max_grad_norm` | 0.5 | Global gradient norm cap |
| `target_kl` | 0.03 | Stop additional PPO minibatches above this estimate |
| `reward_scale` | 100 | Divide raw score deltas by this in training only |
| `step_cost` | 0.002 | Per-action task cost in training only |
| `first_visit_bonus` | 0.05 | Episodic first-position-visit intrinsic reward |
| `discovery_bonus` | 0.002 | Intrinsic reward per newly observed floor block |
| `intrinsic_start`, `intrinsic_end` | 1.0, 0.05 | Actor's exploration advantage weight schedule |
| `intrinsic_anneal_fraction` | 0.8 | Fraction of total budget over which intrinsic weight decays |
| `start_level`, `max_level` | 1, 35 | Curriculum bounds |
| `curriculum_window` | 50 | Recent frontier outcomes used for promotion |
| `curriculum_min_episodes` | 50 | Minimum completed frontier episodes before promotion |
| `curriculum_threshold` | 0.8 | Required frontier training success |
| `rehearsal_probability` | 0.3 | Chance of sampling an earlier level after the first stage |
| `validation_every` | 50 | Validate every this many optimization updates |
| `validation_episodes` | 10 | Fixed-bank cases per validation level |
| `promotion_validation_threshold` | 0.7 | Required frontier validation success |
| `checkpoint_every` | 25 | Periodic latest.pt interval in updates |
| `record_every` | 25 | Record one in this many assigned training episodes |
| `model.patch_size` | 15 | Odd local remembered crop size |
| `model.global_size` | 21 | Pooled remembered map size |
| `model.max_grid` | 127 | Maximum accepted public block-grid dimension, >=83 |
| `model.hidden_size` | 128 | GRU and fusion dimensions |

The NVIDIA GPU preset changes actors to 16, batch sequences to 16, threads to 4, and total
budget to 10 million; architecture stays the same. The optional `configs/mac-mps.json` preset keeps the CPU-sized batch, selects Apple
MPS, and disables strict deterministic kernels. It is untested on hardware and
MPS must be selected explicitly; CPU remains the default. Sampling uses a CPU
generator on every backend. Checkpoint structure depends on
the model config. Start a new run when changing learning parameters; resume rejects
incompatible changes.

## 11. Resource planning — estimates only

The default model is small: roughly 222k float32 parameters, around 0.85 MiB of
weight storage by shape arithmetic. The raw uint8 spatial rollout for 8 x 256
steps is about 17 MiB; cached hidden states add about 1 MiB. Optimizer state,
duplicate stacking buffers, activations, Python, PyTorch, and the environment
need substantially more. A consumer computer with 8 GB RAM is a reasonable
starting budget; a few GB of GPU memory should accommodate the small preset.
Neither peak memory nor speed has been measured. Larger batches/configs change
these estimates, and the Python environment can bottleneck a small GPU model.

Plan for hours to days. Pure arithmetic: 5 million steps at a hypothetical measured
50 steps/s is about 28 hours; at 500 steps/s it is about 2.8 hours, before extra
validation time. Those rates are examples for budgeting, not predicted performance.
The code never runs a speed benchmark automatically. SQLite records, JSON logs,
and two checkpoints consume disk; sampled recordings reduce growth. Monitor free
space during long jobs.

## 12. Troubleshooting and maintenance

| Symptom | Action |
| --- | --- |
| Dormant error | Wait for authorization, then edit the one READY switch and start a new process. Do not bypass guards. |
| Missing checkpoint | Train after activation, then choose latest.pt/best.pt; no weights ship with the agent. |
| Missing `agents.codex` | Run from repository root or install the nested editable contestant package. |
| An unrelated installed `agents` package shadows the contestant namespace | Use the fresh virtual environment above, without the unrelated package. |
| Missing `maze_game` | Install the shared project into the same virtual environment. |
| No PyTorch wheel / import error | Use a supported native Python version and matching CPU/CUDA/MPS-capable wheel; retain the versions reported in checkpoints. |
| CUDA unavailable / deterministic kernel error | Use `--device cpu`, or correct the wheel/driver. A new config may disable deterministic kernels, at the cost of repeatability. |
| CUDA out of memory | For a new run reduce sequences_per_batch, actors, or model dimensions. For resume keep saved parameters and switch device to CPU. |
| Slow training / pause takes time | Synchronous Python generation and validation are real work. Wait for the current update. The JSON log reports update_seconds. |
| Never unlocks the next level | Inspect training/validation success and returns. More budget or a new experiment may be needed. The code makes no convergence guarantee. |
| Reward high but poor goal success | Inspect raw score separately from intrinsic_return. Exploration bonuses can dominate; use a lower intrinsic weight in a new config. |
| Repeated loops at inference | Compare seeded stochastic and greedy on validation only; inspect remembered visits, policy entropy, and checkpoint maturity. No hidden planner will rescue the policy. |
| Nonfinite losses | Do not continue silently. Inspect the last metrics and config; resume a coherent checkpoint after fixing the cause. |
| Resume trajectory mismatch | Shared game code/seed behavior may have changed. Restore the checkpoint's game revision; do not skip the check. |
| No best.pt yet | No validation interval completed yet. latest.pt is usable, but unvalidated performance is unknown. |
| Empty dashboard / fewer episodes than logs | Training replays are sampled and grouped by level; choose the recorded run. Full counts live in JSONL. |
| STOP prevents resume | Remove the run's STOP file when you explicitly want to continue. |

Keep the game revision fixed for comparable runs. Preserve resolved config, logs,
checkpoint, dependency versions, and the source digest. Source is divided into
activation, config, memory, network, PPO math, curriculum, collector/trainer,
checkpoint runtime, inference, evaluation, and recording modules so each can be
reviewed independently. No core function is left as a TODO. Further experiments
should be your explicitly authorized work after activation, not automatic startup.
