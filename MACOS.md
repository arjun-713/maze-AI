# Running MazeMind on your Apple Silicon Mac

The intended Mac setup is native ARM64 Python 3.12 on a current Apple Silicon
Mac, preferably macOS 14 or newer. The game and Training Studio use Pygame's
packaged SDL libraries; the Codex agent uses PyTorch. CPU is the default for the
agent, with optional Apple Metal/MPS acceleration after authorized activation.
No CUDA, Linux display server, system font installation, or external AI service
is required on your Mac.

## First setup

Install [Homebrew](https://brew.sh/) if you do not already have it, then install
[Python 3.12](https://formulae.brew.sh/formula/python@3.12):

```sh
brew install python@3.12
```

Use a normal native Terminal window, with its **Open using Rosetta** option off.
In the complete repository folder:

```sh
python3.12 run.py setup --codex
.venv/bin/python run.py doctor
.venv/bin/python run.py studio
```

If `python3.12` is not on PATH, use `/opt/homebrew/bin/python3.12` for the first
command. The setup helper creates this checkout's `.venv`, installs the game,
Pygame, NumPy, PyTorch, and the independent Codex package, then runs `pip check`.
It does **not** import an agent, initialize a model, create a maze, open a game
window, alter a READY switch, or begin training. Installing dependencies does not
activate the contestant. Omit `--codex` if you only want the game/dashboard now.

The helper uses subprocess argument lists and checkout-relative paths, so folders
with spaces work. Installation needs an internet connection; game play and
checkpoint inference do not need one afterward.

## Open the app next time

From the repository folder:

```sh
.venv/bin/python run.py studio
.venv/bin/python run.py game
```

Or double-click these Finder launchers:

- **Setup macOS.command** — creates the environment and installs all dependencies.
- **Training Studio.command** — opens the dashboard with the Redbull scaffold.
- **Play Maze.command** — opens manual play.

Each launcher resolves its own folder, so Finder's working directory does not
matter. Launchers use the local venv rather than whichever `python` Terminal
happens to select. If a ZIP/copy removed executable permissions, restore them:

```sh
chmod +x "Setup macOS.command" "Training Studio.command" "Play Maze.command"
```

If macOS asks whether to open a downloaded script, use Finder's **Open** action
for the file you trust, or use the Terminal commands above. No system-wide
security setting needs changing. Setup errors stay visible in the Terminal window.

You can select the dashboard's starting level and recording folder:

```sh
.venv/bin/python run.py studio --level 12 --seed 42 --episodes 100
```

The current Redbull scaffold has no implementation and stays Not ready until you
write it. The game remains manually playable. Opening the studio does not select,
activate, or train the Codex contestant.

## Transfer from Linux to your Mac

1. Copy or clone the complete source, including `Redbull/`, `agents/codex/`,
   `maze_game/assets/fonts/`, and the new launchers. Uncommitted files must be
   transferred too; cloning a remote cannot include work you have not committed/pushed.
2. **Recreate `.venv` on the Mac.** Do not copy Linux's `.venv`, wheel caches, or
   `/tmp` dependency folders. Virtual environments and native wheels are OS- and
   architecture-specific. The helper rejects a clearly foreign/incomplete venv
   without deleting it; move an old venv aside before rerunning setup.
3. Copy `.maze-runs/` separately if you want existing dashboard history; it is
   intentionally ignored by Git. Close the old app before copying SQLite files.
4. In the future, copy `Redbull/` checkpoint/data folders and
   `agents/codex/artifacts/` separately when they contain your work. Those runtime
   artifacts are also ignored. No Codex checkpoints are currently supplied.
5. Run setup and doctor, then open the studio or game explicitly.

The helper does not need the Linux-only `wmctrl` command or
`LIBGL_ALWAYS_SOFTWARE` setting used during earlier local development. Do not set
`SDL_VIDEODRIVER=dummy` for normal Mac play; let SDL select Cocoa. Pygame rendering
runs on the main thread. Fonts are bundled with the project. Initial windows
fit the reported Mac desktop while retaining the existing minimum sizes (game:
900x680, studio: 1100x760); use a display scaling setting with enough usable space.

## Codex agent on Apple Silicon — remains dormant

`agents/codex/activation.py` still contains `READY = False`. Installation, doctor,
and the Finder launchers leave that assignment alone. Follow the complete
[Codex handoff](agents/codex/README.md) when you later authorize competition training.

After authorized activation, the existing CPU preset runs on the Mac:

```sh
.venv/bin/python -m agents.codex train \
  --config agents/codex/configs/cpu.json \
  --run agents/codex/artifacts/mac-cpu01
```

The optional Mac GPU preset selects MPS explicitly:

```sh
.venv/bin/python -m agents.codex train \
  --config agents/codex/configs/mac-mps.json \
  --run agents/codex/artifacts/mac-mps01
```

These are **future training commands**, not setup verification commands. MPS model
execution and performance have not been tested here. CPU remains the reference
path; `auto` still chooses CUDA when available and CPU otherwise, so a Mac never
switches to MPS implicitly. The MPS preset disables strict deterministic kernels
because backend coverage differs; results are not promised bit-identical across
CPU/MPS/CUDA. Action sampling uses a CPU generator on every backend, avoiding an
unsupported MPS-generator construction. Checkpoints include MPS RNG state when
the model is on MPS, and loading maps tensors through CPU before device placement.

If MPS is unavailable or a kernel is unsupported, start a new CPU run or resume a
coherent checkpoint with `--device cpu`. Only actual post-activation execution
can establish speed or learning quality. This change did not run the contestant.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Python reports `x86_64` on an Apple Silicon Mac | Use native Terminal/ARM64 Python, then recreate the venv. Avoid mixing `/usr/local` Intel wheels with `/opt/homebrew` ARM64 Python. |
| Copied venv fails to execute | Move it aside and rerun setup on the Mac. |
| Pygame import fails or no window appears | Run setup in the current checkout; use the local `.venv/bin/python`; remove inherited dummy SDL settings. |
| Torch installation has no matching wheel | Use ARM64 Python 3.12 and a current macOS. Modern PyTorch wheels are not available for Intel macOS. |
| `codex-maze` or `maze-game` is not on PATH | Use the explicit local interpreter commands above or activate `.venv`. |
| MPS unavailable | Use CPU. MPS availability depends on the PyTorch build, OS, and hardware. |
| No successful episodes / no checkpoint | The contestant is dormant, or training has not produced one. Setup never creates learned weights. |
| Old recordings absent after a clone | Copy the ignored `.maze-runs/` folder from the old computer. |

The game/dashboard can also run on an Intel Mac with the game-only setup command.
The supported full migration target here is your **Apple Silicon Mac**. This
project does not silently install a legacy Intel PyTorch release.

## What was verified

Current official PyPI wheel metadata was inspected for Python 3.12: pygame-ce 2.5.8
has a macOS universal2 wheel and PyTorch 2.14 has a macOS 14 ARM64 wheel. The
[PyTorch MPS API](https://docs.pytorch.org/docs/stable/notes/mps.html) and checkpoint
RNG API were consulted. The launcher has isolated orchestration tests that mock
installation and window-launch commands; they do not run an agent or a maze.
Shell launchers were syntax-checked, and contestant code was statically audited
with its activation switch false. No macOS machine is available in this workspace,
so a native macOS launch and MPS execution remain unverified.
