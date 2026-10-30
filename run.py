"""Portable local setup and launch helper. Never imports or activates contestant code."""

import argparse
import ast
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def venv_python(root: Path = ROOT) -> Path:
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def command(args: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, check=True, text=True, capture_output=capture)


def environment_info(python: Path) -> dict:
    # Metadata only: no Pygame/Torch/agent import, model construction, or maze reset.
    program = """
import json, platform, sys
from importlib import metadata
packages = {}
for name in ('maze-ai-game', 'pygame-ce', 'mazemind-codex-contestant', 'numpy', 'torch'):
    try:
        packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({'python': sys.version.split()[0], 'prefix': sys.prefix, 'machine': platform.machine(),
                  'system': platform.system(), 'macos': platform.mac_ver()[0], 'packages': packages}))
"""
    return json.loads(command([str(python), "-c", program], capture=True).stdout)


def native_mac() -> None:
    if sys.platform == "darwin" and platform.machine() == "x86_64":
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            raise RuntimeError(
                "Python is running under Rosetta on an Apple Silicon Mac. Open a native Terminal "
                "and use ARM64 Python 3.12 before creating .venv. See MACOS.md."
            )


def setup(include_codex: bool) -> None:
    native_mac()
    if include_codex and sys.platform == "darwin" and platform.machine() != "arm64":
        raise RuntimeError(
            "The current Codex PyTorch dependency targets Apple Silicon on macOS. "
            "Intel Macs can install the game/studio with setup without --codex."
        )
    python = venv_python()
    if not python.exists():
        if (ROOT / ".venv").exists():
            raise RuntimeError(
                ".venv is incomplete or copied from another OS. Move it aside and rerun setup."
            )
        print("Creating the project's local .venv...", flush=True)
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    info = environment_info(python)
    if (
        info["machine"] != platform.machine()
        or info["system"] != platform.system()
        or Path(info["prefix"]).resolve() != (ROOT / ".venv").resolve()
    ):
        raise RuntimeError(
            ".venv belongs to a different architecture/OS. Recreate it on this computer."
        )
    command([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    command([str(python), "-m", "pip", "install", "-e", f"{ROOT}[pygame]"])
    if include_codex:
        command(
            [str(python), "-m", "pip", "install", "-e", str(ROOT / "agents" / "codex")]
        )
    command([str(python), "-m", "pip", "check"])
    print("Setup complete. No game, agent, or training session was started.")
    launcher = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print(f"Next: {launcher} run.py studio   or   {launcher} run.py game")
    if not include_codex:
        print("Optional dormant contestant dependencies: python3 run.py setup --codex")


def doctor() -> None:
    python = venv_python()
    if not python.exists():
        raise RuntimeError("No local .venv. Run: python3.12 run.py setup --codex")
    info = environment_info(python)
    print(
        f"MazeMind environment: {info['system']} {info['machine']}, Python {info['python']}"
    )
    print(f"Interpreter: {python}")
    for name, version in info["packages"].items():
        print(f"  {name}: {version or 'not installed'}")
    switch = ROOT / "agents" / "codex" / "activation.py"
    values = [
        node.value.value
        for node in ast.parse(switch.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(
            isinstance(target, ast.Name) and target.id == "READY"
            for target in node.targets
        )
    ]
    print(
        f"Codex READY: {values[0] if len(values) == 1 else 'unknown (inspect activation.py)'}"
    )
    fonts = ROOT / "maze_game" / "assets" / "fonts"
    for name in ("Fredoka-SemiBold.ttf", "Nunito-Medium.ttf", "Nunito-Bold.ttf"):
        if not (fonts / name).is_file():
            raise RuntimeError(
                f"Missing bundled font: {name}; restore the complete checkout."
            )
    if not info["packages"]["maze-ai-game"] or not info["packages"]["pygame-ce"]:
        raise RuntimeError("Game dependencies are missing. Run: python3 run.py setup")
    print("Bundled fonts present. No model, environment, or window was initialized.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set up or open MazeMind using this checkout's local .venv."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    installation = commands.add_parser(
        "setup", help="create .venv and install the game; starts nothing"
    )
    installation.add_argument(
        "--codex",
        action="store_true",
        help="also install dormant contestant dependencies",
    )
    commands.add_parser(
        "doctor", help="inspect installed package metadata without importing agents"
    )
    commands.add_parser("game", help="open the manual maze game")
    studio = commands.add_parser(
        "studio", help="open Training Studio with your current Redbull scaffold"
    )
    studio.add_argument("--level", type=int, default=1, choices=range(1, 36))
    studio.add_argument("--seed", type=int, default=42)
    studio.add_argument("--episodes", type=int, default=100)
    studio.add_argument("--runs-dir", type=Path, default=ROOT / ".maze-runs")
    args = parser.parse_args()
    try:
        if sys.version_info < (3, 10):  # noqa: UP036 -- diagnose older system Python explicitly
            raise RuntimeError(
                "Python 3.10+ is required; Python 3.12 is recommended on macOS."
            )
        if args.command == "setup":
            setup(args.codex)
        elif args.command == "doctor":
            doctor()
        else:
            python = venv_python()
            if not python.exists():
                raise RuntimeError(
                    "Run python3.12 run.py setup first. Do not copy a Linux .venv to your Mac."
                )
            launch = [str(python), "-m", "maze_game"]
            if args.command == "studio":
                launch += [
                    "--dashboard",
                    "--level",
                    str(args.level),
                    "--seed",
                    str(args.seed),
                    "--episodes",
                    str(args.episodes),
                    "--runs-dir",
                    str(args.runs_dir),
                ]
            # A normal foreground child keeps Pygame on the child's main thread.
            command(launch)
        return 0
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        print(
            f"MazeMind command failed (exit {error.returncode}). See the output above and MACOS.md.",
            file=sys.stderr,
        )
        return error.returncode if error.returncode > 0 else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"MazeMind: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("MazeMind interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
