"""Portability utilities only: no game, agent, model, or episode is executed."""

import platform
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import run
from maze_game._window import initial_size


class PortableLauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(prefix="MazeMind with spaces ")
        self.root = Path(self.directory.name)
        self.python = run.venv_python(self.root)
        self.python.parent.mkdir(parents=True)
        self.python.touch()

    def tearDown(self):
        self.directory.cleanup()

    def info(self):
        return {
            "machine": platform.machine(),
            "system": platform.system(),
            "prefix": str(self.root / ".venv"),
        }

    def test_setup_only_installs_packages_with_paths_as_single_arguments(self):
        with (
            patch.object(run, "ROOT", self.root),
            patch.object(run, "venv_python", return_value=self.python),
            patch.object(run, "environment_info", return_value=self.info()),
            patch.object(run, "native_mac"),
            patch.object(run, "command") as execute,
            redirect_stdout(StringIO()),
        ):
            run.setup(include_codex=True)
        self.assertEqual(
            [item.args[0][2:4] for item in execute.call_args_list],
            [
                ["pip", "install"],
                ["pip", "install"],
                ["pip", "install"],
                ["pip", "check"],
            ],
        )
        self.assertEqual(execute.call_args_list[1].args[0][-1], f"{self.root}[pygame]")
        self.assertEqual(
            execute.call_args_list[2].args[0][-1], str(self.root / "agents" / "codex")
        )
        self.assertFalse(
            any(
                "maze_game" in call.args[0] or "agents.codex" in call.args[0]
                for call in execute.call_args_list
            )
        )

    def test_setup_rejects_foreign_venv_before_installing(self):
        info = self.info()
        info["system"] = "a different operating system"
        with (
            patch.object(run, "ROOT", self.root),
            patch.object(run, "venv_python", return_value=self.python),
            patch.object(run, "environment_info", return_value=info),
            patch.object(run, "native_mac"),
            patch.object(run, "command") as execute,
            self.assertRaisesRegex(RuntimeError, "different architecture/OS"),
        ):
            run.setup(include_codex=False)
        execute.assert_not_called()

    def test_setup_rejects_incomplete_venv_without_overwriting_it(self):
        self.python.unlink()
        with (
            patch.object(run, "ROOT", self.root),
            patch.object(run, "venv_python", return_value=self.python),
            patch.object(run, "native_mac"),
            patch.object(run.venv, "EnvBuilder") as builder,
            self.assertRaisesRegex(RuntimeError, "incomplete or copied"),
        ):
            run.setup(include_codex=False)
        builder.assert_not_called()

    def test_studio_passes_arguments_without_selecting_a_competition_agent(self):
        with (
            patch.object(run, "ROOT", self.root),
            patch.object(run, "venv_python", return_value=self.python),
            patch.object(run, "command") as execute,
            patch("sys.argv", ["run.py", "studio", "--level", "35", "--seed", "73"]),
        ):
            self.assertEqual(run.main(), 0)
        arguments = execute.call_args.args[0]
        self.assertEqual(
            arguments[:4], [str(self.python), "-m", "maze_game", "--dashboard"]
        )
        self.assertIn("35", arguments)
        self.assertNotIn("--agent", arguments)
        self.assertNotIn("--train-headless", arguments)

    def test_subprocess_uses_checkout_directory_and_does_not_use_a_shell(self):
        with (
            patch.object(run, "ROOT", self.root),
            patch.object(run.subprocess, "run") as execute,
        ):
            run.command([str(self.python), "-m", "pip", "check"])
        self.assertEqual(execute.call_args.kwargs["cwd"], self.root)
        self.assertNotIn("shell", execute.call_args.kwargs)

    def test_mac_laptop_window_fits_without_changing_minimums(self):
        pygame = SimpleNamespace(
            display=SimpleNamespace(get_desktop_sizes=lambda: [(1440, 900)])
        )
        with patch("sys.platform", "darwin"):
            self.assertEqual(
                initial_size(pygame, (1560, 980), (1100, 760)), (1376, 804)
            )
            self.assertEqual(initial_size(pygame, (1380, 880), (900, 680)), (1376, 804))

    def test_other_platforms_keep_their_existing_window_sizes(self):
        with patch("sys.platform", "linux"):
            self.assertEqual(initial_size(None, (1560, 980), (1100, 760)), (1560, 980))


if __name__ == "__main__":
    unittest.main()
