"""Write sampled episodes in the existing Training Studio's SQLite schema."""

from pathlib import Path

from .activation import require_ready


class StudioRecorder:
    def __init__(self, directory: Path, name: str, run_ids: dict | None = None):
        require_ready()
        from maze_game.training import RunStore

        self.store = RunStore(directory / "training.sqlite3")
        self.name = name
        self.run_ids = {int(k): v for k, v in (run_ids or {}).items()}
        existing = {item["id"] for item in self.store.runs()}
        self.run_ids = {level: key for level, key in self.run_ids.items() if key in existing}

    def save(self, record, master_seed: int) -> None:
        require_ready()
        from maze_game.training import RunConfig

        level = record.level
        if level not in self.run_ids:
            self.run_ids[level] = self.store.create_run(
                f"{self.name} / L{level:02d}", "Codex Memory PPO", RunConfig(level, master_seed, 100_000)
            )
        run_id = self.run_ids[level]
        self.store.save_episode(run_id, record)
        self.store.set_status(run_id, "running")

    def close(self, status: str = "paused") -> None:
        require_ready()
        for run_id in self.run_ids.values():
            self.store.set_status(run_id, status)
        self.store.close()

    def rollback(self, assigned_episodes: int, active_records: list, master_seed: int) -> None:
        require_ready()
        # Later recordings from a crash or a newer checkpoint must not remain
        # mixed with this resumed branch. Active episodes may have subsequently
        # finished, so replace them with the checkpoint's partial snapshots too.
        owned = {
            item["id"]
            for item in self.store.runs()
            if item["name"].startswith(self.name + " / L") and item["agent"] == "Codex Memory PPO"
        }
        with self.store.connection:
            for run_id in owned | set(self.run_ids.values()):
                self.store.connection.execute(
                    "DELETE FROM episodes WHERE run_id=? AND number>?", (run_id, assigned_episodes)
                )
                if run_id not in self.run_ids.values():
                    self.store.connection.execute("UPDATE runs SET status='superseded' WHERE id=?", (run_id,))
        for record in active_records:
            if record is not None:
                self.save(record, master_seed)
