"""The only activation switch. No CLI flag or environment variable overrides it."""

READY = False


class DormantError(RuntimeError):
    """An executable contestant operation was requested before activation."""


def require_ready() -> None:
    if READY is not True:
        raise DormantError(
            "Codex contestant is dormant (READY = False). No operation started. "
            "Activate agents/codex/activation.py only when the championship permits it."
        )
