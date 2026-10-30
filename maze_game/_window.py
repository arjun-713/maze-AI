"""Mac laptop window sizing, without opening a window or creating an environment."""

import sys


def initial_size(
    pygame, preferred: tuple[int, int], minimum: tuple[int, int]
) -> tuple[int, int]:
    if sys.platform != "darwin":
        return preferred
    desktops = pygame.display.get_desktop_sizes()
    if not desktops:
        return preferred
    width, height = desktops[0]
    # Leave space for the macOS menu bar, title bar, and Dock while retaining
    # each interface's existing minimum. Larger displays keep the preferred size.
    return (
        min(preferred[0], max(minimum[0], width - 64)),
        min(preferred[1], max(minimum[1], height - 96)),
    )
