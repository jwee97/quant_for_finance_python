"""A root folder of its own for a test app.

An app keeps its state under ``<root>/data/user`` (the record of every background run, downloads, the iTick counters), and its guides under ``<root>/docs``. A test that gave the app the repository as its root
wrote those records into the working tree. This gives it a new folder instead: the repository's guides and downloaded market data are linked in, and ``data/user`` is new and empty.
"""

from __future__ import annotations

from pathlib import Path


def isolated_root(base: Path, config) -> Path:
    """``base`` made into an app root: ``docs`` and every folder of the repository's ``data`` except ``user`` are links to the real ones; ``data/user`` is a new, empty folder."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    (base / "docs").symlink_to(config.root / "docs", target_is_directory=True)
    data = base / "data"
    data.mkdir()
    for child in (config.root / "data").iterdir():
        if child.name != "user":
            (data / child.name).symlink_to(child, target_is_directory=child.is_dir())
    return base
