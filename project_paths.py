from __future__ import annotations

from pathlib import Path


def project_root_from(file_path: str, levels_up: int = 1) -> Path:
    # Keep the invocation-facing path instead of resolving symlinks.
    return Path(file_path).absolute().parents[levels_up]


def repo_rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except Exception:
        return p.as_posix() if p.is_absolute() else str(p)
