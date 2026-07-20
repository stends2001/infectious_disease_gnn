from pathlib import Path
import sys

def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()

    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists() or (path / ".git").exists():
            return path

    raise RuntimeError("Could not find project root.")

ROOT = find_project_root()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))