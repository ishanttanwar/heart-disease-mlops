"""Phase 0: scaffold the heart-disease MLOps project.

Run once from the repository root:

    python setup_project.py

Safe to re-run: existing files are never overwritten.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folders to create. Empty ones get a .gitkeep so Git tracks them.
DIRS = [
    "data/raw",
    "data/processed",
    "src/api",
    "notebooks",
    "tests",
    "models",
    "k8s",
    "monitoring",
    "report",
    ".github/workflows",
    "screenshots/eda",
    "screenshots/mlflow",
    "screenshots/api",
    "screenshots/docker",
    "screenshots/ci",
    "screenshots/k8s",
    "screenshots/monitoring",
]

# Python packages (folders that need an __init__.py).
PACKAGES = ["src", "src/api", "tests"]

# Small config files written only if they do not exist yet.
FILES = {
    "pytest.ini": (
        "[pytest]\n"
        "testpaths = tests\n"
        "pythonpath = .\n"
        "addopts = -v\n"
    ),
    ".flake8": (
        "[flake8]\n"
        "max-line-length = 100\n"
        "exclude = .git,__pycache__,.venv,venv,mlruns,notebooks\n"
    ),
    # Keep line endings as LF so Dockerfiles and scripts behave the same on
    # Windows, Linux containers, and GitHub Actions.
    ".gitattributes": "* text=auto eol=lf\n",
    ".gitignore": (
        "# Python\n"
        "__pycache__/\n"
        "*.py[cod]\n"
        ".pytest_cache/\n"
        ".ipynb_checkpoints/\n"
        "\n"
        "# Virtual environments\n"
        ".venv/\n"
        "venv/\n"
        "env/\n"
        "\n"
        "# MLflow local tracking data (screenshots serve as evidence)\n"
        "mlruns/\n"
        "mlartifacts/\n"
        "mlflow.db\n"
        "\n"
        "# Secrets and local config\n"
        ".env\n"
        "\n"
        "# Editors and OS\n"
        ".vscode/\n"
        ".idea/\n"
        ".DS_Store\n"
        "Thumbs.db\n"
    ),
}


def write_if_missing(path: Path, content: str = "") -> bool:
    """Create `path` with `content` unless it already exists."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    created = []

    for d in DIRS:
        folder = ROOT / d
        folder.mkdir(parents=True, exist_ok=True)
        if write_if_missing(folder / ".gitkeep"):
            created.append(f"{d}/.gitkeep")

    for pkg in PACKAGES:
        if write_if_missing(ROOT / pkg / "__init__.py"):
            created.append(f"{pkg}/__init__.py")

    for name, content in FILES.items():
        if write_if_missing(ROOT / name, content):
            created.append(name)

    if created:
        print("Created:")
        for item in created:
            print(f"  + {item}")
    else:
        print("Nothing to do: scaffold already exists.")


if __name__ == "__main__":
    main()
