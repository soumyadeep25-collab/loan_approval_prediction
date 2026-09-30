import os

# backend/src -> repo root (two levels up)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def data_path(name: str) -> str:
    """Absolute path to a file inside the repo's data/ directory."""
    return os.path.join(REPO_ROOT, "data", name)


def models_path(name: str) -> str:
    """Absolute path to a file inside the repo's models/ directory."""
    return os.path.join(REPO_ROOT, "models", name)


def images_path(name: str) -> str:
    """Absolute path to a file inside the repo's images/ directory."""
    return os.path.join(REPO_ROOT, "images", name)
