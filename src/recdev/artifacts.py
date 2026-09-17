"""Saídas versionadas por execução e informações de reprodutibilidade."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def environment(root: Path) -> dict:
    versions = {}
    for name in ("recdev", "numpy", "scikit-learn", "torch", "torchvision", "facenet-pytorch", "pillow", "matplotlib"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                capture_output=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True,
                                   capture_output=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(python=sys.version, platform=platform.platform(), versions=versions,
                git_commit=commit, git_dirty=dirty)


def new_run(root: Path, dataset: str, label: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / "artifacts" / "runs" / dataset / f"{stamp}-{label}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
