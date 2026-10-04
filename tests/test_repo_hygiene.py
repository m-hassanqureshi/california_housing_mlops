"""Repository guard rails that CI enforces even when a contributor skipped pre-commit."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAX_TRACKED_BYTES = 1024 * 1024  # same 1 MB limit as the pre-commit hook
DATA_SUFFIXES = {".csv", ".parquet", ".pkl", ".joblib", ".onnx", ".h5"}


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )


def _tracked_files() -> list[Path]:
    result = _git("ls-files", "-z")
    if result.returncode != 0:
        pytest.skip("not inside a git checkout")
    return [ROOT / name for name in result.stdout.split("\0") if name]


def test_no_large_files_tracked_by_git():
    too_big = [
        str(path.relative_to(ROOT))
        for path in _tracked_files()
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES
    ]
    assert not too_big, f"Track these with DVC instead of Git: {too_big}"


def test_no_data_or_model_files_tracked_by_git():
    leaked = [
        str(path.relative_to(ROOT))
        for path in _tracked_files()
        if path.suffix in DATA_SUFFIXES
    ]
    assert not leaked, f"Data/model files must be tracked by DVC: {leaked}"


def test_dvc_pointer_files_are_not_git_ignored():
    pointers = sorted(ROOT.glob("data/**/*.dvc"))
    if not pointers:
        pytest.skip("no DVC pointer files yet")
    for pointer in pointers:
        # --no-index: test the ignore rules themselves, even for already-tracked
        # pointers, so a new `dvc add` in the same folder would not be blocked
        ignored = _git(
            "check-ignore", "--no-index", "-q", str(pointer.relative_to(ROOT))
        )
        assert ignored.returncode == 1, f"{pointer.name} is git-ignored"
