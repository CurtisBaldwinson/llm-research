"""Stage the official TRACE benchmark data onto the GPU pod.

The official TRACE data (Wang et al., 2024) is distributed as a single archive
on Google Drive, linked from the repo README
(https://github.com/BeyonderXX/TRACE):

    https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view

It unpacks to one folder per task, each containing train.json / eval.json /
test.json, every record being {"prompt": ..., "answer": ...}. This module
downloads + extracts that archive and normalizes the result to:

    <data_root>/TRACE/<Task>/train.json
    <data_root>/TRACE/<Task>/test.json      (held-out eval)

so benchmarks/trace/trace_data.py can load it directly.

This runs ON THE POD (fetch-on-pod), not on the operator's Mac, so the code
tarball stays small. It is idempotent: if the 8 task folders already exist
with train/test json, it does nothing.

Usage (on the pod, invoked by the benchmark wrapper before run.py):
    python3 benchmarks/trace/fetch_data.py --data-root data
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# Google Drive file id from the official TRACE README ("Trace Benchmark" link).
TRACE_GDRIVE_ID = "1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV"

TRACE_TASKS = [
    "ScienceQA", "FOMC", "MeetingBank", "C-STANCE",
    "20Minuten", "Py150", "NumGLUE-cm", "NumGLUE-ds",
]


def _already_staged(trace_dir: Path) -> bool:
    for task in TRACE_TASKS:
        d = trace_dir / task
        if not (d / "train.json").exists() or not (d / "test.json").exists():
            return False
    return True


def _pip_install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _download_archive(dest: Path) -> Path:
    """Download the TRACE archive from Google Drive via gdown."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        _pip_install("gdown")
        import gdown  # noqa: F401

    import gdown
    out = dest / "trace_raw_download"
    # gdown handles the Drive confirm-token dance for large files.
    got = gdown.download(id=TRACE_GDRIVE_ID, output=str(out), quiet=False)
    if not got:
        raise RuntimeError(
            "gdown failed to fetch the TRACE archive. If this is a quota/"
            "confirm error, download the file manually from "
            "https://drive.google.com/file/d/%s/view and place it at %s."
            % (TRACE_GDRIVE_ID, out)
        )
    return Path(got)


def _extract(archive: Path, dest: Path) -> None:
    """Extract a zip or tar archive into dest."""
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    else:
        raise RuntimeError(f"Unrecognized archive format: {archive}")


def _find_task_dirs(root: Path) -> dict:
    """Locate each task folder anywhere under root (the archive may nest them
    under a top-level dir). Returns {task: Path}."""
    found = {}
    for task in TRACE_TASKS:
        # exact-name match, searched at any depth
        matches = [p for p in root.rglob(task) if p.is_dir()]
        # prefer a dir that actually contains a train.json
        matches.sort(key=lambda p: 0 if (p / "train.json").exists() else 1)
        if matches:
            found[task] = matches[0]
    return found


def stage_trace(data_root: Path) -> Path:
    """Ensure <data_root>/TRACE/<task>/{train,test}.json exists. Idempotent."""
    data_root = Path(data_root)
    trace_dir = data_root / "TRACE"
    if _already_staged(trace_dir):
        print(f"[fetch_data] TRACE already staged at {trace_dir}; skipping.")
        return trace_dir

    trace_dir.mkdir(parents=True, exist_ok=True)
    scratch = data_root / "_trace_extract"
    scratch.mkdir(parents=True, exist_ok=True)

    print("[fetch_data] downloading official TRACE archive from Google Drive...")
    archive = _download_archive(scratch)
    print(f"[fetch_data] extracting {archive} ...")
    _extract(archive, scratch)

    task_dirs = _find_task_dirs(scratch)
    missing = [t for t in TRACE_TASKS if t not in task_dirs]
    if missing:
        raise RuntimeError(
            f"After extraction, could not locate task folders: {missing}. "
            f"Inspect {scratch} and adjust _find_task_dirs()."
        )

    import shutil
    for task, srcdir in task_dirs.items():
        dstdir = trace_dir / task
        dstdir.mkdir(parents=True, exist_ok=True)
        for fname in ("train.json", "eval.json", "test.json"):
            src = srcdir / fname
            if src.exists():
                shutil.copy2(src, dstdir / fname)
        # test.json is the held-out eval; if the archive only ships eval.json,
        # fall back to it so trace_data.py's expectation holds.
        if not (dstdir / "test.json").exists() and (dstdir / "eval.json").exists():
            shutil.copy2(dstdir / "eval.json", dstdir / "test.json")

    if not _already_staged(trace_dir):
        raise RuntimeError(
            f"Staging incomplete under {trace_dir}; some tasks lack train/test json."
        )
    print(f"[fetch_data] TRACE staged: {trace_dir} ({len(TRACE_TASKS)} tasks).")
    return trace_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=Path("data"),
                    help="dir under which TRACE/ is created (default: ./data)")
    args = ap.parse_args()
    stage_trace(args.data_root)


if __name__ == "__main__":
    main()
