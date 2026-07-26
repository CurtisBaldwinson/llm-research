"""Stage the official TRACE benchmark data onto the GPU pod.

The official TRACE data (Wang et al., 2024) is distributed as a single archive
on Google Drive, linked from the repo README
(https://github.com/BeyonderXX/TRACE):

    https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view

It unpacks to one folder per task, each containing train.json / eval.json /
test.json, every record being {"prompt": ..., "answer": ...}. Crucially, the
archive ships MULTIPLE size variants of each task -- the full 5,000-example
train split (the officially reported TRACE size; Wang et al. 2024, Table 1) plus
reduced 1,000/500 ablation splits -- typically nested under sibling directories
named like ``LLM-CL-Benchmark_5000`` / ``LLM-CL-Benchmark_1000`` /
``LLM-CL-Benchmark_500``. Every variant contains a ``<Task>/train.json``, so a
naive "find <Task> anywhere and take the first hit" is nondeterministic across
filesystems and can silently stage the WRONG (smaller) split.

This module downloads + extracts that archive, DETERMINISTICALLY pins the
canonical 5,000-example variant, verifies the exact per-task record counts, and
normalizes the result to:

    <data_root>/TRACE/<Task>/train.json     (5000 records)
    <data_root>/TRACE/<Task>/test.json      (2000 records, held-out eval)

so benchmarks/trace/trace_data.py can load it directly. It also records the
SHA-256 of the downloaded archive and a per-task record-count manifest to
    <data_root>/TRACE/_provenance.json
for reproducibility.

This runs ON THE POD (fetch-on-pod), not on the operator's Mac, so the code
tarball stays small. It is idempotent: if the 8 task folders already exist with
train/test json AND pass the record-count check, it does nothing.

Usage (on the pod, invoked by the benchmark wrapper before run.py):
    python3 benchmarks/trace/fetch_data.py --data-root data
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# --- Dataset-variant pinning (Claim 1 fix) -------------------------------
# The archive ships several size variants of the TRACE split. We pin the
# canonical full-size variant: 5,000 train + 2,000 test examples per task,
# exactly as reported in TRACE (Wang et al., 2024, Table 1) and matching this
# project's registered design (run.py: n_train=5000, n_eval=2000).
#
# We identify the correct variant by a substring token that appears in the
# variant directory name. If the archive layout ever changes and no directory
# matches this token, we do NOT silently fall back to another variant -- we
# fail loudly (see _pin_variant_root). The definitive guarantee is the
# record-count assertion in _assert_record_counts, which rejects any variant
# whose per-task counts don't match EXPECTED_* below regardless of naming.
VARIANT_DIR_TOKEN = "LLM-CL-Benchmark_5000"
EXPECTED_TRAIN_COUNT = 5000
EXPECTED_TEST_COUNT = 2000


def _already_staged(trace_dir: Path) -> bool:
    for task in TRACE_TASKS:
        d = trace_dir / task
        if not (d / "train.json").exists() or not (d / "test.json").exists():
            return False
    return True


def _pip_install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _count_records(json_path: Path) -> int:
    """Count records in a TRACE task json (a top-level list of examples)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise RuntimeError(
            f"{json_path} is not a JSON list of records (got {type(data).__name__})."
        )
    return len(data)


def _pin_variant_root(scratch: Path) -> Path:
    """Return the archive subtree for the pinned 5,000-example variant.

    Deterministic: among all directories whose name contains VARIANT_DIR_TOKEN,
    pick the lexicographically-smallest full path that actually contains at
    least one of the TRACE task folders. Raises if none is found -- we never
    silently fall back to a different-size variant.
    """
    candidates = sorted(
        (p for p in scratch.rglob("*")
         if p.is_dir() and VARIANT_DIR_TOKEN in p.name),
        key=lambda p: str(p),
    )
    for cand in candidates:
        if any((cand / task).is_dir() for task in TRACE_TASKS):
            return cand
    raise RuntimeError(
        "Could not locate the pinned TRACE variant directory "
        f"(name containing {VARIANT_DIR_TOKEN!r}) with task folders under "
        f"{scratch}. The archive layout may have changed. Inspect the extracted "
        "tree and update VARIANT_DIR_TOKEN; do NOT stage an unverified variant."
    )


def _find_task_dirs(variant_root: Path) -> dict:
    """Locate each task folder under the PINNED variant root.

    Deterministic: searches only within the pinned 5,000-example variant, and
    among any duplicate matches picks the lexicographically-smallest path that
    contains a train.json. Returns {task: Path}.
    """
    found = {}
    for task in TRACE_TASKS:
        matches = sorted(
            (p for p in variant_root.rglob(task) if p.is_dir()),
            key=lambda p: str(p),
        )
        # deterministic preference for a dir that actually has train.json,
        # breaking ties by sorted path (already sorted above).
        matches.sort(key=lambda p: (0 if (p / "train.json").exists() else 1, str(p)))
        if matches:
            found[task] = matches[0]
    return found


def _assert_record_counts(trace_dir: Path) -> dict:
    """Hard-verify each staged task has EXACTLY the expected record counts.

    This is the definitive guard against staging the wrong-size variant
    (Claim 1). Returns a {task: {"train": n, "test": m}} manifest. Raises on
    any mismatch.
    """
    manifest = {}
    errors = []
    for task in TRACE_TASKS:
        d = trace_dir / task
        n_train = _count_records(d / "train.json")
        n_test = _count_records(d / "test.json")
        manifest[task] = {"train": n_train, "test": n_test}
        if n_train != EXPECTED_TRAIN_COUNT:
            errors.append(
                f"{task}: train.json has {n_train} records, "
                f"expected {EXPECTED_TRAIN_COUNT}"
            )
        if n_test != EXPECTED_TEST_COUNT:
            errors.append(
                f"{task}: test.json has {n_test} records, "
                f"expected {EXPECTED_TEST_COUNT}"
            )
    if errors:
        raise RuntimeError(
            "TRACE record-count verification FAILED -- the staged data does not "
            "match the canonical 5,000-train / 2,000-test variant. This almost "
            "certainly means the wrong size variant was staged.\n  "
            + "\n  ".join(errors)
        )
    return manifest


def stage_trace(data_root: Path) -> Path:
    """Ensure <data_root>/TRACE/<task>/{train,test}.json exists, pinned to the
    5,000-example variant and record-count verified. Idempotent."""
    data_root = Path(data_root)
    trace_dir = data_root / "TRACE"
    if _already_staged(trace_dir):
        # Even when already staged, re-verify counts so a previously-staged
        # WRONG variant can't slip through on a re-run.
        _assert_record_counts(trace_dir)
        print(f"[fetch_data] TRACE already staged + verified at {trace_dir}; skipping.")
        return trace_dir

    trace_dir.mkdir(parents=True, exist_ok=True)
    scratch = data_root / "_trace_extract"
    scratch.mkdir(parents=True, exist_ok=True)

    print("[fetch_data] downloading official TRACE archive from Google Drive...")
    archive = _download_archive(scratch)
    archive_sha256 = _sha256_file(archive)
    print(f"[fetch_data] archive sha256: {archive_sha256}")
    print(f"[fetch_data] extracting {archive} ...")
    _extract(archive, scratch)

    variant_root = _pin_variant_root(scratch)
    print(f"[fetch_data] pinned variant root: {variant_root}")

    task_dirs = _find_task_dirs(variant_root)
    missing = [t for t in TRACE_TASKS if t not in task_dirs]
    if missing:
        raise RuntimeError(
            f"After extraction, could not locate task folders under the pinned "
            f"variant {variant_root}: {missing}. Inspect it and adjust "
            f"_find_task_dirs()."
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

    # Definitive guard: verify exact record counts before declaring success.
    manifest = _assert_record_counts(trace_dir)

    # Persist provenance for reproducibility. Record the variant root relative
    # to the scratch dir when possible (portable across Python versions).
    try:
        variant_root_str = str(variant_root.relative_to(scratch))
    except ValueError:
        variant_root_str = str(variant_root)
    provenance = {
        "gdrive_id": TRACE_GDRIVE_ID,
        "archive_sha256": archive_sha256,
        "pinned_variant_token": VARIANT_DIR_TOKEN,
        "pinned_variant_root": variant_root_str,
        "expected_train_count": EXPECTED_TRAIN_COUNT,
        "expected_test_count": EXPECTED_TEST_COUNT,
        "record_counts": manifest,
    }
    with open(trace_dir / "_provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)

    print(
        f"[fetch_data] TRACE staged + verified: {trace_dir} "
        f"({len(TRACE_TASKS)} tasks, {EXPECTED_TRAIN_COUNT} train / "
        f"{EXPECTED_TEST_COUNT} test each). Provenance -> {trace_dir/'_provenance.json'}."
    )
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
