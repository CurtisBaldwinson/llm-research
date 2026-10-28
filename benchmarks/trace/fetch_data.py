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
    <data_root>/TRACE/<Task>/test.json      (nonempty, per-task held-out counts)

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
# canonical full-size variant by its TRAIN size: 5,000 train examples per task
# (the real, uniform discriminator between the _5000 / _1000 / _500 variants),
# matching this project's registered design (run.py / trace_data.py:
# n_train=5000).
#
# TEST counts are deliberately NOT asserted against a uniform expectation.
# TRACE's held-out test sets are per-task and non-uniform (e.g. FOMC 496,
# MeetingBank 692, 20Minuten 200, NumGLUE-cm 81, NumGLUE-ds 325); there is no
# canonical "2,000 test each". The eval consumer in trace_data.py takes
# records[:n_eval] (a CAP, not a requirement -- see _tokenize_examples), so a
# task with fewer test records is valid and simply evaluates on all it has.
# CHANGED 2026-07-26: a prior EXPECTED_TEST_COUNT=2000 assertion was rejecting
# the correct full-size variant (train=5000 passed for every task; only the
# fabricated uniform test expectation failed). We now assert train==5000 and
# test>=MIN_TEST_COUNT (>0), and record the ACTUAL per-task test counts in the
# manifest/provenance for honest logs. Wrong-variant detection is preserved by
# the train==5000 check (the small variants ship 500/1000 train, not 5000).
#
# We identify the correct variant by a substring token that appears in the
# variant directory name. If the archive layout ever changes and no directory
# matches this token, we do NOT silently fall back to another variant -- we
# fail loudly (see _pin_variant_root). The definitive guarantee is the
# train-count assertion in _assert_record_counts, which rejects any variant
# whose per-task train counts don't match EXPECTED_TRAIN_COUNT regardless of
# naming.
VARIANT_DIR_TOKEN = "LLM-CL-Benchmark_5000"
EXPECTED_TRAIN_COUNT = 5000
# Minimum acceptable held-out test records per task (>0). NOT a uniform target;
# see the note above on why per-task test counts vary.
MIN_TEST_COUNT = 1


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
    """Verify each staged task is the canonical full-size variant.

    Variant guard = TRAIN count. We assert train == EXPECTED_TRAIN_COUNT
    (5,000), which is the real, uniform discriminator between the archive's
    _5000 / _1000 / _500 variants. TEST counts are per-task and non-uniform in
    TRACE, and the eval path caps at records[:n_eval], so we only require
    test >= MIN_TEST_COUNT (>0) and RECORD the actual per-task test counts in
    the returned manifest (for honest provenance logs) rather than asserting a
    fabricated uniform target. See the module-level note dated 2026-07-26.

    Returns a {task: {"train": n, "test": m}} manifest. Raises on any train
    mismatch or an empty test split.
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
                f"expected {EXPECTED_TRAIN_COUNT} (wrong-size variant?)"
            )
        if n_test < MIN_TEST_COUNT:
            errors.append(
                f"{task}: test.json has {n_test} records, "
                f"expected >= {MIN_TEST_COUNT} (empty/missing held-out split)"
            )
    if errors:
        raise RuntimeError(
            "TRACE variant verification FAILED -- the staged data is not the "
            "canonical 5,000-train variant (or a task has an empty test split). "
            "Per-task test counts vary by design and are NOT asserted against a "
            "uniform target; see fetch_data's module-level note.\n  "
            + "\n  ".join(errors)
        )
    return manifest


def _write_provenance(trace_dir: Path, manifest: dict,
                      archive_sha256: str | None = None,
                      variant_root_str: str | None = None) -> None:
    """Write _provenance.json in the CURRENT schema.

    Called from BOTH the full-staging path and the already-staged fast path so
    a rerun over a directory staged by an older version is migrated to the
    current provenance schema rather than left with stale metadata
    (CodeRabbit 2026-07-28). On the fast path nothing was downloaded, so
    archive_sha256 / pinned_variant_root are unknown; we record an explicit
    sentinel instead of a stale value.
    """
    provenance = {
        "gdrive_id": TRACE_GDRIVE_ID,
        "archive_sha256": archive_sha256 if archive_sha256 is not None
            else "unknown (fast-path rerun; archive not re-downloaded)",
        "pinned_variant_token": VARIANT_DIR_TOKEN,
        "pinned_variant_root": variant_root_str if variant_root_str is not None
            else "unknown (fast-path rerun; archive not re-extracted)",
        "expected_train_count": EXPECTED_TRAIN_COUNT,
        "min_test_count": MIN_TEST_COUNT,
        "test_counts_note": (
            "per-task test counts vary by design; actual counts recorded in "
            "record_counts below (not asserted against a uniform target)"
        ),
        "record_counts": manifest,
    }
    with open(trace_dir / "_provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)


def stage_trace(data_root: Path) -> Path:
    """Ensure <data_root>/TRACE/<task>/{train,test}.json exists, pinned to the
    5,000-example variant and record-count verified. Idempotent."""
    data_root = Path(data_root)
    trace_dir = data_root / "TRACE"
    if _already_staged(trace_dir):
        # Even when already staged, re-verify counts so a previously-staged
        # WRONG variant can't slip through on a re-run.
        manifest = _assert_record_counts(trace_dir)
        # Rewrite provenance in the current schema so a directory staged by an
        # older version doesn't retain stale/missing metadata (CodeRabbit).
        _write_provenance(trace_dir, manifest)
        test_counts_str = ", ".join(
            f"{task}:{manifest[task]['test']}" for task in TRACE_TASKS
        )
        print(
            f"[fetch_data] TRACE already staged + verified at {trace_dir}; "
            f"skipping download (per-task test counts [{test_counts_str}]); "
            f"provenance refreshed."
        )
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
    _write_provenance(trace_dir, manifest, archive_sha256=archive_sha256,
                      variant_root_str=variant_root_str)

    test_counts_str = ", ".join(
        f"{task}:{manifest[task]['test']}" for task in TRACE_TASKS
    )
    print(
        f"[fetch_data] TRACE staged + verified: {trace_dir} "
        f"({len(TRACE_TASKS)} tasks, {EXPECTED_TRAIN_COUNT} train each; "
        f"per-task test counts [{test_counts_str}]). "
        f"Provenance -> {trace_dir/'_provenance.json'}."
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
