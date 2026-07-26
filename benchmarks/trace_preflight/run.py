#!/usr/bin/env python3
"""trace_preflight -- one-command Batch 2 equivalence preflight on RunPod.

This is a THIN benchmark entry point. Its only job is to run the REAL
``preflight_batch2`` logic that lives in the sibling ``trace`` benchmark
(pulled onto the pod via ``bench_deps.txt``) against the real Llama-3.2-3B +
LoRA model, and to drop the resulting ``_batch2_preflight.json`` artifact into
the run's results dir so the dispatcher pulls it home automatically.

Deploy with ZERO extra effort:

    make run BENCH=trace_preflight GPU=A40 CLOUD=SECURE

The pod wrapper execs ``python3 -u run.py {--smoke} {extra_args} --out <OUT>``
and exports ``SYNTH_RESULTS_DIR`` (and ``SYNTH_GIT_COMMIT`` when the dispatcher
knows the operator HEAD). We honor that contract:
  * ``--out``  -- the wrapper's OUT_JSON path; we ALSO write the canonical
                  ``_batch2_preflight.json`` next to it (same results dir) so
                  the archive/pull captures the artifact under the gate's
                  expected filename.
  * ``--smoke``-- plumbing check only: same real model_id path but CPU +
                  smaller probe counts. Still downloads/loads the real model
                  (slow). NEVER evidence for a scored run. On a real
                  preflight, DO NOT pass --smoke.

Everything else (model id, data root, tolerance, probe counts) is forwarded to
``preflight_batch2.main`` with pod-sensible defaults, overridable via
``--extra-args`` on the make command, e.g.:

    make run BENCH=trace_preflight GPU=A40 CLOUD=SECURE \\
        EXTRA_FLAGS="--extra-args '--tol 1e-5 --n_probe_batches 6'"
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the sibling trace benchmark on the pod.
#
# push_code lays the tar out as REMOTE_BASE/{sm_src, trace_preflight, trace}.
# This file is REMOTE_BASE/trace_preflight/run.py, so the trace folder is a
# sibling. Locally (make run BENCH=trace_preflight from a dev box) the same
# relative layout holds under benchmarks/.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
TRACE_DIR = HERE.parent / "trace"
SM_SRC = HERE.parent.parent / "sm_src"  # local layout; on pod it's REMOTE_BASE/sm_src

# On the pod sm_src sits at REMOTE_BASE/sm_src == HERE.parent/sm_src.
_pod_sm_src = HERE.parent / "sm_src"
for p in (TRACE_DIR, _pod_sm_src, SM_SRC):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

if not TRACE_DIR.exists():
    print(
        "[trace_preflight] FATAL: sibling 'trace' benchmark not found at "
        f"{TRACE_DIR}. Ensure benchmarks/trace_preflight/bench_deps.txt lists "
        "'trace' so the dispatcher pushes it alongside this benchmark.",
        file=sys.stderr, flush=True,
    )
    # Emit a done sentinel the finalize path understands, then fail.
    rc = 2
    rd = os.environ.get("SYNTH_RESULTS_DIR")
    if rd:
        Path(rd, f"_DONE.{rc}").write_text(str(rc))
    sys.exit(rc)


def _stage_data_root(*, require: bool) -> str:
    """Stage official TRACE data and return the task-folder parent path.

    ``fetch_data.stage_trace(parent)`` creates ``<parent>/TRACE/<task>/...`` and
    returns the ``.../TRACE`` directory that ``build_trace_sequence`` reads.
    Same contract as ``trace/run.py``'s ``_trace_sequence``.
    """
    import fetch_data  # from the trace folder (now on sys.path)

    parent = TRACE_DIR / "data"
    if os.environ.get("TRACE_DATA_ROOT"):
        # Operator override: treat as already-staged task parent if it exists,
        # else as the stage parent (stage_trace appends TRACE/).
        override = Path(os.environ["TRACE_DATA_ROOT"])
        if (override / "ScienceQA" / "train.json").exists():
            return str(override)
        parent = override
    print(
        f"[trace_preflight] staging TRACE data via fetch_data.stage_trace({parent}) ...",
        flush=True,
    )
    try:
        staged = fetch_data.stage_trace(parent)
    except Exception as exc:  # noqa: BLE001
        if require:
            print(
                f"[trace_preflight] FATAL: TRACE data staging failed: {exc!r}",
                file=sys.stderr, flush=True,
            )
            raise SystemExit(2) from exc
        print(
            f"[trace_preflight] NOTE: data staging failed ({exc!r}); "
            "smoke will fail later if data is absent.",
            flush=True,
        )
        return str(parent / "TRACE")
    return str(staged)


def main() -> None:
    # Parse ONLY the wrapper-injected args we must intercept; forward the rest
    # verbatim to preflight_batch2.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--data_root", default=None)
    known, passthrough = ap.parse_known_args()

    results_dir = None
    if known.out:
        results_dir = Path(known.out).resolve().parent
    elif os.environ.get("SYNTH_RESULTS_DIR"):
        results_dir = Path(os.environ["SYNTH_RESULTS_DIR"]).resolve()
    else:
        results_dir = HERE  # last resort (local hand-run)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_id = known.model_id or os.environ.get(
        "PREFLIGHT_MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")

    # Stage (or locate) TRACE data. Real preflight fails hard if staging fails;
    # smoke may continue and surface a clearer missing-data error downstream.
    if known.data_root:
        data_root = known.data_root
    else:
        data_root = _stage_data_root(require=not known.smoke)

    # Build the argv preflight_batch2.main expects. The artifact is written
    # into results_dir so the dispatcher's `tar ... results` pull brings it
    # home; the gate looks for exactly this filename.
    argv = [
        "preflight_batch2.py",
        "--model_id", model_id,
        "--data_root", data_root,
        "--out", str(results_dir),
    ]
    if known.smoke:
        # Smoke: shrink probes and use CPU. Still loads the real model_id
        # (no synthetic substitute). NOT a valid equivalence proof.
        argv += ["--device", "cpu", "--n_probe_batches", "2",
                 "--n_probe_tasks", "1", "--n_tasks", "1"]
        print(
            "[trace_preflight] SMOKE mode: plumbing check only (real model on "
            "CPU, smaller probes). NOT evidence.",
            flush=True,
        )
    argv += passthrough  # --extra-args the operator supplied via the dispatcher

    print(f"[trace_preflight] model_id={model_id}", flush=True)
    print(f"[trace_preflight] data_root={data_root}", flush=True)
    print(
        f"[trace_preflight] SYNTH_GIT_COMMIT="
        f"{os.environ.get('SYNTH_GIT_COMMIT', '')!r}",
        flush=True,
    )
    print(f"[trace_preflight] artifact -> {results_dir/'_batch2_preflight.json'}",
          flush=True)
    print(f"[trace_preflight] argv={argv}", flush=True)

    # Run the REAL preflight as __main__ so there is a single source of truth.
    sys.argv = argv
    rc = 0
    try:
        runpy.run_path(str(TRACE_DIR / "preflight_batch2.py"),
                       run_name="__main__")
    except SystemExit as e:  # preflight_batch2 calls sys.exit(0/1)
        rc = int(e.code) if isinstance(e.code, int) else (0 if not e.code else 1)
    except Exception as exc:  # noqa: BLE001
        print(f"[trace_preflight] preflight raised: {exc!r}", file=sys.stderr,
              flush=True)
        rc = 1

    # Mirror the JSON the wrapper expected at --out (some finalize paths key off
    # OUT_JSON existing) so a green run always leaves both files behind.
    art = results_dir / "_batch2_preflight.json"
    if known.out and art.exists():
        try:
            Path(known.out).write_text(art.read_text())
        except OSError:
            pass

    verdict = "PASS" if rc == 0 else "FAIL"
    print(f"[trace_preflight] {verdict} (rc={rc}). Artifact: {art}", flush=True)
    print("[trace_preflight] After pull, commit benchmarks/trace/"
          "_batch2_preflight.json so the scored run's gate authorizes "
          "--crop-batches/--sort-eval-batches for this model + commit.",
          flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
