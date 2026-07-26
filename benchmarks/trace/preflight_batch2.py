#!/usr/bin/env python3
"""Batch 2 real-model equivalence preflight (run on the target A40).

Proves, against the REAL model + tokenizer that a scored run uses, that the two
Batch 2 efficiency transforms are output-preserving, then writes a signed
artifact (``_batch2_preflight.json``) that run.py's scored-run gate requires
before it will launch a matrix with ``--crop-batches`` / ``--sort-eval-batches``.

What it checks (both use the SAME real code paths the scored run uses -- no
parallel reimplementation):

  crop:
    Take real padded batches. Run the model forward on each batch and on its
    ``crop_batch_padding`` version. Assert, on REAL (non-pad) positions:
        max |logits_cropped - logits_full|  <= --tol   (default 1e-5)
        |loss_cropped - loss_full|          <= --tol
    This is the load-bearing check: it catches any CUDA attention-kernel
    padding leakage on THIS model, which is the only reason cropping could
    change results.

  sort_eval:
    Compute evaluate() on an eval loader (a) as-is and (b) after
    sort_eval_loader_by_length + crop. Assert the aggregate accuracy is
    EXACTLY equal (integer correct/total is invariant to batch grouping under
    a correct mask). Uses the real evaluate() from run.py.

Exit code 0 iff every enabled check passes; non-zero otherwise. The artifact is
written either way (passed flag reflects the result) so a failure is on record.

Usage (on the pod, same env as the fleet):
    python preflight_batch2.py \
        --model_id meta-llama/Llama-3.2-3B-Instruct \
        --data_root /workspace/synth/trace/data/TRACE \
        [--tol 1e-5] [--batch_size 8] [--max_len 512] \
        [--n_probe_batches 4] [--out <dir>] [--device cuda] \
        [--only crop|sort_eval]

``--data_root`` is the directory that directly contains the task folders
(ScienceQA, FOMC, ...), i.e. the path returned by ``fetch_data.stage_trace``.
Run it once per (model_id, git commit). run.py binds the artifact to both;
on pods the commit is read from ``SYNTH_GIT_COMMIT`` (dispatcher-stamped).
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Ensure the trace package dir is importable (same layout the pod uses).
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _git_commit() -> str:
    """Resolve the code commit this preflight proves.

    Preference order:
      1. ``SYNTH_GIT_COMMIT`` -- stamped by the dispatcher wrapper from the
         operator's HEAD (pods have no ``.git`` after tar-push).
      2. ``git rev-parse HEAD`` from this file's directory (local hand-runs).
      3. Empty string (gate then matches model_id only, with a warning).
    """
    import os
    env = (os.environ.get("SYNTH_GIT_COMMIT") or "").strip()
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(HERE),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return ""


def _real_token_logits(logits, attention_mask, keep_cols=None):
    """Gather logits at real (mask==1) positions.

    If ``keep_cols`` (bool [seq]) is given, first restrict to those columns so a
    cropped tensor lines up with the corresponding columns of the full tensor.
    Returns a flat [n_real, vocab] tensor, ordered identically for full vs
    cropped as long as keep_cols selects exactly the crop columns.
    """
    if keep_cols is not None:
        logits = logits[:, keep_cols, :]
        attention_mask = attention_mask[:, keep_cols]
    m = attention_mask.bool()
    return logits[m]  # [sum(mask), vocab]


def check_crop(model, sequence, torch, unpack_lm_batch, crop_batch_padding,
               pad_token_id, tol, n_probe_batches, device):
    """Real-model forward equivalence for padding-crop. Returns a result dict."""
    model.eval()
    max_logit_diff = 0.0
    max_loss_diff = 0.0
    n_checked = 0
    n_cropped = 0  # batches that actually had shared padding to drop
    details = []
    with torch.no_grad():
        # Probe eval loaders (they hold the ragged-length batches that exercise
        # cropping hardest). Fall back to train loaders if needed.
        loaders = [ev for (_tr, ev) in sequence] + [tr for (tr, _ev) in sequence]
        for loader in loaders:
            for batch in loader:
                if n_checked >= n_probe_batches:
                    break
                mask = batch["attention_mask"]
                keep = mask.any(dim=0)
                cropped = crop_batch_padding(batch)
                # Only informative when something was actually dropped.
                did_crop = bool((~keep).any())

                ids_f, labels_f, am_f = unpack_lm_batch(
                    batch, device, pad_token_id=pad_token_id)
                ids_c, labels_c, am_c = unpack_lm_batch(
                    cropped, device, pad_token_id=pad_token_id)

                out_f = model(ids_f, labels=labels_f, attention_mask=am_f)
                out_c = model(ids_c, labels=labels_c, attention_mask=am_c)

                keep_dev = keep.to(out_f.logits.device)
                lf = _real_token_logits(out_f.logits, am_f, keep_cols=keep_dev)
                lc = _real_token_logits(out_c.logits, am_c, keep_cols=None)
                # lf and lc must be the same real positions in the same order.
                if lf.shape != lc.shape:
                    details.append({
                        "error": "real-token logit shape mismatch",
                        "full": list(lf.shape), "cropped": list(lc.shape),
                    })
                    return {
                        "passed": False, "reason": "shape mismatch",
                        "details": details,
                    }
                ldiff = float((lf.float() - lc.float()).abs().max().item()) \
                    if lf.numel() else 0.0
                lossdiff = float(abs(out_f.loss.item() - out_c.loss.item()))
                max_logit_diff = max(max_logit_diff, ldiff)
                max_loss_diff = max(max_loss_diff, lossdiff)
                n_checked += 1
                n_cropped += int(did_crop)
                details.append({
                    "seq_full": int(mask.shape[1]),
                    "seq_cropped": int(cropped["attention_mask"].shape[1]),
                    "did_crop": did_crop,
                    "max_logit_diff": ldiff,
                    "loss_diff": lossdiff,
                })
            if n_checked >= n_probe_batches:
                break
    # Require at least one batch that actually dropped shared pad columns;
    # otherwise this is only an identity check and does not prove crop safety.
    passed = (max_logit_diff <= tol) and (max_loss_diff <= tol) \
        and n_checked > 0 and n_cropped > 0
    reason = None
    if n_checked == 0:
        reason = "no batches probed"
    elif n_cropped == 0:
        reason = (
            "no probed batch had shared padding to crop; refuse identity-only "
            "pass (increase n_probe_batches or use more ragged eval data)"
        )
    elif max_logit_diff > tol or max_loss_diff > tol:
        reason = "logit/loss diff exceeded tol"
    return {
        "passed": bool(passed),
        "tol": tol,
        "max_logit_diff": max_logit_diff,
        "max_loss_diff": max_loss_diff,
        "n_batches_checked": n_checked,
        "n_batches_actually_cropped": n_cropped,
        "reason": reason,
        "note": (
            "Real-token logits + loss preserved to <= tol on the real model. "
            "n_batches_actually_cropped>0 is required so the check exercised "
            "real padding removal, not just no-op identity batches."
        ),
        "details": details,
    }


def check_sort_eval(model, sequence, torch, evaluate,
                    sort_eval_loader_by_length, crop_loader_padding,
                    pad_token_id, batch_size, n_probe_tasks):
    """Aggregate-accuracy invariance for length-sorted (+cropped) eval."""
    model.eval()
    max_acc_diff = 0.0
    n_checked = 0
    details = []
    for (_tr, eval_loader) in sequence[:n_probe_tasks]:
        acc_plain = evaluate(model, eval_loader, pad_token_id=pad_token_id)
        sorted_loader = sort_eval_loader_by_length(eval_loader, batch_size)
        sorted_cropped = crop_loader_padding(sorted_loader)
        acc_sorted = evaluate(model, sorted_cropped, pad_token_id=pad_token_id)
        d = abs(acc_plain - acc_sorted)
        max_acc_diff = max(max_acc_diff, d)
        n_checked += 1
        details.append({
            "acc_plain": acc_plain,
            "acc_sorted_cropped": acc_sorted,
            "abs_diff": d,
        })
    # Aggregate accuracy is integer correct/total: it must match EXACTLY.
    passed = (max_acc_diff == 0.0) and n_checked > 0
    return {
        "passed": bool(passed),
        "max_acc_diff": max_acc_diff,
        "n_tasks_checked": n_checked,
        "note": (
            "Aggregate next-token accuracy is invariant to eval batch grouping "
            "under a correct mask; required exact (0.0) diff. Caveat: proves "
            "AGGREGATE only -- not per-example metrics."
        ),
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--data_root", type=Path, required=True,
                    help="dir that directly contains TRACE task folders "
                         "(…/TRACE with ScienceQA/, FOMC/, …)")
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max allowed |logit| and |loss| diff for crop check")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_tasks", type=int, default=2,
                    help="how many TRACE tasks to load for the probe (2 is "
                         "plenty; more only lengthens the preflight)")
    ap.add_argument("--n_probe_batches", type=int, default=4,
                    help="crop: number of real batches to forward-check")
    ap.add_argument("--n_probe_tasks", type=int, default=2,
                    help="sort_eval: number of eval loaders to check")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--only", choices=["crop", "sort_eval"], default=None,
                    help="run only one check (default: both)")
    ap.add_argument("--out", type=Path, default=None,
                    help="dir to write _batch2_preflight.json (default: "
                         "run.py directory so the gate finds it)")
    args = ap.parse_args()

    import torch
    from trace_data import (
        build_trace_sequence, build_real_model,
        crop_batch_padding, crop_loader_padding, sort_eval_loader_by_length,
    )
    from synthetic_myelination import unpack_lm_batch
    from run import evaluate

    print(f"[preflight] loading real model: {args.model_id}", flush=True)
    model, tokenizer = build_real_model(args.model_id, args.device, lora=True)
    pad_token_id = tokenizer.pad_token_id

    print(f"[preflight] building real TRACE sequence "
          f"(n_tasks={args.n_tasks})", flush=True)
    sequence = build_trace_sequence(
        tokenizer, args.data_root,
        n_train=args.batch_size * 8,   # small train slice; crop check is forward-only
        n_eval=args.batch_size * 16,
        max_len=args.max_len,
        batch_size=args.batch_size,
        tasks=None,
    )[:args.n_tasks]

    checks = {}
    do_crop = args.only in (None, "crop")
    do_sort = args.only in (None, "sort_eval")

    if do_crop:
        print("[preflight] crop: real-model forward equivalence ...", flush=True)
        checks["crop"] = check_crop(
            model, sequence, torch, unpack_lm_batch, crop_batch_padding,
            pad_token_id, args.tol, args.n_probe_batches, args.device)
        c = checks["crop"]
        print(f"[preflight] crop passed={c['passed']} "
              f"max_logit_diff={c['max_logit_diff']:.3e} "
              f"max_loss_diff={c['max_loss_diff']:.3e} "
              f"cropped_batches={c['n_batches_actually_cropped']}/"
              f"{c['n_batches_checked']}", flush=True)

    if do_sort:
        print("[preflight] sort_eval: aggregate-accuracy invariance ...",
              flush=True)
        checks["sort_eval"] = check_sort_eval(
            model, sequence, torch, evaluate,
            sort_eval_loader_by_length, crop_loader_padding,
            pad_token_id, args.batch_size, args.n_probe_tasks)
        s = checks["sort_eval"]
        print(f"[preflight] sort_eval passed={s['passed']} "
              f"max_acc_diff={s['max_acc_diff']:.3e} "
              f"tasks={s['n_tasks_checked']}", flush=True)

    overall = all(v.get("passed", False) for v in checks.values()) \
        and len(checks) > 0

    artifact = {
        "artifact": "batch2_preflight",
        "passed": bool(overall),
        "model_id": args.model_id,
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tol": args.tol,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "device": args.device,
        "checks": checks,
        "gate_note": (
            "run.py's _check_batch2_preflight requires passed=true, a matching "
            "model_id + git_commit, and checks.<transform>.passed=true for each "
            "enabled transform before a scored run may use it."
        ),
    }

    out_dir = args.out or HERE
    out_dir.mkdir(parents=True, exist_ok=True)
    art_path = out_dir / "_batch2_preflight.json"
    art_path.write_text(json.dumps(artifact, indent=2))
    print(f"[preflight] {'PASS' if overall else 'FAIL'} -> {art_path}",
          flush=True)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
