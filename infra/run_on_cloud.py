#!/usr/bin/env python3
"""Multi-cloud GPU dispatcher for synthetic-myelination benchmarks.

This is the provider-agnostic orchestrator.  Cloud-specific control planes
(create instance, poll for ready, get SSH endpoint, destroy) live behind the
``cloud_providers.CloudProvider`` plugin interface; provider-specific code
sits in ``scripts/cloud_providers/<name>.py``.

End-to-end pipeline:

  1. Parse CLI flags; look up the requested provider plugin.
  2. Stage the per-launch directory under
     ``<PROJECT_ROOT>/lab_reports/_staging/[_smokes/]<benchmark>_results_<utc>/``
     and self-tee stdout/stderr into ``dispatcher.log`` there.
  3. Load the benchmark's ``bench_caps.json`` (if present) and filter the
     operator-supplied GPU candidate list down to entries that meet the
     declared requirements.  Drop sm_120 (Blackwell) cards if the default
     image's PyTorch wheel doesn't support them.
  4. For each surviving candidate in order, call ``provider.create``.
     ``CapacityError`` is a soft failure (try the next candidate);
     ``AuthError`` / ``BalanceError`` are hard aborts.
  5. Write ``pod_provisioning.json`` (provider + instance provenance + run
     metadata) alongside ``dispatcher.log`` in the staging dir.
  6. ``provider.wait_for_ready``; ``push_code`` via tar-over-SSH; launch the
     remote wrapper under nohup; poll for completion via the ``_DONE.<rc>``
     sentinel; final-pull on completion.
  7. ``provider.destroy`` on ANY sentinel-confirmed exit (rc==0 or rc!=0):
     once the sentinel is written and ``run.log`` + ``results/`` are pulled,
     the pod has zero remaining diagnostic value and is just idling on the
     billing meter.  The instance is kept alive ONLY when the run never
     produced a sentinel (KeyboardInterrupt, poll-loop SSH-failure abort) —
     the remote python may still be mid-execution and destroying it would
     discard whatever's been computed but not pulled.  Pass ``--keep-on-fail``
     to also keep on non-zero sentinel (for the rare non-reproducible bug
     where live SSH inspection is genuinely needed).

Usage::

    # Default provider (runpod), default GPU candidate list, smoke run:
    python3 scripts/run_on_cloud.py --benchmark 07_gradient_aligned_m --smoke

    # Try Vultr first, fall back to Lambda Labs if Vultr is out of capacity
    # (multi-provider fallback is a future feature; for now operators run
    # the dispatcher twice with different --provider values):
    python3 scripts/run_on_cloud.py --benchmark gpt2_v1 \\
        --provider vultr --gpu-type A40,A6000,L40S

    # Headline run on RunPod SECURE with a specific candidate list:
    python3 scripts/run_on_cloud.py --benchmark 07_gradient_aligned_m \\
        --provider runpod --cloud SECURE \\
        --gpu-type A100-80GB,A40,L40S,RTX-6000-ADA
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from coolname import generate_slug

# Make `cloud_providers` importable when the script is run directly from the
# scripts/ directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from cloud_providers import (
    AuthError,
    BalanceError,
    CapacityError,
    CloudInstance,
    CloudProvider,
    GPU_CANONICAL,
    ProvisioningError,
    UnknownGpuError,
    get_provider,
    gpu_supports_pytorch_2_4,
    list_providers,
)
from cloud_providers._b2_fetch import (
    B2CredentialsMissing,
    B2FetchConfig,
    B2FetchResult,
    ManifestNotFound,
    fetch_and_decrypt,
)
from cloud_providers._bench_caps import (
    BenchCaps,
    gpu_meets_caps,
    load_bench_caps,
)
from cloud_providers._ssh import (
    REMOTE_BASE,
    FinalizePayloadError,
    _TeeWriter,
    poll_until_done,
    pull_results_into,
    push_code,
    push_finalize_payload,
    remote_paths_for,
    run_remote_experiment,
    wait_for_remote_finalize,
)


PROJECT_ROOT = _SCRIPT_DIR.parent
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# ---------------------------------------------------------------------------
# Staging layout (matches scripts/finalize_run.py's expectations).
# ---------------------------------------------------------------------------
DEFAULT_STAGING_DIR = PROJECT_ROOT / "lab_reports" / "_staging"
SMOKE_STAGING_SUBDIR = "_smokes"

_LAUNCH_DIR_SLUG_RETRIES = 5  # 1-in-10^7 collision rate per draw; 5 draws is paranoia


def _create_unique_launch_dir(
    *,
    launch_root: Path,
    benchmark: str,
    stamp: str,
) -> Path:
    """Create a per-launch staging directory with a unique-by-construction name.

    Returns ``launch_root / f'{benchmark}_results_{stamp}_{slug}'`` where
    ``slug`` is a coolname adjective-noun pair (e.g. ``serene-dragon``).

    Uses ``mkdir(parents=True, exist_ok=False)`` so two dispatchers landing
    in the same UTC second cannot silently share a directory (the legacy
    ``exist_ok=True`` behavior was the root cause of the 2026-05-26 PP20
    watcher iter-14 race: three concurrent dispatchers collapsed onto one
    staging dir, only one's CLI args won, and the others' nohup logs all
    parsed the same shared dir as 'in-flight').

    With ~10^7 unique adjective-noun slug pairs in the coolname corpus, the
    odds of two simultaneous dispatchers drawing the same ``stamp + slug``
    are ~1 in 10^7 per attempt; on the rare collision we just draw a fresh
    slug.  Five retries is far beyond defensive.

    Raises ``RuntimeError`` only if every retry collides -- effectively
    impossible in practice; documented for paranoia.
    """
    last_exc: Optional[FileExistsError] = None
    for _ in range(_LAUNCH_DIR_SLUG_RETRIES):
        slug = generate_slug(2)
        candidate = launch_root / f"{benchmark}_results_{stamp}_{slug}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"Failed to create a unique staging dir under {launch_root} after "
        f"{_LAUNCH_DIR_SLUG_RETRIES} coolname slug draws (this should be "
        f"statistically impossible).  Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def _filter_candidates(
    raw_candidates: List[str],
    *,
    caps: BenchCaps,
    image_supports_pytorch_2_4: bool = True,
) -> List[str]:
    """Apply image-compatibility + bench_caps filters to a candidate list.

    Returns the filtered list in the original order.  Drops:
      * GPUs the benchmark's bench_caps.json declares incompatible.
      * Blackwell (sm_120) cards when the default image is PyTorch 2.4.0.
    Unknown canonical names pass through unchanged; the provider plugin
    will raise UnknownGpuError if it doesn't support them.
    """
    out: List[str] = []
    for cand in raw_candidates:
        spec = GPU_CANONICAL.get(cand)
        if spec is None:
            # We don't know this canonical name; let the provider decide.
            out.append(cand)
            continue
        if image_supports_pytorch_2_4 and not gpu_supports_pytorch_2_4(cand):
            print(
                f"!! Filtering out {cand!r}: compute capability "
                f"{spec.compute_capability} is not supported by the default "
                f"PyTorch 2.4.0 image.  Upgrade --image to a PyTorch 2.5+ "
                f"build to use this GPU.",
                file=sys.stderr,
            )
            continue
        if not gpu_meets_caps(spec, caps):
            print(
                f"!! Filtering out {cand!r}: does not meet bench_caps "
                f"(VRAM {spec.vram_gb} GB, compute {spec.compute_capability}; "
                f"benchmark needs >= {caps.min_vram_gb} GB / "
                f"{caps.required_compute_capability or 'any'}).",
                file=sys.stderr,
            )
            continue
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Provider selection: try candidates in order until one accepts
# ---------------------------------------------------------------------------

def _try_create(
    provider: CloudProvider,
    *,
    candidates: List[str],
    image: str,
    label: str,
    gpu_count: int,
    container_disk_gb: int,
    volume_gb: int,
    regions: Optional[List[str]],
) -> Optional[CloudInstance]:
    """Walk the candidate list, returning the first successfully-created instance.

    ``AuthError`` / ``BalanceError`` abort the loop with ``sys.exit(2)``;
    ``CapacityError`` / ``UnknownGpuError`` are soft failures (skip to next).
    """
    last_error: Optional[str] = None
    for canonical in candidates:
        if canonical in provider.approval_gated_gpus:
            print(
                f"!! Warning: {canonical!r} typically requires capacity "
                f"approval on {provider.name}.  If creation fails, the "
                f"dispatcher will try the next entry.",
                file=sys.stderr,
            )
        print(f"    trying canonical={canonical!r} ...", flush=True)
        try:
            return provider.create(
                canonical_gpu=canonical,
                image=image,
                name=label,
                gpu_count=gpu_count,
                container_disk_gb=container_disk_gb,
                volume_gb=volume_gb,
                regions=regions,
            )
        except UnknownGpuError as exc:
            print(f"    -> not offered: {exc}", flush=True)
            last_error = str(exc)
            continue
        except CapacityError as exc:
            print(f"    -> no capacity: {exc}", flush=True)
            last_error = str(exc)
            continue
        except (AuthError, BalanceError) as exc:
            print(f"\n!! {type(exc).__name__}: {exc}", file=sys.stderr)
            sys.exit(2)
    print(
        f"\n!! No GPU candidate had capacity on {provider.name}.",
        file=sys.stderr,
    )
    if last_error:
        print(f"   Last error: {last_error}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Destruction policy decision (pure function, separately unit-tested)
# ---------------------------------------------------------------------------

def _should_keep_alive(
    *,
    run_completed: bool,
    rc: int,
    keep_on_fail: bool,
    remote_may_be_running: bool = True,
) -> bool:
    """Return True iff the dispatcher should leave the instance running.

    Decision table:
      * ``run_completed=False, remote_may_be_running=True``  -> keep alive.
        The remote python may still be mid-execution (KeyboardInterrupt
        after launch, or poll-loop SSH-failure abort). Destroying would
        discard in-progress work that has not yet been pulled.
      * ``run_completed=False, remote_may_be_running=False`` -> destroy.
        Pre-run failure: wait_for_ready timeout, push_code failure,
        remote_finalize setup error, etc. Nothing useful is running on
        the pod; keeping it alive is pure orphan billing.
      * ``run_completed=True, rc==0``                        -> destroy.
        Success path: results pulled, nothing more to do.
      * ``run_completed=True, rc!=0, keep_on_fail=False``    -> destroy.
        Sentinel-confirmed error exit; dispatcher already pulled logs.
      * ``run_completed=True, rc!=0, keep_on_fail=True``     -> keep alive.
        Explicit opt-in for live SSH debugging of a failed run.
    """
    if not run_completed:
        return bool(remote_may_be_running)
    return rc != 0 and keep_on_fail


# ---------------------------------------------------------------------------
# Post-poll handlers (shared between fresh-launch and --reattach paths)
# ---------------------------------------------------------------------------

def _print_summary(results_dir: Optional[Path]) -> None:
    """Print the SUMMARY block from ``results/results_summary.txt``.

    Accepts either the launch staging dir (SCP contract:
    ``<launch>/results/results_summary.txt``) or the B2 pivot dest
    (``<launch>/results/`` directly), so both recovery paths print the
    same summary.
    """
    print("\n==> SUMMARY")
    if results_dir is None:
        print("(no results_dir; run never started or aborted before producing one)")
        return
    candidates = [
        results_dir / "results" / "results_summary.txt",
        results_dir / "results_summary.txt",
    ]
    summary_path = next((p for p in candidates if p.exists()), None)
    if summary_path is not None:
        print(summary_path.read_text())
    else:
        print(f"(no results_summary.txt under {results_dir})")
    print(f"\n==> All results in: {results_dir}")


# ---------------------------------------------------------------------------
# Remote-finalize wiring (pod-side encrypt + upload + self-terminate)
# ---------------------------------------------------------------------------

# Env vars consumed when bench_caps.remote_finalize is True.  All must be
# present or the dispatcher refuses to enable remote finalize.  Sourced
# from the operator's ``.env`` via the existing Makefile include line.
# RunPod-only: self-terminate uses the RunPod REST API.
_REMOTE_FINALIZE_ENV_KEYS = (
    "RUNPOD_API_KEY",
    "B2_KEY_ID_POD",
    "B2_KEY_POD",
    "B2_BUCKET",
)
_DEFAULT_PUBKEY_PATH = PROJECT_ROOT / "infra" / "keys" / "synth-results.pub"
_DESTROY_RETRIES = 3
_DESTROY_RETRY_SLEEP_S = 2.0
_REMOTE_FINALIZE_WAIT_S = 180


def _build_remote_finalize_config(*, launch_dir: Path, pod_id: str) -> dict:
    """Assemble the dict consumed by ``_ssh.build_remote_wrapper``.

    Reads the required secrets from the process environment (which the
    Makefile populates from ``.env``).  Uses the launch directory's
    basename as the ``run_id``, which keeps the B2 layout
    (``<bench>/<run_id>/results.synth``) aligned with the local staging
    layout (``lab_reports/_staging/<run_id>/``) for one-to-one mapping
    during reattach.  The ``pod_id`` is forwarded into the wrapper as
    ``RUNPOD_POD_ID`` so ``pod_finalize.py`` can self-terminate the
    pod on the RunPod side after the B2 upload (RunPod does NOT inject
    this env var into SSH login sessions, so the dispatcher has to
    forward it explicitly).

    Raises ``RuntimeError`` listing every missing env var so the
    operator can fix .env in a single pass.
    """
    missing = [k for k in _REMOTE_FINALIZE_ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "remote_finalize: missing required env vars: "
            + ", ".join(missing)
            + ".  Populate .env (or infra/.env.template) and re-source."
        )
    if not pod_id:
        raise RuntimeError(
            "remote_finalize: empty pod_id; remote finalize is RunPod-only "
            "and requires a live pod id from provider.create()."
        )
    return {
        "run_id": launch_dir.name,
        "b2_bucket": os.environ["B2_BUCKET"],
        "b2_key_id": os.environ["B2_KEY_ID_POD"],
        "b2_key": os.environ["B2_KEY_POD"],
        "runpod_api_key": os.environ["RUNPOD_API_KEY"],
        "pod_id": pod_id,
    }


def _maybe_push_finalize_payload(
    *,
    caps: BenchCaps,
    provider: CloudProvider,
    instance: CloudInstance,
    ssh_target: List[str],
    pubkey_path: Path,
    launch_dir: Path,
) -> Optional[dict]:
    """Push finalize payload and return remote_finalize cfg, or None.

    Remote finalize is RunPod-only (self-terminate uses RunPod REST +
    ``pod_id``).  When bench_caps opts in on a non-RunPod provider, we
    log and fall back to the legacy SCP-pull path rather than crashing
    after the pod is already billing.
    """
    if not caps.remote_finalize:
        return None
    if getattr(provider, "name", None) != "runpod":
        print(
            f"!! bench_caps.remote_finalize=True but provider="
            f"{getattr(provider, 'name', '?')!r}; remote finalize is "
            f"RunPod-only. Falling back to SCP-pull + dispatcher destroy.",
            flush=True,
        )
        return None
    pod_id = getattr(instance, "pod_id", None)
    if not pod_id:
        print(
            "!! remote_finalize requested but instance has no pod_id; "
            "falling back to SCP-pull + dispatcher destroy.",
            flush=True,
        )
        return None
    print(
        f"==> bench_caps.remote_finalize=True; pushing finalize payload "
        f"(pubkey={pubkey_path})",
        flush=True,
    )
    push_finalize_payload(
        project_root=PROJECT_ROOT,
        ssh_target=ssh_target,
        pubkey_path=pubkey_path,
    )
    return _build_remote_finalize_config(launch_dir=launch_dir, pod_id=str(pod_id))


def _instance_id_for_orphan(instance: CloudInstance) -> str:
    """Best-effort stable id for ORPHAN_POD sidecars across providers."""
    for attr in ("pod_id", "instance_id"):
        val = getattr(instance, attr, None)
        if val:
            return str(val)
    try:
        prov = instance.provenance()
        for key in ("pod_id", "instance_id", "id"):
            if prov.get(key):
                return str(prov[key])
    except Exception:
        pass
    return instance.label()


def _write_orphan_sidecar(
    *,
    launch_dir: Optional[Path],
    instance: CloudInstance,
    provider: CloudProvider,
    reason: str,
) -> None:
    """Persist orphan metadata so ``make reap`` / operators can find leaks."""
    if launch_dir is None:
        return
    payload = {
        "provider": getattr(provider, "name", None),
        "instance_id": _instance_id_for_orphan(instance),
        "label": instance.label(),
        "reason": reason,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": None,
    }
    try:
        payload["provenance"] = instance.provenance()
    except Exception as exc:
        payload["provenance_error"] = f"{type(exc).__name__}: {exc}"
    path = launch_dir / "ORPHAN_POD.json"
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
        print(f"!! Wrote orphan sidecar: {path}", flush=True)
    except OSError as exc:
        print(f"!! Failed to write ORPHAN_POD.json: {exc}", flush=True)


def _destroy_with_retry(
    *,
    provider: CloudProvider,
    instance: CloudInstance,
    launch_dir: Optional[Path],
    reason: str,
) -> bool:
    """Retry provider.destroy; write ORPHAN_POD.json on persistent failure.

    Returns True iff destroy reported success.
    """
    last_ok = False
    for attempt in range(1, _DESTROY_RETRIES + 1):
        try:
            result = provider.destroy(instance)
            # Providers may return None (legacy) or bool.
            last_ok = True if result is None else bool(result)
        except Exception as exc:
            last_ok = False
            print(
                f"!! destroy attempt {attempt}/{_DESTROY_RETRIES} raised "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        if last_ok:
            if attempt > 1:
                print(f"==> destroy succeeded on attempt {attempt}", flush=True)
            return True
        if attempt < _DESTROY_RETRIES:
            time.sleep(_DESTROY_RETRY_SLEEP_S * attempt)
    _write_orphan_sidecar(
        launch_dir=launch_dir,
        instance=instance,
        provider=provider,
        reason=f"destroy_failed_after_{_DESTROY_RETRIES}_retries: {reason}",
    )
    print(
        "!! DESTROY FAILED after retries — instance may still be billing. "
        f"See ORPHAN_POD.json under {launch_dir}.",
        flush=True,
    )
    return False


def _decide_and_destroy(
    *,
    rc: int,
    run_completed: bool,
    instance: CloudInstance,
    provider: CloudProvider,
    keep_on_fail: bool,
    results_dir: Optional[Path],
    remote_may_be_running: bool = True,
    launch_dir: Optional[Path] = None,
    skip_destroy: bool = False,
) -> bool:
    """Apply the destruction policy.  Returns True iff instance was destroyed.

    Pure routing on top of :func:`_should_keep_alive`; the side effects
    (print + ``provider.destroy``) live here so the routing decision stays
    independently unit-testable.
    """
    if skip_destroy:
        print(
            "==> Skipping dispatcher destroy (remote_finalize already "
            "self-terminated or _UPLOADED confirmed; pod expected gone).",
            flush=True,
        )
        return True
    keep_alive = _should_keep_alive(
        run_completed=run_completed,
        rc=rc,
        keep_on_fail=keep_on_fail,
        remote_may_be_running=remote_may_be_running,
    )
    if keep_alive:
        if not run_completed and remote_may_be_running:
            reason = "no sentinel — remote may still be running"
        elif not run_completed:
            reason = "no sentinel but remote was never launched (should destroy)"
        else:
            reason = f"rc={rc} and --keep-on-fail set"
        print(f"!! INSTANCE LEFT ALIVE ({reason}).")
        print(f"   SSH: {instance.label()}")
        if results_dir is not None:
            print(f"   Partial results so far: {results_dir}")
        print("   To destroy manually, see the provider plugin "
              "documentation or your provider's console.")
        if not run_completed and remote_may_be_running:
            _write_orphan_sidecar(
                launch_dir=launch_dir,
                instance=instance,
                provider=provider,
                reason="kept_alive_remote_may_be_running",
            )
        return False
    ok = _destroy_with_retry(
        provider=provider,
        instance=instance,
        launch_dir=launch_dir,
        reason=f"rc={rc} run_completed={run_completed}",
    )
    if ok:
        outcome = "success" if rc == 0 else f"failure rc={rc}"
        if not run_completed:
            outcome = "pre-run failure (nothing useful running)"
        print(f"==> Instance destroyed ({outcome}).  Billing stopped.")
    return ok


def _maybe_auto_finalize(
    *,
    rc: int,
    destroyed: bool,
    launch_dir: Path,
    benchmark: str,
    slug: Optional[str],
    operator: Optional[str],
    ledger_description: Optional[str],
) -> None:
    """If ``--auto-finalize-slug`` was set and the run succeeded + destroyed,
    subprocess-invoke ``scripts/finalize_run.py`` to materialize a partial
    lab report into ``lab_reports/<date>_<HHMM>_<slug>/``.

    Skipped on:
      * No slug provided (the flag is the opt-in toggle).
      * ``rc != 0`` (finalize_run.py would refuse to render a failed run
        as a "headline"-shaped report; manual curation required).
      * Instance kept alive (the staging dir's results/ may still be
        growing; finalize would move incomplete data).
      * Operator unset (finalize_run.py requires it; we don't pick a
        default to avoid silent attribution errors).

    Failures of the chained finalize do NOT change the dispatcher's exit
    code.  The benchmark itself succeeded; the operator can rerun
    ``finalize_run.py`` manually with the same args printed below.
    """
    if not slug:
        return
    if rc != 0:
        print(f"!! Skipping --auto-finalize: rc={rc} (only run on rc==0).",
              flush=True)
        return
    if not destroyed:
        print("!! Skipping --auto-finalize: instance was kept alive "
              "(results may still be growing).", flush=True)
        return
    if not operator:
        print("!! --auto-finalize-slug set but --auto-finalize-operator "
              "missing; skipping auto-finalize.", file=sys.stderr)
        return
    cmd: List[str] = [
        sys.executable,
        str(_SCRIPT_DIR / "finalize_run.py"),
        "--results-dir", str(launch_dir / "results"),
        "--benchmark", benchmark,
        "--slug", slug,
        "--operator", operator,
    ]
    if ledger_description:
        cmd.extend(["--ledger-description", ledger_description])
    print(f"\n==> Auto-finalize: {' '.join(cmd)}", flush=True)
    finalize_rc = subprocess.run(cmd, check=False).returncode
    if finalize_rc == 0:
        print("==> Auto-finalize succeeded.", flush=True)
    else:
        print(f"!! Auto-finalize failed with rc={finalize_rc}; "
              f"rerun manually:\n   {' '.join(cmd)}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Re-attach flow: skip provisioning/push/launch, jump straight to poll
# ---------------------------------------------------------------------------

def _try_b2_pivot(
    *,
    benchmark: str,
    launch_dir: Path,
    original_error: Exception,
) -> Optional[B2FetchResult]:
    """Attempt the B2 reattach pivot when the pod is gone.

    Returns the :class:`B2FetchResult` on success, ``None`` when the
    pivot is not configured or the manifest doesn't exist.  Re-raises
    :class:`ManifestVerificationError` / :class:`CiphertextNotFound` as
    HARD errors -- those mean the manifest exists but the data is
    untrustworthy or has been tampered with, and the operator must
    investigate (the original 'pod gone' error is no longer the
    interesting failure).

    The ``run_id`` is the launch directory's basename; this matches the
    convention :func:`_build_remote_finalize_config` uses on the
    fresh-launch path so the B2 layout is round-trippable.
    """
    print(
        "==> Pod is gone; attempting B2 pivot "
        f"(b2://<bucket>/{benchmark}/{launch_dir.name}/manifest.json)",
        flush=True,
    )
    try:
        config = B2FetchConfig.from_env(
            bench=benchmark,
            run_id=launch_dir.name,
        )
    except B2CredentialsMissing as exc:
        print(f"   B2 pivot not configured: {exc}", flush=True)
        print(f"   Original error: {type(original_error).__name__}: "
              f"{original_error}", flush=True)
        return None

    try:
        result = fetch_and_decrypt(config, dest_dir=launch_dir)
    except ManifestNotFound as exc:
        print(f"   {exc}", flush=True)
        return None

    print(
        f"==> B2 pivot succeeded: extracted {len(result.extracted_files)} "
        f"files into {result.dest_dir} "
        f"(ciphertext {result.ciphertext_size} bytes; "
        f"completion_utc={result.manifest.get('completion_utc')!r})",
        flush=True,
    )
    return result


# Default retry budget for the fresh-launch B2 pivot.  Sized to give
# pod_finalize.py enough wall-time to: tar+encrypt the results dir
# (sub-second at smoke scale, up to ~10s at production), upload the
# ciphertext (small payloads <1MB are sub-second, large ones tens of
# seconds at A100 egress speeds), upload the manifest (always tiny),
# and HEAD-check both.  Total expected: <60s; budget gives 3x headroom.
_FRESH_LAUNCH_B2_PIVOT_RETRIES = 4
_FRESH_LAUNCH_B2_PIVOT_DELAY_S = 30


def _try_b2_pivot_with_retry(
    *,
    benchmark: str,
    launch_dir: Path,
    original_error: Exception,
    retries: int = _FRESH_LAUNCH_B2_PIVOT_RETRIES,
    delay_s: float = _FRESH_LAUNCH_B2_PIVOT_DELAY_S,
    sleep_fn=time.sleep,
) -> Optional[B2FetchResult]:
    """Like :func:`_try_b2_pivot` but tolerates pod-still-finalizing.

    On the fresh-launch path the SSH-failure-budget can be exhausted
    BEFORE ``pod_finalize.py`` finishes its encrypt + upload sequence
    (sshd may go away when the wrapper exits, while pod_finalize is
    still running its terminate-after-upload sequence).  In that
    window ``manifest.json`` doesn't exist yet — which
    :func:`_try_b2_pivot` reports as ``None``.  This wrapper retries
    that case up to ``retries`` times with ``delay_s`` between attempts,
    so the dispatcher waits the bounded amount of wall-time the pod
    needs to commit its results to B2.

    Returns the populated :class:`B2FetchResult` on success, ``None``
    only after every retry returned no manifest (genuine pod-stuck
    case — caller should keep the instance alive for investigation).
    Re-raises hard errors (``ManifestVerificationError``,
    ``CiphertextNotFound``, ``B2CredentialsMissing``) unchanged.
    """
    last_err: Exception = original_error
    for attempt in range(1, retries + 1):
        try:
            pivot = _try_b2_pivot(
                benchmark=benchmark,
                launch_dir=launch_dir,
                original_error=last_err,
            )
        except B2CredentialsMissing:
            # _try_b2_pivot already prints + returns None on this; but
            # if it ever raises (future change), surface it immediately
            # rather than burning the retry budget on a misconfig.
            raise
        if pivot is not None:
            return pivot
        if attempt < retries:
            print(
                f"   B2 manifest not yet present (attempt {attempt}/{retries}); "
                f"waiting {delay_s:.0f}s for pod_finalize to commit.",
                flush=True,
            )
            sleep_fn(delay_s)
    return None


def _write_b2_pivot_done_sentinel(launch_dir: Path) -> None:
    """Write ``_DONE.0`` into the launch dir's ``results/`` after a
    successful fresh-launch B2 pivot.

    The fresh-launch path's polling loop never observed ``_DONE.<rc>``
    over SSH (the pod went away first), so downstream consumers that
    gate on the sentinel — the overnight-dispatch watcher, manual
    ``ls results/_DONE.*`` checks — would otherwise see a missing
    sentinel.  The B2 tarball already contained an authentic
    ``_DONE.0`` written by the wrapper on the pod side; this writes
    one again in case the operator inspects the launch dir before the
    pivot's untar (the untar will overwrite it identically).
    """
    results_dir = launch_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    sentinel = results_dir / "_DONE.0"
    if not sentinel.exists():
        sentinel.write_text("")


def _handle_pod_gone(
    *,
    args: argparse.Namespace,
    benchmark: str,
    launch_dir: Path,
    original_error: Exception,
) -> None:
    """Pod-gone branch of the reattach flow.

    Tries the B2 pivot; on success treats the run as a normal rc=0
    completion (results extracted into the launch dir from B2, pod is
    already gone so destroy is a no-op), runs auto-finalize if
    requested, and exits 0.  On failure re-prints the original
    'pod gone' error and exits 2.

    Always terminates the process via :func:`sys.exit`; never returns.
    """
    pivot = _try_b2_pivot(
        benchmark=benchmark,
        launch_dir=launch_dir,
        original_error=original_error,
    )
    if pivot is None:
        print(
            f"\n!! Reattach failed: {type(original_error).__name__}: "
            f"{original_error}",
            file=sys.stderr,
        )
        sys.exit(2)
    _maybe_auto_finalize(
        rc=0,
        destroyed=True,
        launch_dir=launch_dir,
        benchmark=benchmark,
        slug=args.auto_finalize_slug,
        operator=args.auto_finalize_operator,
        ledger_description=args.auto_finalize_ledger_description,
    )
    sys.exit(0)


def _run_reattach(args: argparse.Namespace) -> None:
    """Re-attach to a launch directory whose dispatcher died.

    Reads ``pod_provisioning.json`` from ``args.reattach``, instantiates
    the provider plugin, reconstructs the live :class:`CloudInstance` via
    :meth:`CloudProvider.attach_from_provisioning`, jumps directly into
    :func:`poll_until_done` against the same remote paths the launch path
    would have used, and runs the same destroy + auto-finalize sequence.

    When the pod is GONE (typically because ``pod_finalize`` self-
    terminated it after a successful encrypt + B2 upload), the SSH
    path is unavailable; in that case the dispatcher pivots to
    :func:`_try_b2_pivot` to recover the run from B2 instead.

    Calls ``sys.exit()`` directly; the caller (``main``) should return
    immediately after invoking this.
    """
    launch_dir = args.reattach.resolve()
    if not launch_dir.is_dir():
        print(f"ERROR: --reattach launch dir does not exist: {launch_dir}",
              file=sys.stderr)
        sys.exit(2)
    prov_path = launch_dir / "pod_provisioning.json"
    if not prov_path.is_file():
        print(f"ERROR: missing {prov_path}; cannot reattach.",
              file=sys.stderr)
        sys.exit(2)
    try:
        prov = json.loads(prov_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"ERROR: {prov_path} is not valid JSON: {exc}",
              file=sys.stderr)
        sys.exit(2)
    benchmark = prov.get("benchmark")
    if not benchmark:
        print(f"ERROR: {prov_path} missing 'benchmark' field; cannot reattach.",
              file=sys.stderr)
        sys.exit(2)
    provider_name = prov.get("provider") or args.provider

    # Tee dispatcher.log in APPEND mode so the reattach session is appended
    # below the original dispatcher's output rather than overwriting it.
    # The boundary marker below makes the resume point unambiguous in logs.
    dispatcher_log_path = launch_dir / "dispatcher.log"
    dispatcher_log_f = dispatcher_log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, dispatcher_log_f)
    sys.stderr = _TeeWriter(sys.__stderr__, dispatcher_log_f)
    print(
        f"\n==> Reattached at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        flush=True,
    )
    print(f"    launch_dir   = {launch_dir}", flush=True)
    print(f"    benchmark    = {benchmark}", flush=True)
    print(f"    provider     = {provider_name}", flush=True)
    print(f"    pod_id       = {prov.get('pod_id')!r}", flush=True)

    if shutil.which("ssh") is None:
        print("ERROR: ssh must be on PATH", file=sys.stderr)
        sys.exit(2)

    # Reconstruct provider plugin using the cloud_type the original launch
    # was provisioned with (RunPod-specific; other providers ignore it).
    provider_opts = {}
    if prov.get("cloud_type"):
        provider_opts["cloud_type"] = prov["cloud_type"]
    try:
        provider: CloudProvider = get_provider(provider_name, **provider_opts)
    except (AuthError, ValueError) as exc:
        print(f"\n!! Provider {provider_name!r} init failed: {exc}",
              file=sys.stderr)
        sys.exit(2)

    # Reconstruct the live CloudInstance + refresh transient SSH endpoint
    # from current API state.  This is the only network round-trip the
    # reattach path makes before entering the poll loop -- unless the
    # pod is gone, in which case _handle_pod_gone pivots to B2.
    try:
        instance = provider.attach_from_provisioning(prov)
    except NotImplementedError as exc:
        print(f"\n!! {exc}", file=sys.stderr)
        sys.exit(2)
    except ProvisioningError as exc:
        _handle_pod_gone(
            args=args, benchmark=benchmark,
            launch_dir=launch_dir, original_error=exc,
        )
    except Exception as exc:
        print(f"\n!! Reattach failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        sys.exit(2)
    print(f"    SSH ready    = {instance.label()}", flush=True)

    # Use the same deterministic remote paths the launch path would have.
    paths = remote_paths_for(benchmark)

    rc = 1
    results_dir: Optional[Path] = None
    run_completed = False
    # Reattach implies the remote was already launched.
    remote_may_be_running = True
    skip_destroy = False
    try:
        rc, results_dir = poll_until_done(
            instance.ssh_target(),
            benchmark,
            log_path=paths["log"],
            done_glob=paths["done_glob"],
            dest=launch_dir,
            poll_interval_s=args.poll_interval,
        )
        run_completed = rc >= 0
        if rc != 0:
            print(f"!! Remote experiment exited with code {rc}",
                  file=sys.stderr)
        _print_summary(results_dir)
    except KeyboardInterrupt:
        print(
            "\n!! Re-attach interrupted by operator.  Instance left alive "
            "so the remote run can finish; rerun --reattach (or destroy "
            "manually) when you decide.",
            file=sys.stderr,
        )
    finally:
        destroyed = _decide_and_destroy(
            rc=rc,
            run_completed=run_completed,
            instance=instance,
            provider=provider,
            keep_on_fail=args.keep_on_fail,
            results_dir=results_dir,
            remote_may_be_running=remote_may_be_running,
            launch_dir=launch_dir,
            skip_destroy=skip_destroy,
        )
        _maybe_auto_finalize(
            rc=rc,
            destroyed=destroyed,
            launch_dir=launch_dir,
            benchmark=benchmark,
            slug=args.auto_finalize_slug,
            operator=args.auto_finalize_operator,
            ledger_description=args.auto_finalize_ledger_description,
        )

    if rc == 0:
        sys.exit(0)
    sys.exit(rc if rc > 0 else 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    available_providers = list_providers()
    parser = argparse.ArgumentParser(
        description="Run a benchmark on a fresh GPU instance from any "
                    "registered cloud provider.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider", "-p", default="runpod", choices=available_providers,
        help=f"Cloud provider plugin to use (default: runpod). "
             f"Registered: {available_providers}.",
    )
    parser.add_argument(
        "--benchmark", "-b", default=None,
        help="Benchmark directory name under benchmarks/ "
             "(e.g. gpt2_v1, 07_gradient_aligned_m).  Required for fresh "
             "launches; optional with --reattach (read from "
             "pod_provisioning.json instead).",
    )
    parser.add_argument(
        "--gpu-type", default=None,
        help="Comma-separated list of canonical GPU names (e.g. "
             "A40,A6000,L40S), tried in order until one has capacity. "
             "Defaults to the benchmark's bench_caps.json preferred_gpus, "
             "or to the provider's default GPU if that is also absent. "
             "Canonical names: "
             f"{', '.join(sorted(GPU_CANONICAL.keys()))}.",
    )
    parser.add_argument(
        "--gpu-count", type=int, default=1,
        help="Number of GPUs per instance (default 1).",
    )
    parser.add_argument(
        "--image", default=None,
        help="Provider-appropriate image tag.  Defaults to "
             "<provider>.default_image.",
    )
    parser.add_argument(
        "--cloud", default=None,
        help="Provider-specific cloud tier.  RunPod accepts SECURE/COMMUNITY; "
             "other providers may ignore this flag.",
    )
    parser.add_argument(
        "--community", action="store_true",
        help="Shortcut for --cloud COMMUNITY (RunPod compatibility).",
    )
    parser.add_argument(
        "--container-disk-gb", type=int, default=50,
    )
    parser.add_argument(
        "--volume-gb", type=int, default=20,
    )
    parser.add_argument(
        "--regions", "--data-centers", nargs="+", default=None,
        help="Optional provider-specific region / datacenter IDs to pin.",
    )
    parser.add_argument(
        "--project-root", type=Path, default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--results-dest", type=Path, default=DEFAULT_STAGING_DIR,
        help="Local destination for tar-pulled results.  Default: "
             "<PROJECT_ROOT>/lab_reports/_staging/.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Pass --smoke to the remote benchmark and stage under "
             "_staging/_smokes/ so finalize_run.py refuses to promote.",
    )
    parser.add_argument(
        "--extra-args", default="",
        help="Extra CLI args forwarded verbatim to the remote python "
             "invocation, e.g. '--lambda_sweep' or "
             "'--lambda_reg 0.01 --seeds 42 7 1337'.  No shell-quoting is "
             "applied; pass a single whitespace-separated string.",
    )
    parser.add_argument(
        "--keep-on-fail", action="store_true",
        help="Don't destroy the instance if the remote benchmark exits "
             "non-zero.  Default behaviour is to destroy on ANY "
             "sentinel-confirmed exit (zero or non-zero) since the "
             "dispatcher has already pulled run.log and results/ at that "
             "point and the pod is just idling and billing.  Pods are "
             "kept alive when the remote wrapper was launched but no "
             "_DONE sentinel appeared (KeyboardInterrupt after launch, "
             "SSH-failure abort mid-run) — the remote may still be "
             "running.  Pre-run failures (ready timeout, push failure) "
             "ALWAYS destroy: nothing useful is running on the pod.",
    )
    parser.add_argument(
        "--ready-timeout", type=int, default=900,
        help="Seconds to wait for the instance to become SSH-ready "
             "(default 900).",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=60,
        help="Seconds between poll cycles during the run (default 60).",
    )
    parser.add_argument(
        "--reattach", type=Path, default=None, metavar="LAUNCH_DIR",
        help="Re-attach to a launch directory whose dispatcher died. "
             "Skips provisioning, push, and launch; reads "
             "pod_provisioning.json from LAUNCH_DIR, reconstructs the "
             "live instance via the provider plugin, jumps directly into "
             "the poll loop, and runs the same destroy + (optional) "
             "auto-finalize sequence the fresh-launch path would have. "
             "Most other --provider/--gpu/--cloud flags are ignored when "
             "--reattach is set.",
    )
    parser.add_argument(
        "--auto-finalize-slug", default=None,
        help="If set, after a successful run + destroy, automatically "
             "chain into scripts/finalize_run.py with this slug to "
             "render a partial lab report.  Skipped on rc != 0 or when "
             "the instance is kept alive.  Requires "
             "--auto-finalize-operator.",
    )
    parser.add_argument(
        "--auto-finalize-operator", default=None,
        help="Operator string for --auto-finalize-slug "
             "(e.g. 'agent:cascade-2026-05-24').  Required when "
             "--auto-finalize-slug is set; we don't pick a default to "
             "avoid silent attribution errors.",
    )
    parser.add_argument(
        "--auto-finalize-ledger-description", default=None,
        help="Optional ledger-description string passed through to "
             "finalize_run.py --ledger-description when --auto-finalize-slug "
             "is set.",
    )
    args = parser.parse_args()

    # --benchmark is required for fresh launches but optional for --reattach
    # (in which case it's read from pod_provisioning.json).  We can't use
    # argparse's required= attribute since the requirement is conditional.
    if not args.reattach and not args.benchmark:
        parser.error(
            "--benchmark is required (unless --reattach is set, in which "
            "case the benchmark name is read from pod_provisioning.json)."
        )

    # --reattach is mutually exclusive with the fresh-launch flow; dispatch
    # to the dedicated handler before any fresh-launch state is created.
    if args.reattach is not None:
        _run_reattach(args)
        return  # _run_reattach calls sys.exit(); this is defensive.

    if args.community:
        args.cloud = "COMMUNITY"

    # --- Stage the per-launch directory + start dispatcher.log self-tee -----
    stamp = time.strftime("%Y%m%d-%H%M%S")
    launch_root = args.results_dest
    if args.smoke:
        launch_root = launch_root / SMOKE_STAGING_SUBDIR
    launch_dir = _create_unique_launch_dir(
        launch_root=launch_root, benchmark=args.benchmark, stamp=stamp,
    )

    dispatcher_log_path = launch_dir / "dispatcher.log"
    dispatcher_log_f = dispatcher_log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeWriter(sys.__stdout__, dispatcher_log_f)
    sys.stderr = _TeeWriter(sys.__stderr__, dispatcher_log_f)
    print(f"==> Staging launch under {launch_dir}", flush=True)
    print(f"    dispatcher.log -> {dispatcher_log_path}", flush=True)
    print(f"    provider       -> {args.provider}", flush=True)

    if shutil.which("ssh") is None:
        print("ERROR: ssh must be on PATH", file=sys.stderr)
        sys.exit(2)

    # --- Load bench_caps + build the GPU candidate list ---------------------
    benchmark_dir = BENCHMARKS_DIR / args.benchmark
    if not benchmark_dir.is_dir():
        print(
            f"ERROR: benchmark directory {benchmark_dir} does not exist.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        caps = load_bench_caps(benchmark_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    if caps.notes:
        print(f"    bench_caps     -> {caps.notes}", flush=True)

    # --- Instantiate the provider plugin -----------------------------------
    provider_opts = {}
    if args.cloud is not None:
        provider_opts["cloud_type"] = args.cloud
    try:
        provider: CloudProvider = get_provider(args.provider, **provider_opts)
    except (AuthError, ValueError) as exc:
        print(f"\n!! Provider {args.provider!r} init failed: {exc}",
              file=sys.stderr)
        sys.exit(2)

    image = args.image or provider.default_image

    # Default candidate list: explicit CLI, then bench_caps.preferred_gpus,
    # then the provider's default GPU.
    if args.gpu_type:
        raw_candidates = [g.strip() for g in args.gpu_type.split(",") if g.strip()]
    elif caps.preferred_gpus:
        raw_candidates = list(caps.preferred_gpus)
    else:
        raw_candidates = [provider.default_gpu]
    if not raw_candidates:
        print("ERROR: --gpu-type produced an empty candidate list",
              file=sys.stderr)
        sys.exit(2)

    candidates = _filter_candidates(
        raw_candidates,
        caps=caps,
        image_supports_pytorch_2_4=(image == provider.default_image),
    )
    if not candidates:
        print(
            "ERROR: every requested GPU was filtered out (incompatibility or "
            "bench_caps violation).  See messages above; either pass --image "
            "with a more recent PyTorch build, or pass --gpu-type with cards "
            "that meet the benchmark's bench_caps.json requirements.",
            file=sys.stderr,
        )
        sys.exit(2)

    # --- Create the instance ------------------------------------------------
    label = f"synth-{args.benchmark}-{time.strftime('%Y%m%d-%H%M%S')}"
    print(f"==> Creating GPU instance: provider={provider.name} "
          f"name={label} image={image}")
    if args.cloud:
        print(f"    cloud_tier     -> {args.cloud}")
    print(f"    container disk = {args.container_disk_gb} GB, "
          f"volume = {args.volume_gb} GB")
    print(f"    GPU candidates (tried in order): {candidates}")

    instance = _try_create(
        provider,
        candidates=candidates,
        image=image,
        label=label,
        gpu_count=args.gpu_count,
        container_disk_gb=args.container_disk_gb,
        volume_gb=args.volume_gb,
        regions=args.regions,
    )
    if instance is None:
        sys.exit(2)

    print(f"==> Instance created: {provider.provisioning_dict(instance)}")

    # --- Write pod_provisioning.json sidecar -------------------------------
    provisioning = dict(provider.provisioning_dict(instance))
    provisioning.update({
        "benchmark":   args.benchmark,
        "smoke":       args.smoke,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    (launch_dir / "pod_provisioning.json").write_text(
        json.dumps(provisioning, indent=2, sort_keys=True, default=str)
    )
    print(f"    pod_provisioning.json -> "
          f"cost_per_hr_usd={provisioning.get('cost_per_hr_usd')!r}",
          flush=True)

    rc = 1
    results_dir: Optional[Path] = None
    run_completed = False
    # False until the remote wrapper is actually launched. Pre-run
    # failures (ready timeout, push, finalize setup) must destroy the
    # pod — nothing useful is running and keep-alive would orphan-bill.
    remote_may_be_running = False
    remote_finalize_cfg: Optional[dict] = None
    skip_destroy = False
    try:
        print(f"==> Waiting for instance SSH access (up to "
              f"{args.ready_timeout}s)...")
        instance = provider.wait_for_ready(
            instance, timeout_s=args.ready_timeout
        )
        ssh_target = instance.ssh_target()
        print(f"    SSH ready: {instance.label()}")

        push_code(args.project_root, args.benchmark, ssh_target)

        # Remote-finalize opt-in (bench_caps.remote_finalize=true):
        # RunPod-only encrypt/upload/self-terminate. Non-RunPod providers
        # fall back to SCP-pull + dispatcher destroy.
        remote_finalize_cfg = _maybe_push_finalize_payload(
            caps=caps,
            provider=provider,
            instance=instance,
            ssh_target=ssh_target,
            pubkey_path=_DEFAULT_PUBKEY_PATH,
            launch_dir=launch_dir,
        )

        # From here the remote wrapper may be running under nohup.
        remote_may_be_running = True
        rc, results_dir = run_remote_experiment(
            ssh_target, args.benchmark,
            smoke=args.smoke,
            extra_args=args.extra_args,
            launch_dir=launch_dir,
            poll_interval_s=args.poll_interval,
            remote_finalize=remote_finalize_cfg,
        )
        run_completed = rc >= 0

        # After a successful sentinel with remote_finalize, wait for
        # _UPLOADED (or SSH death) before dispatcher destroy so we do
        # not kill pod_finalize mid-upload.
        if (
            remote_finalize_cfg is not None
            and run_completed
            and rc == 0
            and results_dir is not None
        ):
            fin = wait_for_remote_finalize(
                ssh_target,
                args.benchmark,
                timeout_s=_REMOTE_FINALIZE_WAIT_S,
                poll_interval_s=min(15, args.poll_interval),
            )
            if fin in ("uploaded", "ssh_gone"):
                # Pod self-terminate is expected; dispatcher destroy is
                # best-effort idempotent cleanup only.
                skip_destroy = fin == "ssh_gone"
                # Final pull may already have the results; refresh once
                # if SSH is still up (uploaded but not yet terminated).
                if fin == "uploaded":
                    pull_results_into(ssh_target, args.benchmark, launch_dir)

        # B2 pivot on the fresh-launch path: when remote_finalize is
        # enabled and the poll loop exhausted its SSH-failure budget
        # (rc==-1), the expected cause is the pod's wrapper exiting
        # and pod_finalize.py taking sshd down as part of self-
        # terminate.  In that case "no SSH" IS the success signal,
        # provided B2 has the ciphertext + manifest pair.  Treat a
        # successful pivot identically to a normal rc=0 completion.
        if rc == -1 and remote_finalize_cfg is not None:
            print(
                "==> SSH failure budget exhausted with remote_finalize=true; "
                "pivoting to B2 (pod is expected to have self-terminated after upload).",
                flush=True,
            )
            pivot = _try_b2_pivot_with_retry(
                benchmark=args.benchmark,
                launch_dir=launch_dir,
                original_error=RuntimeError(
                    "ssh poll budget exhausted; remote_finalize=true; "
                    "attempting B2 pivot"
                ),
            )
            if pivot is not None:
                _write_b2_pivot_done_sentinel(launch_dir)
                rc = 0
                run_completed = True
                # SCP contract: results_dir is the launch staging dir
                # (contents under launch_dir/results/). B2 pivot extracts
                # into the same layout; keep the launch_dir handle so
                # _print_summary path resolution stays consistent.
                results_dir = launch_dir
                skip_destroy = True
                remote_may_be_running = False

        if rc != 0:
            print(f"!! Remote experiment exited with code {rc}",
                  file=sys.stderr)
        _print_summary(results_dir)
    except KeyboardInterrupt:
        if remote_may_be_running:
            print(
                "\n!! Interrupted by operator.  Instance will be left alive so "
                "the remote run can finish; rerun the dispatcher (or destroy "
                "manually) when you decide.",
                file=sys.stderr,
            )
        else:
            print(
                "\n!! Interrupted before remote launch.  Instance will be "
                "destroyed to stop billing.",
                file=sys.stderr,
            )
        rc = 1
        run_completed = False
    except Exception as exc:
        # Pre-run failures must not silently orphan the pod (finally
        # destroys when remote_may_be_running is False). Mid-run errors
        # keep the pod so in-progress work is not discarded.
        print(
            f"\n!! Dispatcher error ({type(exc).__name__}: {exc}). "
            f"remote_may_be_running={remote_may_be_running}.",
            file=sys.stderr,
        )
        rc = 1
        run_completed = False
    finally:
        # Destruction policy lives in _decide_and_destroy:
        #   * Sentinel-confirmed exit (run_completed=True) → destroy
        #     (unless remote_finalize already took the pod down).
        #   * No sentinel + remote never launched → destroy (orphan fix).
        #   * No sentinel + remote may be running → keep alive.
        #   * --keep-on-fail → keep alive on non-zero rc.
        destroyed = _decide_and_destroy(
            rc=rc,
            run_completed=run_completed,
            instance=instance,
            provider=provider,
            keep_on_fail=args.keep_on_fail,
            results_dir=results_dir,
            remote_may_be_running=remote_may_be_running,
            launch_dir=launch_dir,
            skip_destroy=skip_destroy,
        )
        _maybe_auto_finalize(
            rc=rc,
            destroyed=destroyed,
            launch_dir=launch_dir,
            benchmark=args.benchmark,
            slug=args.auto_finalize_slug,
            operator=args.auto_finalize_operator,
            ledger_description=args.auto_finalize_ledger_description,
        )
        # Surface destroy failures as a non-zero exit even on rc==0 so
        # operators notice orphan billing.
        if not destroyed and not _should_keep_alive(
            run_completed=run_completed,
            rc=rc,
            keep_on_fail=args.keep_on_fail,
            remote_may_be_running=remote_may_be_running,
        ) and not skip_destroy:
            rc = max(rc, 1) if rc != 0 else 3

    if rc == 0:
        sys.exit(0)
    sys.exit(rc if rc > 0 else 1)


if __name__ == "__main__":
    main()
