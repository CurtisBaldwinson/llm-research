#!/usr/bin/env python3
"""pod_finalize: pack, encrypt, upload, and self-terminate from inside a pod.

Runs on the cloud GPU pod after the benchmark's Python exits cleanly.
End-to-end sequence (matches docs/2026-05-25_remote-finalize-architecture.md):

  1.  Tar the ``results/`` directory into an in-memory bytestream.
  2.  Encrypt with ``synth_crypto`` (X25519 + ML-KEM-768 hybrid) using
      the recipient public key committed to ``infra/keys/synth-results.pub``.
  3.  Upload the ciphertext to Backblaze B2 at
      ``<B2_BUCKET>/<bench>/<run_id>/results.synth``.
  4.  Build + upload a small ``manifest.json`` (sha256, file list,
      sizes, completion timestamp) at
      ``<B2_BUCKET>/<bench>/<run_id>/manifest.json``.  The manifest is
      uploaded LAST so its presence acts as a commit marker — the
      local-side reattach path treats a missing manifest as "ciphertext
      not yet trustworthy, fall back to SCP".
  5.  HEAD-check both objects.
  6.  Write an ``_UPLOADED`` sentinel into ``results/``.
  7.  Call RunPod's REST API to terminate this pod.

Failure handling
================

Any step may fail (B2 transient down, transient RunPod API hiccup, OOM,
disk full).  Each step retries with exponential backoff (3 attempts,
1s → 2s → 4s) and on persistent failure the script exits non-zero
WITHOUT calling ``terminate``.  In that state the operator's
``make reattach`` flow still works (the pod is alive, SCP-pull is the
fallback path).

The script is intentionally NOT idempotent on the upload step: if you
re-run after a partial upload, you'll re-encrypt and re-upload.  B2
versioning + the lifecycle policy on the bucket clean up stale
versions.  The terminate step IS idempotent (RunPod returns 404 if the
pod is already destroyed, which we treat as success).

Environment
===========

Reads from environment (the dispatcher / bootstrap script populates):

  RUNPOD_POD_ID         - injected by the RunPod control plane; this
                          script's own pod identifier
  RUNPOD_API_KEY        - for the terminate REST call
  B2_KEY_ID_POD         - B2 application key id (write-only scope)
  B2_KEY_POD            - B2 application key secret
  B2_BUCKET             - target B2 bucket name (e.g., "synthetic-results")

CLI flags supersede the env where useful (test mode, dry runs).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import tarfile
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Local imports come last to keep "what runs at import time" minimal.
# These imports CAN fail if the pod's bootstrap didn't install the
# crypto stack; the script catches that case and exits cleanly.
import synth_crypto  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIPHERTEXT_OBJECT_NAME = "results.synth"
MANIFEST_OBJECT_NAME = "manifest.json"
SENTINEL_FILENAME = "_UPLOADED"
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_MULTIPLIER = 2.0
RUNPOD_REST_API = "https://rest.runpod.io/v1"


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


def _with_retries(label: str, fn, *args, **kwargs):
    """Call ``fn`` with exponential backoff (3 attempts).  Re-raises the
    last exception if all attempts fail.
    """
    delay = INITIAL_BACKOFF_S
    last_exc: Optional[BaseException] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentional broad catch
            last_exc = exc
            if attempt == MAX_RETRIES:
                _log(
                    f"[pod_finalize] {label}: attempt {attempt}/{MAX_RETRIES} "
                    f"FAILED: {exc!r}"
                )
                raise
            _log(
                f"[pod_finalize] {label}: attempt {attempt}/{MAX_RETRIES} "
                f"failed ({exc!r}); retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER
    # Unreachable: the loop above either returns or re-raises.
    raise RuntimeError(f"{label}: unreachable retry state ({last_exc!r})")


def _log(msg: str) -> None:
    """Stdout logging with a UTC timestamp prefix."""
    print(
        f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Step 1: tar
# ---------------------------------------------------------------------------


def _tar_results(results_dir: Path) -> Tuple[bytes, List[Dict[str, Any]]]:
    """Return (tar.gz bytes, file-list metadata) for ``results_dir``.

    The file-list metadata is what we'll include in the manifest so the
    local-side reattach can verify what's in the archive without
    decrypting.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    file_list: List[Dict[str, Any]] = []
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(results_dir.rglob("*")):
            if not path.is_file():
                continue
            arcname = str(path.relative_to(results_dir))
            tf.add(path, arcname=arcname)
            file_list.append({
                "name": arcname,
                "size": path.stat().st_size,
            })
    return buf.getvalue(), file_list


# ---------------------------------------------------------------------------
# Step 2: encrypt
# ---------------------------------------------------------------------------


def _encrypt(plaintext: bytes, pubkey_path: Path) -> bytes:
    if not pubkey_path.is_file():
        raise FileNotFoundError(f"recipient pubkey missing: {pubkey_path}")
    pub = synth_crypto.PublicKey.from_file(pubkey_path)
    return synth_crypto.encrypt(plaintext, pub)


# ---------------------------------------------------------------------------
# Step 3 / 4: B2 upload
# ---------------------------------------------------------------------------


def _b2_client():
    """Return an authorized b2sdk InMemoryAccountInfo + B2Api."""
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "b2sdk is required for B2 upload.  Install via "
            "`pip install b2sdk` (already in scripts/requirements.txt)."
        ) from exc
    api = B2Api(InMemoryAccountInfo())
    key_id = os.environ.get("B2_KEY_ID_POD")
    key = os.environ.get("B2_KEY_POD")
    if not key_id or not key:
        raise RuntimeError(
            "B2_KEY_ID_POD / B2_KEY_POD env vars are not set.  "
            "The pod's bootstrap must populate these from the dispatcher's .env."
        )
    api.authorize_account("production", key_id, key)
    return api


def _b2_upload_bytes(
    api,
    bucket_name: str,
    object_key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> Dict[str, Any]:
    """Upload ``data`` to B2 as ``bucket_name/object_key``.  Returns the
    server response dict (includes ``fileId``, ``contentSha1``, ``size``).
    """
    bucket = api.get_bucket_by_name(bucket_name)
    file_info = bucket.upload_bytes(
        data,
        object_key,
        content_type=content_type,
    )
    return {
        "file_id": getattr(file_info, "id_", None),
        "content_sha1": getattr(file_info, "content_sha1", None),
        "size": getattr(file_info, "size", None),
        "object_key": object_key,
    }


def _b2_head(api, bucket_name: str, object_key: str) -> Dict[str, Any]:
    """HEAD an object in B2; returns file metadata or raises if missing."""
    bucket = api.get_bucket_by_name(bucket_name)
    info = bucket.get_file_info_by_name(object_key)
    return {
        "file_id": getattr(info, "id_", None),
        "size": getattr(info, "size", None),
        "content_sha1": getattr(info, "content_sha1", None),
    }


# ---------------------------------------------------------------------------
# Step 7: RunPod self-terminate
# ---------------------------------------------------------------------------


def _runpod_terminate(pod_id: str, api_key: str) -> None:
    """Call RunPod REST DELETE /pods/{pod_id}.  Treats 404 as success
    (the pod is already gone).
    """
    url = f"{RUNPOD_REST_API}/pods/{pod_id}"
    req = urllib.request.Request(
        url,
        method="DELETE",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            _log(
                f"[pod_finalize] RunPod terminate({pod_id}) returned "
                f"HTTP {resp.status}"
            )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            _log(
                f"[pod_finalize] RunPod terminate({pod_id}) -> 404 "
                f"(already destroyed); treating as success"
            )
            return
        raise
    except (urllib.error.URLError, socket.timeout) as exc:
        # Network blip; re-raise so _with_retries kicks in.
        raise RuntimeError(f"runpod terminate network error: {exc!r}") from exc


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _build_manifest(
    *,
    bench: str,
    run_id: str,
    ciphertext_size: int,
    ciphertext_sha256: str,
    file_list: List[Dict[str, Any]],
    completion_utc: str,
    wire_format_version: int,
) -> bytes:
    """Build the manifest JSON.  Contains everything the local-side
    reattach needs to verify the ciphertext before decryption.
    """
    return json.dumps({
        "schema": "synth-finalize/v1",
        "bench": bench,
        "run_id": run_id,
        "completion_utc": completion_utc,
        "wire_format_version": wire_format_version,
        "ciphertext": {
            "object_key": f"{bench}/{run_id}/{CIPHERTEXT_OBJECT_NAME}",
            "size": ciphertext_size,
            "sha256": ciphertext_sha256,
        },
        "files": file_list,
    }, indent=2, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def finalize(args: argparse.Namespace) -> int:
    """Top-level orchestrator; returns process exit code."""
    results_dir = Path(args.results_dir).expanduser().resolve()
    pubkey_path = Path(args.recipient_pubkey).expanduser().resolve()
    bench = args.bench
    run_id = args.run_id
    bucket = args.b2_bucket or os.environ.get("B2_BUCKET")
    if not bucket:
        _log("[pod_finalize] FATAL: --b2-bucket / B2_BUCKET not set")
        return 2

    pod_id = args.pod_id or os.environ.get("RUNPOD_POD_ID")
    runpod_api_key = args.runpod_api_key or os.environ.get("RUNPOD_API_KEY")
    if not args.no_terminate and not (pod_id and runpod_api_key):
        _log(
            "[pod_finalize] FATAL: --pod-id / RUNPOD_POD_ID and "
            "--runpod-api-key / RUNPOD_API_KEY are required unless "
            "--no-terminate is set"
        )
        return 2

    _log(f"[pod_finalize] results_dir   = {results_dir}")
    _log(f"[pod_finalize] pubkey        = {pubkey_path}")
    _log(f"[pod_finalize] bench/run_id  = {bench}/{run_id}")
    _log(f"[pod_finalize] b2 bucket     = {bucket}")
    _log(f"[pod_finalize] pod_id        = {pod_id or '(--no-terminate)'}")

    # Step 1: tar
    _log("[pod_finalize] step 1: tarring results dir...")
    tar_bytes, file_list = _tar_results(results_dir)
    _log(
        f"[pod_finalize] tarred {len(file_list)} files, "
        f"{len(tar_bytes)} bytes (gzipped)"
    )

    # Step 2: encrypt
    _log("[pod_finalize] step 2: encrypting (X25519 + ML-KEM-768 hybrid)...")
    ciphertext = _encrypt(tar_bytes, pubkey_path)
    ct_sha256 = hashlib.sha256(ciphertext).hexdigest()
    _log(
        f"[pod_finalize] ciphertext {len(ciphertext)} bytes, "
        f"sha256={ct_sha256[:16]}..."
    )

    # Step 3: B2 upload (ciphertext first)
    _log("[pod_finalize] step 3: B2 client auth...")
    b2 = _with_retries("b2_authorize", _b2_client)

    ct_object_key = f"{bench}/{run_id}/{CIPHERTEXT_OBJECT_NAME}"
    mf_object_key = f"{bench}/{run_id}/{MANIFEST_OBJECT_NAME}"

    _log(f"[pod_finalize] step 4: uploading ciphertext -> b2://{bucket}/{ct_object_key}")
    ct_info = _with_retries(
        "b2_upload_ciphertext",
        _b2_upload_bytes, b2, bucket, ct_object_key, ciphertext,
        "application/x-synth-crypto",
    )
    _log(f"[pod_finalize] ciphertext uploaded: {ct_info}")

    # Step 4: build + upload manifest LAST (acts as commit marker)
    completion_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_bytes = _build_manifest(
        bench=bench,
        run_id=run_id,
        ciphertext_size=len(ciphertext),
        ciphertext_sha256=ct_sha256,
        file_list=file_list,
        completion_utc=completion_utc,
        wire_format_version=synth_crypto.WIRE_VERSION,
    )
    _log(f"[pod_finalize] step 5: uploading manifest -> b2://{bucket}/{mf_object_key}")
    mf_info = _with_retries(
        "b2_upload_manifest",
        _b2_upload_bytes, b2, bucket, mf_object_key, manifest_bytes,
        "application/json",
    )
    _log(f"[pod_finalize] manifest uploaded: {mf_info}")

    # Step 5: HEAD-check both
    _log("[pod_finalize] step 6: HEAD-check ciphertext + manifest")
    ct_head = _with_retries(
        "b2_head_ciphertext", _b2_head, b2, bucket, ct_object_key
    )
    mf_head = _with_retries(
        "b2_head_manifest", _b2_head, b2, bucket, mf_object_key
    )
    if ct_head.get("size") != len(ciphertext):
        raise RuntimeError(
            f"ciphertext HEAD size mismatch: server={ct_head.get('size')} "
            f"local={len(ciphertext)}"
        )
    if mf_head.get("size") != len(manifest_bytes):
        raise RuntimeError(
            f"manifest HEAD size mismatch: server={mf_head.get('size')} "
            f"local={len(manifest_bytes)}"
        )
    _log("[pod_finalize] HEAD-check OK")

    # Step 6: local sentinel (operator-visible via SCP if pod is still up)
    sentinel = results_dir / SENTINEL_FILENAME
    sentinel.write_text(json.dumps({
        "object_key_ciphertext": ct_object_key,
        "object_key_manifest": mf_object_key,
        "completion_utc": completion_utc,
        "ciphertext_sha256": ct_sha256,
        "b2_bucket": bucket,
    }, indent=2))
    _log(f"[pod_finalize] step 7: wrote {sentinel}")

    # Step 7: RunPod self-terminate
    if args.no_terminate:
        _log("[pod_finalize] --no-terminate set; skipping pod terminate")
        return 0

    _log(f"[pod_finalize] step 8: RunPod terminate pod={pod_id}")
    _with_retries(
        "runpod_terminate", _runpod_terminate, pod_id, runpod_api_key,
    )
    _log("[pod_finalize] done; pod will be reaped within ~30 s")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pod_finalize",
        description="Pack, encrypt, upload, and self-terminate from inside a pod.",
    )
    p.add_argument(
        "--results-dir", required=True,
        help="path to the results directory to package",
    )
    p.add_argument(
        "--recipient-pubkey", required=True,
        help="path to the recipient public key (committed to infra/keys/)",
    )
    p.add_argument(
        "--bench", required=True,
        help="benchmark name (becomes the first path component in B2)",
    )
    p.add_argument(
        "--run-id", required=True,
        help="unique run identifier (e.g., the staging-dir timestamp)",
    )
    p.add_argument(
        "--b2-bucket",
        help="B2 bucket name (default: $B2_BUCKET env var)",
    )
    p.add_argument(
        "--pod-id",
        help="RunPod pod id to self-terminate (default: $RUNPOD_POD_ID env var)",
    )
    p.add_argument(
        "--runpod-api-key",
        help="RunPod API key (default: $RUNPOD_API_KEY env var)",
    )
    p.add_argument(
        "--no-terminate", action="store_true",
        help="skip the pod-terminate step (test mode; leaves the pod running)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return finalize(args)
    except Exception as exc:  # noqa: BLE001
        # On any uncaught exception: DO NOT terminate.  Log a clear
        # error to stderr so the operator can SCP-pull instead.
        _log(f"[pod_finalize] FATAL: {exc!r}")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
