"""Local-side B2 fetch + decrypt + untar for the dispatcher reattach path.

Companion to ``scripts/pod_finalize.py``: the pod uploads
``<bucket>/<bench>/<run_id>/results.synth`` (encrypted tar.gz) and
``<bucket>/<bench>/<run_id>/manifest.json`` (the commit marker) before
self-terminating.  When the dispatcher reattaches and discovers the pod
is gone (``cp.ProvisioningError`` from ``attach_from_provisioning``),
this module is the recovery path: download both objects via the
operator's read+write LOCAL B2 application key, verify the ciphertext's
sha256 against the manifest, decrypt with the sovereign private key, and
untar into the launch directory's ``results/``.

Object naming MUST match the pod side; the constants are duplicated here
intentionally rather than imported from ``pod_finalize`` because:
  * ``pod_finalize`` is intended to run inside the pod where the
    operator's privkey is never present;
  * importing it locally would drag ``urllib.request`` etc. into the
    dispatcher's import graph for no benefit.

Wire-format expectations (must stay aligned with
``pod_finalize._build_manifest``):

    {
      "schema": "synth-finalize/v1",
      "bench":  <str>,
      "run_id": <str>,
      "completion_utc": <ISO-8601>,
      "wire_format_version": <int>,
      "ciphertext": {
        "object_key": "<bench>/<run_id>/results.synth",
        "size":   <int>,
        "sha256": <hex>,
      },
      "files": [{"name": <str>, "size": <int>}, ...],
    }

The dispatcher's reattach logic decides WHEN to call this; this module
is purely "given a (bench, run_id, bucket) and credentials, fetch +
decrypt + untar idempotently or raise a typed error."
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Wire-format constants (must match scripts/pod_finalize.py)
# ---------------------------------------------------------------------------

CIPHERTEXT_OBJECT_NAME = "results.synth"
MANIFEST_OBJECT_NAME = "manifest.json"
MANIFEST_SCHEMA = "synth-finalize/v1"


# ---------------------------------------------------------------------------
# Typed error hierarchy
# ---------------------------------------------------------------------------


class B2FetchError(Exception):
    """Base class for all errors raised by this module."""


class B2CredentialsMissing(B2FetchError):
    """Required env vars (``B2_KEY_ID_LOCAL`` / ``B2_KEY_LOCAL`` /
    ``B2_BUCKET``) or the recipient private key are absent.  The
    dispatcher should treat this as "B2 pivot not configured; bubble up
    the original ProvisioningError".
    """


class ManifestNotFound(B2FetchError):
    """No ``manifest.json`` exists at the expected B2 prefix.  The pod
    either never finalized successfully or the run was on the legacy
    SCP-pull path.  The dispatcher should re-raise the underlying
    "pod gone" error rather than treating this as a recoverable state.
    """


class CiphertextNotFound(B2FetchError):
    """``manifest.json`` was found but the ciphertext object it points
    to is missing.  This is a HARD error: the manifest is the commit
    marker, so its presence implies the ciphertext was uploaded first.
    A missing ciphertext at this point means manual interference,
    bucket misconfiguration, or a B2 outage during upload that wasn't
    caught by ``pod_finalize``'s HEAD-check.
    """


class ManifestVerificationError(B2FetchError):
    """Manifest schema mismatch, sha256 hash mismatch, or other
    integrity check failure.  Surfaced as a hard error: the
    ciphertext is not trustworthy, so we refuse to decrypt it.
    """


# ---------------------------------------------------------------------------
# Default privkey resolution (mirrors Makefile SYNTH_KEY_PRIV default)
# ---------------------------------------------------------------------------


def default_privkey_path() -> Path:
    """Return the operator's default sovereign private key path.

    Mirrors the Makefile's ``SYNTH_KEY_PRIV`` default
    (``$(HOME)/.config/synth/keys/synth-results.key``) so the
    dispatcher and the keygen ceremony agree without any explicit
    config.  Override via the ``SYNTH_KEY_PRIV`` environment variable.
    """
    explicit = os.environ.get("SYNTH_KEY_PRIV")
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".config" / "synth" / "keys" / "synth-results.key"


# ---------------------------------------------------------------------------
# Lazy b2sdk import (keeps the dispatcher importable when b2sdk isn't
# installed, e.g. on a fresh checkout before `make scripts-venv`).
# ---------------------------------------------------------------------------


def _b2_authorize(key_id: str, key: str):
    try:
        from b2sdk.v2 import B2Api, InMemoryAccountInfo
    except ImportError as exc:  # pragma: no cover - venv has b2sdk
        raise B2FetchError(
            "b2sdk is required for the B2 reattach path.  "
            "Install via `make scripts-venv` (already in "
            "scripts/requirements.txt)."
        ) from exc
    api = B2Api(InMemoryAccountInfo())
    api.authorize_account("production", key_id, key)
    return api


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B2FetchConfig:
    """All inputs needed to recover one run from B2.

    The dispatcher constructs this from environment variables + the
    persisted ``pod_provisioning.json`` + the launch-dir basename.
    """
    bench: str
    run_id: str
    bucket: str
    key_id: str
    key: str
    privkey_path: Path

    @classmethod
    def from_env(
        cls,
        *,
        bench: str,
        run_id: str,
        privkey_path: Optional[Path] = None,
    ) -> "B2FetchConfig":
        """Build a :class:`B2FetchConfig` from process environment.

        Raises :class:`B2CredentialsMissing` listing every missing piece
        so the operator can fix ``.env`` in a single pass.  The
        dispatcher catches this and falls back to "bubble up the
        original ProvisioningError".
        """
        missing: List[str] = []
        bucket = os.environ.get("B2_BUCKET")
        key_id = os.environ.get("B2_KEY_ID_LOCAL")
        key = os.environ.get("B2_KEY_LOCAL")
        if not bucket:
            missing.append("B2_BUCKET")
        if not key_id:
            missing.append("B2_KEY_ID_LOCAL")
        if not key:
            missing.append("B2_KEY_LOCAL")
        resolved_privkey = privkey_path or default_privkey_path()
        if not resolved_privkey.is_file():
            missing.append(f"private key at {resolved_privkey}")
        if missing:
            raise B2CredentialsMissing(
                "B2 reattach pivot not configured; missing: "
                + ", ".join(missing)
                + ".  Populate .env (B2_KEY_ID_LOCAL / B2_KEY_LOCAL / "
                "B2_BUCKET) and ensure the sovereign privkey is "
                "present (default $HOME/.config/synth/keys/"
                "synth-results.key, or override via SYNTH_KEY_PRIV)."
            )
        # Type checker: we proved bucket/key_id/key are non-None above.
        assert bucket is not None and key_id is not None and key is not None
        return cls(
            bench=bench,
            run_id=run_id,
            bucket=bucket,
            key_id=key_id,
            key=key,
            privkey_path=resolved_privkey,
        )


# ---------------------------------------------------------------------------
# Object-key helpers
# ---------------------------------------------------------------------------


def manifest_object_key(bench: str, run_id: str) -> str:
    return f"{bench}/{run_id}/{MANIFEST_OBJECT_NAME}"


def ciphertext_object_key(bench: str, run_id: str) -> str:
    return f"{bench}/{run_id}/{CIPHERTEXT_OBJECT_NAME}"


# ---------------------------------------------------------------------------
# B2 download primitives
# ---------------------------------------------------------------------------


def _download_bytes(api, bucket_name: str, object_key: str) -> bytes:
    """Download a single B2 object into memory.  Returns the raw bytes.

    Raises ``b2sdk`` exceptions on network / auth failure; the caller
    distinguishes "object missing" via ``_try_download_bytes`` below.
    """
    bucket = api.get_bucket_by_name(bucket_name)
    downloaded = bucket.download_file_by_name(object_key)
    buf = io.BytesIO()
    downloaded.save(buf)
    return buf.getvalue()


def _try_download_bytes(api, bucket_name: str, object_key: str) -> Optional[bytes]:
    """Like :func:`_download_bytes` but returns ``None`` on a 404
    (file_not_present) rather than raising.  Other errors propagate.
    """
    # b2sdk's exception hierarchy lives under v2; import lazily so this
    # module imports cleanly in environments without b2sdk.
    from b2sdk.v2 import exception as b2_exc
    try:
        return _download_bytes(api, bucket_name, object_key)
    except b2_exc.FileNotPresent:
        return None
    # b2sdk also surfaces a 404 as the more general FileOrBucketNotFound
    # depending on version; catch the broader bucket-not-present too,
    # but only when the message indicates the file path (not the bucket).
    except b2_exc.UnknownError as exc:  # pragma: no cover - version drift
        if "404" in str(exc):
            return None
        raise


# ---------------------------------------------------------------------------
# Manifest verification
# ---------------------------------------------------------------------------


def _parse_and_verify_manifest(
    raw: bytes, *, bench: str, run_id: str,
) -> Dict[str, Any]:
    """Parse + sanity-check the manifest JSON.

    Verifies schema label, bench/run_id agreement (defense against
    cross-prefix object swaps in B2), and that the ciphertext metadata
    fields the caller will need are well-formed.
    """
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestVerificationError(
            f"manifest.json is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ManifestVerificationError(
            f"manifest.json top-level must be a JSON object, "
            f"got {type(manifest).__name__}"
        )
    schema = manifest.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ManifestVerificationError(
            f"manifest.json schema mismatch: expected "
            f"{MANIFEST_SCHEMA!r}, got {schema!r}"
        )
    if manifest.get("bench") != bench:
        raise ManifestVerificationError(
            f"manifest.json bench mismatch: expected {bench!r}, "
            f"got {manifest.get('bench')!r}.  Possible cross-prefix "
            "object swap in B2."
        )
    if manifest.get("run_id") != run_id:
        raise ManifestVerificationError(
            f"manifest.json run_id mismatch: expected {run_id!r}, "
            f"got {manifest.get('run_id')!r}"
        )
    ct = manifest.get("ciphertext")
    if not isinstance(ct, dict):
        raise ManifestVerificationError(
            "manifest.json missing or malformed 'ciphertext' object"
        )
    for required_field in ("object_key", "size", "sha256"):
        if required_field not in ct:
            raise ManifestVerificationError(
                f"manifest.json ciphertext missing required field "
                f"{required_field!r}"
            )
    return manifest


def _verify_ciphertext_against_manifest(
    ciphertext: bytes, manifest: Dict[str, Any],
) -> None:
    expected_size = manifest["ciphertext"]["size"]
    expected_sha = manifest["ciphertext"]["sha256"]
    if len(ciphertext) != expected_size:
        raise ManifestVerificationError(
            f"ciphertext size mismatch: manifest={expected_size}, "
            f"actual={len(ciphertext)}"
        )
    actual_sha = hashlib.sha256(ciphertext).hexdigest()
    if actual_sha != expected_sha:
        raise ManifestVerificationError(
            f"ciphertext sha256 mismatch: manifest={expected_sha}, "
            f"actual={actual_sha}.  The ciphertext on B2 has been "
            "tampered with or corrupted."
        )


# ---------------------------------------------------------------------------
# Decrypt + untar
# ---------------------------------------------------------------------------


def _decrypt_ciphertext(ciphertext: bytes, privkey_path: Path) -> bytes:
    """Decrypt with the sovereign hybrid private key.

    Imports ``synth_crypto`` lazily so this module can be imported in
    environments where the crypto stack isn't yet installed (e.g.,
    bare ``python3 -m unittest discover`` from a fresh checkout).
    """
    import synth_crypto  # noqa: WPS433 - intentional lazy import
    privkey = synth_crypto.PrivateKey.from_file(privkey_path)
    return synth_crypto.decrypt(ciphertext, privkey)


def _safe_extract_tar(tar_bytes: bytes, dest_dir: Path) -> List[str]:
    """Extract a gzipped tar into ``dest_dir`` with path-traversal
    guarding.  Returns the list of extracted member names.

    The pod side writes plain relative paths under ``results/``; we
    refuse any member whose resolved destination escapes ``dest_dir``.
    Defense in depth against a hypothetically-tampered tar even though
    the AEAD already proved the bytes are authentic.
    """
    dest_dir = dest_dir.resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[str] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for member in tf.getmembers():
            target = (dest_dir / member.name).resolve()
            try:
                target.relative_to(dest_dir)
            except ValueError as exc:
                raise ManifestVerificationError(
                    f"tar member escapes dest_dir: {member.name!r}"
                ) from exc
            if member.isdev() or member.issym() or member.islnk():
                raise ManifestVerificationError(
                    f"refusing non-regular tar member: "
                    f"{member.name!r} (type={member.type!r})"
                )
            # filter='data' is the Python 3.12+ safe default that
            # rejects absolute paths, links, and device nodes; we've
            # already enforced stricter checks above, so this is
            # belt-and-suspenders for forward compatibility (Python
            # 3.14 will require an explicit filter).
            tf.extract(member, dest_dir, filter="data")
            extracted.append(member.name)
    return extracted


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class B2FetchResult:
    """What the dispatcher's reattach path needs after a successful pivot."""
    manifest: Dict[str, Any]
    extracted_files: List[str]
    dest_dir: Path
    ciphertext_size: int


def fetch_and_decrypt(config: B2FetchConfig, dest_dir: Path) -> B2FetchResult:
    """Recover one run from B2: download + verify + decrypt + untar.

    Sequence (matches the inverse of ``pod_finalize.finalize``):

      1. Authorize B2 with the LOCAL read+write key.
      2. Download ``manifest.json`` first; raise :class:`ManifestNotFound`
         if absent (the dispatcher treats this as "pod_finalize never
         committed; bubble up the original 'pod gone' error").
      3. Parse + verify manifest schema, bench/run_id agreement, and
         well-formedness.
      4. Download the ciphertext referenced by the manifest; raise
         :class:`CiphertextNotFound` if absent (hard error: manifest
         existence implies ciphertext was uploaded first).
      5. Verify ciphertext size + sha256 against manifest.
      6. Decrypt with the sovereign privkey (synth_crypto AEAD).
      7. Untar into ``dest_dir/results/`` with path-traversal guards.

    Any step's failure raises a :class:`B2FetchError` subclass; the
    caller distinguishes recoverable (``ManifestNotFound``) from hard
    failures (``ManifestVerificationError`` + ``CiphertextNotFound``).
    """
    api = _b2_authorize(config.key_id, config.key)

    mf_key = manifest_object_key(config.bench, config.run_id)
    mf_raw = _try_download_bytes(api, config.bucket, mf_key)
    if mf_raw is None:
        raise ManifestNotFound(
            f"no manifest at b2://{config.bucket}/{mf_key}.  Pod may "
            "have crashed before commit, or the run used the legacy "
            "SCP-pull path.  Falling back to original error."
        )
    manifest = _parse_and_verify_manifest(
        mf_raw, bench=config.bench, run_id=config.run_id,
    )

    ct_key = manifest["ciphertext"]["object_key"]
    ct_raw = _try_download_bytes(api, config.bucket, ct_key)
    if ct_raw is None:
        raise CiphertextNotFound(
            f"manifest claims ciphertext at b2://{config.bucket}/"
            f"{ct_key} but the object is missing.  The manifest is "
            "the commit marker; its presence implies the ciphertext "
            "was uploaded first.  Possible manual interference."
        )
    _verify_ciphertext_against_manifest(ct_raw, manifest)

    plaintext = _decrypt_ciphertext(ct_raw, config.privkey_path)

    results_dir = dest_dir / "results"
    extracted = _safe_extract_tar(plaintext, results_dir)
    return B2FetchResult(
        manifest=manifest,
        extracted_files=extracted,
        dest_dir=results_dir,
        ciphertext_size=len(ct_raw),
    )
