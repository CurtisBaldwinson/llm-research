"""Provider-agnostic SSH transport, remote-wrapper templating, and polling.

Every cloud-GPU provider has a different control plane (create pod / instance,
poll for ready, get an SSH endpoint, destroy), but once a pod or VM is up and
the operator has its SSH target the rest of the dispatch is identical across
providers:

  1. Push code via tar-over-SSH.
  2. Drop a self-contained bash wrapper on the remote and launch it under
     ``nohup`` so it survives any dispatcher-side disconnect.
  3. Poll the remote for the ``_DONE.<rc>`` sentinel, tar-pulling partial
     results each cycle.
  4. Pull final results on completion.

This module owns that middle 70% of the pipeline.  Provider plugins
(:mod:`cloud_providers.runpod`, :mod:`cloud_providers.vultr`,
:mod:`cloud_providers.lambda_labs`) own the cloud-specific control plane; the
orchestrator (``scripts/run_on_cloud.py``) wires them together with the helpers
defined here.

The functions here MUST stay provider-agnostic: nothing in this file should
mention a specific cloud, an API endpoint, a GPU SKU, or an authentication
scheme.  If a function starts wanting to know "is this RunPod?" or "is this
Vultr?", that is a sign the function belongs in the provider plugin instead.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import IO, List, Tuple

# ---------------------------------------------------------------------------
# Common SSH options.  -T disables PTY allocation (some proxy SSHes reject
# PTY requests); ServerAlive keeps long sessions alive across short network
# blips; ConnectTimeout makes outages visible quickly instead of hanging on
# the kernel's default 75-second TCP timeout.
# ---------------------------------------------------------------------------
SSH_OPTS: List[str] = [
    "-T",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=10",
    "-o", "ConnectTimeout=30",
    "-i", os.path.expanduser("~/.ssh/id_ed25519"),
]

# ---------------------------------------------------------------------------
# GNU tar discovery (macOS bsdtar embeds Apple-only PAX headers that GNU tar
# on the remote can't read).  Forcing --format=gnu and stripping ownership
# produces a clean archive that extracts cleanly anywhere.
# ---------------------------------------------------------------------------

def _find_gtar() -> str:
    """Return the path to GNU tar, or exit with a clear error."""
    for candidate in ("gtar", "/opt/homebrew/bin/gtar", "/usr/local/bin/gtar"):
        path = shutil.which(candidate)
        if path:
            return path
    system_tar = shutil.which("tar")
    if system_tar:
        try:
            out = subprocess.run(
                [system_tar, "--version"], capture_output=True, text=True, timeout=5
            )
            if "GNU tar" in (out.stdout or ""):
                return system_tar
        except Exception:
            pass
    print("ERROR: GNU tar (gtar) not found.", file=sys.stderr)
    print("  On macOS: brew install gnu-tar", file=sys.stderr)
    print("  On Linux: apt-get install tar  (usually pre-installed)", file=sys.stderr)
    sys.exit(2)


GTAR = _find_gtar()

GTAR_CREATE: List[str] = [
    GTAR,
    "--format=gnu",        # pure GNU format, no PAX extended headers
    "--owner=0",           # root ownership in archive (no macOS uid leak)
    "--group=0",
    "--numeric-owner",
    "--no-xattrs",
    "--no-acls",
    "-czf", "-",           # compress to stdout
]

# Paths and patterns that should never be pushed to a remote pod, even when a
# benchmark dir is listed as a sibling dependency.
#
#   * Local-only build/cache artefacts (venvs, pyc caches, prior results,
#     test caches).  The remote regenerates these as needed.
#   * Documentation (``*.md``).  The runner never reads markdown at execution
#     time; READMEs, SPECs, and continuation-claim drafts are author-side
#     artefacts that don't belong on a third-party-operated host.  Keeping
#     them off the wire reduces incidental exposure of patent-adjacent prose
#     and design rationale to whichever host happens to schedule the pod.
TAR_EXCLUDES: List[str] = [
    "--exclude=.venv",
    "--exclude=__pycache__",
    "--exclude=.DS_Store",
    "--exclude=cache",
    "--exclude=results",
    "--exclude=.pytest_cache",
    "--exclude=.mypy_cache",
    "--exclude=*.md",
]

# Belt-and-suspenders against macOS sprinkling AppleDouble metadata into
# archives via the host tar.
TAR_ENV = {**os.environ, "COPYFILE_DISABLE": "1"}

# Remote layout under the persistent volume mount.  Provider plugins are
# expected to mount their volume here so the dispatcher's tar layout works
# uniformly across clouds.
REMOTE_BASE = "/workspace/synth"


# ---------------------------------------------------------------------------
# _TeeWriter — duplicates writes to two file-like streams so dispatcher
# output ends up both on the operator's terminal and in dispatcher.log.
# ---------------------------------------------------------------------------

class _TeeWriter:
    """File-like that duplicates writes to two underlying streams."""

    def __init__(self, primary: IO[str], secondary: IO[str]) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
        except Exception:
            pass  # never let a broken log file kill the dispatcher
        return n

    def flush(self) -> None:
        try:
            self._primary.flush()
        finally:
            try:
                self._secondary.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._primary.fileno()


# ---------------------------------------------------------------------------
# SSH command helpers
# ---------------------------------------------------------------------------

def ssh_cmd(ssh_target: List[str]) -> List[str]:
    """Return ``["ssh", *SSH_OPTS, *ssh_target]`` for direct invocation."""
    return ["ssh"] + SSH_OPTS + ssh_target


def ssh_run(
    ssh_target: List[str], command: str, timeout: int = 60
) -> subprocess.CompletedProcess:
    """Run a one-shot SSH command and capture stdout+stderr."""
    return subprocess.run(
        ssh_cmd(ssh_target) + [command],
        capture_output=True, text=True, timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Code transport: tar-over-SSH push and pull
# ---------------------------------------------------------------------------

def _read_bench_deps(exp_path: Path) -> List[str]:
    """Return the list of sibling benchmark dir names declared in bench_deps.txt."""
    deps_file = exp_path / "bench_deps.txt"
    if not deps_file.exists():
        return []
    out: List[str] = []
    for raw in deps_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def push_code(project_root: Path, benchmark: str, ssh_target: List[str]) -> None:
    """Push ``sm_src/`` + ``benchmarks/<benchmark>/`` + declared sibling deps
    to ``REMOTE_BASE`` on the remote.

    Uses GNU tar with ``--format=gnu`` and explicit exclusions so the archive
    is clean of macOS metadata and local-only artefacts.  Markdown files are
    excluded by :data:`TAR_EXCLUDES`.
    """
    sm_path = project_root / "sm_src"
    benchmarks_root = project_root / "benchmarks"
    exp_path = benchmarks_root / benchmark
    if not sm_path.exists():
        raise FileNotFoundError(f"Missing sm_src at {sm_path}")
    if not exp_path.exists():
        raise FileNotFoundError(f"Missing benchmark {benchmark} at {exp_path}")
    if not (exp_path / "run.sh").exists():
        raise FileNotFoundError(f"Benchmark {benchmark} has no run.sh at {exp_path}")

    dep_names = _read_bench_deps(exp_path)
    dep_paths: List[Path] = []
    for dep in dep_names:
        dep_path = benchmarks_root / dep
        if not dep_path.exists():
            raise FileNotFoundError(
                f"bench_deps.txt for {benchmark} lists {dep!r}, "
                f"but {dep_path} does not exist"
            )
        dep_paths.append(dep_path)

    dep_label = (" + " + " + ".join(dep_names)) if dep_names else ""
    print(f"==> Pushing sm_src + {benchmark}{dep_label} to {REMOTE_BASE}/", flush=True)

    subprocess.run(ssh_cmd(ssh_target) + [f"mkdir -p {REMOTE_BASE}"], check=True)

    for src in [sm_path, exp_path, *dep_paths]:
        print(f"    tar-push {src.name}/", flush=True)
        tar = subprocess.Popen(
            GTAR_CREATE + TAR_EXCLUDES + ["-C", str(src.parent), src.name],
            stdout=subprocess.PIPE,
            env=TAR_ENV,
        )
        ssh = subprocess.Popen(
            ssh_cmd(ssh_target) + [f"tar xzf - --no-same-owner -C {REMOTE_BASE}"],
            stdin=tar.stdout,
        )
        if tar.stdout is not None:
            tar.stdout.close()  # allow tar to receive SIGPIPE if ssh exits
        ssh.communicate()
        if ssh.returncode != 0:
            raise RuntimeError(f"Failed to push {src.name} (ssh rc={ssh.returncode})")


def pull_results_into(
    ssh_target: List[str], benchmark: str, dest_dir: Path
) -> bool:
    """tar-pull the remote ``results/`` into ``dest_dir``.  Returns True on success.

    Idempotent: each call overwrites the local files with the latest remote
    state, so a sequence of calls during the poll loop progressively grows
    the local snapshot as the remote produces more rows.

    Tar warnings are surfaced (not swallowed) so the dispatcher.log records
    anomalies the older ``2>/dev/null`` form silently discarded.  After the
    2026-05-18 RunPod failure (final-pulled run.log was 3 minutes shorter
    than the dispatcher's live tail had captured), we want to *hear* tar
    say things like "file shrank by N bytes" or "file removed before we
    read it" instead of finding out forensically.  We suppress only
    ``file-changed`` because it's *expected* during partial pulls of a
    growing ``run.log`` and would otherwise dominate the log.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Use a tempfile for the remote (ssh) stderr instead of a PIPE: tar can
    # emit warnings before all of stdout has been read by the local tar, and
    # a PIPE that nobody drains could (in principle) deadlock if the buffer
    # filled.  The tempfile is bounded only by disk, never blocks.
    remote_stderr_tmpf = tempfile.TemporaryFile(mode="w+b")
    try:
        ssh = subprocess.Popen(
            ssh_cmd(ssh_target) + [
                "tar --warning=all --warning=no-file-changed "
                f"-czf - -C {REMOTE_BASE}/{benchmark} results"
            ],
            stdout=subprocess.PIPE,
            stderr=remote_stderr_tmpf,
        )
        tar = subprocess.Popen(
            ["tar", "xzf", "-", "-C", str(dest_dir)],
            stdin=ssh.stdout,
            stderr=subprocess.PIPE,
        )
        if ssh.stdout is not None:
            ssh.stdout.close()  # allow tar to receive SIGPIPE if ssh exits
        _, local_stderr_bytes = tar.communicate()
        ssh.wait()

        remote_stderr_tmpf.seek(0)
        remote_stderr_bytes = remote_stderr_tmpf.read()
    finally:
        remote_stderr_tmpf.close()

    remote_stderr = remote_stderr_bytes.decode("utf-8", errors="replace").strip()
    local_stderr = (local_stderr_bytes or b"").decode("utf-8", errors="replace").strip()
    if remote_stderr:
        print(
            "  [pull] remote tar warnings:\n"
            + "\n".join(f"    {line}" for line in remote_stderr.splitlines()),
            file=sys.stderr, flush=True,
        )
    if local_stderr:
        print(
            "  [pull] local tar warnings:\n"
            + "\n".join(f"    {line}" for line in local_stderr.splitlines()),
            file=sys.stderr, flush=True,
        )
    return tar.returncode == 0 and ssh.returncode == 0


# ---------------------------------------------------------------------------
# Remote-finalize payload push
#
# (Distinct from run_on_cloud.py's `--auto-finalize-slug` feature, which
# runs scripts/finalize_run.py locally after a successful dispatch.
# Remote-finalize is the POD-side encrypt-upload-self-terminate stack;
# the two features are independent and may be enabled separately.)
# ---------------------------------------------------------------------------

# Where pod_finalize.py + synth_crypto.py + the recipient pubkey live on
# the pod when remote-finalize is enabled.  Kept out of the benchmark
# dir so that pushing a new benchmark version doesn't churn these files.
REMOTE_FINALIZE_DIR = f"{REMOTE_BASE}/_finalize"

# Minimal pip requirements installed on the pod when remote-finalize is
# on.  Distinct from scripts/requirements.txt (which carries dev-only
# extras like pytest); this list is intentionally as small as possible
# so a pip failure on the pod has a tiny blast radius.
_FINALIZE_PIP_REQUIREMENTS = (
    "cryptography>=42.0,<49.0",
    "pqcrypto>=0.4.0,<0.5.0",
    "b2sdk>=2.0,<3.0",
)


class FinalizePayloadError(RuntimeError):
    """Raised when the local finalize payload (scripts/pubkey) is missing
    or malformed; bubbled up to the dispatcher so it can refuse to launch
    a run that wouldn't be able to finalize itself.
    """


def push_finalize_payload(
    project_root: Path,
    ssh_target: List[str],
    *,
    pubkey_path: Path,
) -> None:
    """Push the minimum files needed for the pod to encrypt + upload + self-terminate.

    The pod's auto-finalize stack consists of THREE files plus a small
    requirements list:

      * ``scripts/synth_crypto.py``  -- the hybrid PQ encryption module
      * ``scripts/pod_finalize.py``  -- the orchestrator
      * ``<pubkey_path>``            -- the recipient public key the pod
                                        encrypts to

    All three are placed under :data:`REMOTE_FINALIZE_DIR` so the wrapper
    can reference them with deterministic paths.

    Pip dependencies (``cryptography``, ``pqcrypto``, ``b2sdk``) get
    written into a small ``requirements.txt`` in the same dir; the wrapper
    picks them up via the existing benchmark-deps install loop.

    Raises :class:`FinalizePayloadError` if any required file is missing
    locally -- the dispatcher uses this to fail fast rather than launch a
    run that would silently lose its remote-finalize capability.
    """
    sc_path = project_root / "scripts" / "synth_crypto.py"
    pf_path = project_root / "scripts" / "pod_finalize.py"
    pk_path = pubkey_path.expanduser().resolve()

    for label, p in [("synth_crypto.py", sc_path),
                     ("pod_finalize.py", pf_path),
                     ("recipient pubkey", pk_path)]:
        if not p.is_file():
            raise FinalizePayloadError(
                f"auto-finalize payload missing: {label} at {p}.  "
                "Run `make synth-keygen` first (or restore the pubkey from "
                "the committed copy at infra/keys/synth-results.pub)."
            )

    print(f"==> Pushing remote-finalize payload to {REMOTE_FINALIZE_DIR}/", flush=True)

    # Stage the three files + a minimal requirements.txt into a tmpdir;
    # we ship the lot in one tar to avoid four round-trips.
    with tempfile.TemporaryDirectory() as staging:
        staging_path = Path(staging) / "_finalize"
        staging_path.mkdir()
        shutil.copy2(sc_path, staging_path / "synth_crypto.py")
        shutil.copy2(pf_path, staging_path / "pod_finalize.py")
        shutil.copy2(pk_path, staging_path / "synth-results.pub")
        (staging_path / "requirements.txt").write_text(
            "# Auto-finalize stack only; installed on the pod by the\n"
            "# benchmark-deps loop in the remote wrapper.\n"
            + "\n".join(_FINALIZE_PIP_REQUIREMENTS) + "\n"
        )

        subprocess.run(
            ssh_cmd(ssh_target) + [f"mkdir -p {REMOTE_FINALIZE_DIR}"], check=True,
        )

        tar = subprocess.Popen(
            GTAR_CREATE + TAR_EXCLUDES
            + ["-C", str(staging_path.parent), staging_path.name],
            stdout=subprocess.PIPE,
            env=TAR_ENV,
        )
        ssh = subprocess.Popen(
            ssh_cmd(ssh_target) + [
                f"tar xzf - --no-same-owner --strip-components=1 -C {REMOTE_FINALIZE_DIR}"
            ],
            stdin=tar.stdout,
        )
        if tar.stdout is not None:
            tar.stdout.close()
        ssh.communicate()
        if ssh.returncode != 0:
            raise RuntimeError(
                f"Failed to push finalize payload (ssh rc={ssh.returncode})"
            )


# ---------------------------------------------------------------------------
# Remote wrapper template
# ---------------------------------------------------------------------------

def build_remote_wrapper(
    benchmark: str,
    smoke: bool,
    extra_args: str = "",
    *,
    remote_finalize: dict | None = None,
) -> str:
    """Return the body of ``/workspace/synth/<benchmark>/results/_run.sh``.

    The wrapper performs the CUDA sanity check, installs requirements
    (filtering out torch so the host's CUDA-matched torch wheel stays put),
    and execs the entry point under ``stdbuf -oL -eL python3 -u`` so the
    polling tail sees fresh log lines without waiting for full buffer flushes.
    ``stdbuf`` covers libc-level streams; ``-u`` covers Python's own stdout
    buffering (a defense-in-depth pair after the 2026-05-18 RunPod failure
    where ``run.log`` was missing the last ~3 minutes of output that the
    dispatcher's live tail had captured).

    Three sentinels are written for finalize_run.py:
      * ``_STARTED.<utc>``   at wrapper launch (filesystem-level start time).
      * ``_DONE.<rc>``       at exit (existing contract: dispatcher uses rc).
      * ``_FINISHED.<utc>``  at exit (filesystem-level end time).

    Exit-path diagnostics:
      * ``set -x`` is enabled AFTER python returns so the trailing echo /
        df / free / nvidia-smi / touch commands are xtrace-logged into
        run.log.  If a future failure produces an unexpected rc, we want
        an unambiguous record of which trailing commands ran (and which
        didn't).
      * Disk (``df``), RAM (``free``), and GPU (``nvidia-smi``) state is
        captured at exit so we can correlate a non-zero rc with resource
        exhaustion without having to rerun.
    """
    smoke_flag = "--smoke" if smoke else ""
    # The benchmark-deps install loop globs `{REMOTE_BASE}/*/requirements.txt`,
    # which automatically picks up `{REMOTE_BASE}/_finalize/requirements.txt`
    # iff push_finalize_payload() ran for this launch.  No special-case
    # needed.
    install_cmd = (
        "for req in "
        f'"{REMOTE_BASE}/{benchmark}/requirements.txt" '
        f'"{REMOTE_BASE}"/*/requirements.txt; do '
        'if [ -f "$req" ]; then '
        '  echo "    [pip] installing from $req (skipping torch)"; '
        "  grep -v -E '^[[:space:]]*torch([[:space:]]|==|>=|<=|~=|!=|$)' \"$req\" "
        "    | grep -v -E '^[[:space:]]*(#|$)' "
        "    | xargs -r pip install --quiet --no-cache-dir; "
        "fi; "
        "done"
    )
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_export = f'export HF_TOKEN="{hf_token}"' if hf_token else "# HF_TOKEN not set"

    # Remote-finalize block, injected into the wrapper's EXIT path (after
    # the _DONE / _FINISHED sentinels are written, so the dispatcher's
    # poll loop has already detected completion and pulled results).
    # Encrypts, uploads to B2, and self-terminates the pod.  Failures
    # here are non-fatal: if any step blows up, we leave _DONE /
    # _FINISHED in place and exit with the original RC so the operator's
    # SCP-pull / make reattach is still the fallback path.
    finalize_block = _build_finalize_block(benchmark, remote_finalize)

    return f"""#!/usr/bin/env bash
# Auto-generated by scripts/run_on_cloud.py.  Runs the benchmark entry point
# under nohup so it survives any SSH disconnect on the dispatcher side.
set -u
{hf_export}
RESULTS_DIR="{REMOTE_BASE}/{benchmark}/results"
mkdir -p "$RESULTS_DIR"
cd "{REMOTE_BASE}/{benchmark}"

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Filesystem-level start sentinel.  Stable across any future churn in the
# `[wrapper] starting at ...` log line format (which finalize_run.py used to
# regex-scrape).
touch "$RESULTS_DIR/_STARTED.$STARTED_UTC"
echo "[wrapper] starting at $STARTED_UTC"
echo "[wrapper] benchmark={benchmark} smoke={'1' if smoke else '0'}"
echo "[wrapper] PID=$$"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not detected'; print('CUDA OK:', torch.cuda.get_device_name(0))" || {{
    echo 999 > "$RESULTS_DIR/_DONE.999"
    exit 999
}}

{install_cmd}

if [ -f run.py ]; then
    ENTRY=run.py
elif [ -f run_experiment.py ]; then
    ENTRY=run_experiment.py
else
    echo 'ERROR: no run.py or run_experiment.py found' >&2
    echo 2 > "$RESULTS_DIR/_DONE.2"
    exit 2
fi

echo "[wrapper] entry=$ENTRY  smoke_flag={smoke_flag!r}  extra_args={extra_args!r}"
# python3 -u: disable Python's own stdout/stderr buffering.
# stdbuf -oL -eL: line-buffer libc-level streams.
# Both layers together ensure each `print(..., flush=True)` is on disk
# by the time the next line of Python executes.
stdbuf -oL -eL python3 -u "$ENTRY" {smoke_flag} {extra_args}
RC=$?
FINISHED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Trace the trailing exit-path commands.  xtrace output goes to stderr,
# which is merged with stdout into run.log, so every trailing command
# appears verbatim in the log with a `+` prefix.  Diagnostic-only;
# verbosity is bounded (constant number of commands).
set -x
echo "[wrapper] python exit=$RC at $FINISHED_UTC"
# Exit-state snapshot.  Each tool is best-effort so a missing binary
# (slim images sometimes omit `free` or `nvidia-smi`) never changes RC.
echo "[wrapper] --- exit-state snapshot ---"
df -h "$RESULTS_DIR" /workspace 2>&1 | sed 's/^/[wrapper][df] /' || true
if command -v free >/dev/null 2>&1; then
    free -h 2>&1 | sed 's/^/[wrapper][free] /' || true
else
    echo "[wrapper][free] not available on this image"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
        --format=csv 2>&1 | sed 's/^/[wrapper][nvidia-smi] /' || true
else
    echo "[wrapper][nvidia-smi] not available on this image"
fi
echo "[wrapper] --- end exit-state snapshot ---"
touch "$RESULTS_DIR/_DONE.$RC"
touch "$RESULTS_DIR/_FINISHED.$FINISHED_UTC"
# Best-effort fsync so the sentinel + tail of run.log are durable on the
# pod's filesystem before the dispatcher's final pull.  Catches the case
# where a network-backed volume mount has unflushed writes that would be
# lost if the pod were destroyed/replaced before the kernel's dirty-page
# writeback fires (the leading hypothesis for the truncated run.log on
# the 2026-05-18 RunPod failure).
sync || true

{finalize_block}

exit $RC
"""


def _build_finalize_block(benchmark: str, remote_finalize: dict | None) -> str:
    """Return the bash snippet that encrypts + uploads + self-terminates.

    Returns a placeholder comment when ``remote_finalize`` is None (the
    existing behavior: dispatcher SCP-pulls and destroys; no pod-side
    action).

    When enabled, the snippet:

      * Skips the whole block if the benchmark exited non-zero (we never
        encrypt-and-upload a failed run; the operator will want to SSH
        in and debug).
      * Skips if the finalize payload isn't on the pod (defensive; should
        never happen because the dispatcher's launch path runs
        push_finalize_payload first when this block is non-empty).
      * Exports B2 + RunPod secrets, then calls pod_finalize.py.
      * Captures pod_finalize's stdout/stderr into run.log with a
        ``[finalize]`` prefix so the dispatcher's live tail surfaces it.
      * Failure is non-fatal: a ``|| echo ...`` clause keeps the
        wrapper's exit code at the benchmark's RC.
    """
    if not remote_finalize:
        return "# remote-finalize: disabled for this run"

    required = ("run_id", "b2_bucket", "b2_key_id", "b2_key", "runpod_api_key", "pod_id")
    missing = [k for k in required if not remote_finalize.get(k)]
    if missing:
        raise ValueError(
            f"remote_finalize config missing required keys: {missing}"
        )

    # Shell-escape each secret literal.  Using single-quoted bash strings
    # plus the standard "close-quote, escape, reopen-quote" idiom for any
    # embedded single quotes; this is the safest form for secrets that
    # could contain any printable byte.
    def _sh_quote(s: str) -> str:
        return "'" + s.replace("'", "'\"'\"'") + "'"

    run_id = _sh_quote(remote_finalize["run_id"])
    b2_bucket = _sh_quote(remote_finalize["b2_bucket"])
    b2_key_id = _sh_quote(remote_finalize["b2_key_id"])
    b2_key = _sh_quote(remote_finalize["b2_key"])
    runpod_api_key = _sh_quote(remote_finalize["runpod_api_key"])
    pod_id = _sh_quote(remote_finalize["pod_id"])

    return f"""# ---- remote-finalize block ----------------------------------------------
# Encrypt + upload to B2 + self-terminate the pod.  Only runs on a
# successful benchmark exit (RC=0); failed runs are left intact for
# manual investigation via SCP.
if [ "$RC" = "0" ] \\
        && [ -f "{REMOTE_FINALIZE_DIR}/pod_finalize.py" ] \\
        && [ -f "{REMOTE_FINALIZE_DIR}/synth_crypto.py" ] \\
        && [ -f "{REMOTE_FINALIZE_DIR}/synth-results.pub" ]; then
    echo "[wrapper] remote-finalize: invoking pod_finalize.py"
    # Secrets exported only within this subshell; visible to the python
    # process below but not to any later wrapper commands (there are
    # none after this block aside from `exit $RC`).
    (
        export B2_KEY_ID_POD={b2_key_id}
        export B2_KEY_POD={b2_key}
        export B2_BUCKET={b2_bucket}
        export RUNPOD_API_KEY={runpod_api_key}
        # RUNPOD_POD_ID is NOT auto-injected by RunPod's runtime into
        # SSH login sessions, so the dispatcher forwards it explicitly
        # from the provider's create-instance response.  Without this
        # explicit forward, pod_finalize.py would FATAL on the env-var
        # check and the pod would never self-terminate.
        export RUNPOD_POD_ID={pod_id}
        python3 "{REMOTE_FINALIZE_DIR}/pod_finalize.py" \\
            --results-dir "$RESULTS_DIR" \\
            --recipient-pubkey "{REMOTE_FINALIZE_DIR}/synth-results.pub" \\
            --bench '{benchmark}' \\
            --run-id {run_id} \\
            --b2-bucket {b2_bucket} \\
            2>&1 | sed 's/^/[finalize] /'
    ) || echo "[wrapper] remote-finalize: pod_finalize.py FAILED; pod left alive for SCP fallback"
elif [ "$RC" != "0" ]; then
    echo "[wrapper] remote-finalize: skipped (benchmark RC=$RC != 0; leaving pod alive for investigation)"
else
    echo "[wrapper] remote-finalize: payload missing under {REMOTE_FINALIZE_DIR}/; skipping"
fi
# ---- end remote-finalize block ------------------------------------------"""


def launch_remote_detached(
    ssh_target: List[str],
    benchmark: str,
    smoke: bool,
    extra_args: str = "",
    *,
    remote_finalize: dict | None = None,
) -> dict:
    """Write the wrapper to the remote, kick it off under nohup, return run info.

    Pass ``remote_finalize`` (a dict containing run_id, b2_bucket,
    b2_key_id, b2_key, and runpod_api_key) to enable the pod-side
    encrypt + upload + self-terminate stack.  Defaults to ``None``
    which preserves the original behavior (dispatcher SCP-pulls and
    destroys).
    """
    print(f"==> Launching {benchmark} on remote under nohup (smoke={smoke}"
          f"{', extra_args=' + repr(extra_args) if extra_args else ''}"
          f"{', remote_finalize=on' if remote_finalize else ''})",
          flush=True)
    wrapper_body = build_remote_wrapper(
        benchmark, smoke, extra_args, remote_finalize=remote_finalize,
    )
    results_dir = f"{REMOTE_BASE}/{benchmark}/results"
    log_path = f"{results_dir}/run.log"
    pid_path = f"{results_dir}/run.pid"
    wrapper_path = f"{results_dir}/_run.sh"
    done_glob = f"{results_dir}/_DONE.*"

    # Stage 1: create results dir + write the wrapper file via a heredoc.
    setup_cmd = (
        "set -e\n"
        f"mkdir -p {results_dir}\n"
        # Remove stale _DONE sentinels from any prior attempt on the same pod.
        f"rm -f {results_dir}/_DONE.* {pid_path}\n"
        f"cat > {wrapper_path} <<'__SM_RUNNER_EOF__'\n"
        f"{wrapper_body}\n"
        "__SM_RUNNER_EOF__\n"
        f"chmod +x {wrapper_path}\n"
    )
    r = subprocess.run(
        ssh_cmd(ssh_target) + [setup_cmd],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Failed to stage remote wrapper (rc={r.returncode}): {r.stderr}"
        )

    # Stage 2: launch the wrapper fully detached from the SSH session.  The
    # outer subshell whose fds point at /dev/null is the critical trick:
    # without it, OpenSSH waits for the backgrounded process's inherited fds
    # to close before returning, which means the dispatcher's SSH call blocks
    # for the entire run.
    launch_cmd = (
        f"( nohup {wrapper_path} > {log_path} 2>&1 < /dev/null & ) "
        "</dev/null >/dev/null 2>&1"
    )
    r = subprocess.run(
        ssh_cmd(ssh_target) + [launch_cmd],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"Failed to launch remote wrapper (rc={r.returncode}): "
            f"stderr={r.stderr!r} stdout={r.stdout!r}"
        )
    print(f"    [remote] launched detached; log={log_path}", flush=True)
    return {
        "log": log_path,
        "done_glob": done_glob,
        "results_dir": results_dir,
        "wrapper_path": wrapper_path,
    }


# ---------------------------------------------------------------------------
# Polling loop: tail remote log, partial-pull results, watch for _DONE
# ---------------------------------------------------------------------------

def remote_paths_for(benchmark: str) -> dict:
    """Return the deterministic remote paths produced by ``build_remote_wrapper``.

    Centralized so re-attach (which has only the benchmark name from
    ``pod_provisioning.json``) and the launch path agree byte-for-byte.
    """
    results_dir = f"{REMOTE_BASE}/{benchmark}/results"
    return {
        "results_dir": results_dir,
        "log": f"{results_dir}/run.log",
        "done_glob": f"{results_dir}/_DONE.*",
        "wrapper_path": f"{results_dir}/_run.sh",
    }


def poll_until_done(
    ssh_target: List[str],
    benchmark: str,
    *,
    log_path: str,
    done_glob: str,
    dest: Path,
    poll_interval_s: int = 60,
    max_consecutive_ssh_failures: int = 5,
) -> Tuple[int, Path]:
    """Poll a running remote benchmark for the ``_DONE.<rc>`` sentinel.

    Tails ``log_path`` over SSH each cycle, partial-pulls ``results/`` into
    ``dest``, and watches for ``done_glob`` (``ls`` against the wrapper's
    ``_DONE.<rc>`` sentinel).  On detection, performs a final pull and
    returns ``(rc, dest)``.

    Returns ``(exit_code, local_results_dir)``:
      * ``exit_code`` is the python exit code from the ``_DONE.<rc>`` sentinel
      * ``-1`` if the poll loop aborted after exceeding ``max_consecutive_ssh_failures``
      * ``-2`` if a sentinel filename was malformed (no integer suffix)
      * ``-3`` if the final pull failed (pod is left alive in this case)

    Two entry points share this function:
      1. ``run_remote_experiment`` — fresh launch via ``launch_remote_detached``
         then poll.
      2. ``run_on_cloud.py --reattach`` — skip launch, jump straight here
         using paths derived from ``pod_provisioning.json``'s ``benchmark`` field
         via :func:`remote_paths_for`.

    Both entry points share the function body verbatim so re-attach is
    behaviorally indistinguishable from the latter half of a fresh launch.
    """
    dest.mkdir(parents=True, exist_ok=True)
    print(f"==> Polling for completion.  Partial results -> {dest}", flush=True)
    print(f"    (poll interval {poll_interval_s}s; up to "
          f"{max_consecutive_ssh_failures} consecutive SSH failures tolerated)",
          flush=True)

    log_offset = 0
    consecutive_failures = 0
    last_partial_pull = 0.0
    poll_count = 0
    while True:
        poll_count += 1
        # 1. Tail incremental log lines since last cycle.
        tail_cmd = f"tail -c +{log_offset + 1} {log_path} 2>/dev/null || true"
        try:
            tail = ssh_run(ssh_target, tail_cmd, timeout=60)
        except subprocess.TimeoutExpired as exc:
            consecutive_failures += 1
            print(f"  [poll #{poll_count}] tail timeout: {exc}; "
                  f"failure {consecutive_failures}/{max_consecutive_ssh_failures}",
                  flush=True)
            if consecutive_failures >= max_consecutive_ssh_failures:
                print("!! Too many consecutive SSH failures; aborting poll loop",
                      file=sys.stderr)
                return -1, dest
            time.sleep(min(poll_interval_s, 30))
            continue
        if tail.returncode != 0:
            consecutive_failures += 1
            print(f"  [poll #{poll_count}] tail failed rc={tail.returncode}: "
                  f"{tail.stderr.strip()!r}; "
                  f"failure {consecutive_failures}/{max_consecutive_ssh_failures}",
                  flush=True)
            if consecutive_failures >= max_consecutive_ssh_failures:
                return -1, dest
            time.sleep(min(poll_interval_s, 30))
            continue
        consecutive_failures = 0
        new_text = tail.stdout or ""
        if new_text:
            sys.stdout.write(new_text)
            sys.stdout.flush()
            log_offset += len(new_text.encode("utf-8", errors="replace"))

        # 2. Periodic partial pull.
        if time.time() - last_partial_pull >= poll_interval_s:
            ok = pull_results_into(ssh_target, benchmark, dest)
            if ok:
                print(f"  [poll #{poll_count}] partial pull -> {dest}/results/ ok",
                      flush=True)
            else:
                print(f"  [poll #{poll_count}] partial pull failed (will retry)",
                      flush=True)
            last_partial_pull = time.time()

        # 3. Completion sentinel check.
        check = ssh_run(
            ssh_target,
            f"ls {done_glob} 2>/dev/null | head -1",
            timeout=30,
        )
        sentinel = (check.stdout or "").strip()
        if sentinel:
            try:
                rc = int(sentinel.rsplit(".", 1)[-1])
            except ValueError:
                rc = -2  # malformed sentinel
            print(f"==> Remote run finished. exit_code={rc} sentinel={sentinel}",
                  flush=True)
            # Final pull to capture anything written between the last partial
            # pull and the sentinel touch.
            final_ok = pull_results_into(ssh_target, benchmark, dest)
            if not final_ok:
                print("!! Final results pull failed.  Pod will NOT be auto-destroyed.",
                      file=sys.stderr)
                return rc if rc != 0 else -3, dest
            print(f"    final pull -> {dest}/results/ ok", flush=True)
            return rc, dest

        time.sleep(poll_interval_s)


def run_remote_experiment(
    ssh_target: List[str],
    benchmark: str,
    *,
    smoke: bool = False,
    extra_args: str = "",
    launch_dir: Path,
    poll_interval_s: int = 60,
    max_consecutive_ssh_failures: int = 5,
    remote_finalize: dict | None = None,
) -> Tuple[int, Path]:
    """Launch the benchmark under nohup and poll until completion.

    Returns ``(exit_code, local_results_dir)``.  ``exit_code`` is the python
    exit code from the ``_DONE.<rc>`` sentinel, or a negative sentinel value
    documented in :func:`poll_until_done`.  ``launch_dir`` is the pre-created
    staging folder for this launch (caller is responsible for its lifecycle
    so ``dispatcher.log`` and ``pod_provisioning.json`` can live alongside
    the ``results/`` directory it populates).

    ``remote_finalize`` is an optional config dict that opts the run into
    the pod-side encrypt-upload-self-terminate stack.  When supplied, the
    caller is responsible for having already called
    :func:`push_finalize_payload` so the pod has the scripts + pubkey
    available.  See docs/2026-05-25_remote-finalize-architecture.md.

    Implemented as ``launch_remote_detached(...)`` followed by
    ``poll_until_done(...)``; the re-attach path in ``run_on_cloud.py`` calls
    only the latter, against the same ``benchmark``-derived remote paths.
    """
    info = launch_remote_detached(
        ssh_target, benchmark, smoke, extra_args,
        remote_finalize=remote_finalize,
    )
    return poll_until_done(
        ssh_target,
        benchmark,
        log_path=info["log"],
        done_glob=info["done_glob"],
        dest=launch_dir,
        poll_interval_s=poll_interval_s,
        max_consecutive_ssh_failures=max_consecutive_ssh_failures,
    )
