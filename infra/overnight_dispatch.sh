#!/usr/bin/env bash
# overnight_dispatch.sh
#
# *** QUARANTINED (Synthetic2 TRACE era) ***
# This script targets v1 imp05 make targets and paths under scripts/
# (run_on_cloud.py, .venv) that no longer exist. Running it will fail
# mid-dispatch or launch the wrong experiment. Do not use until rewritten
# for benchmarks/trace + infra/run_on_cloud.py + make run-parallel.
#
# Watcher that polls a B2-smoke staging dir until it succeeds (or fails),
# then auto-dispatches the three v2.0.1 / v3.0 multi-seed sweeps:
#
#   PP18 — A_strong multi-seed confirmation   (4 arms)
#   PP16 — KTO multi-seed confirmation         (4 arms)
#   PP17 — IPO operating-point debug           (3 arms)
#
# Total post-smoke compute: 11 parallel pods.  All pods self-terminate to
# B2 via bench_caps.remote_finalize=True.
#
# Success criterion (smoke validated):
#   * <smoke_staging_dir>/results/_DONE.0     exists locally (rc=0)
#   * smoke dispatcher PID exits cleanly (this means the dispatcher's
#     own destroy + finalize hooks ran).
#
# With the B2 pivot patch in run_on_cloud.py (_try_b2_pivot_with_retry):
#   The local _DONE.0 sentinel is written by the dispatcher via TWO
#   paths.  (a) Legacy SCP-pull: poll loop sees the pod's _DONE.<rc>
#   over SSH and pulls it.  (b) Remote-finalize B2 pivot: the poll
#   loop's ssh-budget exhausts (pod self-terminated sshd), dispatcher
#   pivots to B2, downloads + decrypts the ciphertext, untars it into
#   results/ which restores the authentic _DONE.0 written on the pod
#   side, plus a belt-and-suspenders rewrite via _write_b2_pivot_done_
#   sentinel.  Either path writes _DONE.0; this watcher's gate works
#   for both.
#
# Failure criterion (do NOT dispatch sweeps):
#   * <smoke_staging_dir>/results/_DONE.<rc>  with rc != 0, OR
#   * smoke dispatcher PID dies before _DONE appears, OR
#   * timeout reached.
# On the "dispatcher died before _DONE" branch the watcher runs an
# informational B2 status check (b2_smoke_check) before exiting, so
# the morning-debug operator instantly sees whether the pod actually
# finalized (manifest+ciphertext both in B2) and the dispatcher just
# crashed post-finalize, vs. the pod genuinely failed mid-run.
#
# Designed for `caffeinate -i nohup ./scripts/overnight_dispatch.sh ... &`
# so the laptop can sleep (display-off only; -i prevents idle sleep).

set -u

echo "FATAL: infra/overnight_dispatch.sh is quarantined." >&2
echo "It still references v1 imp05 make targets and scripts/run_on_cloud.py." >&2
echo "Use: make run-parallel BENCH=trace  (via infra/launch_parallel_sweep.py)" >&2
echo "Rewrite this watcher before un-quarantining." >&2
exit 99

SMOKE_STAGE="${1:-}"
SMOKE_PID_FILE="${2:-}"
TIMEOUT_S="${3:-3600}"   # default 60 min smoke ceiling
POLL_S=30

if [[ -z "$SMOKE_STAGE" || -z "$SMOKE_PID_FILE" ]]; then
    echo "usage: $0 <smoke_staging_dir> <smoke_pid_file> [timeout_seconds]" >&2
    echo "" >&2
    echo "Example:" >&2
    echo "  $0 lab_reports/_staging/_smokes/05_m_coupled_kl_rlhf_results_TIMESTAMP \\" >&2
    echo "     lab_reports/_staging/_logs/imp05_b2smoke_TIMESTAMP.pid 3600" >&2
    exit 2
fi

SMOKE_PID=$(cat "$SMOKE_PID_FILE")
LOG_DIR="lab_reports/_staging/_logs"
WATCHER_LOG="$LOG_DIR/overnight_dispatch_$(date -u +%Y%m%dT%H%M%SZ).log"

mkdir -p "$LOG_DIR"

log() {
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[$ts] $*" | tee -a "$WATCHER_LOG"
}

# Informational helper: if the smoke dispatcher died unexpectedly, peek
# at B2 to tell the morning-debug operator whether the pod actually
# finalized (manifest+ciphertext both present) and the dispatcher
# crashed post-success, vs. the pod genuinely failed mid-run.  Never
# touches the watcher's exit status -- emits log lines only.
b2_smoke_check() {
    local bench="05_m_coupled_kl_rlhf"
    local run_id
    run_id=$(basename "$SMOKE_STAGE")
    log "  B2 status check for ${bench}/${run_id}:"
    if [[ ! -d scripts/.venv ]]; then
        log "    skipped (scripts/.venv not present; run 'make scripts-venv' to enable)"
        return
    fi
    scripts/.venv/bin/python - "$bench" "$run_id" <<'PY' 2>&1 | sed 's/^/    /' | tee -a "$WATCHER_LOG"
import os, sys
bench, run_id = sys.argv[1], sys.argv[2]
for line in open(".env"):
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1); os.environ[k] = v
try:
    from b2sdk.v2 import B2Api, InMemoryAccountInfo
    api = B2Api(InMemoryAccountInfo())
    api.authorize_account("production", os.environ["B2_KEY_ID_LOCAL"], os.environ["B2_KEY_LOCAL"])
    b = api.get_bucket_by_name(os.environ["B2_BUCKET"])
    prefix = f"{bench}/{run_id}/"
    have_ct = have_mf = False
    for f, _ in b.ls(folder_to_list=prefix, latest_only=False, recursive=True):
        if f.file_name.endswith("/results.synth"): have_ct = True
        if f.file_name.endswith("/manifest.json"): have_mf = True
    if have_ct and have_mf:
        print("VERDICT: pod finalized to B2 successfully; dispatcher crashed post-success.")
        print("         recover with: python scripts/run_on_cloud.py --reattach " + os.path.relpath(os.environ.get("SMOKE_STAGE", "<stage>")))
    elif have_mf:
        print("VERDICT: manifest present but ciphertext absent -- impossible under contract; bucket corrupted.")
    elif have_ct:
        print("VERDICT: ciphertext uploaded but manifest commit-marker missing; pod crashed mid-finalize.")
    else:
        print("VERDICT: pod never reached B2 upload step (no objects at this prefix).")
except Exception as exc:
    print(f"VERDICT: B2 check itself failed ({type(exc).__name__}: {exc})")
PY
}

log "watcher starting"
log "  smoke_stage   = $SMOKE_STAGE"
log "  smoke_pid     = $SMOKE_PID (file: $SMOKE_PID_FILE)"
log "  timeout_s     = $TIMEOUT_S"
log "  poll_interval = ${POLL_S}s"

START=$(date +%s)
while :; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START ))

    # Look for _DONE.<rc> sentinel
    DONE_FILE=$(ls -t "$SMOKE_STAGE/results/_DONE."* 2>/dev/null | head -1 || true)

    if [[ -n "$DONE_FILE" ]]; then
        RC_PART=${DONE_FILE##*_DONE.}
        RC=${RC_PART:0:1}
        # _DONE.0 has just one digit; longer rcs (e.g. _DONE.137) need stricter parse
        RC_NUMERIC=$(echo "${DONE_FILE##*_DONE.}" | grep -oE '^[0-9]+' || echo 1)
        log "smoke _DONE seen: $DONE_FILE (rc=$RC_NUMERIC)"

        if [[ "$RC_NUMERIC" != "0" ]]; then
            log "FAIL: smoke returned non-zero; not dispatching sweeps"
            exit 1
        fi

        # Wait for the dispatcher itself to exit (it's the one that
        # performs the destroy + B2 pivot or SCP fallback after _DONE).
        log "waiting for smoke dispatcher PID $SMOKE_PID to exit"
        WAIT_START=$(date +%s)
        while kill -0 "$SMOKE_PID" 2>/dev/null; do
            sleep 5
            if (( $(date +%s) - WAIT_START > 600 )); then
                log "FAIL: dispatcher hung after _DONE.0 for >10 min; not dispatching"
                exit 2
            fi
        done
        log "smoke dispatcher exited cleanly"

        # Optional informational signal
        if [[ -f "$SMOKE_STAGE/results/_UPLOADED" ]]; then
            log "B2 path validated (informational): _UPLOADED sentinel present"
        else
            log "B2 path NOT exercised (informational): _UPLOADED sentinel absent."
            log "      Most likely the SCP-pull fallback ran.  This is acceptable"
            log "      gating for tonight, but means the B2 path is not yet smoke-"
            log "      validated end-to-end.  Investigate before relying on it for"
            log "      the v3.0 PP3 cross-model run later."
        fi

        log "smoke green; dispatching 3 sweeps"

        # Source .env for env vars (RUNPOD_API_KEY, B2_*)
        set -a
        # shellcheck disable=SC1091
        source .env
        set +a

        log "dispatching imp05-v201-pp18-a-strong-confirm"
        make imp05-v201-pp18-a-strong-confirm 2>&1 | tee -a "$WATCHER_LOG" || {
            log "ERROR: PP18 dispatch failed"
            exit 3
        }
        sleep 5

        log "dispatching imp05-v30-pp16-kto-multiseed"
        make imp05-v30-pp16-kto-multiseed 2>&1 | tee -a "$WATCHER_LOG" || {
            log "ERROR: PP16 dispatch failed"
            exit 4
        }
        sleep 5

        log "dispatching imp05-v30-pp17-ipo-debug"
        make imp05-v30-pp17-ipo-debug 2>&1 | tee -a "$WATCHER_LOG" || {
            log "ERROR: PP17 dispatch failed"
            exit 5
        }

        log "ALL THREE SWEEPS DISPATCHED"
        log "  follow each sweep:  tail -f lab_reports/_staging/_logs/imp05_v*/_arm_*.nohup.log"
        log "  morning task:       run aggregators + scaffold lab reports + verify"
        exit 0
    fi

    # No _DONE yet; check the dispatcher PID is still alive
    if ! kill -0 "$SMOKE_PID" 2>/dev/null; then
        log "FAIL: smoke dispatcher PID $SMOKE_PID died before _DONE; not dispatching"
        log "      tail the dispatcher.log under $SMOKE_STAGE for the cause"
        SMOKE_STAGE="$SMOKE_STAGE" b2_smoke_check
        exit 6
    fi

    if (( ELAPSED >= TIMEOUT_S )); then
        log "FAIL: timeout (${TIMEOUT_S}s) reached without _DONE sentinel"
        log "      smoke is probably stuck; not dispatching sweeps"
        exit 7
    fi

    if (( ELAPSED % 300 == 0 )); then
        log "still waiting; elapsed=${ELAPSED}s, dispatcher alive"
    fi

    sleep "$POLL_S"
done
