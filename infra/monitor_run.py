#!/usr/bin/env python3
"""
Live monitor for a dispatched RunPod benchmark run.

Shows a tail -f style log stream with a live countdown to the next
poll cycle, and lets you press 'p' to trigger an immediate manual
results pull.

Usage:
    python3 scripts/monitor_run.py logs/cag_v101_30seed_20260511-163506.log

    # Or with explicit poll interval (default: auto-detected from log)
    python3 scripts/monitor_run.py logs/cag_v101_30seed_20260511-163506.log --poll-interval 120

Keys:
    p  — trigger an immediate results pull (tar-over-SSH)
    q  — quit the monitor (does NOT stop the run)
    s  — show current progress summary (seeds completed per condition)

The monitor is read-only with respect to the running dispatcher: it
tails the log file and parses SSH/results-dir info from it, then does
its own independent tar-pull when you press 'p'.  It never writes to
the log, never touches the pod lifecycle, and can be started/stopped
freely without affecting the run.
"""
from __future__ import annotations

import argparse
import os
import re
import select
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path


# ANSI escape codes for the status bar
SAVE_CURSOR = "\033[s"
RESTORE_CURSOR = "\033[u"
CLEAR_LINE = "\033[2K"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"
MOVE_TO_BOTTOM = "\033[999B"  # move cursor to bottom of terminal


def parse_ssh_target(log_text: str) -> list | None:
    """Extract the SSH target from dispatcher log output."""
    # Look for: SSH ready: ssh root@1.2.3.4 -p 12345  (method=direct)
    m = re.search(r'SSH ready: ssh (root@[\d.]+) -p (\d+)', log_text)
    if m:
        return ["-p", m.group(2), m.group(1)]
    # Look for proxy: ssh user@ssh.runpod.io
    m = re.search(r'SSH ready: ssh (\S+@ssh\.runpod\.io)', log_text)
    if m:
        return [m.group(1)]
    return None


def parse_results_dir(log_text: str) -> Path | None:
    """Extract the local results directory from dispatcher log output."""
    m = re.search(r'Partial results -> (/\S+)', log_text)
    if m:
        return Path(m.group(1))
    return None


def parse_benchmark(log_text: str) -> str | None:
    """Extract the benchmark name from dispatcher log output."""
    m = re.search(r'Launching (\S+) on pod', log_text)
    if m:
        return m.group(1)
    return None


def parse_last_poll_time(log_text: str) -> float | None:
    """Find the wall-clock time of the most recent [poll #N] line."""
    # We track this by file position: each time we see a poll line,
    # record the wall time.  Called from the main loop, not from log text.
    return None  # handled in-loop instead


# A Phase B completion in the current runner is signalled by a line of the
# form `  [<cond> | seed <s> | step <N>/<N>]  dpo_loss=...` where the
# step count equals the total step count.  Captured with a backreference:
# (\d+)/\1 only matches when numerator == denominator.
_COMPLETION_RE = re.compile(
    r'\[\s*(?P<cond>\S+)\s*\|\s*seed\s+(?P<seed>\d+)\s*\|'
    r'\s*step\s+(?P<n>\d+)/(?P=n)\s*\]'
)


def count_completions(log_text: str) -> dict:
    """Count Phase B completions per condition (final-step line)."""
    counts: dict = {}
    # Use a set on (cond, seed) so multiple matches of the same final-step
    # line (we print it whenever step == n_steps) only count once.
    seen: set = set()
    for m in _COMPLETION_RE.finditer(log_text):
        key = (m.group("cond"), m.group("seed"))
        if key in seen:
            continue
        seen.add(key)
        counts[m.group("cond")] = counts.get(m.group("cond"), 0) + 1
    return counts


def parse_total_runs(log_text: str) -> int | None:
    """Auto-detect total expected (condition, seed) runs from the log.

    Recognises two shapes:
      * SPEC §6 λ sweep mode:
          [lambda_sweep] seeds=[42], conditions=vanilla + M_coupled@λ∈[a, b, c, d]
        → (1 + len(λ values)) × len(seeds).
      * Regular mode:
          [main] seeds: [42, 7, 1337]
          [main] conditions: ['vanilla_DPO', 'low_beta_DPO', ...]
        → len(seeds) × len(conditions).

    Returns None if neither shape is present yet (e.g. log is still in
    the very early pod-creation stage).
    """
    # Sweep mode preamble takes precedence.  This line is emitted by run.py
    # before the main loop and carries the authoritative λ list.
    m = re.search(
        r'\[lambda_sweep\] seeds=\[(.*?)\],\s*conditions=vanilla \+ '
        r'M_coupled@\u03bb\u2208\[(.*?)\]',
        log_text,
    )
    if m:
        seeds = [s for s in m.group(1).split(",") if s.strip()]
        lambdas = [s for s in m.group(2).split(",") if s.strip()]
        if seeds and lambdas:
            return (1 + len(lambdas)) * len(seeds)

    # Regular mode: parse the two `[main] ...:` lines.
    seeds_m = re.search(r'\[main\] seeds:\s*\[(.*?)\]', log_text)
    conds_m = re.search(r'\[main\] conditions:\s*\[(.*?)\]', log_text)
    if seeds_m and conds_m:
        n_seeds = len([s for s in seeds_m.group(1).split(",") if s.strip()])
        n_conds = len([s for s in conds_m.group(1).split(",") if s.strip()])
        if n_seeds and n_conds:
            return n_seeds * n_conds

    return None


def estimate_eta(completions: dict, run_start: float, total_runs: int = 90) -> str:
    """Estimate time remaining based on completed runs and elapsed time."""
    done = sum(completions.values())
    if done == 0:
        return "estimating..."
    elapsed = time.time() - run_start
    per_run = elapsed / done
    remaining = (total_runs - done) * per_run
    if remaining <= 0:
        return "finishing..."
    h = int(remaining) // 3600
    m = (int(remaining) % 3600) // 60
    eta_wall = time.strftime("%H:%M", time.localtime(time.time() + remaining))
    if h > 0:
        return f"~{h}h{m:02d}m left (ETA {eta_wall})"
    return f"~{m}m left (ETA {eta_wall})"


def parse_run_start(log_text: str) -> float | None:
    """Parse the wrapper start time from the log to anchor ETA calculations."""
    m = re.search(r'\[wrapper\] starting at (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z', log_text)
    if m:
        import calendar
        from datetime import datetime, timezone
        dt = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def timestamp_prefix() -> str:
    """Return a dim timestamp prefix for log lines."""
    return f"{DIM}{time.strftime('%H:%M:%S')}{RESET} "


def do_manual_pull(ssh_target: list, benchmark: str, dest: Path) -> bool:
    """Run a tar-over-SSH pull of results/ into dest.  Returns True on success."""
    SSH_OPTS = [
        "-T",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ServerAliveInterval=60",
        "-o", "ServerAliveCountMax=10",
        "-o", "ConnectTimeout=30",
        "-i", os.path.expanduser("~/.ssh/id_ed25519"),
    ]
    remote_base = "/workspace/synth"
    dest.mkdir(parents=True, exist_ok=True)
    ssh = subprocess.Popen(
        ["ssh"] + SSH_OPTS + ssh_target +
        [f"tar czf - -C {remote_base}/{benchmark} results 2>/dev/null"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    tar = subprocess.Popen(
        ["tar", "xzf", "-", "-C", str(dest)],
        stdin=ssh.stdout,
        stderr=subprocess.DEVNULL,
    )
    if ssh.stdout is not None:
        ssh.stdout.close()
    tar.communicate(timeout=120)
    return tar.returncode == 0


def format_countdown(seconds: float) -> str:
    """Format seconds as M:SS."""
    if seconds <= 0:
        return "now"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def get_terminal_width() -> int:
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live monitor for a dispatched RunPod benchmark run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("logfile", type=Path,
                        help="Path to the dispatcher log file (e.g. logs/cag_v101_*.log)")
    parser.add_argument("--poll-interval", type=int, default=0,
                        help="Expected poll interval in seconds (0 = auto-detect from log, default)")
    args = parser.parse_args()

    if not args.logfile.exists():
        print(f"Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    # Read the existing log to extract connection info
    existing = args.logfile.read_text()
    ssh_target = parse_ssh_target(existing)
    results_dir = parse_results_dir(existing)
    benchmark = parse_benchmark(existing)
    poll_interval = args.poll_interval

    # Auto-detect poll interval from log if not specified
    if poll_interval == 0:
        m = re.search(r'poll interval (\d+)s', existing)
        if m:
            poll_interval = int(m.group(1))
        else:
            poll_interval = 120  # fallback

    # Print header
    w = get_terminal_width()
    print(f"{BOLD}{'═' * w}{RESET}")
    print(f"{BOLD}  RunPod Monitor{RESET}  │  log: {args.logfile}")
    print(f"  SSH: {CYAN}{' '.join(ssh_target) if ssh_target else '(not yet detected)'}{RESET}")
    print(f"  Results: {results_dir or '(not yet detected)'}")
    print(f"  Benchmark: {benchmark or '(not yet detected)'}")
    print(f"  Poll interval: {poll_interval}s")
    print(f"{DIM}  Keys: [p] pull now  [s] summary  [q] quit{RESET}")
    print(f"{BOLD}{'═' * w}{RESET}")
    print()

    # Set up non-blocking stdin
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())

        # Open log file and seek to current end for tail behavior.
        # But first, print completions from existing content.
        completions = count_completions(existing)
        if completions:
            print(f"{DIM}  (existing progress: {completions}){RESET}")
            print()

        fp = open(args.logfile, "r")
        fp.seek(0, 2)  # seek to end

        last_poll_wall = time.time()  # assume a poll just happened
        run_start = parse_run_start(existing) or time.time()
        pulling = False

        while True:
            # --- Check for new log content ---
            new_text = fp.read()
            if new_text:
                # Re-parse SSH target / results dir if we didn't have them
                if ssh_target is None:
                    ssh_target = parse_ssh_target(new_text)
                if results_dir is None:
                    results_dir = parse_results_dir(new_text)
                if benchmark is None:
                    benchmark = parse_benchmark(new_text)

                # Detect poll events to reset countdown
                if "[poll #" in new_text and "partial pull" in new_text:
                    last_poll_wall = time.time()

                # Detect completion
                if "Remote run finished" in new_text:
                    sys.stdout.write(new_text)
                    sys.stdout.flush()
                    print(f"\n{GREEN}{BOLD}  ✓ Run complete!{RESET}")
                    break

                # Print the log content with timestamps on significant lines
                for line in new_text.splitlines(True):
                    stripped = line.strip()
                    # Add timestamp to lines that mark progress milestones
                    if (stripped.startswith("[") or
                        stripped.startswith("step ") or
                        "CUDA OK" in stripped or
                        "[pip]" in stripped or
                        "[poll" in stripped):
                        sys.stdout.write(f"{timestamp_prefix()}{line}")
                    else:
                        sys.stdout.write(line)
                sys.stdout.flush()

            # --- Update status bar ---
            elapsed_since_poll = time.time() - last_poll_wall
            next_poll_in = max(0, poll_interval - elapsed_since_poll)
            countdown_str = format_countdown(next_poll_in)

            # Compute ETA periodically (every ~10s to avoid thrash)
            eta_str = ""
            total_runs: int | None = None
            if time.time() % 10 < 0.5:
                full_text = args.logfile.read_text()
                completions = count_completions(full_text)
                total_done = sum(completions.values())
                total_runs = parse_total_runs(full_text)
                eta_str = estimate_eta(
                    completions, run_start, total_runs=total_runs or 1
                )
            else:
                total_done = -1  # skip update this cycle

            status_parts = [
                f"{DIM}next poll:{RESET} {YELLOW}{countdown_str}{RESET}",
            ]
            if eta_str:
                status_parts.append(f"{DIM}ETA:{RESET} {CYAN}{eta_str}{RESET}")
            if total_done >= 0:
                denom = str(total_runs) if total_runs else "?"
                status_parts.append(f"{DIM}{total_done}/{denom}{RESET}")
            status_parts.append(f"{DIM}[p]ull [s]ummary [q]uit{RESET}")
            status = "  " + "  │  ".join(status_parts)
            # Write status on current line, then restore
            sys.stdout.write(f"\r{CLEAR_LINE}{status}")
            sys.stdout.flush()

            # --- Check for keypress (non-blocking, 0.5s timeout) ---
            rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
            if rlist:
                ch = sys.stdin.read(1)
                # Clear the status line before printing response
                sys.stdout.write(f"\r{CLEAR_LINE}")

                if ch.lower() == "q":
                    print(f"\n{DIM}  Monitor exiting. Run continues in background.{RESET}")
                    break

                elif ch.lower() == "p":
                    if ssh_target and benchmark and results_dir:
                        print(f"\n{CYAN}  ⟳ Manual pull...{RESET}", flush=True)
                        ok = do_manual_pull(ssh_target, benchmark, results_dir)
                        if ok:
                            print(f"{GREEN}  ✓ Pull complete → {results_dir}/results/{RESET}")
                        else:
                            print(f"{YELLOW}  ✗ Pull failed (SSH error or no results yet){RESET}")
                    else:
                        print(f"{YELLOW}  ✗ Can't pull yet — SSH target not detected in log{RESET}")

                elif ch.lower() == "s":
                    full_text = args.logfile.read_text()
                    completions = count_completions(full_text)
                    total = sum(completions.values())
                    total_runs = parse_total_runs(full_text)
                    denom = str(total_runs) if total_runs else "?"
                    bar_width = max(total_runs or 1, 5)
                    eta = estimate_eta(
                        completions, run_start, total_runs=total_runs or 1
                    )
                    print(f"\n{BOLD}  Progress: {total}/{denom} condition×seed runs  │  {eta}{RESET}")
                    for cond, n in sorted(completions.items()):
                        # Per-condition bar scales to the per-condition expected
                        # count when we know the total layout, else just shows n.
                        per_cond_expected = (
                            (total_runs // max(len(completions), 1))
                            if total_runs else 0
                        )
                        if per_cond_expected:
                            bar = "█" * min(n, per_cond_expected) + "░" * max(0, per_cond_expected - n)
                            print(f"    {cond:<18}  {bar}  {n}/{per_cond_expected}")
                        else:
                            print(f"    {cond:<18}  {n} done")
                    elapsed = time.time() - run_start
                    if total > 0:
                        print(f"{DIM}    avg {elapsed/total:.0f}s per run  │  "
                              f"elapsed {elapsed/60:.0f}m{RESET}")
                    print()

    finally:
        # Restore terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        try:
            fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
