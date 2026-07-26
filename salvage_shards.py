#!/usr/bin/env python3
"""Salvage calib shards from run.log stdout (pods killed at w=80 before writing JSON).

Reconstructs the SELECTION-FAITHFUL subset of each point dict from the [calib] log
lines and reuses run._build_lookup_and_recommend (no reimplementation) to produce
the same shard JSON make merge-calib consumes. Marked source="log_salvage" and
w=80 recorded as missing. Percent fields are 1-decimal (log precision); frac_mid
is derived as 1 - near_zero - locked_hi (the three M bins partition [0,1]).
"""
import sys, re, json, importlib.util, datetime
from pathlib import Path

REPO = Path("/Users/chunter/code/curtis/Synthetic2")
sys.path.insert(0, str(REPO / "benchmarks" / "trace"))
spec = importlib.util.spec_from_file_location("run", REPO / "benchmarks/trace/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
import fetch_data
TRACE_TASKS = fetch_data.TRACE_TASKS
SIGNAL_TYPE = "grad_l2_per_neuron"
TARGETS = run.PROTECTED_FRAC_TARGETS

LINES = Path("/tmp/calib_salvage_lines.txt").read_text().splitlines()

# [calib] tau=0.10 w=1 task=0 steps=150 meanM=0.95.. %~0=1.9 %hi=93.7 A>t=91.9 hard=n mass=ok
PT = re.compile(
    r"^\[calib\]\s+tau=([\d.]+)\s+w=(\d+)\s+task=(\d+)\s+steps=(\d+)\s+"
    r"meanM=([\d.]+)\s+%~0=([\d.]+)\s+%hi=([\d.]+)\s+A>t=([\d.]+)\s+"
    r"hard=(\w+)\s+mass=(\S+)")

# points grouped by shard (a shard = the set of task ids seen in one FILE block)
shards = {}          # shard_label -> dict(task_ids=set, points=[], header=..., windows_seen=set)
cur = None
for ln in LINES:
    if ln.startswith("###FILE###"):
        cur = {"dir": ln.split(None, 1)[1].strip(), "points": [], "task_ids": set(),
               "windows_seen": set(), "taus_seen": set()}
        continue
    if ln.startswith("[calib] INIT-SIGNAL"):
        # tasks=[0] shard_label='0'
        m = re.search(r"shard_label='([^']*)'", ln)
        cur["shard_label"] = m.group(1) if m else None
        continue
    m = PT.match(ln)
    if not m or cur is None:
        continue
    tau, w, ti, steps, meanM, z, hi, at, hard, mass = m.groups()
    tau = round(float(tau), 4); w = int(w); ti = int(ti); steps = int(steps)
    meanM = float(meanM)
    z = float(z) / 100.0          # frac_near_zero
    hi = float(hi) / 100.0        # frac_locked_hi
    at = float(at) / 100.0        # frac_A_above_tau_mean
    mid = max(0.0, 1.0 - z - hi)  # frac_mid (bins partition [0,1])
    hard_bool = (hard.lower() == "y")
    mass_ok = (mass == "ok")
    beta = round(run._bt_beta_for_breakeven(0.1, tau), 6)
    pt = dict(
        tau=tau, ema_window=w, alpha=0.1, beta=beta,
        task_idx=ti, task=(TRACE_TASKS[ti] if ti < len(TRACE_TASKS) else str(ti)),
        steps=steps, hard_degenerate=hard_bool, bc_warning=None,
        mass_split_ok=mass_ok, mass_split_note=("" if mass_ok else mass),
        frac_A_above_tau_mean=at, frac_A_above_tau_n_steps=steps,
        n=None, mean_M=meanM, frac_near_zero=z, frac_locked_hi=hi,
        frac_mid=mid, bc=None,
        salvage_source="log_salvage_1dp",
    )
    cur["points"].append(pt)
    cur["task_ids"].add(ti); cur["windows_seen"].add(w); cur["taus_seen"].add(tau)
    if cur.get("shard_label") is None:
        cur["shard_label"] = str(ti)
    shards.setdefault(cur["shard_label"], cur)

print(f"Parsed {sum(len(s['points']) for s in shards.values())} points "
      f"across {len(shards)} shards")

out_dir = Path("/tmp/salvaged_shards")
out_paths = []
for label, s in sorted(shards.items(), key=lambda kv: kv[0]):
    pts = s["points"]
    taus = sorted(s["taus_seen"])
    windows = sorted(s["windows_seen"])       # what we actually HAVE (no 80)
    task_ids = sorted(s["task_ids"])
    table, recommended, pmap, frontier = run._build_lookup_and_recommend(
        pts, taus, windows, 0.1)
    payload = dict(
        harness="calibrate_signal",
        design_note="02_signal_normalization_pivot",
        calibration_scope="init_signal",
        calibration_scope_note=("Measured on fresh LoRA-init weights with TRACE task "
            "batches. NOT post-task M."),
        normalization="rank", signal_type=SIGNAL_TYPE, alpha=0.1, base_steps=150,
        taus=taus, ema_windows=windows,
        task_shard=task_ids, task_shard_full_run=False,
        task_shard_label=label, requested_tasks=task_ids, dropped_tasks=[],
        n_trace_tasks=len(TRACE_TASKS),
        points=pts, lookup_table=table, protected_frac_map=pmap,
        frontier_endpoints=frontier, recommended=recommended,
        protected_frac_targets=TARGETS,
        # --- SALVAGE HONESTY MARKERS ---
        salvage=dict(
            source="log_salvage_from_run_log_stdout",
            reason=("pods killed at w=80 step before writing native shard JSON; "
                    "reconstructed from [calib] stdout lines"),
            precision="percent fields 1-decimal (log rounding); frac_mid derived "
                      "as 1-near_zero-locked_hi; n/bc/bc_warning unavailable",
            windows_present=windows, windows_missing=[80],
            reconstructed_at=datetime.datetime.now().isoformat(timespec="seconds"),
        ),
        note="SALVAGED shard. w=80 absent (pods killed before that block).",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"calib_signal_shard{label}.json"
    p.write_text(json.dumps(payload, indent=2) + "\n")
    out_paths.append(str(p))
    rec = recommended
    print(f"  shard {label} (task {task_ids}): {len(pts)} pts, w={windows}, "
          f"rec={'None' if not rec else (rec.get('tau'), rec.get('ema_window'))}")

print("\nWrote:")
for p in out_paths:
    print(" ", p)
