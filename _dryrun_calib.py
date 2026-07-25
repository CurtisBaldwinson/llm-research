import sys, json
sys.path.insert(0, "/Users/chunter/code/curtis/Synthetic2/benchmarks/trace")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "run", "/Users/chunter/code/curtis/Synthetic2/benchmarks/trace/run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

print("MAX_ADOPTABLE_EMA_WINDOW =", run.MAX_ADOPTABLE_EMA_WINDOW)
print("MIN_TASKS_FOR_VERDICT    =", run.MIN_TASKS_FOR_VERDICT)

def pt(tau, w, task_idx, hi, mid, z, steps=150, hard=False, mass_ok=True):
    return dict(
        tau=round(tau,4), ema_window=w, alpha=0.1,
        beta=round(run._bt_beta_for_breakeven(0.1, tau),6),
        task_idx=task_idx, task=f"t{task_idx}", steps=steps,
        hard_degenerate=hard, bc_warning=False,
        mass_split_ok=mass_ok, mass_split_note="",
        frac_A_above_tau_mean=1-tau, frac_A_above_tau_n_steps=steps,
        n=1000, mean_M=0.5, frac_near_zero=z, frac_locked_hi=hi, frac_mid=mid, bc=0.8,
    )

points = []
# A) HONEST GOOD: w=5 (adoptable), 3 tasks, protected~0.50, real spread -> the ONLY valid winner
for ti,(hi,mid,z) in enumerate([(0.50,0.30,0.20),(0.48,0.32,0.20),(0.52,0.28,0.20)]):
    points.append(pt(0.50, 5, ti, hi, mid, z))
# B) SINGLE-TASK TRAP: w=5, tau=0.5, ONE task, protected exactly 0.50 -> std would be fake 0.0
points.append(pt(0.50, 5, 7, 0.50, 0.30, 0.20))   # lands in its own (tau=0.50 dup? no-> use diff tau)
# put B at a distinct tau so it forms its own single-task row
points[-1]["tau"] = 0.40
points[-1]["beta"] = round(run._bt_beta_for_breakeven(0.1,0.40),6)
# C) OVER-SMOOTH TRAP: w=80 (non-adoptable), 3 tasks, GLOBALLY lowest std + perfect 0.50 protected
for ti,(hi,mid,z) in enumerate([(0.50,0.30,0.20),(0.50,0.30,0.20),(0.50,0.30,0.20)]):
    points.append(pt(0.50, 80, ti, hi, mid, z))
# D) w=40 (non-adoptable per cap=15) 3 tasks, also very tight
for ti in range(3):
    points.append(pt(0.50, 40, ti, 0.49, 0.31, 0.20))
# E) HARD-DEGENERATE: w=5 adoptable, 3 tasks, but hard_degenerate -> must be excluded
for ti in range(3):
    points.append(pt(0.60, 5, ti, 0.05, 0.10, 0.85, hard=True))

taus = [0.40, 0.50, 0.60]
ema_windows = [1, 5, 15, 40, 80]
table, recommended, pmap, frontier = run._build_lookup_and_recommend(points, taus, ema_windows, 0.1)

print("\n=== ALL ROWS (tau,w,n_tasks,prot,std,hard,selectable) ===")
for t in table:
    if t["n_tasks"]==0: continue
    sel = run._row_selectable_for_verdict(t)
    print(f"tau={t['tau']:.2f} w={t['ema_window']:>2} n={t['n_tasks']} "
          f"prot={t['protected_frac_mean']} std={t['protected_frac_std']} "
          f"hard={t['n_hard_degenerate_tasks']} SELECTABLE={sel}")

print("\n=== RECOMMENDED ===")
print(json.dumps(recommended, indent=2) if recommended else "None")

print("\n=== ASSERTIONS ===")
assert recommended is not None, "expected a valid winner (row A)"
assert recommended["ema_window"] == 5, f"winner must be w=5, got w={recommended['ema_window']}"
assert recommended["tau"] == 0.50, f"winner tau must be 0.50, got {recommended['tau']}"
assert recommended["n_tasks"] >= run.MIN_TASKS_FOR_VERDICT, "winner under-sampled!"
assert recommended["protected_frac_std"] is not None, "winner has fake std!"
# w=80 and w=40 rows exist in the map (measured) but are NOT selectable
w80 = [t for t in table if t["ema_window"]==80 and t["n_tasks"]>0]
w40 = [t for t in table if t["ema_window"]==40 and t["n_tasks"]>0]
assert w80 and all(not run._row_selectable_for_verdict(t) for t in w80), "w=80 leaked into selectable!"
assert w40 and all(not run._row_selectable_for_verdict(t) for t in w40), "w=40 leaked into selectable!"
assert "80" in pmap and pmap["80"], "w=80 missing from protected_frac_map (should be mapped!)"
# single-task row (tau=0.40,w=5) present but std None and not selectable
b = [t for t in table if t["ema_window"]==5 and abs(t["tau"]-0.40)<1e-9]
assert b and b[0]["n_tasks"]==1, "single-task row missing"
assert b[0]["protected_frac_std"] is None, f"single-task std should be None, got {b[0]['protected_frac_std']}"
assert not run._row_selectable_for_verdict(b[0]), "single-task row leaked into selectable!"
# hard-degenerate row not selectable
e = [t for t in table if abs(t["tau"]-0.60)<1e-9 and t["ema_window"]==5]
assert e and not run._row_selectable_for_verdict(e[0]), "hard-degenerate row leaked into selectable!"
print("ALL ASSERTIONS PASSED")
