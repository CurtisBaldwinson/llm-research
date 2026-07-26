import importlib.util, json
spec = importlib.util.spec_from_file_location(
    "run", "/Users/chunter/code/curtis/Synthetic2/benchmarks/trace/run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)

print("MAX_ADOPTABLE_EMA_WINDOW =", run.MAX_ADOPTABLE_EMA_WINDOW, "(expect 40)")
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
# HONEST GOOD w=5 (3 tasks) - a valid selectable row
for ti,(hi,mid,z) in enumerate([(0.50,0.30,0.20),(0.48,0.32,0.20),(0.52,0.28,0.20)]):
    points.append(pt(0.50, 5, ti, hi, mid, z))
# w=40 (3 tasks): NOW adoptable. Give it a perfect 0.50 + tight spread so it is a
# legitimate selectable competitor (should be selectable, may even win).
for ti,(hi,mid,z) in enumerate([(0.50,0.30,0.20),(0.50,0.30,0.20),(0.50,0.30,0.20)]):
    points.append(pt(0.50, 40, ti, hi, mid, z))
# w=80 (3 tasks): globally lowest std, perfect 0.50 -> must STILL be non-selectable.
for ti in range(3):
    points.append(pt(0.50, 80, ti, 0.50, 0.30, 0.20))

taus = [0.50]
ema_windows = [1, 5, 15, 40, 80]
table, recommended, pmap, frontier = run._build_lookup_and_recommend(points, taus, ema_windows, 0.1)

print("\n=== ROWS (w, n_tasks, prot, std, SELECTABLE) ===")
for t in table:
    if t["n_tasks"]==0: continue
    print(f"w={t['ema_window']:>2} n={t['n_tasks']} prot={t['protected_frac_mean']} "
          f"std={t['protected_frac_std']} SELECTABLE={run._row_selectable_for_verdict(t)}")

print("\n=== RECOMMENDED w =", recommended["ema_window"] if recommended else None)

w40 = [t for t in table if t["ema_window"]==40 and t["n_tasks"]>0][0]
w80 = [t for t in table if t["ema_window"]==80 and t["n_tasks"]>0][0]
assert run._row_selectable_for_verdict(w40), "w=40 should now be SELECTABLE"
assert not run._row_selectable_for_verdict(w80), "w=80 must remain NON-selectable"
assert "80" in pmap and pmap["80"], "w=80 must still be MAPPED (measured, on frontier)"
assert "40" in pmap and pmap["40"], "w=40 must be mapped"
assert recommended is not None and recommended["ema_window"] in (5, 40), \
    f"winner must be an adoptable window, got w={recommended['ema_window']}"
print("\nBOUNDARY ASSERTIONS PASSED: w<=40 selectable, w=80 probe-only, both mapped")
