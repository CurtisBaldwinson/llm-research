"""Eval-triangle equivalence proof.

Runs ONE continual-learning cell twice on the identical trained trajectory:
  * R_new: the shipped code (evaluates only j<=i; upper triangle stays 0.0)
  * R_old: the previous behaviour (evaluates the full TxT grid every task)

Both share the same init weights, same seed, same optimizer trajectory, so any
difference in R can only come from the eval loop bound -- which is exactly the
change under test. We then assert the metric-bearing entries (lower triangle:
diagonal + final row + all j<=i) are BITWISE identical, and report how many
upper-triangle evals the new code skipped.

Run from smoke_proof/trace/ so the repo's own sys.path bootstrap resolves.
"""
import copy
import json
import os
import sys
import torch

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "eval_triangle_proof_results.json")

import run as R  # the repo module under test (NEW triangle eval)
from synthetic_myelination import unpack_lm_batch  # noqa: F401  (same import run uses)


def run_arm_full_grid(arm, model, sequence, lr, epochs, pad_token_id=None):
    """EXACT copy of run.run_arm_over_sequence but with the OLD full-width eval
    loop (evaluate every task after every task). Everything else identical so
    the training trajectory matches bit-for-bit."""
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    T = len(sequence)
    Rmat = [[0.0] * T for _ in range(T)]
    step_i = 0
    arm.on_train_start(model)
    prev_loader = None
    for i, (train_loader, _eval_loader) in enumerate(sequence):
        arm.on_task_start(model, i, prev_loader)
        model.train()
        for _ep in range(epochs):
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                device = next(model.parameters()).device
                ids, labels, attention_mask = unpack_lm_batch(
                    batch, device, pad_token_id=pad_token_id,
                )
                try:
                    base = model(ids, labels=labels, attention_mask=attention_mask).loss
                except TypeError:
                    base = model(ids, labels=labels).loss
                aux = arm.aux_loss(model)
                loss = base if aux is None else (base + aux)
                loss.backward()
                opt.step()
                arm.on_step(model, step_i, batch)
                step_i += 1
        arm.on_task_end(model, i, train_loader)
        prev_loader = train_loader
        # OLD behaviour: full TxT grid (this is what we are removing)
        for j, (_tl, ev) in enumerate(sequence):
            Rmat[i][j] = R.evaluate(model, ev, pad_token_id=pad_token_id)
    arm.on_train_end(model)
    return Rmat


def build_cell(seed, arm_name="vanilla"):
    torch.manual_seed(seed)
    model = R.build_model("smoke")
    sequence = R.build_task_sequence("smoke", n_tasks=5, seed=0)
    ArmCls = R.ALL_ARMS[arm_name]
    arm = R._construct_arm(ArmCls, 0.1, seed, pad_token_id=None)
    return model, sequence, arm


def run_variant(which, seed, arm_name):
    """Reset to identical init + seed, then run the chosen eval variant."""
    model, sequence, arm = build_cell(seed, arm_name)
    pristine = copy.deepcopy(model.state_dict())
    model.load_state_dict(pristine, strict=True)
    torch.manual_seed(seed)  # reseed AFTER model build so training RNG matches
    if which == "new":
        out = R.run_arm_over_sequence(arm, model, sequence, lr=1e-2, epochs=1,
                                      pad_token_id=None)
        return out["R"], out["metrics"]
    else:
        Rmat = run_arm_full_grid(arm, model, sequence, lr=1e-2, epochs=1,
                                 pad_token_id=None)
        return Rmat, R.summarize(Rmat)


def compare(arm_name, seed=0):
    # Build ONE model, snapshot init, run both variants from that exact snapshot.
    torch.manual_seed(seed)
    model = R.build_model("smoke")
    sequence = R.build_task_sequence("smoke", n_tasks=5, seed=0)
    init = copy.deepcopy(model.state_dict())

    # NEW
    model.load_state_dict(init, strict=True)
    torch.manual_seed(seed)
    arm_new = R._construct_arm(R.ALL_ARMS[arm_name], 0.1, seed, pad_token_id=None)
    out_new = R.run_arm_over_sequence(arm_new, model, sequence, lr=1e-2,
                                      epochs=1, pad_token_id=None)
    R_new, M_new = out_new["R"], out_new["metrics"]

    # OLD (reset to identical init + seed)
    model.load_state_dict(init, strict=True)
    torch.manual_seed(seed)
    arm_old = R._construct_arm(R.ALL_ARMS[arm_name], 0.1, seed, pad_token_id=None)
    R_old = run_arm_full_grid(arm_old, model, sequence, lr=1e-2, epochs=1,
                              pad_token_id=None)
    M_old = R.summarize(R_old)

    T = len(R_new)
    # Lower triangle incl diagonal (the metric-bearing entries)
    lower_identical = all(
        R_new[i][j] == R_old[i][j] for i in range(T) for j in range(i + 1)
    )
    # Upper triangle: new must be exactly 0.0 (never evaluated)
    upper_new_zero = all(
        R_new[i][j] == 0.0 for i in range(T) for j in range(i + 1, T)
    )
    # How many evals the old code did that the new code skipped
    old_evals = T * T
    new_evals = sum(i + 1 for i in range(T))
    metrics_identical = all(
        (M_new[k] == M_old[k]) or (isinstance(M_new[k], float)
                                   and M_new[k] != M_new[k]
                                   and M_old[k] != M_old[k])  # nan==nan
        for k in ("bwt", "acc", "target_task_acc", "learning_acc")
    )
    return {
        "arm": arm_name,
        "seed": seed,
        "n_tasks": T,
        "lower_triangle_bitwise_identical": lower_identical,
        "upper_triangle_new_all_zero": upper_new_zero,
        "metrics_bitwise_identical": metrics_identical,
        "metrics_new": {k: M_new[k] for k in ("bwt", "acc", "target_task_acc", "learning_acc")},
        "metrics_old": {k: M_old[k] for k in ("bwt", "acc", "target_task_acc", "learning_acc")},
        "evals_old_full_grid": old_evals,
        "evals_new_triangle": new_evals,
        "evals_skipped": old_evals - new_evals,
        "eval_reduction_pct": round(100 * (old_evals - new_evals) / old_evals, 1),
        "R_new": R_new,
        "R_old": R_old,
    }


if __name__ == "__main__":
    results = []
    # Test across multiple arms: a lambda-free one, the mechanism arm, ewc, replay.
    for arm in ["vanilla", "gradient_aligned_m", "ewc", "replay", "elastic_reset"]:
        try:
            results.append(compare(arm, seed=0))
        except Exception as e:
            results.append({"arm": arm, "error": repr(e)})
    json.dump(results, open(_OUT, "w"), indent=2)
    for r in results:
        if "error" in r:
            print(f"[{r['arm']:20s}] ERROR: {r['error']}")
            continue
        ok = (r["lower_triangle_bitwise_identical"]
              and r["upper_triangle_new_all_zero"]
              and r["metrics_bitwise_identical"])
        print(f"[{r['arm']:20s}] identical={ok}  "
              f"lower_tri={r['lower_triangle_bitwise_identical']}  "
              f"upper_zero={r['upper_triangle_new_all_zero']}  "
              f"metrics={r['metrics_bitwise_identical']}  "
              f"skipped {r['evals_skipped']}/{r['evals_old_full_grid']} evals "
              f"({r['eval_reduction_pct']}%)")
