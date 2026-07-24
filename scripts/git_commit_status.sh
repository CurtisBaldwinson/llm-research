#!/usr/bin/env bash
# One-shot: initialize the Synthetic2 repo and commit the current status.
# Run from anywhere; it cd's to the project root.
set -euo pipefail

cd "$(dirname "$0")/.."   # -> Synthetic2/
echo "[git] repo root: $(pwd)"

# Strip macOS AppleDouble junk that the v1 copy dragged in (safe: resource forks).
find . -name '._*' -not -path './.git/*' -delete 2>/dev/null || true
find . -name '.DS_Store' -not -path './.git/*' -delete 2>/dev/null || true

if [ ! -d .git ]; then
  git init
fi

git add -A
git status --short

git commit -m "Synthetic2: pruned sm_core + Experiment 1 harness + real TRACE loader

- sm_core pruned to the tested core: scalar-lambda EWC-family penalty
  (m_coupled/fisher/uniform), MyelinState + asymptotic lock,
  grad_l2_per_neuron + max-norm signals, auto_assign_roles, D1/D2 +
  setup-agnostic check_degenerate_M guard. Removed all per-pathway
  machinery and the LoRA-category-coupled D5 check. 11/11 tests pass.

- docs/design_notes/01_temporal_fidelity.md: EWC = instantaneous
  curvature snapshot; M = temporal integral + discrete lock; the lock
  is the thing Exp 1 tests; pre-training-as-scaffold deferred to a
  possible Exp 3 gated on Exp 1.

- baselines/: full mandatory set (vanilla, ewc/diagonal-Fisher,
  elastic_reset, model_averaging/AMA, replay) behind a uniform Arm
  interface so the harness never special-cases an arm.

- benchmarks/trace/: Experiment 1 harness. gradient_aligned_m (M+lock)
  vs raw_fisher_signal (same signal, no lock) + baselines. BWT metric
  of record, >2sigma-at-n>=5 gate, refuses <5 seeds in the record path.
  --smoke preflight verified end-to-end; lock confirmed to engage and
  mutate M.

- benchmarks/trace/trace_data.py: real 8-task TRACE loader (official
  ScienceQA..NumGLUE-ds order, prompt/answer JSON, fixed sequence) +
  Llama-3.2-3B + LoRA builder. Fails loudly if data is not staged, so a
  short sequence can never silently corrupt BWT. bench_caps.json +
  requirements.txt added for the GPU dispatcher.

- eval/metrics.py: BWT / ACC / target-task accuracy + 2-sigma gate."

echo
echo "[git] done. Latest commit:"
git --no-pager log --oneline -1
