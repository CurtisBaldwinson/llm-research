# Neural Network Research & Engineering

Independent large language model (LLM) research & benchmarking.

This repository records my work on continual learning: how neural networks can
learn new tasks while retaining earlier capabilities. It includes experimental
mechanisms, comparative baselines, evaluation code, research decisions, and cloud
experiment infrastructure.

**Research status:** exploratory. Current work centers on calibration, preflight
checks, and controlled comparisons against established baselines.

## Current investigation

The first project, developed under the internal name **Synthetic2**, explores
myelin-inspired consolidation and release of parameter-importance weights.
The central question is whether those dynamics add measurable value beyond
ordinary importance-weighted regularization.

- **Consolidation:** compare a soft consolidation lock with a control using the
  same gradient-derived importance signal without the lock.
- **Release:** investigate whether releasing stale constraints preserves the
  ability to learn later tasks. This remains a research objective, not a result.
- **Comparisons:** Elastic Weight Consolidation (EWC), Elastic Reset, model
  averaging, replay, and an unregularized sanity baseline.

The real-data harness targets the [TRACE continual-learning benchmark](https://github.com/BeyonderXX/TRACE).
Phase 1 measures backward transfer (BWT) using a pad-masked next-token accuracy
(NTA) proxy. It does **not** implement the full set of task-specific TRACE scores,
so these measurements must not be presented as benchmark leaderboard results.

The [original research protocol](docs/2026-10-27_research-protocol.md) preserves
the hypotheses, comparison requirements, and stopping criteria. A positive result
requires at least five seeds, improvement beyond seed variation, a meaningful
stability/plasticity trade-off region, and cost accounting. Negative and
inconclusive findings remain part of the record.

## Explore the work

| Area | Contents |
| --- | --- |
| [Mechanism](sm_src/synthetic_myelination/) | Activity capture, importance state, soft locks, and regularization |
| [Baselines](baselines/) | Comparative continual-learning methods |
| [Experiment harness](benchmarks/trace/) | Data staging, calibration, preflights, sweeps, and result decisions |
| [Metrics](eval/metrics.py) | Forgetting and learning measurements with explicit proxy scope |
| [Infrastructure](infra/) | Cloud dispatch, monitoring, and result collection |
| [Mechanism tests](sm_src/tests/) | Signal, initialization, and normalization regression checks |
| [Data-staging tests](tests/test_trace_staging.py) | Offline checks for split counts, invalid data, and rerun provenance |

The benchmark contains copies of the baseline and metric modules so they can
travel with remote runs. Those copies need to remain consistent with their
root-level counterparts.

Research decisions:

- [Temporal fidelity and the isolation experiment](docs/design_notes/01_temporal_fidelity.md)
- [Signal normalization and protected-fraction calibration](docs/design_notes/02_signal_normalization_pivot.md)
- [Evaluation cost and experiment-matrix efficiency](docs/design_notes/03_compute_efficiency.md)

These are historical design notes. Their recorded measurements and verification
claims have not been independently reproduced for this publication.

## Local checks

The code uses Python, PyTorch, Transformers, and parameter-efficient fine-tuning
(PEFT). Dependency ranges are recorded in
[the benchmark requirements](benchmarks/trace/requirements.txt) and
[the infrastructure requirements](infra/requirements.txt). PyTorch is provided
separately by the execution environment. There is no complete dependency lockfile.

With those dependencies and pytest available, run the regression suite from the
repository root:

```sh
python3 -m pytest -q
```

A central processing unit (CPU) smoke run uses a tiny randomly initialized model
and synthetic tasks. It checks the experiment plumbing and is not research
evidence:

```sh
python3 benchmarks/trace/run.py --smoke --out results/smoke.json
```

Real-data runs require a suitable graphics processing unit (GPU), access to the
chosen base model, and the benchmark data. Cloud commands can provision paid
resources and require operator configuration. The repository does not distribute
model weights or benchmark datasets; their respective access and license terms
apply.
