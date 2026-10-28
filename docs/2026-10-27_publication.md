# Neural Network Research & Engineering: publication record

Date: 2026-10-27, America/Vancouver.
Destination: `CurtisBaldwinson/llm-research`, public, default branch `main`.
Base revision: `01fbf98`, with all 14 original commits retained.
The publication commit identifies the source snapshot covered by this record.

## Scope

The public introduction now uses the title “Neural Network Research & Engineering.”
The original Synthetic2 research protocol is preserved verbatim in
[the protocol archive](2026-10-27_research-protocol.md), with a historical-context
notice. Existing design notes remain historical evidence rather than fresh
verification claims.

The four pre-existing modified benchmark files are included in the publication:
data-count validation and provenance refresh, attention-backend selection and
recording, and inclusion of a tuned baseline point in its sweep. Publication
preparation adds offline data-staging regressions and corrects stale explanatory
comments, without changing those implementations. Local staging artifacts and
credentials remain on disk; ignore rules prevent adding new staging artifacts
and common private-key formats by default. Previously tracked operational
manifests remain tracked and in history. No history rewrite was performed.

## Verification environment and commands

Environment: macOS 26.7.2 on Apple silicon, Python 3.12.13.
Commands run from the repository root:

```sh
git diff --check
python3 -m compileall -q baselines benchmarks/trace benchmarks/trace_preflight eval sm_src infra/*.py infra/cloud_providers scripts tests _*.py salvage_shards.py
python3 -m unittest discover -s tests -p 'test_trace_staging.py' -v
infra/.venv/bin/python -m pytest -q
```

- Whitespace and Python syntax checks passed. Syntax checks do not exercise imports
  or model execution.
- All five offline data-staging regressions passed: unequal held-out counts are
  retained; a smaller training variant, an empty held-out split, and an object
  instead of a record list are rejected; repeated staging refreshes legacy
  provenance without a download or a fabricated archive hash.
- The full pytest attempt stopped during collection: PyTorch was unavailable,
  causing two collection errors and one skipped module. This is an incomplete
  gate, not a passing suite. No dependencies were installed for publication.
- Shell syntax was checked with `bash -n` for each tracked shell script.
- Local assertions checked README destinations, exact preservation of the
  original protocol, equality of the vendored baseline and metric modules, and
  seven excluded paths plus three allowed source/public-key paths.

The complete model suite, synthetic model smoke run, real-data runs, cloud-provider
integration, numerical equivalence, and performance measurements were not run.

## Publication review

A targeted local scan examined 183 file blobs reachable from the original history
and the current publishable files. It checked private-key headers, common provider
and GitHub token formats, credential-bearing addresses, and exact matches to six
local credential values without printing those values. It found no matches.
This is bounded publication screening, not a comprehensive security audit.

Local review covered the introduction, research-status claims, ignore rules,
original history, existing code changes, and the absence of distributed datasets
or model weights in the tracked tree. No browser rendering or assistive-technology
evaluation was performed. No application accessibility, security, privacy,
reliability, or performance certification is claimed.

Automatic approval review rejected the proposed CodeRabbit review upload because
it would send uncommitted repository contents to that third-party service without
explicit authorization for that destination. The upload did not run; review was
completed locally.

## Known research limits

- Phase 1 uses next-token accuracy (NTA) as a proxy, not full task-specific
  benchmark scoring. No validated improvement over comparative methods is claimed.
- The attention backend is recorded by `preflight_batch2.py`, but the scored-run
  gate in `run.py` does not compare it. Do not treat that gate as proof that
  preflight and scored runs used the same attention implementation.
- Dependency files contain version ranges and no complete lockfile. Historical
  results have not been reproduced with a freshly resolved environment.
- Some inherited pathway test files are explicitly retired and contain no active
  tests. They provide no verification evidence.
- There is no repository license file. No new licensing decision was made during
  publication preparation.

Publication records incomplete research software. It does not establish a
compliant software release or validate the historical scientific conclusions.
