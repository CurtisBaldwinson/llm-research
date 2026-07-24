include .env
 export

# ---------------------------------------------------------------------------
# Synthetic2 multi-cloud GPU benchmark runner.
#
# Ported from v1. ONE structural change from v1: the dispatcher now lives in
# infra/ (not scripts/), so CLOUD_SCRIPT / RUNPOD_SCRIPT point at infra/.
# v1-specific convenience wrappers (imp05 etc.) are intentionally dropped;
# Synthetic2 has a single benchmark family (trace).
#
# The dispatcher (infra/run_on_cloud.py) is provider-agnostic; cloud-specific
# control planes live behind a plugin interface in infra/cloud_providers/.
# Select a provider with PROVIDER=...; the default is runpod.
#
# Typical usage:
#   make scripts-venv                                  # one-time: build dispatcher venv
#   make smoke BENCH=trace                             # RunPod A40 SECURE, --smoke preflight
#   make run   BENCH=trace GPU=A40 CLOUD=SECURE        # real run
#   make run-bg BENCH=trace GPU=A40 CLOUD=SECURE       # detached (survives terminal close)
#   make tail                                          # follow latest dispatcher.log
#   make status                                        # dispatcher alive? pod info?
# ---------------------------------------------------------------------------

CLOUD_SCRIPT  := infra/run_on_cloud.py
RUNPOD_SCRIPT := infra/run_on_runpod.py   # legacy CLI shim; delegates to CLOUD_SCRIPT

# Dispatcher venv (b2sdk, coolname, cryptography, pqcrypto, pytest, requests).
SCRIPTS_VENV := infra/.venv
SCRIPTS_PY   ?= $(SCRIPTS_VENV)/bin/python

# Outer nohup logs + PID files for detached launches. The dispatcher writes
# its own richer dispatcher.log under lab_reports/_staging/<bench>_results_*/.
LOG_DIR := lab_reports/_staging/_logs

PROVIDER ?= runpod
# GPU is a comma-separated canonical list (A40,A6000,L40S, ...). Leave unset
# to use the benchmark's bench_caps.json preferred_gpus.
GPU      ?=
# RunPod-only tier selector (SECURE or COMMUNITY). Ignored by other providers.
CLOUD    ?= SECURE

GPU_FLAG    := $(if $(strip $(GPU)),--gpu-type $(GPU))
CLOUD_FLAG  := $(if $(filter runpod,$(PROVIDER)),--cloud $(CLOUD))
EXTRA_FLAGS ?=

# ---------- API-key checks (per provider) ---------------------------------

.PHONY: key-runpod
key-runpod:
	@test -n "$(RUNPOD_API_KEY)" || \
	    (echo "ERROR: RUNPOD_API_KEY not set in .env (https://www.console.runpod.io/user/settings)" && exit 1)

.PHONY: key-lambda_labs
key-lambda_labs:
	@test -n "$(LAMBDA_API_KEY)" || \
	    (echo "ERROR: LAMBDA_API_KEY not set in .env (https://cloud.lambdalabs.com/api-keys)" && exit 1)

.PHONY: key-vultr
key-vultr:
	@test -n "$(VULTR_API_KEY)" || \
	    (echo "ERROR: VULTR_API_KEY not set in .env (https://my.vultr.com/settings/#settingsapi)" && exit 1)

# ---------- Dispatcher venv ------------------------------------------------

## Build (or rebuild) the dispatcher venv from infra/requirements.txt.
.PHONY: scripts-venv
scripts-venv:
	python3 -m venv $(SCRIPTS_VENV)
	$(SCRIPTS_VENV)/bin/pip install --upgrade pip
	$(SCRIPTS_VENV)/bin/pip install -r infra/requirements.txt
	@echo "Dispatcher venv ready at $(SCRIPTS_VENV)"

# ---------- Headline runs -------------------------------------------------

## Run a benchmark on a fresh GPU instance.
## Usage: make run BENCH=<benchmark> [PROVIDER=runpod|lambda_labs|vultr] [GPU=...] [CLOUD=SECURE|COMMUNITY]
run: key-$(PROVIDER)
	@test -n "$(BENCH)" || \
	    (echo "Usage: make run BENCH=<benchmark> [PROVIDER=...] [GPU=...] [CLOUD=SECURE|COMMUNITY] [EXTRA_FLAGS='...']" && exit 1)
	$(SCRIPTS_PY) $(CLOUD_SCRIPT) --provider $(PROVIDER) --benchmark $(BENCH) $(GPU_FLAG) $(CLOUD_FLAG) $(EXTRA_FLAGS)

## Smoke-test a benchmark on a fresh GPU instance (--smoke flag).
## Usage: make smoke BENCH=<benchmark> [PROVIDER=...] [GPU=...] [CLOUD=...]
smoke: key-$(PROVIDER)
	@test -n "$(BENCH)" || \
	    (echo "Usage: make smoke BENCH=<benchmark> [PROVIDER=...] [GPU=...] [CLOUD=SECURE|COMMUNITY] [EXTRA_FLAGS='...']" && exit 1)
	$(SCRIPTS_PY) $(CLOUD_SCRIPT) --provider $(PROVIDER) --benchmark $(BENCH) --smoke $(GPU_FLAG) $(CLOUD_FLAG) $(EXTRA_FLAGS)

# ---------- Detached (background) runs + monitoring ----------------------

## Run a benchmark detached on a fresh GPU instance.
run-bg: key-$(PROVIDER)
	@test -n "$(BENCH)" || \
	    (echo "Usage: make run-bg BENCH=<benchmark> [PROVIDER=...] [GPU=...] [CLOUD=SECURE|COMMUNITY] [EXTRA_FLAGS='...']" && exit 1)
	@mkdir -p $(LOG_DIR)
	@STAMP=$$(date -u +%Y%m%dT%H%M%SZ); \
	 LOGFILE=$(LOG_DIR)/$(BENCH)_run_$$STAMP.nohup.log; \
	 PIDFILE=$(LOG_DIR)/$(BENCH)_run_$$STAMP.pid; \
	 nohup $(SCRIPTS_PY) $(CLOUD_SCRIPT) --provider $(PROVIDER) --benchmark $(BENCH) $(GPU_FLAG) $(CLOUD_FLAG) $(EXTRA_FLAGS) > $$LOGFILE 2>&1 & \
	 echo $$! > $$PIDFILE; \
	 echo "Launched detached run"; \
	 echo "  PID:     $$(cat $$PIDFILE)"; \
	 echo "  Logfile: $$LOGFILE"; \
	 echo ""; \
	 echo "  make tail     # follow dispatcher.log of latest launch"; \
	 echo "  make status   # PID alive? pod info? last poll line?"

## Smoke-test a benchmark detached on a fresh GPU instance.
smoke-bg: key-$(PROVIDER)
	@test -n "$(BENCH)" || \
	    (echo "Usage: make smoke-bg BENCH=<benchmark> [PROVIDER=...] [GPU=...] [CLOUD=SECURE|COMMUNITY] [EXTRA_FLAGS='...']" && exit 1)
	@mkdir -p $(LOG_DIR)
	@STAMP=$$(date -u +%Y%m%dT%H%M%SZ); \
	 LOGFILE=$(LOG_DIR)/$(BENCH)_smoke_$$STAMP.nohup.log; \
	 PIDFILE=$(LOG_DIR)/$(BENCH)_smoke_$$STAMP.pid; \
	 nohup $(SCRIPTS_PY) $(CLOUD_SCRIPT) --provider $(PROVIDER) --benchmark $(BENCH) --smoke $(GPU_FLAG) $(CLOUD_FLAG) $(EXTRA_FLAGS) > $$LOGFILE 2>&1 & \
	 echo $$! > $$PIDFILE; \
	 echo "Launched detached smoke"; \
	 echo "  PID:     $$(cat $$PIDFILE)"; \
	 echo "  Logfile: $$LOGFILE"; \
	 echo ""; \
	 echo "  make tail     # follow dispatcher.log of latest launch"; \
	 echo "  make status   # PID alive? pod info? last poll line?"

## Fan out the TRACE Exp1 matrix as ONE POD PER ARM (7 pods by default).
## Each pod runs its full seed x lambda grid internally. This is the arm-axis
## parallelization (matches v1's proven per-arm fan-out); NOT a flat 175-pod run.
## Fire-and-forget: spawns all arms, writes a manifest, then returns.
##
## Usage:
##   make run-parallel BENCH=trace GPU=A6000 CLOUD=SECURE
##   make run-parallel BENCH=trace GPU=A6000 SEEDS="0 1 2" LAMBDAS="0.01 0.1"
##   make run-parallel BENCH=trace GPU=A6000 ARMS="gradient_aligned_m raw_fisher_signal"
ARMS    ?=
SEEDS   ?= 0 1 2 3 4
LAMBDAS ?= 0.001 0.01 0.03 0.1 0.3
ARMSPEC_DIR := lab_reports/_staging/_armspecs

.PHONY: run-parallel
run-parallel: key-$(PROVIDER)
	@test -n "$(BENCH)" || \
	    (echo "Usage: make run-parallel BENCH=trace [GPU=A6000] [CLOUD=SECURE] [ARMS='...'] [SEEDS='...'] [LAMBDAS='...']" && exit 1)
	@mkdir -p $(ARMSPEC_DIR) $(LOG_DIR)
	@STAMP=$$(date -u +%Y%m%dT%H%M%SZ); \
	 SPEC=$(ARMSPEC_DIR)/$(BENCH)_exp1_$$STAMP.json; \
	 $(SCRIPTS_PY) infra/make_trace_armspec.py \
	     $(if $(strip $(ARMS)),--arms $(ARMS)) \
	     --seeds $(SEEDS) --lambdas $(LAMBDAS) --check --out $$SPEC || exit 1; \
	 echo ""; \
	 echo "==> Launching parallel sweep from $$SPEC"; \
	 $(SCRIPTS_PY) infra/launch_parallel_sweep.py \
	     --name $(BENCH)_exp1_$$STAMP \
	     --benchmark $(BENCH) \
	     --provider $(PROVIDER) \
	     $(GPU_FLAG) $(CLOUD_FLAG) \
	     --arm-spec $$SPEC; \
	 echo ""; \
	 echo "  make tail     # follow dispatcher.log of the most recent arm"; \
	 echo "  make status   # PID alive? pod info? last poll line?"

## Follow the dispatcher.log of the most recent launch.
tail:
	@latest=$$(ls -td lab_reports/_staging/*_results_*/ lab_reports/_staging/_smokes/*_results_*/ 2>/dev/null | head -n1); \
	 if [ -z "$$latest" ]; then echo "No staging dirs under lab_reports/_staging/"; exit 1; fi; \
	 log="$${latest}dispatcher.log"; \
	 if [ ! -f "$$log" ]; then echo "No dispatcher.log at $$log (run may not have reached SSH stage yet)"; exit 1; fi; \
	 echo "==> Following $$log  (Ctrl-C detaches; dispatcher + pod keep running)"; echo ""; \
	 tail -f "$$log"

## Status of the most recent launch (read-only; safe any time).
status:
	@latest=$$(ls -td lab_reports/_staging/*_results_*/ lab_reports/_staging/_smokes/*_results_*/ 2>/dev/null | head -n1); \
	 if [ -z "$$latest" ]; then echo "No staging dirs under lab_reports/_staging/"; exit 0; fi; \
	 echo "Latest staging dir: $$latest"; echo ""; \
	 if [ -f "$${latest}pod_provisioning.json" ]; then echo "--- pod_provisioning.json ---"; cat "$${latest}pod_provisioning.json"; echo ""; fi; \
	 latest_pidfile=$$(ls -t $(LOG_DIR)/*.pid 2>/dev/null | head -n1); \
	 if [ -n "$$latest_pidfile" ]; then \
	     pid=$$(cat "$$latest_pidfile"); \
	     if kill -0 "$$pid" 2>/dev/null; then echo "Dispatcher PID $$pid: ALIVE  (from $$latest_pidfile)"; \
	     else echo "Dispatcher PID $$pid: GONE   (from $$latest_pidfile)"; \
	          echo "  (remote python is nohup-detached and may still be running; 'make tail' shows whether dispatcher.log is growing)"; fi; echo ""; \
	 else echo "No PID file in $(LOG_DIR)/ (launched via make smoke-bg / run-bg?)"; echo ""; fi; \
	 if [ -f "$${latest}dispatcher.log" ]; then echo "--- last 15 lines of dispatcher.log ---"; tail -n 15 "$${latest}dispatcher.log"; fi

## Run the test suite (sm_core + eval + baselines).
.PHONY: test
test:
	python3 -m pytest -q

.PHONY: help
help:
	@grep -E '^##' Makefile | sed 's/^## //'
