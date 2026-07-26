"""Mechanism arms for Experiment 1 (and the committed continuous Phase-2 arm).

README Exp 1 question: does the M state machinery (asymptotic soft
penalty-based consolidation lock, bimodal plasticity) add value over raw
diagonal Fisher **when both use the same gradient-derived importance
signal**?

Phase 1 isolation screen (task-boundary M build)
------------------------------------------------
Both boundary arms share:
  - the same importance signal          (signal_type="grad_l2_per_neuron")
  - the same penalty form               (lambda * sum_n W_n (theta - theta_ref)^2)
  - the same re-anchor schedule         (snapshot at each task boundary)
  - M built from a short post-task integral (NOT online during SGD)

The ONLY difference:
  - GradientAlignedMArm : W = M, with the asymptotic LOCK enabled
  - RawFisherSignalArm  : W = the same per-neuron gradient signal, frozen
                          as a static diagonal — NO lock

This is an isolation screen: same signal, same penalty, only
``use_asymptotic_lock`` differs. It does **not** claim continuous temporal
consolidation during training — that is the Phase-2 arm below.

Phase 2 continuous consolidation (on_step)
------------------------------------------
GradientAlignedMOnlineArm keeps a live MyelinState + ActivityCapture and
calls ``myelin.update`` from ``on_step`` so lock counters integrate real
training steps. Still a soft penalty (no hard freeze / apply_gate).

M is intentionally NOT cumulative across tasks the way online-EWC Fisher
is. Multi-task consolidation for M is carried by the asymptotic lock
dynamics (floors that latch and stay); that is the independent variable,
not a baseline to standardize onto EWC accumulation.
"""

from __future__ import annotations

import os
from pathlib import Path

from baselines.base import Arm, DegenerateSignalError


_SIGNAL_TYPE = "grad_l2_per_neuron"
# Design Note 02 (2026-07-24): smoothed-rank signal normalization makes the
# protected fraction distribution-invariant. "rank" routes the per-neuron
# signal through EMA-then-fractional-rank in ActivityCapture._get_signals_per_neuron.
_NORMALIZATION = "rank"

# Defaults before signal_tuning.json is loaded (stale τ≈0.33 if β stays 0.05).
_DEFAULT_ALPHA = 0.1
_DEFAULT_BETA = 0.05
_DEFAULT_EMA_WINDOW = 1


def load_signal_tuning(path=None) -> dict:
    """Load signal_tuning.json (calib → arm autowire). Empty dict if absent."""
    import json
    import os
    from pathlib import Path

    candidates = []
    if path is not None:
        candidates.append(Path(path))
    env = os.environ.get("SYNTH_SIGNAL_TUNING")
    if env:
        candidates.append(Path(env))
    cwd = Path.cwd()
    candidates.extend([
        cwd / "results" / "signal_tuning.json",
        cwd / "signal_tuning.json",
        Path(__file__).resolve().parent / "signal_tuning.json",
    ])
    for p in candidates:
        try:
            if p.is_file():
                data = json.loads(p.read_text())
                if isinstance(data, dict):
                    data = dict(data)
                    data["_loaded_from"] = str(p.resolve())
                    return data
        except Exception:
            continue
    return {}


def _loaded_from_env(tuning: dict) -> bool:
    """True iff this tuning dict was selected via SYNTH_SIGNAL_TUNING.

    An env-selected file is an EXPLICIT request (per _apply_signal_tuning's
    contract) and must hard-raise on unconfirmed, not be silently skipped like a
    filesystem auto-load. We compare the loader's resolved ``_loaded_from``
    against the resolved env path so a relative env value still matches.
    """
    if not tuning:
        return False
    env = os.environ.get("SYNTH_SIGNAL_TUNING")
    if not env:
        return False
    loaded = tuning.get("_loaded_from")
    if not loaded:
        return False
    try:
        return Path(env).resolve() == Path(loaded).resolve()
    except Exception:
        return False


def _apply_signal_tuning(arm, tuning: dict, allow_unconfirmed: bool = False,
                         on_autoload: bool = False) -> None:
    """Overwrite alpha/beta/ema_window from calib recommendation when present.

    SAFETY GATE (DN02 "measured != selectable"): a signal_tuning.json produced
    from init-signal calibration carries ``confirmation_required: true`` until a
    post-1-epoch-task0 confirmation cell agrees on the mass split. We refuse to
    silently lock unconfirmed calibration into a scored arm -- the merge writer
    already gates *writing* the file behind --allow-unconfirmed, and this closes
    the same gap at the *consumption* point. Set env SM_ALLOW_UNCONFIRMED_TUNING=1
    or pass allow_unconfirmed=True to intentionally use unconfirmed tuning.

    AUTO-LOAD vs EXPLICIT (fixes the regression flagged in the 2026-07-25 review):
    the confirmation file must live under benchmarks/trace/ so the dispatcher
    uploads it to the pod -- but that dir is ALSO the silent auto-load search
    path. Hard-raising on auto-load therefore broke *every* default mechanism-arm
    run (GradientAlignedMArm() with no flags). Resolution: distinguish intent.
      * on_autoload=True  (silent filesystem pickup, no explicit request):
            unconfirmed -> WARN and KEEP DEFAULTS (do not apply, do not raise).
            A real run can never be silently shaped by unconfirmed tuning; it
            simply runs on built-in defaults until the flag is cleared.
      * on_autoload=False (explicit --signal-tuning-json / signal_tuning= dict /
            env request): unconfirmed -> hard-raise as before (the one case that
            must never be silent -- someone explicitly asked for this file).
    Confirmed files (confirmation_required=false) auto-adopt in BOTH paths, so
    autowire keeps working the moment confirmation clears the flag.
    """
    if not tuning:
        return
    needs_conf = bool(tuning.get("confirmation_required", False))
    env_ok = os.environ.get("SM_ALLOW_UNCONFIRMED_TUNING", "") == "1"
    # Silent auto-load of an unconfirmed file: never contaminate a real run --
    # skip the tuning, keep defaults, and warn loudly. (Explicit override via
    # allow_unconfirmed/env still applies below.)
    if needs_conf and on_autoload and not (allow_unconfirmed or env_ok):
        src = tuning.get("_loaded_from", "<unknown>")
        print(
            "[arm] WARNING: found UNCONFIRMED signal tuning on the auto-load "
            f"path ({src}, confirmation_required=true). KEEPING BUILT-IN "
            "DEFAULTS for this run -- unconfirmed calibration will not shape a "
            "scored run. Run `make confirm-tuning` to clear the flag, or set "
            "SM_ALLOW_UNCONFIRMED_TUNING=1 to intentionally auto-adopt it.",
            flush=True,
        )
        arm._signal_tuning_source = None
        arm._signal_tuning_confirmed = False
        return
    if needs_conf and not (allow_unconfirmed or env_ok):
        src = tuning.get("_loaded_from", "<unknown>")
        raise RuntimeError(
            "refusing to apply UNCONFIRMED signal tuning to a mechanism arm: "
            f"{src} has confirmation_required=true. Run the post-1-epoch-task0 "
            "confirmation cell and clear the flag, or set "
            "SM_ALLOW_UNCONFIRMED_TUNING=1 (or allow_unconfirmed=True) to "
            "intentionally proceed. "
            f"note: {tuning.get('confirmation_note', '')}"
        )
    if needs_conf:
        print(
            "[arm] WARNING: applying UNCONFIRMED signal tuning "
            f"(confirmation_required=true) from {tuning.get('_loaded_from')} "
            "because override was set. Results are exploratory, not locked.",
            flush=True,
        )
    if "alpha" in tuning:
        arm.alpha = float(tuning["alpha"])
    if "beta" in tuning:
        arm.beta = float(tuning["beta"])
    if "ema_window" in tuning:
        arm.ema_window = int(tuning["ema_window"])
    arm._signal_tuning_source = tuning.get("_loaded_from")
    arm._signal_tuning_confirmed = not needs_conf


class _SignalRegArmBase(Arm):
    """Shared machinery: gradient-aligned per-neuron signal + re-anchoring."""

    family = "mechanism"

    def __init__(self, lambda_reg: float = 0.1, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self._reg = None
        self._myelin = None
        self._pad_token_id = kw.get("pad_token_id", None)
        self._consolidation_mode = "task_boundary_isolation_screen"
        self.ema_window = int(kw.get("ema_window", _DEFAULT_EMA_WINDOW))
        self._signal_tuning_source = None
        # Tuning can arrive three ways; only the last is a SILENT auto-load.
        # We track which so _apply_signal_tuning can be lenient on auto-load and
        # strict on explicit requests (see its docstring / 2026-07-25 review).
        st = kw.get("signal_tuning")                       # explicit dict
        explicit = st is not None
        if st is None and kw.get("signal_tuning_json"):    # explicit path
            st = load_signal_tuning(kw.get("signal_tuning_json"))
            explicit = True
        elif st is None and kw.get("load_signal_tuning", True):  # loader chain
            st = load_signal_tuning()
            # The loader chain includes SYNTH_SIGNAL_TUNING (an EXPLICIT request
            # per the gate contract) *and* silent filesystem auto-load paths.
            # Only the latter should get the lenient on_autoload treatment: an
            # env-selected file must hard-raise on unconfirmed like any other
            # explicit request. Detect env selection by matching the resolved
            # _loaded_from against SYNTH_SIGNAL_TUNING.
            explicit = _loaded_from_env(st)
        if st:
            _apply_signal_tuning(self, st, on_autoload=not explicit)

    def on_train_start(self, model) -> None:
        from synthetic_myelination import (
            MCoupledRegularizer,
            MyelinState,
            check_degenerate_M,
            check_m_hard_degenerate,
        )
        self._MCoupledRegularizer = MCoupledRegularizer
        self._MyelinState = MyelinState
        self._check_degenerate_M = check_degenerate_M
        self._check_m_hard = check_m_hard_degenerate
        self._m_distribution_warning = None  # last task only (convenience)
        # Per-task BC diagnostics — must land in cell checkpoints for go/kill.
        self._m_distribution_warnings: list = []

    def aux_loss(self, model):
        if self._reg is None:
            return None
        return self._reg.penalty()

    def _assert_m_ok(self, myelin, task_idx: int) -> None:
        # Hard abort: empty / all-zero / NaN / Inf (blocking).
        # Same corruption set as Fisher via check_degenerate_importance.
        hard = self._check_m_hard(myelin)
        if hard is not None:
            raise DegenerateSignalError(hard, details={
                "arm": self.name,
                "task_idx": task_idx,
                "kind": "M_hard",
                "consolidation_mode": self._consolidation_mode,
                "m_distribution_warnings": list(self._m_distribution_warnings),
            })
        # Distributional diagnostic (BC / degenerate-bimodal): always run,
        # recorded in telemetry + top-level cell fields. Does not abort the
        # Phase-1 isolation screen — short post-task integrals often have
        # not yet developed BC>0.7.
        warn = self._check_degenerate_M(myelin)
        self._m_distribution_warning = warn
        self._m_distribution_warnings.append({
            "task_idx": int(task_idx),
            "warning": warn,  # None means distribution looked healthy
        })

    def _torch_ref(self):
        return __import__("torch")


class GradientAlignedMArm(_SignalRegArmBase):
    """Phase-1 isolation screen: W=M with asymptotic soft lock, task-boundary build.

    M is estimated with a short post-task integral after each task ends.
    This isolates ``use_asymptotic_lock`` against RawFisherSignalArm under
    an identical signal — it is NOT continuous-during-training consolidation.
    """

    name = "gradient_aligned_m"

    def __init__(
        self,
        lambda_reg: float = 0.1,
        lock_consec_steps: int = 20,
        theta_retain: float = 0.9,
        alpha: float = _DEFAULT_ALPHA,
        beta: float = _DEFAULT_BETA,
        ema_window: int = _DEFAULT_EMA_WINDOW,
        **kw,
    ):
        kw.setdefault("ema_window", ema_window)
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.lock_consec_steps = int(lock_consec_steps)
        self.theta_retain = float(theta_retain)
        # signal_tuning (if loaded in base) already overwrote alpha/beta/ema.
        if not getattr(self, "_signal_tuning_source", None):
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.ema_window = int(ema_window)
        self._consolidation_mode = "task_boundary_isolation_screen"

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        # Build/refresh M from the gradient-aligned signal on the finished
        # task, with the asymptotic soft lock enabled, then re-anchor.
        # Note: M is not cumulative across tasks; lock floors carry
        # consolidation within this post-task integral only for the
        # isolation screen. See GradientAlignedMOnlineArm for live floors
        # that persist across tasks via a long-lived MyelinState.
        myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=True,
            lock_consec_steps=self.lock_consec_steps,
            theta_retain=self.theta_retain,
            signal_type=_SIGNAL_TYPE,
            normalization=_NORMALIZATION,
            ema_window=self.ema_window,
        )
        _accumulate_signal_into_M(
            model, task_loader, myelin, self._torch_ref(),
            pad_token_id=self._pad_token_id,
            ema_window=self.ema_window,
        )
        self._assert_m_ok(myelin, task_idx)
        self._myelin = myelin
        self._reg = self._MCoupledRegularizer(
            model=model,
            myelin_state=myelin,
            lambda_reg=self.lambda_reg,
            regularizer_type="m_coupled",
        )
        self._reg.snapshot_reference()

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({
            "lock_consec_steps": self.lock_consec_steps,
            "theta_retain": self.theta_retain,
            "alpha": self.alpha,
            "beta": self.beta,
            "lock_activations": getattr(self._myelin, "lock_activations", 0),
            "lock_kind": "asymptotic_soft_penalty_based_consolidation",
            "consolidation_mode": self._consolidation_mode,
            "m_cumulative": False,
            "m_distribution_warning": self._m_distribution_warning,
            "m_distribution_warnings": list(self._m_distribution_warnings),
            "signal_health": "ok",
            "importance_kind": "M",
            "ema_window": self.ema_window,
            "normalization": _NORMALIZATION,
            "signal_tuning_source": self._signal_tuning_source,
        })
        return t


class RawFisherSignalArm(_SignalRegArmBase):
    """Phase-1 isolation control: same gradient signal, NO lock.

    Identical to GradientAlignedMArm except ``use_asymptotic_lock=False``.
    Distinct from ``EWCArm`` (canonical cumulative Fisher baseline).
    """

    name = "raw_fisher_signal"

    def __init__(self, lambda_reg: float = 0.1, alpha: float = _DEFAULT_ALPHA,
                 beta: float = _DEFAULT_BETA, ema_window: int = _DEFAULT_EMA_WINDOW,
                 **kw):
        kw.setdefault("ema_window", ema_window)
        super().__init__(lambda_reg=lambda_reg, **kw)
        if not getattr(self, "_signal_tuning_source", None):
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.ema_window = int(ema_window)
        self._consolidation_mode = "task_boundary_isolation_screen"

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=False,   # <-- the only structural difference
            signal_type=_SIGNAL_TYPE,
            normalization=_NORMALIZATION,
            ema_window=self.ema_window,
        )
        _accumulate_signal_into_M(
            model, task_loader, myelin, self._torch_ref(),
            pad_token_id=self._pad_token_id,
            ema_window=self.ema_window,
        )
        self._assert_m_ok(myelin, task_idx)
        # Static snapshot of the gradient-aligned importance — no further
        # myelin.update calls after this boundary.
        self._myelin = myelin
        self._reg = self._MCoupledRegularizer(
            model=model,
            myelin_state=myelin,
            lambda_reg=self.lambda_reg,
            regularizer_type="m_coupled",
        )
        self._reg.snapshot_reference()

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({
            "alpha": self.alpha,
            "beta": self.beta,
            "lock": False,
            "lock_kind": "none",
            "consolidation_mode": self._consolidation_mode,
            "m_cumulative": False,
            "m_distribution_warning": self._m_distribution_warning,
            "m_distribution_warnings": list(self._m_distribution_warnings),
            "signal_health": "ok",
            # Importance diagonal is M (same hard-check set as Fisher);
            # not estimate_fisher_diagonal — isolation control for the lock.
            "importance_kind": "M_as_static_diagonal",
            "ema_window": self.ema_window,
            "normalization": _NORMALIZATION,
            "signal_tuning_source": self._signal_tuning_source,
        })
        return t


class GradientAlignedMOnlineArm(_SignalRegArmBase):
    """Phase-2 continuous consolidation: M/lock updated from ``on_step``.

    Live MyelinState persists across tasks. ActivityCapture is registered
    for the duration of each task so lock_consec_steps counts real SGD
    steps under sustained activity — the temporal claim the isolation
    screen deliberately does not make.

    Still asymptotic (soft, penalty-based) consolidation — no hard freeze
    / apply_gate in this phase.
    """

    name = "gradient_aligned_m_online"

    def __init__(
        self,
        lambda_reg: float = 0.1,
        lock_consec_steps: int = 20,
        theta_retain: float = 0.9,
        alpha: float = _DEFAULT_ALPHA,
        beta: float = _DEFAULT_BETA,
        ema_window: int = _DEFAULT_EMA_WINDOW,
        **kw,
    ):
        kw.setdefault("ema_window", ema_window)
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.lock_consec_steps = int(lock_consec_steps)
        self.theta_retain = float(theta_retain)
        if not getattr(self, "_signal_tuning_source", None):
            self.alpha = float(alpha)
            self.beta = float(beta)
            self.ema_window = int(ema_window)
        self._consolidation_mode = "online_on_step"
        self._capture = None
        self._step_updates = 0

    def on_train_start(self, model) -> None:
        super().on_train_start(model)
        from synthetic_myelination import ActivityCapture
        self._ActivityCapture = ActivityCapture
        self._myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=True,
            lock_consec_steps=self.lock_consec_steps,
            theta_retain=self.theta_retain,
            signal_type=_SIGNAL_TYPE,
            normalization=_NORMALIZATION,
            ema_window=self.ema_window,
        )
        self._reg = None
        self._step_updates = 0

    def on_task_start(self, model, task_idx: int, prev_loader) -> None:
        if self._capture is not None:
            self._capture.remove()
            self._capture = None
        self._capture = self._ActivityCapture(
            model, signal_type=_SIGNAL_TYPE, normalization=_NORMALIZATION,
            ema_window=self.ema_window,
        )
        self._capture.register()

    def on_step(self, model, step: int, batch) -> None:
        if self._capture is None or self._myelin is None:
            return
        # Gradients from the just-completed backward are still on params.
        signals = self._capture.get_signals(normalize=True)
        self._myelin.update(signals)
        self._step_updates += 1
        self._capture.reset()

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        if self._capture is not None:
            self._capture.remove()
            self._capture = None
        self._assert_m_ok(self._myelin, task_idx)
        # Re-anchor penalty reference at the boundary; M/floors persist.
        self._reg = self._MCoupledRegularizer(
            model=model,
            myelin_state=self._myelin,
            lambda_reg=self.lambda_reg,
            regularizer_type="m_coupled",
        )
        self._reg.snapshot_reference()

    def on_train_end(self, model) -> None:
        if self._capture is not None:
            self._capture.remove()
            self._capture = None

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({
            "lock_consec_steps": self.lock_consec_steps,
            "theta_retain": self.theta_retain,
            "alpha": self.alpha,
            "beta": self.beta,
            "lock_activations": getattr(self._myelin, "lock_activations", 0),
            "lock_kind": "asymptotic_soft_penalty_based_consolidation",
            "consolidation_mode": self._consolidation_mode,
            "m_cumulative": False,
            "on_step_updates": self._step_updates,
            "m_distribution_warning": self._m_distribution_warning,
            "m_distribution_warnings": list(self._m_distribution_warnings),
            "signal_health": "ok",
            "importance_kind": "M",
            "ema_window": self.ema_window,
            "normalization": _NORMALIZATION,
            "signal_tuning_source": self._signal_tuning_source,
        })
        return t


def _accumulate_signal_into_M(
    model, dataloader, myelin, torch, n_steps=50, pad_token_id=None,
    ema_window: int = 1, track_tau: float = None,
):
    """Run gradient-aligned signal capture over the task data into M.

    Uses ActivityCapture with the grad_l2_per_neuron signal so both
    boundary arms get exactly the same importance signal; only the
    MyelinState config (lock vs no-lock) differs between them.

    Pad positions are excluded from the CE used to form gradients.

    If ``track_tau`` is set, returns a dict with mean fraction of
    conditioned signal A above that break-even (for calib lag diagnostics).
    Otherwise returns None.
    """
    from synthetic_myelination import ActivityCapture, unpack_lm_batch

    capture = ActivityCapture(
        model, signal_type=_SIGNAL_TYPE, normalization=_NORMALIZATION,
        ema_window=int(ema_window),
    )
    capture.register()
    was_training = model.training
    model.train()
    device = next(model.parameters()).device
    a_exceed = []
    try:
        for step, batch in enumerate(dataloader):
            if step >= n_steps:
                break
            input_ids, labels, attention_mask = unpack_lm_batch(
                batch, device, pad_token_id=pad_token_id,
            )
            model.zero_grad(set_to_none=True)
            try:
                out = model(
                    input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            except TypeError:
                out = model(input_ids, labels=labels)
            out.loss.backward()
            signals = capture.get_signals(normalize=True)
            if track_tau is not None and signals:
                fracs = []
                for v in signals.values():
                    if torch.is_tensor(v) and v.numel() > 0:
                        fracs.append(float((v > track_tau).float().mean().item()))
                if fracs:
                    a_exceed.append(sum(fracs) / len(fracs))
            myelin.update(signals)
            capture.reset()
    finally:
        capture.remove()
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
    if track_tau is None:
        return None
    return {
        "frac_A_above_tau_mean": (
            sum(a_exceed) / len(a_exceed) if a_exceed else None
        ),
        "frac_A_above_tau_n_steps": len(a_exceed),
    }


ARM_REGISTRY = {
    "gradient_aligned_m": GradientAlignedMArm,
    "raw_fisher_signal": RawFisherSignalArm,
    "gradient_aligned_m_online": GradientAlignedMOnlineArm,
}
