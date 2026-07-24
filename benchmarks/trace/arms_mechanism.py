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

from baselines.base import Arm, DegenerateSignalError


_SIGNAL_TYPE = "grad_l2_per_neuron"


class _SignalRegArmBase(Arm):
    """Shared machinery: gradient-aligned per-neuron signal + re-anchoring."""

    family = "mechanism"

    def __init__(self, lambda_reg: float = 0.1, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self._reg = None
        self._myelin = None
        self._pad_token_id = kw.get("pad_token_id", None)
        self._consolidation_mode = "task_boundary_isolation_screen"

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
        alpha: float = 0.1,
        beta: float = 0.05,
        **kw,
    ):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.lock_consec_steps = int(lock_consec_steps)
        self.theta_retain = float(theta_retain)
        self.alpha = float(alpha)
        self.beta = float(beta)
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
        )
        _accumulate_signal_into_M(
            model, task_loader, myelin, self._torch_ref(),
            pad_token_id=self._pad_token_id,
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
        })
        return t


class RawFisherSignalArm(_SignalRegArmBase):
    """Phase-1 isolation control: same gradient signal, NO lock.

    Identical to GradientAlignedMArm except ``use_asymptotic_lock=False``.
    Distinct from ``EWCArm`` (canonical cumulative Fisher baseline).
    """

    name = "raw_fisher_signal"

    def __init__(self, lambda_reg: float = 0.1, alpha: float = 0.1,
                 beta: float = 0.05, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._consolidation_mode = "task_boundary_isolation_screen"

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=False,   # <-- the only structural difference
            signal_type=_SIGNAL_TYPE,
        )
        _accumulate_signal_into_M(
            model, task_loader, myelin, self._torch_ref(),
            pad_token_id=self._pad_token_id,
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
        alpha: float = 0.1,
        beta: float = 0.05,
        **kw,
    ):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.lock_consec_steps = int(lock_consec_steps)
        self.theta_retain = float(theta_retain)
        self.alpha = float(alpha)
        self.beta = float(beta)
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
        )
        self._reg = None
        self._step_updates = 0

    def on_task_start(self, model, task_idx: int, prev_loader) -> None:
        if self._capture is not None:
            self._capture.remove()
            self._capture = None
        self._capture = self._ActivityCapture(model, signal_type=_SIGNAL_TYPE)
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
        })
        return t


def _accumulate_signal_into_M(
    model, dataloader, myelin, torch, n_steps=50, pad_token_id=None,
):
    """Run gradient-aligned signal capture over the task data into M.

    Uses ActivityCapture with the grad_l2_per_neuron signal so both
    boundary arms get exactly the same importance signal; only the
    MyelinState config (lock vs no-lock) differs between them.

    Pad positions are excluded from the CE used to form gradients.
    """
    from synthetic_myelination import ActivityCapture, unpack_lm_batch

    capture = ActivityCapture(model, signal_type=_SIGNAL_TYPE)
    capture.register()
    was_training = model.training
    model.train()
    device = next(model.parameters()).device
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
            myelin.update(signals)
            capture.reset()
    finally:
        capture.remove()
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()


ARM_REGISTRY = {
    "gradient_aligned_m": GradientAlignedMArm,
    "raw_fisher_signal": RawFisherSignalArm,
    "gradient_aligned_m_online": GradientAlignedMOnlineArm,
}
