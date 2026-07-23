"""The two mechanism arms that Experiment 1 compares head to head.

README Exp 1 question: does the M state machinery (asymptotic lock, bimodal
plasticity) add value over raw diagonal Fisher **when both use the same
gradient-derived importance signal**?

To make that a fair test, both arms below share:
  - the same importance signal          (signal_type="grad_l2_per_neuron")
  - the same penalty form               (lambda * sum_n W_n (theta - theta_ref)^2)
  - the same re-anchor schedule         (snapshot at each task boundary)

The ONLY difference:
  - GradientAlignedMArm : W = M, with the asymptotic LOCK enabled
                          (discrete/bimodal consolidation over time)
  - RawFisherSignalArm  : W = the same per-neuron gradient signal, frozen as
                          a static Fisher-style diagonal — NO lock

So a win for GradientAlignedMArm over RawFisherSignalArm is attributable to
the lock and nothing else. This is the operationalization of Design Note 01:
"the lock is the thing under test."

Note: RawFisherSignalArm is deliberately distinct from the standalone
`EWCArm` baseline. EWCArm uses the canonical Fisher estimate (its own
signal) and is the classical floor. RawFisherSignalArm uses the *identical*
gradient signal as the M arm, isolating the lock. Both are reported.
"""

from __future__ import annotations

from baselines.base import Arm


_SIGNAL_TYPE = "grad_l2_per_neuron"


class _SignalRegArmBase(Arm):
    """Shared machinery: gradient-aligned per-neuron signal + re-anchoring."""

    family = "mechanism"

    def __init__(self, lambda_reg: float = 0.1, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self._reg = None
        self._myelin = None

    def on_train_start(self, model) -> None:
        from synthetic_myelination import MCoupledRegularizer, MyelinState
        self._MCoupledRegularizer = MCoupledRegularizer
        self._MyelinState = MyelinState

    def aux_loss(self, model):
        if self._reg is None:
            return None
        return self._reg.penalty()


class GradientAlignedMArm(_SignalRegArmBase):
    """W = M with the asymptotic LOCK on. The mechanism under test."""

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

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        # Build/refresh M from the gradient-aligned signal on the finished
        # task, with the lock enabled, then re-anchor the penalty reference.
        myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=True,
            lock_consec_steps=self.lock_consec_steps,
            theta_retain=self.theta_retain,
            signal_type=_SIGNAL_TYPE,
        )
        _accumulate_signal_into_M(model, task_loader, myelin, self._torch_ref())
        self._myelin = myelin
        self._reg = self._MCoupledRegularizer(
            model=model,
            myelin_state=myelin,
            lambda_reg=self.lambda_reg,
            regularizer_type="m_coupled",
        )
        self._reg.snapshot_reference()

    def _torch_ref(self):
        return __import__("torch")

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({
            "lock_consec_steps": self.lock_consec_steps,
            "theta_retain": self.theta_retain,
            "alpha": self.alpha,
            "beta": self.beta,
            "lock_activations": getattr(self._myelin, "lock_activations", 0),
        })
        return t


class RawFisherSignalArm(_SignalRegArmBase):
    """W = the SAME gradient signal, frozen as a static diagonal. NO lock.

    Identical to GradientAlignedMArm in every respect except that M is built
    with the lock DISABLED and never latches, so the penalty is a static
    snapshot of the gradient-aligned importance — the "raw diagonal Fisher"
    control with the confound of a different signal removed.
    """

    name = "raw_fisher_signal"

    def __init__(self, lambda_reg: float = 0.1, alpha: float = 0.1,
                 beta: float = 0.05, **kw):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def on_task_end(self, model, task_idx: int, task_loader) -> None:
        myelin = self._MyelinState(
            model,
            alpha=self.alpha,
            beta=self.beta,
            use_asymptotic_lock=False,   # <-- the only structural difference
            signal_type=_SIGNAL_TYPE,
        )
        _accumulate_signal_into_M(model, task_loader, myelin, __import__("torch"))
        # freeze: use M as a static diagonal via the m_coupled path (no more
        # updates happen because we do not call myelin.update after this).
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
        t.update({"alpha": self.alpha, "beta": self.beta, "lock": False})
        return t


def _accumulate_signal_into_M(model, dataloader, myelin, torch, n_steps=50):
    """Run gradient-aligned signal capture over the task data into M.

    Uses ActivityCapture with the grad_l2_per_neuron signal so both arms get
    exactly the same importance signal; only the MyelinState config (lock vs
    no-lock) differs between them.
    """
    from synthetic_myelination import ActivityCapture

    capture = ActivityCapture(model, signal_type=_SIGNAL_TYPE)
    capture.register()
    was_training = model.training
    model.train()
    try:
        for step, batch in enumerate(dataloader):
            if step >= n_steps:
                break
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch
            input_ids = input_ids.to(next(model.parameters()).device)
            model.zero_grad(set_to_none=True)
            out = model(input_ids, labels=input_ids)
            out.loss.backward()
            signals = capture.get_signals(normalize=True)
            myelin.update(signals)
    finally:
        capture.remove()
        model.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()


ARM_REGISTRY = {
    "gradient_aligned_m": GradientAlignedMArm,
    "raw_fisher_signal": RawFisherSignalArm,
}
