"""Experience-replay baseline (rehearsal).

Keeps a small reservoir of examples from previous tasks and mixes them into
each training batch, so the loss on old tasks is directly rehearsed. This is
the standard rehearsal reference on TRACE.

Replay is a different family again (rehearsal, not regularization or
weight-space). Its inclusion means the mechanism is measured against the
three main CL families at once; a paradigm that cannot beat a small replay
buffer has limited practical value.

Reservoir sampling keeps the buffer an unbiased sample of everything seen
without unbounded memory.
"""

from __future__ import annotations

import random

from .base import Arm


class ReplayArm(Arm):
    name = "replay"
    family = "rehearsal"

    def __init__(
        self,
        lambda_reg: float = 1.0,   # weight on the replay loss term
        buffer_size: int = 256,
        replay_batch: int = 8,
        seed: int = 0,
        **kw,
    ):
        super().__init__(lambda_reg=lambda_reg, **kw)
        self.buffer_size = int(buffer_size)
        self.replay_batch = int(replay_batch)
        self._buffer = []          # list of input_id tensors (1 example each)
        self._seen = 0
        self._rng = random.Random(seed)

    def on_train_start(self, model) -> None:
        self._torch = __import__("torch")

    def _ce_loss_on_sample(self, model, sample):
        from synthetic_myelination import unpack_lm_batch
        device = next(model.parameters()).device
        if isinstance(sample, dict):
            batch = {
                "input_ids": sample["input_ids"].unsqueeze(0)
                if sample["input_ids"].dim() == 1 else sample["input_ids"],
            }
            if "attention_mask" in sample:
                am = sample["attention_mask"]
                batch["attention_mask"] = (
                    am.unsqueeze(0) if am.dim() == 1 else am
                )
        else:
            ids = sample.unsqueeze(0) if sample.dim() == 1 else sample
            batch = ids
        input_ids, labels, attention_mask = unpack_lm_batch(
            batch, device, pad_token_id=self._cfg.get("pad_token_id"),
        )
        try:
            out = model(
                input_ids, labels=labels, attention_mask=attention_mask,
            )
        except TypeError:
            out = model(input_ids, labels=labels)
        return out.loss

    def aux_loss(self, model):
        if not self._buffer:
            return None
        k = min(self.replay_batch, len(self._buffer))
        picks = self._rng.sample(self._buffer, k)
        losses = [self._ce_loss_on_sample(model, s) for s in picks]
        return self.lambda_reg * (sum(losses) / len(losses))

    def on_step(self, model, step: int, batch) -> None:
        # reservoir-sample examples (ids + mask) into the buffer
        if isinstance(batch, dict):
            ids = batch["input_ids"]
            mask = batch.get("attention_mask")
        else:
            ids = batch
            mask = None
        for i in range(ids.shape[0]):
            ex = {"input_ids": ids[i].detach().cpu().clone()}
            if mask is not None:
                ex["attention_mask"] = mask[i].detach().cpu().clone()
            self._seen += 1
            if len(self._buffer) < self.buffer_size:
                self._buffer.append(ex)
            else:
                j = self._rng.randint(0, self._seen - 1)
                if j < self.buffer_size:
                    self._buffer[j] = ex

    def telemetry(self) -> dict:
        t = super().telemetry()
        t.update({
            "buffer_size": self.buffer_size,
            "replay_batch": self.replay_batch,
            "buffer_filled": len(self._buffer),
            "examples_seen": self._seen,
        })
        return t
