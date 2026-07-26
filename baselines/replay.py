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
        # Escape hatch: restore the original k-sequential-call replay path
        # (kept only for numerical-equivalence testing; the batched path in
        # aux_loss is output-preserving and the default).
        if not hasattr(self, "use_sequential_replay"):
            self.use_sequential_replay = False

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

    def _padded_batch(self, model, picks):
        """Right-pad ``picks`` into one batch (ids, labels, attention_mask).

        Mirrors the tokenization/label convention of the per-sample path
        (unpack_lm_batch) but stacks all k samples so a single forward pass
        replaces k sequential single-example calls. Labels are padded with
        -100 so padded positions never enter the loss (same masking the
        per-sample path gets from unpack_lm_batch + ignore_index).
        """
        from synthetic_myelination import unpack_lm_batch
        torch = self._torch
        device = next(model.parameters()).device
        pad_id = self._cfg.get("pad_token_id")

        # Normalize each pick to 1-D (ids, mask) on host, then find max length.
        rows = []
        max_len = 0
        for s in picks:
            if isinstance(s, dict):
                ids = s["input_ids"]
                ids = ids.squeeze(0) if ids.dim() == 2 else ids
                am = s.get("attention_mask")
                if am is not None:
                    am = am.squeeze(0) if am.dim() == 2 else am
            else:
                ids = s.squeeze(0) if s.dim() == 2 else s
                am = None
            rows.append((ids, am))
            max_len = max(max_len, ids.shape[0])

        n = len(rows)
        if pad_id is None:
            # No pad id available -> fall back to per-sample path semantics by
            # requiring uniform length (the only safe batched case).
            pad_id = 0
        ids_batch = torch.full((n, max_len), pad_id, dtype=torch.long)
        am_batch = torch.zeros((n, max_len), dtype=torch.long)
        for i, (ids, am) in enumerate(rows):
            L = ids.shape[0]
            ids_batch[i, :L] = ids
            if am is not None:
                am_batch[i, :L] = am
            else:
                am_batch[i, :L] = 1

        # Route through unpack_lm_batch so labels are constructed EXACTLY as in
        # the per-sample path (same shift/ignore_index convention).
        batch = {"input_ids": ids_batch, "attention_mask": am_batch}
        input_ids, labels, attention_mask = unpack_lm_batch(
            batch, device, pad_token_id=pad_id,
        )
        return input_ids, labels, attention_mask

    def _batched_replay_loss(self, model, picks):
        """Equal-example replay loss via ONE forward pass.

        Numerically reproduces the sequential mean-of-per-sample-losses:
            (1/k) * sum_s [ (1/N_s) * sum_{t in s} CE_{s,t} ]
        i.e. each sample is token-mean-normalized (matching HF ``out.loss``),
        then samples are averaged with equal weight. We compute per-token CE
        with reduction='none', mask padded/ignored positions, take each row's
        own token-mean, then average across rows. This preserves the current
        equal-example loss while replacing k single-example calls with one
        padded batch. (2026-07-26 efficiency review.)
        """
        torch = self._torch
        import torch.nn.functional as F
        input_ids, labels, attention_mask = self._padded_batch(model, picks)
        try:
            logits = model(
                input_ids, attention_mask=attention_mask,
            ).logits
        except TypeError:
            logits = model(input_ids).logits
        # Causal shift, identical to HF internal loss.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        B, Tm1, V = shift_logits.shape
        tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, Tm1)
        valid = (shift_labels != -100).to(tok_ce.dtype)
        per_sample_tok = valid.sum(dim=1)  # N_s per sample
        # Per-sample token-mean == HF out.loss for that sample.
        per_sample_loss = (tok_ce * valid).sum(dim=1) / per_sample_tok.clamp(min=1)
        # Equal-example average across the k samples.
        return per_sample_loss.mean()

    def aux_loss(self, model):
        if not self._buffer:
            return None
        k = min(self.replay_batch, len(self._buffer))
        picks = self._rng.sample(self._buffer, k)
        # Batched single-forward path (output-preserving; see
        # _batched_replay_loss). Set self.use_sequential_replay=True to restore
        # the original k-sequential-call path (kept for equivalence testing).
        if getattr(self, "use_sequential_replay", False):
            losses = [self._ce_loss_on_sample(model, s) for s in picks]
            return self.lambda_reg * (sum(losses) / len(losses))
        return self.lambda_reg * self._batched_replay_loss(model, picks)

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
