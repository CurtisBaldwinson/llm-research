"""Official TRACE benchmark loader + real base-model builder (GPU path).

TRACE (Wang et al., 2024; https://github.com/BeyonderXX/TRACE) is the
standard continual-learning-for-LLMs benchmark. It is an 8-task sequence,
trained in a FIXED order, 5,000 training examples per task and non-uniform
held-out test counts (evaluation caps at 2,000 examples per task), all
standardized to a unified JSON format where each record is:

    {"prompt": "...", "answer": "..."}

The official data is distributed as one folder per task, each containing
train.json / eval.json / test.json. Download it once onto the GPU pod's
`data/` dir (the dispatcher stages this). This module does NOT invent tasks:
it reads the real files in the published order and tokenizes them for
causal-LM next-token training, exposing the SAME (train_loader, eval_loader)
tuple interface that the smoke path uses, so benchmarks/trace/run.py's core
loop is identical for smoke and real runs.

Canonical order (paper Table 4):
    1 ScienceQA   2 FOMC       3 MeetingBank  4 C-STANCE
    5 20Minuten   6 Py150      7 NumGLUE-cm   8 NumGLUE-ds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

# Fixed task order (folder names as distributed by the official TRACE repo).
TRACE_TASK_ORDER = [
    "ScienceQA",
    "FOMC",
    "MeetingBank",
    "C-STANCE",
    "20Minuten",
    "Py150",
    "NumGLUE-cm",
    "NumGLUE-ds",
]

# Per-task primary metric from the TRACE paper. Phase 1 does NOT use these
# for the dependent variable — Exp 1 reports BWT under a uniform
# next-token-accuracy (NTA) proxy (see eval/metrics.py METRIC_SCOPE).
# Faithful per-task scoring is deferred to Phase 2; this map is kept so
# Phase 2 can wire task-correct metrics without rediscovering the table.
TRACE_TASK_METRIC = {
    "ScienceQA": "accuracy",
    "FOMC": "accuracy",
    "MeetingBank": "rougeL",
    "C-STANCE": "accuracy",
    "20Minuten": "sari",
    "Py150": "edit_sim",
    "NumGLUE-cm": "accuracy",
    "NumGLUE-ds": "accuracy",
}

PHASE1_METRIC_SCOPE = "nta_proxy"


def _read_json_records(path: Path) -> List[dict]:
    """Read a TRACE split file (either a JSON array or JSONL)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        return json.loads(text)
    # JSONL fallback
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _tokenize_examples(records, tokenizer, max_len: int, n_cap: int):
    """Turn {prompt, answer} records into padded input_ids + attention_mask.

    The training target is the full prompt+answer sequence (standard
    instruction-tuning next-token objective). Returns
    ``(input_ids [N, max_len], attention_mask [N, max_len])``.

    Pad positions are identified solely via ``attention_mask`` (critical when
    ``pad_token_id == eos_token_id``). The harness sets labels=-100 on mask==0
    so padding never enters loss, Fisher, or M signals.
    """
    import torch

    ids = []
    masks = []
    for rec in records[:n_cap]:
        prompt = str(rec.get("prompt", ""))
        answer = str(rec.get("answer", ""))
        text = prompt + "\n" + answer + tokenizer.eos_token
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        ids.append(enc["input_ids"][0])
        masks.append(enc["attention_mask"][0])
    if not ids:
        z = torch.zeros((0, max_len), dtype=torch.long)
        return z, z.clone()
    return torch.stack(ids, dim=0), torch.stack(masks, dim=0)


def _batched_pairs(ids, mask, batch_size: int):
    """Yield dict batches ``{input_ids, attention_mask}`` for the harness."""
    out = []
    i = 0
    n = ids.shape[0]
    while i < n:
        out.append({
            "input_ids": ids[i:i + batch_size],
            "attention_mask": mask[i:i + batch_size],
        })
        i += batch_size
    return out


# ---------------------------------------------------------------------------
# Batch 2 efficiency transforms (2026-07-26 efficiency review).
#
# Both are OFF by default and OUTPUT-PRESERVING under a correctly-masked model.
# They are wired behind CLI flags (--crop-batches / --sort-eval-batches) and a
# scored run refuses to start with either flag on unless preflight_batch2.py
# has recorded a passing real-model equivalence proof for the current model +
# commit.  See preflight_batch2.py for the A40 assertions.
# ---------------------------------------------------------------------------

def crop_batch_padding(batch: dict) -> dict:
    """Return a copy of ``batch`` with columns that are padding for EVERY row
    removed.

    ``_tokenize_examples`` uses ``padding="max_length"`` (max_len=512), so most
    batches carry a large block of pad columns whose ``attention_mask`` is 0 for
    all rows in the batch. A column masked for every row cannot influence any
    real token under a correct attention mask (and labels=-100 there keep it out
    of the loss), so dropping it is forward-equivalent while shrinking the
    sequence dimension the model actually processes.

    Deliberately padding-side-agnostic: we keep only the columns where at least
    one row is real (``attention_mask.any(dim=0)``) and preserve their original
    order. This is correct whether the tokenizer left-pads or right-pads, and
    correct for ragged interior gaps, because selection is per-column by the
    real/pad status across the batch -- never an assumed contiguous side.

    NOTE: this crops only *shared* padding. A batch of mixed lengths still keeps
    every column up to the longest real sequence in the batch; per-row padding
    below that length is untouched (and remains masked). The savings therefore
    scale with how much shorter the batch max is than the global max_len.
    """
    import torch
    mask = batch["attention_mask"]
    if mask.numel() == 0 or mask.shape[0] == 0:
        return batch
    keep = mask.any(dim=0)  # [seq_len] bool: column is real for >=1 row
    if bool(keep.all()):
        return batch  # nothing to crop; return unchanged (identity)
    cropped = dict(batch)
    cropped["input_ids"] = batch["input_ids"][:, keep]
    cropped["attention_mask"] = mask[:, keep]
    # Carry through any extra per-position tensors (e.g. labels) if present, so
    # this stays correct if the batch dict grows. Non-[N, seq] entries pass
    # through untouched.
    for k, v in batch.items():
        if k in ("input_ids", "attention_mask"):
            continue
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == mask.shape[1]:
            cropped[k] = v[:, keep]
    return cropped


def crop_loader_padding(loader):
    """Apply :func:`crop_batch_padding` to every batch in a loader."""
    return [crop_batch_padding(b) for b in loader]


def sort_eval_loader_by_length(eval_loader, batch_size: int):
    """Re-batch an eval loader with examples grouped by real length.

    Sorting examples by their real token count (``attention_mask.sum``) before
    batching means each batch's shared-padding block is much larger, so
    :func:`crop_batch_padding` can trim far more -- roughly halving eval
    positions in practice. Uses a STABLE sort so ties keep their original
    relative order, making the transform deterministic.

    OUTPUT-PRESERVING FOR AGGREGATE METRICS ONLY. ``evaluate()`` accumulates
    integer correct/total counts, which are invariant to how examples are
    grouped into batches (each example's prediction is independent given a
    correct attention mask). It is NOT safe for any future metric that reads
    per-example or per-batch-position eval outputs, because it changes which
    examples share a batch. The scored-run gate records this caveat; do not
    enable alongside a per-example eval metric without re-validating.

    Reuses the real ``_flatten_loader`` / ``_batched_pairs`` path so the only
    change vs. the scored code path is example grouping.
    """
    import torch
    ids, mask = _flatten_loader(eval_loader)
    n = ids.shape[0]
    if n <= 1:
        return eval_loader
    lengths = mask.sum(dim=1)  # [N] real token count per example
    # stable sort by length (ascending); ties preserve original order
    order = torch.argsort(lengths, stable=True)
    return _batched_pairs(ids[order], mask[order], batch_size)


def _flatten_loader(loader):
    """Reconstruct the flat (ids, mask) tensors from a batched loader.

    Inverse of ``_batched_pairs``: concatenates the per-batch slices back into
    the original example-ordered tensors so they can be re-batched (e.g. after
    a seeded permutation) without re-tokenizing.
    """
    import torch
    if not loader:
        z = torch.zeros((0, 0), dtype=torch.long)
        return z, z.clone()
    ids = torch.cat([b["input_ids"] for b in loader], dim=0)
    mask = torch.cat([b["attention_mask"] for b in loader], dim=0)
    return ids, mask


def reshuffle_train_loader(train_loader, seed: int, batch_size: int):
    """Return a NEW train loader with examples permuted by a seeded RNG.

    This is the mechanism that turns the ``seeds`` axis into genuine statistical
    replicates. Continual-learning outcomes (final weights, BWT) depend on the
    order in which training examples are visited within each task, because SGD
    is path-dependent. With a fixed order, every seed produces a bitwise-
    identical run (std == 0) and any 2-sigma gate is vacuous. Permuting the
    within-task example order per seed -- while keeping the *task* sequence
    order fixed (the controlled variable) -- yields independent trajectories
    and a meaningful across-seed variance.

    Reuses the real tokenized tensors (no re-tokenization) and the real
    ``_batched_pairs`` batching path, so the only thing that changes vs. the
    scored code path is example order. Uses a dedicated ``torch.Generator`` so
    it does not perturb, and is not perturbed by, the global torch RNG state.
    """
    import torch
    ids, mask = _flatten_loader(train_loader)
    n = ids.shape[0]
    if n <= 1:
        return train_loader
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    return _batched_pairs(ids[perm], mask[perm], batch_size)


def build_trace_sequence(
    tokenizer,
    data_root: Path,
    *,
    n_train: int = 5000,
    n_eval: int = 2000,
    max_len: int = 512,
    batch_size: int = 8,
    tasks: List[str] = None,
) -> List[Tuple]:
    """Load the real TRACE sequence as [(train_loader, eval_loader), ...].

    `data_root` is the directory containing one folder per task (the layout
    of the official download). Missing tasks raise, because a silently-short
    sequence would corrupt BWT.
    """
    data_root = Path(data_root)
    order = tasks or TRACE_TASK_ORDER
    sequence = []
    for task in order:
        tdir = data_root / task
        train_path = tdir / "train.json"
        # official repo uses test.json for held-out eval
        eval_path = tdir / "test.json"
        if not train_path.exists() or not eval_path.exists():
            raise FileNotFoundError(
                f"TRACE task {task!r} missing train/test json under {tdir}. "
                f"Download the official TRACE data "
                f"(https://github.com/BeyonderXX/TRACE) into {data_root}."
            )
        train_ids, train_mask = _tokenize_examples(
            _read_json_records(train_path), tokenizer, max_len, n_train
        )
        eval_ids, eval_mask = _tokenize_examples(
            _read_json_records(eval_path), tokenizer, max_len, n_eval
        )
        sequence.append((
            _batched_pairs(train_ids, train_mask, batch_size),
            _batched_pairs(eval_ids, eval_mask, batch_size),
        ))
    return sequence


def build_real_model(model_id: str, device: str, lora: bool = True,
                     attn_implementation: str | None = None):
    """Load the real base model + tokenizer, optionally with LoRA adapters.

    Mirrors the v1 07_gradient_aligned_m conventions: bf16, device_map, LoRA
    on the standard Llama attention+MLP projections. Returns (model, tokenizer).

    attn_implementation: if given (e.g. "eager", "sdpa", "flash_attention_2"),
        forwarded to from_pretrained. Default None preserves the prior behavior
        exactly (transformers picks its default, i.e. SDPA on this stack), so
        every existing caller is byte-for-byte unchanged. "eager" is used by the
        Batch 2 equivalence preflight because it computes attention without the
        tile/shape-dependent float reordering of the fused SDPA/Flash kernels,
        which can reduce shape-dependent numerical differences. The preflight
        must still verify equivalence on the actual model and execution stack;
        selecting a backend alone does not establish output preservation.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _from_pretrained_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    # Only pass attn_implementation when explicitly requested, so the default
    # (None) path is identical to the prior code -- no behavior change for any
    # existing caller.
    if attn_implementation is not None:
        _from_pretrained_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **_from_pretrained_kwargs,
    )

    if lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    return model, tokenizer
