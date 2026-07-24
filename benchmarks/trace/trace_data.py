"""Official TRACE benchmark loader + real base-model builder (GPU path).

TRACE (Wang et al., 2024; https://github.com/BeyonderXX/TRACE) is the
standard continual-learning-for-LLMs benchmark. It is an 8-task sequence,
trained in a FIXED order, 5,000 train / 2,000 test examples per task, all
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

# Per-task primary metric (paper Table). Accuracy tasks use exact-match on the
# generated answer; generation tasks (summarization / simplification) fall back
# to token-level next-token accuracy as a cheap proxy in the harness and are
# scored with their proper metric (ROUGE-L / SARI) in eval/ at report time.
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
    """Turn {prompt, answer} records into padded input_id tensors.

    The training target is the full prompt+answer sequence (standard
    instruction-tuning next-token objective). Returns a single 2-D LongTensor
    [N, max_len]; the harness batches it.
    """
    import torch

    ids = []
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
    if not ids:
        return torch.zeros((0, max_len), dtype=torch.long)
    return torch.stack(ids, dim=0)


def _batched(tensor, batch_size: int):
    """Yield row-batches from a 2-D tensor (list of tensors, harness-friendly)."""
    out = []
    i = 0
    n = tensor.shape[0]
    while i < n:
        out.append(tensor[i:i + batch_size])
        i += batch_size
    return out


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
        train_ids = _tokenize_examples(
            _read_json_records(train_path), tokenizer, max_len, n_train
        )
        eval_ids = _tokenize_examples(
            _read_json_records(eval_path), tokenizer, max_len, n_eval
        )
        sequence.append((_batched(train_ids, batch_size),
                         _batched(eval_ids, batch_size)))
    return sequence


def build_real_model(model_id: str, device: str, lora: bool = True):
    """Load the real base model + tokenizer, optionally with LoRA adapters.

    Mirrors the v1 07_gradient_aligned_m conventions: bf16, device_map, LoRA
    on the standard Llama attention+MLP projections. Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
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
