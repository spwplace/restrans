#!/usr/bin/env python3
"""Matched code/structure training for full architecture comparisons.

This runner is intentionally not a frozen-backbone adapter experiment.  It
trains the whole model end-to-end under a shared token/task budget, mixing:

* real code probes from CodeXGLUE/CodeSearchNet/EquiBench-style JSONL files;
* raw code/text LM chunks extracted from those same files; and
* local verifiable algebraic/logical tasks with yes/no answers.

The evaluation is answer-choice likelihood on held-out real code probes and
synthetic/verifiable validation sets.  That keeps the benchmark closer to the
architecture hypothesis: structural streams should help when semantic or
deductive equivalence is the target, not merely because a pretrained backbone
already knows code.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))

from resonance import ResonanceConfig, ResonanceTransformer, StandardConfig, StandardTransformer  # noqa: E402
from resonance.device import enable_deterministic  # noqa: E402


SCALES: dict[str, dict[str, int]] = {
    "20M": {"embed_dim": 512, "layers": 4, "heads": 8, "ff_dim": 1536},
    "125M": {"embed_dim": 768, "layers": 12, "heads": 12, "ff_dim": 3072},
    "350M": {"embed_dim": 1024, "layers": 24, "heads": 16, "ff_dim": 4096},
    "600M": {"embed_dim": 1280, "layers": 28, "heads": 20, "ff_dim": 5120},
}


@dataclass(frozen=True)
class AnswerExample:
    prefix: str
    answer: str
    source: str


@dataclass(frozen=True)
class LMExample:
    text: str
    source: str


class HFTokenizer:
    def __init__(self, name: str, cache_dir: str | None = None) -> None:
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.model_max_length = int(1e9)
        self.pad_id = int(self.tok.pad_token_id)
        self.eos_id = int(self.tok.eos_token_id)
        self.vocab_size = int(self.tok.vocab_size)
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoded = self.tok(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        return list(encoded["input_ids"]), [tuple(x) for x in encoded["offset_mapping"]]


def expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(p) for p in glob.glob(pattern)]
        if matches:
            paths.extend(sorted(matches))
        else:
            paths.append(Path(pattern))
    return paths


def clean_text(value: object, max_chars: int) -> str:
    text = str(value or "")
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return text[:max_chars]


def read_answer_file(path: Path, limit: int | None = None) -> list[AnswerExample]:
    examples: list[AnswerExample] = []
    if not path.exists():
        return examples
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "prefix" in row and "answer" in row:
                examples.append(
                    AnswerExample(
                        prefix=clean_text(row["prefix"], 100_000),
                        answer=str(row["answer"]).strip().lower(),
                        source=str(row.get("source", path.as_posix())),
                    )
                )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def read_pair_file(path: Path, limit: int | None = None) -> list[AnswerExample]:
    examples: list[AnswerExample] = []
    if not path.exists():
        return examples
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "text_a" not in row or "text_b" not in row:
                continue
            answer = "yes" if int(row.get("label", 0)) == 1 else "no"
            source = str(row.get("source", path.as_posix()))
            examples.append(
                AnswerExample(
                    prefix=(
                        f"Program one: {clean_text(row['text_a'], 8000)} "
                        f"Program two: {clean_text(row['text_b'], 8000)} "
                        "Are these programs semantically equivalent? Answer"
                    ),
                    answer=answer,
                    source=source,
                )
            )
            if limit is not None and len(examples) >= limit:
                break
    return examples


def read_lm_examples(paths: Iterable[Path], limit_per_file: int | None, max_chars: int) -> list[LMExample]:
    examples: list[LMExample] = []
    for path in paths:
        if not path.exists():
            continue
        count = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                fields: list[str] = []
                if "text" in row:
                    fields.append(clean_text(row["text"], max_chars))
                if "prefix" in row:
                    fields.append(clean_text(row["prefix"], max_chars))
                if "text_a" in row:
                    fields.append(clean_text(row["text_a"], max_chars))
                if "text_b" in row:
                    fields.append(clean_text(row["text_b"], max_chars))
                for text in fields:
                    if text:
                        examples.append(LMExample(text=text, source=path.as_posix()))
                count += 1
                if limit_per_file is not None and count >= limit_per_file:
                    break
    return examples


def generate_synthetic(task: str, n_train: int, n_val: int, seed: int, depth: int) -> tuple[list[AnswerExample], list[AnswerExample]]:
    if task == "unification":
        from synthetic.unification import UnificationDataset

        train = UnificationDataset(n_examples=n_train, seed=seed, depth=depth)
        val = UnificationDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    elif task == "cap_matching":
        from synthetic.cap_matching import CapMatchingDataset

        train = CapMatchingDataset(n_examples=n_train, seed=seed, depth=depth)
        val = CapMatchingDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    elif task == "algebraic_protocol":
        from synthetic.algebraic_protocol import AlgebraicProtocolDataset

        train = AlgebraicProtocolDataset(n_examples=n_train, seed=seed, depth=depth)
        val = AlgebraicProtocolDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    elif task == "lambda_equivalence":
        from synthetic.lambda_tasks import LambdaEquivalenceDataset

        train = LambdaEquivalenceDataset(n_examples=n_train, seed=seed, depth=depth)
        val = LambdaEquivalenceDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    elif task == "vm_equivalence":
        from synthetic.vm_tasks import VMEquivalenceDataset

        train = VMEquivalenceDataset(n_examples=n_train, seed=seed, depth=depth)
        val = VMEquivalenceDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    elif task == "dfa_equivalence":
        from synthetic.automata_tasks import DFAEquivalenceDataset

        train = DFAEquivalenceDataset(n_examples=n_train, seed=seed, depth=depth)
        val = DFAEquivalenceDataset(n_examples=n_val, seed=seed + 100_000, depth=depth)
    else:
        raise ValueError(f"unknown synthetic task: {task}")

    def convert(ds, source: str) -> list[AnswerExample]:
        out: list[AnswerExample] = []
        for ex in ds.examples:
            answer = str(ex.answer).strip().lower()
            if answer not in {"yes", "no"}:
                continue
            out.append(AnswerExample(prefix=ex.prefix, answer=answer, source=source))
        return out

    return convert(train, task), convert(val, task)


def labels_for_answer(tokenizer: HFTokenizer, prefix: str, answer: str, seq_len: int) -> tuple[list[int], list[int]]:
    prompt = prefix.rstrip() + " "
    text = prompt + answer.strip().lower() + "\n"
    answer_start = len(prompt)
    answer_end = answer_start + len(answer.strip().lower())
    ids, offsets = tokenizer.encode_with_offsets(text)
    if len(ids) > seq_len:
        # Code prompts can be very long.  Keep the suffix containing the answer
        # so answer-only supervision is never silently dropped.
        start = max(0, len(ids) - seq_len)
        ids = ids[start:]
        offsets = offsets[start:]
    else:
        ids = ids[:seq_len]
        offsets = offsets[:seq_len]
    labels = [-100] * len(ids)
    for i, (start, end) in enumerate(offsets):
        if end <= answer_start:
            continue
        if start >= answer_end:
            continue
        labels[i] = ids[i]
    if ids and ids[-1] != tokenizer.eos_id and len(ids) < seq_len:
        ids.append(tokenizer.eos_id)
        labels.append(-100)
    return ids, labels


class MixedCodeDataset(Dataset):
    def __init__(
        self,
        answer_examples: list[AnswerExample],
        lm_examples: list[LMExample],
        tokenizer: HFTokenizer,
        seq_len: int,
        lm_fraction: float,
        seed: int,
    ) -> None:
        self.rows: list[tuple[list[int], list[int], str, str]] = []
        for ex in answer_examples:
            ids, labels = labels_for_answer(tokenizer, ex.prefix, ex.answer, seq_len)
            if any(label != -100 for label in labels):
                self.rows.append((ids, labels, "answer", ex.source))

        lm_rows: list[tuple[list[int], list[int], str, str]] = []
        for ex in lm_examples:
            ids = tokenizer.encode(ex.text)
            if len(ids) < 8:
                continue
            for start in range(0, max(1, len(ids) - 1), seq_len):
                chunk = ids[start : start + seq_len]
                if len(chunk) < 8:
                    continue
                labels = chunk.copy()
                lm_rows.append((chunk, labels, "lm", ex.source))
                if len(lm_rows) >= max(1, int(len(self.rows) * lm_fraction)):
                    break
            if len(lm_rows) >= max(1, int(len(self.rows) * lm_fraction)):
                break
        self.rows.extend(lm_rows)
        random.Random(seed).shuffle(self.rows)
        self.pad_id = tokenizer.pad_id
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        ids, labels, kind, source = self.rows[idx]
        ids = ids[: self.seq_len]
        labels = labels[: self.seq_len]
        pad = self.seq_len - len(ids)
        if pad > 0:
            ids = ids + [self.pad_id] * pad
            labels = labels + [-100] * pad
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "kind": kind,
            "source": source,
        }


def group_by_source(examples: list[AnswerExample], limit_per_source: int | None = None) -> dict[str, list[AnswerExample]]:
    groups: dict[str, list[AnswerExample]] = {}
    for ex in examples:
        source = ex.source.split("/")[0] if "/" in ex.source else ex.source
        bucket = groups.setdefault(source, [])
        if limit_per_source is None or len(bucket) < limit_per_source:
            bucket.append(ex)
    return groups


def make_config(args: argparse.Namespace, condition: str, vocab_size: int):
    scale = SCALES[args.scale].copy()
    if args.embed_dim is not None:
        scale["embed_dim"] = args.embed_dim
    if args.layers is not None:
        scale["layers"] = args.layers
    if args.heads is not None:
        scale["heads"] = args.heads
    if args.ff_dim is not None:
        scale["ff_dim"] = args.ff_dim

    common = {
        "name": condition,
        "vocab_size": vocab_size,
        "max_seq_len": args.seq_len,
        "embed_dim": scale["embed_dim"],
        "n_layers": scale["layers"],
        "n_heads": scale["heads"],
        "ff_dim": scale["ff_dim"],
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.lr,
    }
    base = condition.removesuffix("_normalized")
    if base.startswith("standard"):
        attention_variant = "alibi" if "alibi" in base else "standard"
        return StandardConfig(**common, attention_variant=attention_variant)

    config = ResonanceConfig(
        **common,
        n_frequencies=args.n_frequencies,
        resonance_blend=args.resonance_blend,
        resonance_attn_weight=args.resonance_attn_weight,
        normalize_resonance="normalized" in condition,
    )
    if base in {"phase_dynamic_qk_film", "phase_dynamic_qk_film_alibi"}:
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "mlp"
        config.phase_condition_qk = "film"
    elif base in {"relation_value_qk_film", "relation_value_qk_film_alibi"}:
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_condition_qk = "film"
        config.relation_value_mode = "additive"
    elif base == "resonance_full":
        config.use_phase_stream = True
        config.use_resonance_bias = True
    elif base == "phase_stream_only":
        config.use_phase_stream = True
        config.use_resonance_bias = False
    else:
        raise ValueError(f"unknown resonance condition: {condition}")
    if "alibi" in base:
        config.attention_variant = "alibi"
    return config


def create_model(args: argparse.Namespace, vocab_size: int):
    config = make_config(args, args.condition, vocab_size)
    if isinstance(config, StandardConfig):
        model = StandardTransformer(config)
    else:
        model = ResonanceTransformer(config)
    return model, config


def device_type(device: torch.device) -> str:
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    return "cpu"


def autocast_context(args: argparse.Namespace, device: torch.device):
    if args.amp == "off" or device.type not in {"cuda", "mps"}:
        return torch.autocast("cpu", enabled=False)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast(device_type(device), dtype=dtype)


def batch_loss(model, batch: dict[str, torch.Tensor], device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    with autocast_context(args, device):
        out = model(input_ids, labels=labels)
        return out["loss"]


@torch.no_grad()
def score_candidate(model, tokenizer: HFTokenizer, prefix: str, answer: str, device: torch.device, args: argparse.Namespace) -> float:
    ids, labels = labels_for_answer(tokenizer, prefix, answer, args.seq_len)
    if not any(label != -100 for label in labels):
        return float("-inf")
    pad = args.seq_len - len(ids)
    if pad > 0:
        ids = ids + [tokenizer.pad_id] * pad
        labels = labels + [-100] * pad
    input_ids = torch.tensor([ids[: args.seq_len]], dtype=torch.long, device=device)
    target = torch.tensor([labels[: args.seq_len]], dtype=torch.long, device=device)
    with autocast_context(args, device):
        logits = model(input_ids)["logits"]
    shifted_logits = logits[:, :-1, :]
    shifted_labels = target[:, 1:]
    mask = shifted_labels != -100
    if not mask.any():
        return float("-inf")
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered = log_probs.gather(-1, shifted_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return float((gathered * mask).sum().item())


@torch.no_grad()
def evaluate_answer_groups(
    model,
    tokenizer: HFTokenizer,
    groups: dict[str, list[AnswerExample]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    model.eval()
    results: dict[str, dict[str, float]] = {}
    for source, examples in groups.items():
        total = 0
        correct = 0
        for ex in examples:
            if ex.answer not in {"yes", "no"}:
                continue
            yes_score = score_candidate(model, tokenizer, ex.prefix, "yes", device, args)
            no_score = score_candidate(model, tokenizer, ex.prefix, "no", device, args)
            pred = "yes" if yes_score >= no_score else "no"
            correct += int(pred == ex.answer)
            total += 1
        results[source] = {
            "accuracy": correct / total if total else 0.0,
            "n": float(total),
        }
    return results


def load_data(args: argparse.Namespace) -> tuple[list[AnswerExample], list[AnswerExample], list[LMExample]]:
    train_answer: list[AnswerExample] = []
    val_answer: list[AnswerExample] = []
    train_paths = expand_paths(args.train_answer_files)
    val_paths = expand_paths(args.val_answer_files)
    pair_paths = expand_paths(args.train_pair_files)

    for path in train_paths:
        train_answer.extend(read_answer_file(path, args.limit_per_answer_file))
    for path in val_paths:
        val_answer.extend(read_answer_file(path, args.val_limit_per_file))
    for path in pair_paths:
        pairs = read_pair_file(path, args.limit_per_pair_file)
        random.Random(args.seed + len(train_answer)).shuffle(pairs)
        split = max(1, int(len(pairs) * 0.8))
        train_answer.extend(pairs[:split])
        val_answer.extend(pairs[split:])

    for i, task in enumerate(args.synthetic_tasks):
        train, val = generate_synthetic(
            task,
            args.synthetic_train_examples,
            args.synthetic_val_examples,
            args.seed + i * 10_000,
            args.synthetic_depth,
        )
        train_answer.extend(train)
        val_answer.extend(val)

    lm_paths = train_paths + pair_paths + expand_paths(args.lm_files)
    lm_examples = read_lm_examples(lm_paths, args.limit_lm_rows_per_file, args.max_lm_chars)
    return train_answer, val_answer, lm_examples


def train(args: argparse.Namespace) -> dict[str, object]:
    enable_deterministic(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device(args.device)
    tokenizer = HFTokenizer(args.tokenizer, cache_dir=args.hf_cache)
    train_answer, val_answer, lm_examples = load_data(args)
    if args.train_limit is not None:
        random.Random(args.seed).shuffle(train_answer)
        train_answer = train_answer[: args.train_limit]
    if args.val_limit is not None:
        random.Random(args.seed + 1).shuffle(val_answer)
        val_answer = val_answer[: args.val_limit]

    dataset = MixedCodeDataset(
        train_answer,
        lm_examples,
        tokenizer,
        args.seq_len,
        args.lm_fraction,
        args.seed,
    )
    if len(dataset) == 0:
        raise ValueError("no training rows produced")
    val_groups = group_by_source(val_answer, args.val_per_source)

    model, config = create_model(args, tokenizer.vocab_size)
    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "event": "setup",
        "condition": args.condition,
        "scale": args.scale,
        "params": param_count,
        "train_answer": len(train_answer),
        "val_answer": len(val_answer),
        "lm_examples": len(lm_examples),
        "dataset_rows": len(dataset),
        "device": str(device),
        "amp": args.amp,
    }), flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.max_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and device.type == "cuda"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps({**vars(args), "params": param_count}, indent=2, default=str))

    model.train()
    total_loss = 0.0
    step = 0
    opt_step = 0
    start = time.time()
    loader_iter = iter(loader)
    optimizer.zero_grad(set_to_none=True)

    while step < args.max_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        if scaler.is_enabled():
            with autocast_context(args, device):
                loss = batch_loss(model, batch, device, args) / args.grad_accum
            scaler.scale(loss).backward()
        else:
            loss = batch_loss(model, batch, device, args) / args.grad_accum
            loss.backward()

        total_loss += float(loss.detach().cpu().item()) * args.grad_accum
        step += 1

        if step % args.grad_accum == 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

        if step % args.log_every == 0:
            elapsed = max(time.time() - start, 1e-6)
            event = {
                "event": "train",
                "step": step,
                "opt_step": opt_step,
                "loss": total_loss / args.log_every,
                "lr": scheduler.get_last_lr()[0],
                "examples_per_sec": (step * args.batch_size) / elapsed,
                "elapsed_sec": elapsed,
            }
            print(json.dumps(event), flush=True)
            with history_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            total_loss = 0.0

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_start = time.time()
            metrics = evaluate_answer_groups(model, tokenizer, val_groups, device, args)
            macro = sum(v["accuracy"] for v in metrics.values()) / max(1, len(metrics))
            event = {
                "event": "eval",
                "step": step,
                "macro_accuracy": macro,
                "metrics": metrics,
                "eval_sec": time.time() - eval_start,
            }
            print(json.dumps(event), flush=True)
            with history_path.open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            model.train()

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt = args.output_dir / f"checkpoint_step{step}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "args": vars(args),
                    "step": step,
                    "params": param_count,
                },
                ckpt,
            )

    final_metrics = evaluate_answer_groups(model, tokenizer, val_groups, device, args)
    final = {
        "condition": args.condition,
        "scale": args.scale,
        "params": param_count,
        "steps": step,
        "final_macro_accuracy": sum(v["accuracy"] for v in final_metrics.values()) / max(1, len(final_metrics)),
        "final_metrics": final_metrics,
        "elapsed_sec": time.time() - start,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(final, indent=2))
    if args.save_final:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "args": vars(args),
                "step": step,
                "params": param_count,
            },
            args.output_dir / "checkpoint_final.pt",
        )
    print(json.dumps({"event": "done", **final}), flush=True)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--condition", default="standard_alibi")
    parser.add_argument("--scale", choices=sorted(SCALES), default="125M")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--hf_cache", default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--save_final", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=["off", "bf16", "fp16"], default="bf16")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--torch_threads", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--ff_dim", type=int, default=None)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--lm_fraction", type=float, default=0.35)
    parser.add_argument("--max_lm_chars", type=int, default=8000)
    parser.add_argument("--limit_lm_rows_per_file", type=int, default=5000)
    parser.add_argument("--limit_per_answer_file", type=int, default=8000)
    parser.add_argument("--limit_per_pair_file", type=int, default=2000)
    parser.add_argument("--val_limit_per_file", type=int, default=1200)
    parser.add_argument("--val_per_source", type=int, default=800)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--synthetic_train_examples", type=int, default=3000)
    parser.add_argument("--synthetic_val_examples", type=int, default=600)
    parser.add_argument("--synthetic_depth", type=int, default=5)
    parser.add_argument(
        "--synthetic_tasks",
        nargs="*",
        default=[
            "unification",
            "cap_matching",
            "algebraic_protocol",
            "lambda_equivalence",
            "vm_equivalence",
            "dfa_equivalence",
        ],
    )
    parser.add_argument(
        "--train_answer_files",
        nargs="*",
        default=[
            "data/processed/probes/poj104/train.jsonl",
            "data/processed/probes/bigclonebench/train.jsonl",
            "data/processed/probes/codesearchnet/train.jsonl",
            "data/processed/equibench/*/train.jsonl",
        ],
    )
    parser.add_argument(
        "--val_answer_files",
        nargs="*",
        default=[
            "data/processed/probes/poj104/validation.jsonl",
            "data/processed/probes/bigclonebench/validation.jsonl",
        ],
    )
    parser.add_argument(
        "--train_pair_files",
        nargs="*",
        default=[
            "data/processed/contrastive/equibench/*/train.jsonl",
        ],
    )
    parser.add_argument("--lm_files", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
