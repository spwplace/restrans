#!/usr/bin/env python3
"""Unified mixed trainer: interleave LM and structural-task data.

Train a single model on language modeling (BabyLM, TinyStories, WikiText)
and structural reasoning tasks (unification, cap matching, algebraic protocol).
Evaluate on held-out LM perplexity, task accuracy, and BLiMP minimal pairs.

This tests whether structural pretraining transfers, rather than asking
whether an isolated structural variant wins on an isolated task.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from resonance import ResonanceConfig, ResonanceTransformer, StandardConfig, StandardTransformer
from resonance.device import enable_deterministic

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_jsonl_texts(path: Path, limit: int | None = None) -> list[str]:
    texts: list[str] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
            if limit is not None and len(texts) >= limit:
                break
    return texts


class WordTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.word2id = vocab
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tok.lower() for tok in TOKEN_RE.findall(text))
        vocab = {"<pad>": 0, "<unk>": 1}
        for special in ("yes", "no"):
            if special not in vocab:
                vocab[special] = len(vocab)
        for token, _ in counts.most_common(max(0, vocab_size - len(vocab))):
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, truncation: bool = False) -> list[int]:
        return [self.vocab.get(tok.lower(), self.unk_id) for tok in TOKEN_RE.findall(text)]

    def __len__(self) -> int:
        return len(self.vocab)


class LMDataset(Dataset):
    def __init__(self, texts: Iterable[str], tokenizer: WordTokenizer, seq_len: int, max_examples: int):
        ids: list[int] = []
        for text in texts:
            encoded = tokenizer.encode(text)
            if encoded:
                ids.extend(encoded)
                ids.append(tokenizer.pad_id)
        self.examples: list[list[int]] = []
        for start in range(0, max(0, len(ids) - seq_len), seq_len):
            self.examples.append(ids[start : start + seq_len])
            if len(self.examples) >= max_examples:
                break
        if not self.examples:
            raise ValueError("no LM chunks produced")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.examples[idx]
        x = ids + [0] * (self.seq_len - len(ids))
        x = x[: self.seq_len]
        return {"input_ids": torch.tensor(x, dtype=torch.long), "labels": torch.tensor(x, dtype=torch.long)}


class TaskDataset(Dataset):
    def __init__(self, examples: list[Any], tokenizer: WordTokenizer, seq_len: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        full_tokens = self.tokenizer.encode(ex.full_text)
        answer_id = self.tokenizer.vocab.get(ex.answer, self.tokenizer.unk_id)
        # Find answer position (search from end for robustness)
        answer_pos = None
        for i in range(len(full_tokens) - 1, -1, -1):
            if full_tokens[i] == answer_id:
                answer_pos = i
                break
        if answer_pos is None:
            answer_pos = len(full_tokens) - 1

        # Truncate while trying to keep answer
        if answer_pos >= self.seq_len:
            # Shift window so answer is near end but fits
            start = answer_pos - self.seq_len + 2
            full_tokens = full_tokens[start:]
            answer_pos = self.seq_len - 2

        x = full_tokens + [0] * (self.seq_len - len(full_tokens))
        x = x[: self.seq_len]
        labels = [-100] * self.seq_len
        # Train model to predict answer token (and period after it) given prefix
        if answer_pos < self.seq_len:
            labels[answer_pos] = full_tokens[answer_pos]
            if answer_pos + 1 < len(full_tokens) and answer_pos + 1 < self.seq_len:
                labels[answer_pos + 1] = full_tokens[answer_pos + 1]

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "answer_pos": min(answer_pos, self.seq_len - 1),
            "answer_id": answer_id,
            "task": getattr(ex, "task", "unknown"),
        }


def generate_task_data(task_name: str, num_train: int, num_val: int, seed: int, **kwargs):
    if task_name == "unification":
        from synthetic.unification import UnificationDataset
        train_ds = UnificationDataset(n_examples=num_train, seed=seed, **kwargs)
        val_ds = UnificationDataset(n_examples=num_val, seed=seed + 100_000, **kwargs)
    elif task_name == "cap_matching":
        from synthetic.cap_matching import CapMatchingDataset
        train_ds = CapMatchingDataset(n_examples=num_train, seed=seed, **kwargs)
        val_ds = CapMatchingDataset(n_examples=num_val, seed=seed + 100_000, **kwargs)
    elif task_name == "algebraic_protocol":
        from synthetic.algebraic_protocol import AlgebraicProtocolDataset
        train_ds = AlgebraicProtocolDataset(n_examples=num_train, seed=seed, **kwargs)
        val_ds = AlgebraicProtocolDataset(n_examples=num_val, seed=seed + 100_000, **kwargs)
    else:
        raise ValueError(f"unknown task: {task_name}")
    return train_ds.examples, val_ds.examples


def make_config(args: argparse.Namespace, name: str, vocab_size: int, resonance: bool):
    common = {
        "name": name,
        "vocab_size": vocab_size,
        "max_seq_len": args.seq_len,
        "embed_dim": args.embed_dim,
        "n_layers": args.layers,
        "n_heads": args.heads,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.lr,
    }
    if not resonance:
        attention_variant = "standard"
        if name in {"standard_alibi", "standard_iso_alibi"}:
            attention_variant = "alibi"
        elif name in {"standard_deberta_lite", "standard_iso_deberta_lite"}:
            attention_variant = "deberta_lite"
        return StandardConfig(**common, attention_variant=attention_variant)
    return ResonanceConfig(
        **common,
        n_frequencies=args.n_frequencies,
        resonance_blend=args.resonance_blend,
        resonance_attn_weight=args.resonance_attn_weight,
        normalize_resonance=True,
        init_preset="default",
    )


def configure_condition(config, condition: str):
    base = condition.removesuffix("_normalized")
    if base == "phase_dynamic_qk_film":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "mlp"
        config.phase_condition_qk = "film"
    elif base == "phase_stream_only":
        config.use_phase_stream = True
        config.use_resonance_bias = False
    elif base == "resonance_full":
        config.use_phase_stream = True
        config.use_resonance_bias = True
    elif base == "standard_alibi":
        if hasattr(config, "attention_variant"):
            config.attention_variant = "alibi"
    return config


@torch.no_grad()
def evaluate_lm(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids, labels=labels)
        # Count non-padding tokens in labels
        mask = labels != -100
        ntok = mask.sum().item()
        if ntok > 0:
            # Compute loss manually to get per-example
            logits = out["logits"][:, :-1, :].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="sum", ignore_index=-100)
            total_loss += loss.item()
            total_tokens += ntok
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


@torch.no_grad()
def evaluate_tasks(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = 0
    total = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        answer_pos = batch["answer_pos"]
        answer_id = batch["answer_id"]
        out = model(input_ids)
        logits = out["logits"]
        # logits[:, t] predicts input_ids[:, t+1]
        # We want prediction for answer_pos, which is at logits[:, answer_pos - 1]
        for i in range(len(input_ids)):
            pos = answer_pos[i].item() - 1
            if pos < 0:
                pos = 0
            pred = logits[i, pos].argmax().item()
            if pred == answer_id[i].item():
                correct += 1
            total += 1
    return correct / total if total > 0 else 0.0


def train_unified(args: argparse.Namespace) -> dict[str, Any]:
    enable_deterministic(True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Load LM texts
    all_lm_texts: list[str] = []
    for path in args.lm_train_paths:
        all_lm_texts.extend(read_jsonl_texts(Path(path), args.lm_train_docs_per_file))

    all_lm_val_texts: list[str] = []
    for path in args.lm_val_paths:
        all_lm_val_texts.extend(read_jsonl_texts(Path(path), args.lm_val_docs_per_file))

    # Generate task data
    all_task_train_examples = []
    all_task_val_examples = []
    for task_spec in args.task_names:
        parts = task_spec.split(":")
        task_name = parts[0]
        task_kwargs = {}
        if len(parts) > 1:
            for kv in parts[1].split(","):
                k, v = kv.split("=")
                task_kwargs[k] = int(v)
        train_ex, val_ex = generate_task_data(
            task_name,
            num_train=args.task_train_examples,
            num_val=args.task_val_examples,
            seed=args.seed,
            **task_kwargs,
        )
        # Examples may be frozen dataclasses; wrap if needed
        for ex in train_ex:
            object.__setattr__(ex, "task", task_name)
        for ex in val_ex:
            object.__setattr__(ex, "task", task_name)
        all_task_train_examples.extend(train_ex)
        all_task_val_examples.extend(val_ex)

    # Gather all texts for tokenizer
    all_texts = all_lm_texts.copy()
    for ex in all_task_train_examples:
        all_texts.append(ex.full_text)
    for ex in all_task_val_examples:
        all_texts.append(ex.full_text)

    tokenizer = WordTokenizer.train(all_texts, args.vocab_size)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Build datasets
    lm_train = LMDataset(all_lm_texts, tokenizer, args.seq_len, args.lm_max_examples)
    lm_val = LMDataset(all_lm_val_texts, tokenizer, args.seq_len, args.lm_val_max_examples)
    task_train = TaskDataset(all_task_train_examples, tokenizer, args.seq_len)
    task_val = TaskDataset(all_task_val_examples, tokenizer, args.seq_len)

    print(f"LM train: {len(lm_train)} chunks, LM val: {len(lm_val)} chunks")
    print(f"Task train: {len(task_train)} examples, Task val: {len(task_val)} examples")

    # Build model
    resonance = args.condition not in {"standard", "standard_alibi", "standard_deberta_lite", "standard_iso_deberta_lite"}
    config = make_config(args, args.condition, len(tokenizer), resonance)
    config = configure_condition(config, args.condition)
    model = StandardTransformer(config) if not resonance else ResonanceTransformer(config)
    model = model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(lm_train) + len(task_train)) // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    lm_loader = DataLoader(lm_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    task_loader = DataLoader(task_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    lm_iter = iter(lm_loader)
    task_iter = iter(task_loader)

    history = []
    global_step = 0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_lm_loss = 0.0
        epoch_task_loss = 0.0
        lm_batches = 0
        task_batches = 0

        num_batches = max(len(lm_loader), len(task_loader))
        for step_in_epoch in range(num_batches):
            # Interleave: 2 LM batches, 1 task batch
            if global_step % 3 < 2:
                try:
                    batch = next(lm_iter)
                except StopIteration:
                    lm_iter = iter(lm_loader)
                    batch = next(lm_iter)
                weight = args.lm_weight
                is_lm = True
            else:
                try:
                    batch = next(task_iter)
                except StopIteration:
                    task_iter = iter(task_loader)
                    batch = next(task_iter)
                weight = args.task_weight
                is_lm = False

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids, labels=labels)
            loss = out["loss"] * weight

            loss.backward()
            if (global_step + 1) % args.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_lm:
                epoch_lm_loss += loss.item()
                lm_batches += 1
            else:
                epoch_task_loss += loss.item()
                task_batches += 1

            if step_in_epoch > 0 and step_in_epoch % 200 == 0:
                print(
                    f"  step {step_in_epoch}/{num_batches} | "
                    f"lm_loss: {epoch_lm_loss/max(lm_batches,1):.4f} | "
                    f"task_loss: {epoch_task_loss/max(task_batches,1):.4f}"
                )

            global_step += 1

        # End of epoch evaluation
        lm_ppl = evaluate_lm(model, lm_val, device)
        task_acc = evaluate_tasks(model, task_val, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"LM loss: {epoch_lm_loss / max(lm_batches, 1):.4f} | "
            f"Task loss: {epoch_task_loss / max(task_batches, 1):.4f} | "
            f"LM PPL: {lm_ppl:.2f} | "
            f"Task Acc: {task_acc:.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "lm_loss": epoch_lm_loss / max(lm_batches, 1),
            "task_loss": epoch_task_loss / max(task_batches, 1),
            "lm_ppl": lm_ppl,
            "task_acc": task_acc,
        })

    elapsed = time.time() - start_time
    return {
        "condition": args.condition,
        "seed": args.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/unified"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--condition", default="phase_dynamic_qk_film_normalized")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=768)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vocab_size", type=int, default=12000)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--task_weight", type=float, default=1.0)

    parser.add_argument("--lm_train_paths", nargs="+", default=[
        "data/processed/babylm_strict_small/train.jsonl",
        "data/processed/tinystories/train.jsonl",
    ])
    parser.add_argument("--lm_val_paths", nargs="+", default=[
        "data/processed/babylm_strict_small/validation.jsonl",
        "data/processed/tinystories/validation.jsonl",
    ])
    parser.add_argument("--lm_train_docs_per_file", type=int, default=50_000)
    parser.add_argument("--lm_val_docs_per_file", type=int, default=10_000)
    parser.add_argument("--lm_max_examples", type=int, default=15_000)
    parser.add_argument("--lm_val_max_examples", type=int, default=2_000)

    parser.add_argument("--task_names", nargs="+", default=[
        "unification:depth=5",
        "cap_matching:depth=4",
        "algebraic_protocol:depth=4",
    ])
    parser.add_argument("--task_train_examples", type=int, default=2_000)
    parser.add_argument("--task_val_examples", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = train_unified(args)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
