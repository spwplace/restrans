#!/usr/bin/env python3
"""Train standard vs resonance LMs on a JSONL text corpus.

This is intentionally modest and reproducible: it uses a local word-level
tokenizer, concatenates documents into fixed-length chunks, trains matched
standard/resonance models, and writes a compact report with validation PPL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from resonance import ResonanceConfig, ResonanceTransformer, StandardConfig, StandardTransformer, train_model
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
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tok.lower() for tok in TOKEN_RE.findall(text))
        vocab = {"<pad>": 0, "<unk>": 1}
        for token, _ in counts.most_common(max(0, vocab_size - len(vocab))):
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, truncation: bool = False) -> list[int]:
        return [self.vocab.get(tok.lower(), self.unk_id) for tok in TOKEN_RE.findall(text)]

    def __len__(self) -> int:
        return len(self.vocab)


class ConcatenatedTextDataset(Dataset):
    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: WordTokenizer,
        seq_len: int,
        max_examples: int,
    ) -> None:
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

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids}


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
        return StandardConfig(**common)
    return ResonanceConfig(
        **common,
        n_frequencies=args.n_frequencies,
        resonance_blend=args.resonance_blend,
        resonance_attn_weight=args.resonance_attn_weight,
        normalize_resonance=True,
        init_preset="default",
    )


def train_one(args: argparse.Namespace, condition: str, train_ds, val_ds, vocab_size: int) -> dict:
    resonance = condition != "standard"
    config = make_config(args, condition, vocab_size, resonance=resonance)
    base = condition.removesuffix("_normalized")
    if base == "phase_stream_only":
        config.use_phase_stream = True
        config.use_resonance_bias = False
    if base == "bias_only":
        config.use_phase_stream = False
        config.use_resonance_bias = True
    if base == "resonance_inert":
        config.use_phase_stream = False
        config.use_resonance_bias = False
    if base == "phase_dynamic_mlp":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "mlp"
    if base == "phase_dynamic_attn":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "self_attn"
    if base == "phase_qk_film":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_condition_qk = "film"
    if base == "phase_dynamic_qk_film":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.phase_update_mode = "mlp"
        config.phase_condition_qk = "film"
    if base == "complex_directional":
        config.use_phase_stream = True
        config.use_resonance_bias = True
        config.resonance_kernel = "directional_complex"
    if base == "structural_heads_1":
        config.use_phase_stream = True
        config.use_resonance_bias = False
        config.n_structural_heads = 1
    model = StandardTransformer(config) if condition == "standard" else ResonanceTransformer(config)
    started = time.time()
    history = train_model(
        model,
        config,
        train_ds,
        val_ds,
        n_epochs=args.epochs,
        device=args.device,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
    )
    elapsed = time.time() - started
    return {
        "condition": condition,
        "params": getattr(model, "n_params", sum(p.numel() for p in model.parameters())),
        "history": history,
        "final_val_loss": history["val_loss"][-1],
        "final_val_ppl": history["val_ppl"][-1],
        "elapsed_sec": elapsed,
    }


def write_report(path: Path, results: dict) -> None:
    lines = [
        "# Corpus LM Run",
        "",
        f"Train file: `{results['config']['train_path']}`",
        f"Validation file: `{results['config']['val_path']}`",
        f"Device: `{results['config']['device']}`",
        f"Train chunks: `{results['train_chunks']}`, validation chunks: `{results['val_chunks']}`",
        "",
        "| condition | params | final val loss | final val ppl | elapsed min |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in results["runs"]:
        lines.append(
            f"| {run['condition']} | {run['params']:,} | {run['final_val_loss']:.4f} | "
            f"{run['final_val_ppl']:.2f} | {run['elapsed_sec'] / 60:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", type=Path, default=Path("data/processed/babylm_strict_small/train.jsonl"))
    parser.add_argument("--val_path", type=Path, default=Path("data/processed/babylm_strict_small/validation.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/corpus_lm"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--conditions", default="standard,resonance_full_normalized,phase_stream_only_normalized,bias_only_normalized,resonance_inert_normalized")
    parser.add_argument("--train_docs", type=int, default=200_000)
    parser.add_argument("--val_docs", type=int, default=20_000)
    parser.add_argument("--train_examples", type=int, default=20_000)
    parser.add_argument("--val_examples", type=int, default=2_000)
    parser.add_argument("--vocab_size", type=int, default=12000)
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_texts = read_jsonl_texts(args.train_path, args.train_docs)
    val_texts = read_jsonl_texts(args.val_path, args.val_docs)
    tokenizer = WordTokenizer.train(train_texts, args.vocab_size)
    train_ds = ConcatenatedTextDataset(train_texts, tokenizer, args.seq_len, args.train_examples)
    val_ds = ConcatenatedTextDataset(val_texts, tokenizer, args.seq_len, args.val_examples)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    runs = [train_one(args, condition, train_ds, val_ds, len(tokenizer)) for condition in conditions]
    results = {
        "kind": "corpus_lm",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "vocab_size": len(tokenizer),
        "train_chunks": len(train_ds),
        "val_chunks": len(val_ds),
        "runs": runs,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_report(args.output_dir / "report.md", results)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
