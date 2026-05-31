#!/usr/bin/env python3
"""Evaluate a trained unified model on BLiMP minimal pairs.

Loads a checkpoint produced by train_unified.py --save_checkpoint
and computes per-paradigm and overall accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from resonance.models import StandardTransformer, ResonanceTransformer, StandardConfig, ResonanceConfig


class WordTokenizer:
    """Minimal tokenizer compatible with train_unified.py"""
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab
        self.word2id = vocab
        self.pad_id = vocab.get("<pad>", 0)
        self.unk_id = vocab.get("<unk>", 1)

    def encode(self, text: str) -> list[int]:
        import re
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return [self.vocab.get(tok.lower(), self.unk_id) for tok in tokens]

    def __len__(self) -> int:
        return len(self.vocab)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab = ckpt["tokenizer_vocab"]
    tokenizer = WordTokenizer(vocab)
    condition = ckpt.get("condition", "standard_alibi")
    
    if isinstance(config, ResonanceConfig):
        model = ResonanceTransformer(config)
    else:
        model = StandardTransformer(config)
    
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, tokenizer, condition


def sentence_loglik(model, tokenizer, sentence: str, device: torch.device) -> float:
    ids = tokenizer.encode(sentence)
    if len(ids) < 2:
        return 0.0
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids, labels=input_ids)
        # out["loss"] is average cross-entropy loss
        # log-likelihood = -loss * seq_len
        loss = out["loss"].item()
        seq_len = len(ids) - 1  # labels are shifted
        return -loss * seq_len


def evaluate_paradigm(model, tokenizer, path: Path, device: torch.device) -> dict:
    correct = 0
    total = 0
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        good = ex["sentence_good"]
        bad = ex["sentence_bad"]
        ll_good = sentence_loglik(model, tokenizer, good, device)
        ll_bad = sentence_loglik(model, tokenizer, bad, device)
        if ll_good > ll_bad:
            correct += 1
        total += 1
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--blimp_dir", type=Path, default=Path("data/processed/blimp"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, condition = load_checkpoint(args.checkpoint, device)
    print(f"Loaded {condition} | vocab={len(tokenizer)} | device={device}")

    results = []
    overall_correct = 0
    overall_total = 0

    paradigm_files = sorted(args.blimp_dir.glob("*.jsonl"))
    for pf in paradigm_files:
        if pf.name == "all.jsonl":
            continue
        r = evaluate_paradigm(model, tokenizer, pf, device)
        r["paradigm"] = pf.stem
        results.append(r)
        overall_correct += r["correct"]
        overall_total += r["total"]
        print(f"  {pf.stem:50s} {r['accuracy']:.3f} ({r['correct']}/{r['total']})")

    overall_acc = overall_correct / overall_total if overall_total else 0.0
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall_acc:.4f} ({overall_correct}/{overall_total})")

    report = {
        "condition": condition,
        "checkpoint": str(args.checkpoint),
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "paradigms": results,
    }
    
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Wrote {args.output}")
    else:
        ckpt_dir = args.checkpoint.parent
        out_path = ckpt_dir / "blimp_eval.json"
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
