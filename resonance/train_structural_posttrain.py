#!/usr/bin/env python3
"""Post-train small transformers with verifiable structural rewards.

This harness is deliberately not a web-scale LM trainer.  It targets the part
of the thesis that raw next-token pretraining cannot test well: if a compact
structural stream is useful, it should respond to explicit outcome and process
signals on tasks whose verifier is exact.

Implemented stages:

* SFT warm-start on the answer token.
* DPO over the correct vs. incorrect answer token, using a frozen reference.
* exact-RL expected reward for binary verifiable answers.
* GRPO-style enumerated group update over the two possible answers.

For our current yes/no tasks the algorithms are intentionally lightweight, but
the accounting mirrors RLVR: reward is produced by the verifier, not by a reward
model, and diagnostics track margins, KL to reference, phase geometry, and
causal phase-ablation sensitivity.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from regime_probe import (  # noqa: E402
    jsonify,
    label_balance,
    label_dispersion,
    representation_dispersion,
    supervised_contrastive_loss,
)
from resonance.device import enable_deterministic, get_device, set_seed  # noqa: E402
from resonance.interpretability import phase_intervention  # noqa: E402
from resonance.models import ResonanceTransformer  # noqa: E402
from story_query_eval import collate_examples, encode_supervised_batch  # noqa: E402
from story_topology_eval import build_model  # noqa: E402
from structural_task_probe import TASKS, build_datasets  # noqa: E402
from synthetic.semantic_story import StoryTokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, default="cap_matching")
    parser.add_argument("--output_dir", type=Path, default=Path("resonance/outputs/structural_posttrain"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--conditions", default="standard_alibi,phase_dynamic_qk_film_alibi_normalized,relation_value_qk_film_alibi_normalized")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13])

    parser.add_argument("--sft_epochs", type=int, default=2)
    parser.add_argument("--rl_epochs", type=int, default=4)
    parser.add_argument("--rl_algorithm", choices=["dpo", "exact_rl", "grpo_enum", "sft_only"], default="dpo")
    parser.add_argument("--reference", choices=["init", "after_sft"], default="after_sft")
    parser.add_argument("--curriculum_tasks", default="", help="Comma-separated one-step/local tasks to SFT before the target task.")
    parser.add_argument("--curriculum_epochs", type=int, default=0)
    parser.add_argument("--curriculum_train_examples", type=int, default=0)
    parser.add_argument("--curriculum_val_examples", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.2, help="DPO inverse-temperature or GRPO reward scale.")
    parser.add_argument("--kl_coef", type=float, default=0.03)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ce_anchor_weight", type=float, default=0.1)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--phase_contrastive_weight", type=float, default=0.05)
    parser.add_argument("--phase_contrastive_temperature", type=float, default=0.2)

    parser.add_argument("--train_examples", type=int, default=2048)
    parser.add_argument("--val_examples", type=int, default=768)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--embed_dim", type=int, default=160)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--n_frequencies", type=int, default=32)
    parser.add_argument("--phase_init_std", type=float, default=0.3)
    parser.add_argument("--story_phase_prior_std", type=float, default=1.2)
    parser.add_argument("--story_phase_prior_noise", type=float, default=0.05)
    parser.add_argument("--resonance_attn_weight", type=float, default=0.1)
    parser.add_argument("--resonance_blend", type=float, default=0.3)
    parser.add_argument("--dataset_seed", type=int, default=9300)

    # Shared structural-task knobs copied from structural_task_probe.
    parser.add_argument("--max_attractors", type=int, default=4)
    parser.add_argument("--force_attractors", type=int, default=None)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--val_depth",
        type=int,
        default=None,
        help="Optional validation depth for extrapolation probes.",
    )
    parser.add_argument(
        "--cap_mode",
        choices=["mixed", "hard", "closure"],
        default="mixed",
        help="Cap-matching generator: mixed=random negatives, hard=matched hard negatives, closure=positives require Cap closure.",
    )
    parser.add_argument("--max_size", type=int, default=18)
    parser.add_argument("--max_trace_steps", type=int, default=8)
    parser.add_argument("--modulus", type=int, default=17)
    parser.add_argument("--n_templates", type=int, default=50)
    parser.add_argument("--dyck_mode", choices=["nested", "cross"], default="nested")
    parser.add_argument("--n_types", type=int, default=3)
    parser.add_argument("--n_events", type=int, default=6)
    parser.add_argument("--edge_prob", type=float, default=0.35)
    parser.add_argument("--n_graphs", type=int, default=16)
    parser.add_argument("--n_nodes", type=int, default=16)
    parser.add_argument("--aliases_per_node", type=int, default=3)
    parser.add_argument("--alias_pool_size", type=int, default=0)
    parser.add_argument("--out_degree", type=int, default=2)
    parser.add_argument("--walk_length", type=int, default=4)
    parser.add_argument("--query_steps", type=int, default=1)
    parser.add_argument("--val_graphs", choices=["same", "same_aliases_new_edges", "new"], default="new")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)

    parser.add_argument("--eval_each_epoch", action="store_true")
    parser.add_argument(
        "--skip_before_eval",
        action="store_true",
        help="Skip initial/before validation. Useful for large trace runs where random-policy eval is not informative.",
    )
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--phase_ablation_eval", action="store_true")
    parser.add_argument("--phase_noise_sigma", type=float, default=0.5)
    parser.add_argument("--cascade_eval", action="store_true", help="Evaluate trace validity by composing local one-step judgments.")
    return parser.parse_args()


class AnswerPrompt:
    def __init__(self, prefix: str, answer: str = "yes") -> None:
        self.prefix = prefix
        self.answer = answer
        self.full_text = f"{prefix} {answer}."
        self.graph_id = "cascade"


def answer_log_probs(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    *,
    max_length: int,
    device: torch.device,
    return_states: bool = False,
) -> dict[str, torch.Tensor | None]:
    input_ids, _labels, answer_positions = encode_supervised_batch(
        tokenizer,
        batch,
        max_length=max_length,
        device=device,
    )
    out = model(input_ids, labels=None, return_hidden=return_states)
    logits = out["logits"]
    yes_id = tokenizer.word2id["yes"]
    no_id = tokenizer.word2id["no"]
    answer_states = logits.gather(
        1,
        answer_positions.view(-1, 1, 1).expand(-1, 1, logits.size(-1)),
    ).squeeze(1)
    answer_token_ids = torch.tensor([yes_id, no_id], dtype=torch.long, device=device)
    answer_logits = answer_states.index_select(-1, answer_token_ids)
    log_probs = F.log_softmax(answer_logits, dim=-1)
    probs = log_probs.exp()
    targets = torch.tensor(
        [0 if example.answer == "yes" else 1 for example in batch],
        dtype=torch.long,
        device=device,
    )
    rejected = 1 - targets
    chosen_logp = log_probs.gather(1, targets[:, None]).squeeze(1)
    rejected_logp = log_probs.gather(1, rejected[:, None]).squeeze(1)
    hidden = None
    phase = None
    if return_states:
        hidden_states = out["hidden_states"]
        hidden = hidden_states.gather(
            1,
            answer_positions.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1)),
        ).squeeze(1)
        phase_states = out.get("phase_states")
        if isinstance(phase_states, torch.Tensor):
            mask = (input_ids != tokenizer.pad_id).float().unsqueeze(-1)
            phase = (phase_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return {
        "logits": answer_logits,
        "log_probs": log_probs,
        "probs": probs,
        "targets": targets,
        "rejected": rejected,
        "chosen_logp": chosen_logp,
        "rejected_logp": rejected_logp,
        "hidden": hidden,
        "phase": phase,
    }


@torch.no_grad()
def evaluate_cascade_metrics(
    model: torch.nn.Module,
    dataset,
    tokenizer: StoryTokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Predict whole-trace validity by composing local one-step judgments."""
    trace_rows = []
    prompt_examples: list[AnswerPrompt] = []
    prompt_trace_ids: list[int] = []
    for idx in range(len(dataset)):
        example = dataset[idx]
        metadata = getattr(example, "metadata", {})
        prompts = metadata.get("local_step_prompts") if isinstance(metadata, dict) else None
        if not prompts:
            continue
        trace_rows.append(example)
        trace_id = len(trace_rows) - 1
        for prompt in prompts:
            prompt_examples.append(AnswerPrompt(str(prompt), "yes"))
            prompt_trace_ids.append(trace_id)
    if not trace_rows or not prompt_examples:
        return {}

    per_trace_probs: list[list[float]] = [[] for _ in trace_rows]
    loader = DataLoader(prompt_examples, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    offset = 0
    for batch in loader:
        values = answer_log_probs(
            model,
            tokenizer,
            batch,
            max_length=max_length,
            device=device,
            return_states=False,
        )
        probs = values["probs"]
        assert isinstance(probs, torch.Tensor)
        yes_probs = probs[:, 0].detach().cpu().tolist()
        for local_idx, yes_prob in enumerate(yes_probs):
            trace_idx = prompt_trace_ids[offset + local_idx]
            per_trace_probs[trace_idx].append(float(yes_prob))
        offset += len(yes_probs)

    correct = 0
    pred_valid_count = 0
    min_yes_probs = []
    mean_yes_probs = []
    for example, probs_for_trace in zip(trace_rows, per_trace_probs):
        metadata = getattr(example, "metadata", {})
        terminal_ok = bool(metadata.get("terminal_normal", metadata.get("terminal_matches", True)))
        local_ok = bool(probs_for_trace) and all(prob >= threshold for prob in probs_for_trace)
        pred_valid = local_ok and terminal_ok
        target_valid = getattr(example, "answer", "no") == "yes"
        correct += int(pred_valid == target_valid)
        pred_valid_count += int(pred_valid)
        min_yes_probs.append(min(probs_for_trace) if probs_for_trace else 0.0)
        mean_yes_probs.append(statistics.fmean(probs_for_trace) if probs_for_trace else 0.0)

    return {
        "cascade_acc": correct / max(len(trace_rows), 1),
        "cascade_pred_valid_rate": pred_valid_count / max(len(trace_rows), 1),
        "cascade_min_yes_prob": statistics.fmean(min_yes_probs) if min_yes_probs else 0.0,
        "cascade_mean_yes_prob": statistics.fmean(mean_yes_probs) if mean_yes_probs else 0.0,
        "cascade_n_traces": float(len(trace_rows)),
    }


def phase_auxiliary(
    phase: torch.Tensor | None,
    targets: torch.Tensor,
    *,
    weight: float,
    temperature: float,
) -> torch.Tensor:
    if phase is None or weight <= 0:
        return targets.new_tensor(0.0, dtype=torch.float32)
    return weight * supervised_contrastive_loss(phase, targets, temperature=temperature)


def kl_to_reference(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
) -> torch.Tensor:
    policy_probs = policy_log_probs.exp()
    return (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)


def maybe_sync(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset,
    tokenizer: StoryTokenizer,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
    ref_model: torch.nn.Module | None = None,
    cascade_eval: bool = False,
) -> dict[str, float]:
    model.eval()
    if ref_model is not None:
        ref_model.eval()
    maybe_sync(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_examples)
    total = 0
    correct = 0
    losses: list[float] = []
    margins: list[float] = []
    rewards: list[float] = []
    kls: list[float] = []
    hidden_rows: list[torch.Tensor] = []
    phase_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for batch in loader:
        values = answer_log_probs(
            model,
            tokenizer,
            batch,
            max_length=max_length,
            device=device,
            return_states=True,
        )
        logits = values["logits"]
        targets = values["targets"]
        log_probs = values["log_probs"]
        assert isinstance(logits, torch.Tensor)
        assert isinstance(targets, torch.Tensor)
        assert isinstance(log_probs, torch.Tensor)
        predictions = logits.argmax(dim=-1)
        total += int(targets.numel())
        correct += int((predictions == targets).sum().item())
        losses.extend(F.cross_entropy(logits, targets, reduction="none").detach().cpu().tolist())
        signed_margin = logits[:, 0] - logits[:, 1]
        signed_margin = torch.where(targets == 0, signed_margin, -signed_margin)
        margins.extend(signed_margin.detach().cpu().tolist())
        rewards.extend((predictions == targets).float().detach().cpu().tolist())
        if ref_model is not None:
            ref_values = answer_log_probs(
                ref_model,
                tokenizer,
                batch,
                max_length=max_length,
                device=device,
                return_states=False,
            )
            ref_log_probs = ref_values["log_probs"]
            assert isinstance(ref_log_probs, torch.Tensor)
            kls.extend(kl_to_reference(log_probs, ref_log_probs).detach().cpu().tolist())
        hidden = values["hidden"]
        phase = values["phase"]
        if isinstance(hidden, torch.Tensor):
            hidden_rows.append(hidden.detach().cpu())
            target_rows.append(targets.detach().cpu())
        if isinstance(phase, torch.Tensor):
            phase_rows.append(phase.detach().cpu())

    result = {
        "answer_acc": correct / max(total, 1),
        "answer_loss": statistics.fmean(losses) if losses else 0.0,
        "answer_margin": statistics.fmean(margins) if margins else 0.0,
        "verifier_reward": statistics.fmean(rewards) if rewards else 0.0,
        "kl_to_ref": statistics.fmean(kls) if kls else 0.0,
    }
    if hidden_rows:
        hidden_tensor = torch.cat(hidden_rows, dim=0)
        targets_tensor = torch.cat(target_rows, dim=0)
        result["hidden_dispersion"] = float(representation_dispersion(hidden_tensor).item())
        result.update({f"hidden_{k}": v for k, v in label_dispersion(hidden_tensor, targets_tensor).items()})
    if phase_rows:
        phase_tensor = torch.cat(phase_rows, dim=0)
        targets_tensor = torch.cat(target_rows, dim=0)
        result["phase_dispersion"] = float(representation_dispersion(phase_tensor).item())
        result.update({f"phase_{k}": v for k, v in label_dispersion(phase_tensor, targets_tensor).items()})
    if cascade_eval:
        result.update(
            evaluate_cascade_metrics(
                model,
                dataset,
                tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
        )
    maybe_sync(device)
    return result


def sft_step(
    model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = answer_log_probs(
        model,
        tokenizer,
        batch,
        max_length=args.max_length,
        device=device,
        return_states=True,
    )
    logits = values["logits"]
    targets = values["targets"]
    phase = values["phase"]
    assert isinstance(logits, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    answer_loss = F.cross_entropy(logits, targets)
    phase_aux = phase_auxiliary(
        phase if isinstance(phase, torch.Tensor) else None,
        targets,
        weight=args.phase_contrastive_weight,
        temperature=args.phase_contrastive_temperature,
    )
    loss = answer_loss + phase_aux
    return loss, {
        "sft_loss": float(answer_loss.detach().item()),
        "phase_aux": float(phase_aux.detach().item()),
    }


def dpo_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = answer_log_probs(
        model,
        tokenizer,
        batch,
        max_length=args.max_length,
        device=device,
        return_states=True,
    )
    with torch.no_grad():
        ref_values = answer_log_probs(
            ref_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            device=device,
            return_states=False,
        )
    chosen = values["chosen_logp"]
    rejected = values["rejected_logp"]
    ref_chosen = ref_values["chosen_logp"]
    ref_rejected = ref_values["rejected_logp"]
    targets = values["targets"]
    phase = values["phase"]
    logits = values["logits"]
    assert isinstance(chosen, torch.Tensor) and isinstance(rejected, torch.Tensor)
    assert isinstance(ref_chosen, torch.Tensor) and isinstance(ref_rejected, torch.Tensor)
    assert isinstance(targets, torch.Tensor) and isinstance(logits, torch.Tensor)
    policy_margin = chosen - rejected
    ref_margin = ref_chosen - ref_rejected
    dpo_loss = -F.logsigmoid(args.beta * (policy_margin - ref_margin)).mean()
    ce_anchor = F.cross_entropy(logits, targets)
    phase_aux = phase_auxiliary(
        phase if isinstance(phase, torch.Tensor) else None,
        targets,
        weight=args.phase_contrastive_weight,
        temperature=args.phase_contrastive_temperature,
    )
    loss = dpo_loss + args.ce_anchor_weight * ce_anchor + phase_aux
    return loss, {
        "dpo_loss": float(dpo_loss.detach().item()),
        "ce_anchor": float(ce_anchor.detach().item()),
        "phase_aux": float(phase_aux.detach().item()),
        "policy_margin": float(policy_margin.detach().mean().item()),
        "ref_margin": float(ref_margin.detach().mean().item()),
    }


def exact_rl_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = answer_log_probs(
        model,
        tokenizer,
        batch,
        max_length=args.max_length,
        device=device,
        return_states=True,
    )
    with torch.no_grad():
        ref_values = answer_log_probs(
            ref_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            device=device,
            return_states=False,
        )
    probs = values["probs"]
    log_probs = values["log_probs"]
    ref_log_probs = ref_values["log_probs"]
    targets = values["targets"]
    phase = values["phase"]
    assert isinstance(probs, torch.Tensor)
    assert isinstance(log_probs, torch.Tensor)
    assert isinstance(ref_log_probs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    reward = probs.gather(1, targets[:, None]).squeeze(1)
    kl = kl_to_reference(log_probs, ref_log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    phase_aux = phase_auxiliary(
        phase if isinstance(phase, torch.Tensor) else None,
        targets,
        weight=args.phase_contrastive_weight,
        temperature=args.phase_contrastive_temperature,
    )
    loss = -reward.mean() + args.kl_coef * kl.mean() - args.entropy_coef * entropy.mean() + phase_aux
    return loss, {
        "expected_reward": float(reward.detach().mean().item()),
        "kl": float(kl.detach().mean().item()),
        "entropy": float(entropy.detach().mean().item()),
        "phase_aux": float(phase_aux.detach().item()),
    }


def grpo_enum_step(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: StoryTokenizer,
    batch,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    values = answer_log_probs(
        model,
        tokenizer,
        batch,
        max_length=args.max_length,
        device=device,
        return_states=True,
    )
    with torch.no_grad():
        ref_values = answer_log_probs(
            ref_model,
            tokenizer,
            batch,
            max_length=args.max_length,
            device=device,
            return_states=False,
        )
    log_probs = values["log_probs"]
    ref_log_probs = ref_values["log_probs"]
    targets = values["targets"]
    phase = values["phase"]
    assert isinstance(log_probs, torch.Tensor)
    assert isinstance(ref_log_probs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    rewards = torch.zeros_like(log_probs)
    rewards.scatter_(1, targets[:, None], 1.0)
    advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / rewards.std(dim=-1, keepdim=True).clamp(min=1e-6)
    ratio = (log_probs - ref_log_probs).exp()
    clipped = ratio.clamp(1.0 - args.clip_range, 1.0 + args.clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped * advantages)
    kl = kl_to_reference(log_probs, ref_log_probs)
    phase_aux = phase_auxiliary(
        phase if isinstance(phase, torch.Tensor) else None,
        targets,
        weight=args.phase_contrastive_weight,
        temperature=args.phase_contrastive_temperature,
    )
    loss = -(args.beta * surrogate).mean() + args.kl_coef * kl.mean() + phase_aux
    chosen_adv = advantages.gather(1, targets[:, None]).mean()
    return loss, {
        "grpo_loss": float((-(args.beta * surrogate).mean()).detach().item()),
        "kl": float(kl.detach().mean().item()),
        "phase_aux": float(phase_aux.detach().item()),
        "chosen_advantage": float(chosen_adv.detach().item()),
    }


def train_epoch(
    *,
    model: torch.nn.Module,
    ref_model: torch.nn.Module | None,
    tokenizer: StoryTokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    stage: str,
) -> dict[str, float]:
    model.train()
    if ref_model is not None:
        ref_model.eval()
    rows: dict[str, list[float]] = {}
    for batch in loader:
        if stage == "sft" or args.rl_algorithm == "sft_only":
            loss, metrics = sft_step(model, tokenizer, batch, args, device)
        elif args.rl_algorithm == "dpo":
            assert ref_model is not None
            loss, metrics = dpo_step(model, ref_model, tokenizer, batch, args, device)
        elif args.rl_algorithm == "exact_rl":
            assert ref_model is not None
            loss, metrics = exact_rl_step(model, ref_model, tokenizer, batch, args, device)
        elif args.rl_algorithm == "grpo_enum":
            assert ref_model is not None
            loss, metrics = grpo_enum_step(model, ref_model, tokenizer, batch, args, device)
        else:
            raise ValueError(args.rl_algorithm)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        maybe_sync(device)
        rows.setdefault("loss", []).append(float(loss.detach().item()))
        for key, value in metrics.items():
            rows.setdefault(key, []).append(value)
    return {key: statistics.fmean(value) for key, value in rows.items() if value}


@torch.no_grad()
def phase_ablation_metrics(
    model: torch.nn.Module,
    dataset,
    tokenizer: StoryTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    base: dict[str, float],
) -> dict[str, dict[str, float]]:
    if not isinstance(model, ResonanceTransformer):
        return {}
    out: dict[str, dict[str, float]] = {}
    for mode in ("zero", "permute", "noise"):
        with phase_intervention(model, mode=mode, sigma=args.phase_noise_sigma, seed=args.dataset_seed):
            metrics = evaluate_model(
                model,
                dataset,
                tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                device=device,
            )
        out[mode] = {
            "answer_acc": metrics["answer_acc"],
            "answer_margin": metrics["answer_margin"],
            "answer_acc_delta": metrics["answer_acc"] - base["answer_acc"],
            "answer_margin_delta": metrics["answer_margin"] - base["answer_margin"],
        }
    return out


def build_curriculum_datasets(args: argparse.Namespace):
    tasks = [task.strip() for task in args.curriculum_tasks.split(",") if task.strip()]
    if not tasks or args.curriculum_epochs <= 0:
        return []
    datasets = []
    for idx, task in enumerate(tasks):
        sub_args = copy.copy(args)
        sub_args.task = task
        sub_args.train_examples = args.curriculum_train_examples or args.train_examples
        sub_args.val_examples = args.curriculum_val_examples or min(args.val_examples, 256)
        sub_args.dataset_seed = args.dataset_seed + 700_000 + idx * 100_000
        train_ds, val_ds = build_datasets(sub_args)
        datasets.append((task, train_ds, val_ds))
    return datasets


def train_condition(
    args: argparse.Namespace,
    condition: str,
    seed: int,
    train_ds,
    val_ds,
    tokenizer: StoryTokenizer,
    curriculum_sets,
) -> dict[str, Any]:
    set_seed(seed)
    device = get_device(args.device)
    model, config = build_model(args, condition, tokenizer)
    model.to(device)
    init_ref = copy.deepcopy(model).to(device).eval()
    for param in init_ref.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples)

    print(f"[{condition} seed={seed}] starting train_condition", flush=True)
    if args.skip_before_eval:
        initial_before = {}
    else:
        print(f"[{condition} seed={seed}] initial eval", flush=True)
        initial_before = evaluate_model(
            model,
            val_ds,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            ref_model=init_ref,
        )
    history: list[dict[str, float | str | int]] = []
    start = time.perf_counter()

    for curriculum_task, curriculum_train, _curriculum_val in curriculum_sets:
        curriculum_loader = DataLoader(
            curriculum_train,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_examples,
        )
        for epoch in range(args.curriculum_epochs):
            row = train_epoch(
                model=model,
                ref_model=None,
                tokenizer=tokenizer,
                loader=curriculum_loader,
                optimizer=optimizer,
                args=args,
                device=device,
                stage="sft",
            )
            row.update(
                {
                    "stage": "curriculum_sft",
                    "curriculum_task": curriculum_task,
                    "epoch": epoch + 1,
                }
            )
            if args.eval_each_epoch:
                val = evaluate_model(
                    model,
                    val_ds,
                    tokenizer,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    ref_model=init_ref,
                )
                row.update({f"target_val_{k}": v for k, v in val.items() if isinstance(v, float)})
            history.append(row)
            print(
                f"[{condition} seed={seed}] curriculum {curriculum_task} "
                f"epoch={epoch + 1} loss={row['loss']:.4f}",
                flush=True,
            )

    if args.skip_before_eval:
        before = {}
    else:
        print(f"[{condition} seed={seed}] pre-target eval", flush=True)
        before = evaluate_model(
            model,
            val_ds,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            ref_model=init_ref,
        )

    for epoch in range(args.sft_epochs):
        row = train_epoch(
            model=model,
            ref_model=None,
            tokenizer=tokenizer,
            loader=loader,
            optimizer=optimizer,
            args=args,
            device=device,
            stage="sft",
        )
        row.update({"stage": "sft", "epoch": epoch + 1})
        if args.eval_each_epoch:
            val = evaluate_model(
                model,
                val_ds,
                tokenizer,
                batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    ref_model=init_ref,
            )
            row.update({f"val_{k}": v for k, v in val.items() if isinstance(v, float)})
        history.append(row)
        print(f"[{condition} seed={seed}] sft epoch={epoch + 1} loss={row['loss']:.4f}", flush=True)

    if args.reference == "after_sft":
        ref_model = copy.deepcopy(model).to(device).eval()
        for param in ref_model.parameters():
            param.requires_grad_(False)
    else:
        ref_model = init_ref

    if args.rl_algorithm != "sft_only":
        for epoch in range(args.rl_epochs):
            row = train_epoch(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                loader=loader,
                optimizer=optimizer,
                args=args,
                device=device,
                stage=args.rl_algorithm,
            )
            row.update({"stage": args.rl_algorithm, "epoch": epoch + 1})
            if args.eval_each_epoch:
                val = evaluate_model(
                    model,
                    val_ds,
                    tokenizer,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    device=device,
                    ref_model=ref_model,
                )
                row.update({f"val_{k}": v for k, v in val.items() if isinstance(v, float)})
            history.append(row)
            print(f"[{condition} seed={seed}] {args.rl_algorithm} epoch={epoch + 1} loss={row['loss']:.4f}", flush=True)

    print(f"[{condition} seed={seed}] final eval", flush=True)
    after = evaluate_model(
        model,
        val_ds,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        ref_model=ref_model,
        cascade_eval=args.cascade_eval,
    )
    interventions = (
        phase_ablation_metrics(model, val_ds, tokenizer, args, device, after)
        if args.phase_ablation_eval
        else {}
    )
    checkpoint_path = None
    if args.save_models:
        ckpt_dir = args.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        safe_condition = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in condition)
        checkpoint_path = ckpt_dir / f"{args.task}_{safe_condition}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "condition": condition,
                "seed": seed,
                "args": jsonify(vars(args)),
                "model_config": jsonify(getattr(config, "__dict__", {})),
                "tokenizer": {
                    "vocab_size": tokenizer.vocab_size,
                    "word2id": tokenizer.word2id,
                },
            },
            checkpoint_path,
        )
    return {
        "task": args.task,
        "condition": condition,
        "seed": seed,
        "rl_algorithm": args.rl_algorithm,
        "reference": args.reference,
        "curriculum_tasks": [task for task, _train, _val in curriculum_sets],
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "config": {
            "use_phase_stream": getattr(config, "use_phase_stream", None),
            "use_resonance_bias": getattr(config, "use_resonance_bias", None),
            "resonance_kernel": getattr(config, "resonance_kernel", None),
            "phase_update_mode": getattr(config, "phase_update_mode", None),
            "phase_condition_qk": getattr(config, "phase_condition_qk", None),
            "relation_value_mode": getattr(config, "relation_value_mode", None),
            "attention_variant": getattr(config, "attention_variant", None),
        },
        "initial_before": initial_before,
        "before": before,
        "after": after,
        "interventions": interventions,
        "history": history,
        "elapsed_sec": time.perf_counter() - start,
    }


def summarise(runs: list[dict[str, Any]], majority_acc: float) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["condition"], []).append(run)
    summary: dict[str, dict[str, float]] = {}
    for condition, rows in grouped.items():
        acc = [float(row["after"]["answer_acc"]) for row in rows]
        before_acc = [
            float(row["before"]["answer_acc"])
            for row in rows
            if isinstance(row.get("before"), dict) and "answer_acc" in row["before"]
        ]
        margin = [float(row["after"]["answer_margin"]) for row in rows]
        loss = [float(row["after"]["answer_loss"]) for row in rows]
        phase_gap = [
            float(row["after"].get("phase_between_minus_within", 0.0))
            for row in rows
        ]
        hidden_gap = [
            float(row["after"].get("hidden_between_minus_within", 0.0))
            for row in rows
        ]
        cascade_acc = [
            float(row["after"]["cascade_acc"])
            for row in rows
            if "cascade_acc" in row["after"]
        ]
        summary[condition] = {
            "n": float(len(rows)),
            "params": float(rows[0]["params"]),
            "before_acc_mean": statistics.fmean(before_acc) if before_acc else float("nan"),
            "answer_acc_mean": statistics.fmean(acc),
            "answer_acc_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
            "acc_minus_majority": statistics.fmean(acc) - majority_acc,
            "acc_gain": statistics.fmean(acc) - statistics.fmean(before_acc) if before_acc else float("nan"),
            "answer_margin_mean": statistics.fmean(margin),
            "answer_loss_mean": statistics.fmean(loss),
            "phase_label_gap": statistics.fmean(phase_gap),
            "hidden_label_gap": statistics.fmean(hidden_gap),
            "cascade_acc_mean": statistics.fmean(cascade_acc) if cascade_acc else float("nan"),
        }
    return summary


def write_report(path: Path, args: argparse.Namespace, runs: list[dict[str, Any]], balance: dict[str, float]) -> None:
    summary = summarise(runs, balance["majority_acc"])
    lines = [
        "# Structural Posttraining",
        "",
        f"Task: `{args.task}`.",
        f"Algorithm: `{args.rl_algorithm}`, reference: `{args.reference}`.",
        f"Curriculum tasks: `{args.curriculum_tasks or 'none'}`, curriculum epochs: `{args.curriculum_epochs}`.",
        f"SFT epochs: `{args.sft_epochs}`, RL epochs: `{args.rl_epochs}`.",
        f"Train examples: `{args.train_examples}`, validation examples: `{args.val_examples}`.",
        "",
        "| Condition | Params | Before Acc | Final Acc | Cascade Acc | Gain | Acc-Majority | Margin | Loss | Phase Gap | Hidden Gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in summary.items():
        cascade = "" if math.isnan(row["cascade_acc_mean"]) else f"{row['cascade_acc_mean']:.4f}"
        before = "" if math.isnan(row["before_acc_mean"]) else f"{row['before_acc_mean']:.4f}"
        gain = "" if math.isnan(row["acc_gain"]) else f"{row['acc_gain']:+.4f}"
        lines.append(
            f"| {condition} | {int(row['params']):,} | {before} | "
            f"{row['answer_acc_mean']:.4f} | {cascade} | {gain} | "
            f"{row['acc_minus_majority']:+.4f} | {row['answer_margin_mean']:.4f} | "
            f"{row['answer_loss_mean']:.4f} | {row['phase_label_gap']:.4f} | "
            f"{row['hidden_label_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Contract",
            "",
            "A useful run should improve verifier reward while also making the structural stream more probeable or more causally relevant. Accuracy alone is not enough.",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    enable_deterministic(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = build_datasets(args)
    curriculum_sets = build_curriculum_datasets(args)
    tokenizer = StoryTokenizer(vocab_size=args.vocab_size)
    curriculum_texts: list[str] = []
    for _task, curriculum_train, curriculum_val in curriculum_sets:
        curriculum_texts.extend(curriculum_train.all_texts())
        curriculum_texts.extend(curriculum_val.all_texts())
    tokenizer.train(train_ds.all_texts() + val_ds.all_texts() + curriculum_texts + ["yes no"])

    conditions = [condition.strip() for condition in args.conditions.split(",") if condition.strip()]
    runs = []
    for seed in args.seeds:
        for condition in conditions:
            runs.append(train_condition(args, condition, seed, train_ds, val_ds, tokenizer, curriculum_sets))
    balance = label_balance(val_ds)
    results = {
        "task": args.task,
        "summary": summarise(runs, balance["majority_acc"]),
        "label_balance": balance,
        "runs": runs,
        "config": vars(args),
    }
    (args.output_dir / "results.json").write_text(json.dumps(jsonify(results), indent=2))
    write_report(args.output_dir / "report.md", args, runs, balance)
    print(f"Wrote {args.output_dir / 'results.json'}")
    print(f"Wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
