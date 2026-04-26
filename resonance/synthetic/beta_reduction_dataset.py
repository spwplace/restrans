"""Beta-reduction trace dataset for realizability-topos training.

Generates random simply-typed lambda calculus terms, computes their full
beta-reduction traces, and produces (term, normal_form) pairs for
contrastive equivalence learning.

The core hypothesis: a ResonanceTransformer should learn that a term and
its normal form are "the same" (semantic equivalence) even though they
have different syntax.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .lambda_generator import LambdaGenerator, normal_form
from .proof_walk_generator import ProofWalkDataset


def compute_reduction_trace(term, max_steps: int = 100) -> List:
    """Compute the full beta-reduction trace of a term.

    Returns a list [t0, t1, t2, ..., tn] where t0 is the original term,
    each ti+1 is the result of one beta-reduction step, and tn is in
    normal form (or max_steps was reached).
    """
    from .lambda_generator import _beta_reduce_once
    trace = [term]
    for _ in range(max_steps):
        reduced, new_term = _beta_reduce_once(trace[-1])
        if not reduced:
            break
        trace.append(new_term)
    return trace


class BetaReductionDataset:
    """Dataset of (term, normal_form) pairs with reduction traces.

    Unlike ProofWalkDataset which uses same-type-in-same-context as a
    proxy for equivalence, this dataset uses exact beta-normal-form
    equality. Every positive pair is guaranteed semantically equivalent.
    """

    def __init__(
        self,
        n_pairs: int = 1000,
        max_depth: int = 5,
        max_size: int = 20,
        min_trace_length: int = 2,
        generator_seed: int = 42,
    ) -> None:
        self.n_pairs = n_pairs
        self.pairs: List[dict] = []
        self.traces: List[List[str]] = []

        gen = LambdaGenerator(seed=generator_seed)
        rng = random.Random(generator_seed)

        attempts = 0
        max_attempts = n_pairs * 20

        while len(self.pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            try:
                term = gen.random_term(max_depth=max_depth, max_size=max_size)
                trace = compute_reduction_trace(term, max_steps=50)

                if len(trace) < min_trace_length:
                    continue

                original_str = trace[0].to_string()
                nf_str = trace[-1].to_string()

                # Skip if original is already in normal form
                if original_str == nf_str:
                    continue

                # Deduplicate: skip if we've seen this exact pair
                if any(p["original"] == original_str for p in self.pairs):
                    continue

                self.pairs.append({
                    "original": original_str,
                    "normal_form": nf_str,
                    "trace_length": len(trace),
                    "original_size": trace[0].size(),
                    "nf_size": trace[-1].size(),
                })

                # Also store the full trace as strings
                self.traces.append([t.to_string() for t in trace])

            except Exception:
                continue

        print(f"BetaReductionDataset: generated {len(self.pairs)}/{n_pairs} pairs "
              f"({attempts} attempts, {len(self.pairs)/max(attempts,1):.1%} yield)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        return self.pairs[idx]

    def get_trace(self, idx: int) -> List[str]:
        return self.traces[idx]

    def statistics(self) -> dict:
        if not self.pairs:
            return {}
        trace_lengths = [p["trace_length"] for p in self.pairs]
        orig_sizes = [p["original_size"] for p in self.pairs]
        nf_sizes = [p["nf_size"] for p in self.pairs]
        return {
            "n_pairs": len(self.pairs),
            "mean_trace_length": sum(trace_lengths) / len(trace_lengths),
            "max_trace_length": max(trace_lengths),
            "mean_original_size": sum(orig_sizes) / len(orig_sizes),
            "mean_nf_size": sum(nf_sizes) / len(nf_sizes),
            "size_reduction": (sum(orig_sizes) - sum(nf_sizes)) / sum(orig_sizes),
        }
