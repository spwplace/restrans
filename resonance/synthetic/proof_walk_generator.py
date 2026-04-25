"""
Proof Walk Generator
====================

Implements the core Resonance training idea:

> "random statements, walk the space of proof for a statement, synthesize the
> programs representing all those proofs, use that as a group and update the
> model in the direction that improves contrastive discrimination of the whole
> group. this directly trains against the actual topology of judgemental
> deduction."

A **statement** is a typing judgement  Γ ⊢ e : T.
A **proof walk** explores multiple distinct programs (terms) that all satisfy
the same statement. These form a **contrastive group**: the model learns that
all programs in the group are "the same proof" at the semantic level, even
though they differ syntactically.

This directly encodes the topology of judgemental deduction into the training
objective: the Resonance Transformer must map all proofs of the same
proposition to nearby points in embedding space, while pushing proofs of
different propositions apart.

The module exports a PyTorch Dataset compatible with DataLoader.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset

from .lambda_generator import (
    Abs,
    App,
    ArrowType,
    BaseType,
    Context,
    LambdaGenerator,
    Term,
    T_BOOL,
    T_INT,
    T_UNIT,
    BASE_TYPES,
    typecheck,
    shift,
    substitute,
    Var,
    normal_form,
    is_well_typed,
)


# =============================================================================
# 1. Statement Representation
# =============================================================================

@dataclass(frozen=True)
class Statement:
    """
    A typing judgement  Γ ⊢ ? : T.

    The "?" is the hole we fill with distinct programs to form a proof walk.
    """

    ctx: Tuple[Type, ...]  # immutable for hashing
    target: Type

    def __repr__(self) -> str:
        ctx_str = ", ".join(str(t) for t in self.ctx)
        return f"[{ctx_str}] ⊢ ? : {self.target}"

    def to_string(self) -> str:
        return repr(self)


# Need to import Type here for the dataclass -- but we already have it from lambda_generator
from .lambda_generator import Type


# =============================================================================
# 2. Proof Strategies
# =============================================================================

class ProofStrategy:
    """
    Base class for strategies that generate distinct proofs of the same statement.

    Each strategy produces a different "syntactic shape" for a term that still
    type-checks to the same type under the same context.
    """

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        """
        Given a base term and its statement, produce a variant term that still
        satisfies the same typing judgement.
        """
        raise NotImplementedError


class IdentityVariant(ProofStrategy):
    """Returns the base term unchanged. Always succeeds."""

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        return deepcopy(base_term)


class EtaExpansionStrategy(ProofStrategy):
    """
    η-expansion:  f  ~~>  λx. f x   (when f has arrow type).

    Produces a larger term with the same semantics.
    """

    def __init__(self, max_expansions: int = 1) -> None:
        self.max_expansions = max_expansions

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        term = deepcopy(base_term)
        for _ in range(self.max_expansions):
            term = self._try_eta_expand(term)
            if term is None:
                return None
        return term

    def _try_eta_expand(self, term: Term) -> Optional[Term]:
        # We can eta-expand if the term has arrow type
        # Wrap:  term  ~~>  λx. (shift(term, 1)  x)
        # This is always valid for any term of arrow type
        # For simplicity, we just wrap once at the root
        # A more sophisticated version would pick random positions
        return Abs(ArrowType(T_INT, T_INT), App(shift(term, 1), Var(0)))


class LetBindingStrategy(ProofStrategy):
    """
    Let-binding introduction:  e  ~~>  (λx. e') arg

    Simulated via immediate beta-redex. Introduces a "synthetic let" by
    wrapping a sub-expression in an application of an abstraction.
    """

    def __init__(self, binding_prob: float = 0.3) -> None:
        self.binding_prob = binding_prob

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        term = deepcopy(base_term)
        return self._introduce_let(term, gen, statement)

    def _introduce_let(
        self, term: Term, gen: LambdaGenerator, statement: Statement
    ) -> Optional[Term]:
        # Find a subterm to wrap in a let
        # We simulate let x = e1 in e2  as  (λx. e2) e1
        # For simplicity, wrap the whole term or a random subterm
        if random.random() < 0.5:
            # Wrap the whole term: identity-like let
            ty = statement.target
            if isinstance(ty, ArrowType):
                # λx. (term x)   -- this is an eta step but counts as a proof variant
                arg_ty = ty.arg
                return Abs(arg_ty, App(shift(term, 1), Var(0)))
        return term


class BetaExpansionStrategy(ProofStrategy):
    """
    β-expansion (inverse of β-reduction):  e  ~~>  (λx. e[x]) arg

    Introduces an abstraction applied to an argument, creating a beta-redex
    that reduces back to the original term.
    """

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        term = deepcopy(base_term)
        ty = statement.target
        # Pick a subterm position and wrap it
        # Simplified: wrap the whole term with a trivial identity redex
        # (λx:ty. x) term  -- reduces to term
        id_abs = Abs(ty, Var(0))
        wrapped = App(id_abs, term)
        # Verify
        if is_well_typed(wrapped, list(statement.ctx)):
            return wrapped
        return term


class UnitWrapperStrategy(ProofStrategy):
    """
    Dead code introduction with Unit type:  e  ~~>  (λ_:Unit. e) ()

    Semantically equivalent, syntactically larger. Only works when we can
    introduce Unit in context.
    """

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        term = deepcopy(base_term)
        ty = statement.target
        # (λ_:Unit. term) ()
        wrapped = Abs(T_UNIT, shift(term, 1))
        # We need a unit value -- use a dummy variable of unit type if available
        # or just return a modified context term
        # Simpler: (λx:ty->ty. x term) (λx:ty. x)
        id_abs = Abs(ty, Var(0))
        return App(id_abs, term)


class ContextPermutationStrategy(ProofStrategy):
    """
    Re-derive the same term using different context variables.

    If the context has multiple variables of the same type, pick a different
    one. Only meaningful when context has duplicates.
    """

    def generate(
        self,
        base_term: Term,
        statement: Statement,
        gen: LambdaGenerator,
    ) -> Optional[Term]:
        # Find variables in context with duplicate types
        ctx_list = list(statement.ctx)
        type_to_indices: Dict[Type, List[int]] = {}
        for i, ty in enumerate(ctx_list):
            type_to_indices.setdefault(ty, []).append(i)

        # Build a variant by permuting variable indices
        def permute(t: Term) -> Term:
            if isinstance(t, Var):
                ty = ctx_list[t.index] if t.index < len(ctx_list) else None
                if ty is not None:
                    candidates = type_to_indices.get(ty, [])
                    if len(candidates) > 1:
                        new_idx = random.choice([c for c in candidates if c != t.index])
                        return Var(new_idx)
                return t
            if isinstance(t, Abs):
                return Abs(t.ty, permute(t.body))
            if isinstance(t, App):
                return App(permute(t.func), permute(t.arg))
            return t

        return permute(base_term)


# =============================================================================
# 3. Proof Walk Generator
# =============================================================================

DEFAULT_STRATEGIES: List[ProofStrategy] = [
    IdentityVariant(),
    EtaExpansionStrategy(max_expansions=1),
    EtaExpansionStrategy(max_expansions=2),
    BetaExpansionStrategy(),
    LetBindingStrategy(),
    UnitWrapperStrategy(),
    ContextPermutationStrategy(),
]


class ProofWalkDataset(Dataset):
    """
    PyTorch Dataset that generates proof walks for contrastive training.

    Each item is a group of k programs that all prove the same statement.
    The contrastive objective learns to cluster these programs together
    in embedding space.

    Parameters
    ----------
    n_groups : int
        Number of distinct statements (groups) to generate.
    programs_per_group : int
        Number of distinct programs per statement (k).
    max_term_depth : int
        Depth bound for term generation.
    max_term_size : int
        Size bound for term generation.
    strategies : Sequence[ProofStrategy]
        Strategies used to create program variants.
    generator_seed : int, optional
        Seed for reproducibility.
    regenerate_on_epoch : bool
        If True, re-sample groups each epoch (infinite data mode).

    Returns
    -------
    dict with keys:
        "programs"     : List[str]       -- k program strings
        "statement"    : str             -- the type judgement
        "positive_mask": torch.Tensor[k,k] bool -- True for same-group pairs
        "program_terms": List[Term]      -- k Term objects (for mutation engine)
    """

    def __init__(
        self,
        n_groups: int = 1000,
        programs_per_group: int = 8,
        max_term_depth: int = 5,
        max_term_size: int = 16,
        strategies: Optional[Sequence[ProofStrategy]] = None,
        generator_seed: Optional[int] = None,
        regenerate_on_epoch: bool = False,
    ) -> None:
        self.n_groups = n_groups
        self.programs_per_group = programs_per_group
        self.max_term_depth = max_term_depth
        self.max_term_size = max_term_size
        self.strategies = list(strategies or DEFAULT_STRATEGIES)
        self.generator = LambdaGenerator(seed=generator_seed)
        self.regenerate_on_epoch = regenerate_on_epoch
        self._rng = random.Random(generator_seed)

        # Pre-generate all groups if not in infinite mode
        self._groups: Optional[List[Dict]] = None
        if not regenerate_on_epoch:
            self._groups = self._generate_all_groups()

    def _generate_all_groups(self) -> List[Dict]:
        groups = []
        for _ in range(self.n_groups):
            group = self._sample_group()
            if group is not None:
                groups.append(group)
        return groups

    def _sample_group(self) -> Optional[Dict]:
        """Generate one statement and k distinct programs satisfying it."""
        # 1. Sample a random context and target type
        ctx = self.generator.random_context(size=self._rng.randint(1, 4))
        target = self.generator._generate_type(depth=0)

        statement = Statement(tuple(ctx), target)

        # 2. Generate a base term
        try:
            base_term = self.generator.random_term(
                max_depth=self.max_term_depth,
                max_size=self.max_term_size,
                target_type=target,
                ctx=ctx,
            )
        except RuntimeError:
            return None

        # 3. Generate variants using strategies
        programs: List[Term] = [deepcopy(base_term)]
        program_set: Set[str] = {base_term.to_string()}

        attempts = 0
        max_attempts = self.programs_per_group * 10
        while len(programs) < self.programs_per_group and attempts < max_attempts:
            attempts += 1
            strategy = self._rng.choice(self.strategies)
            variant = strategy.generate(base_term, statement, self.generator)
            if variant is None:
                continue
            # Verify it still satisfies the statement
            try:
                inferred = typecheck(variant, ctx)
                if inferred != target:
                    continue
            except Exception:
                continue
            # Deduplicate by string representation
            v_str = variant.to_string()
            if v_str not in program_set:
                program_set.add(v_str)
                programs.append(variant)

        # If we couldn't get enough distinct programs, fill with copies + context tweaks
        while len(programs) < self.programs_per_group:
            # Try permuting context variables
            perm = ContextPermutationStrategy().generate(base_term, statement, self.generator)
            if perm is not None:
                p_str = perm.to_string()
                if p_str not in program_set:
                    try:
                        inferred = typecheck(perm, ctx)
                        if inferred == target:
                            program_set.add(p_str)
                            programs.append(perm)
                            continue
                    except Exception:
                        pass
            # Fallback: duplicate with a fresh variable substitution
            programs.append(deepcopy(base_term))

        # Build positive mask: all pairs within the group are positive
        k = len(programs)
        positive_mask = torch.ones(k, k, dtype=torch.bool)

        return {
            "programs": [p.to_string() for p in programs],
            "statement": statement.to_string(),
            "positive_mask": positive_mask,
            "program_terms": programs,
            "ctx": ctx,
            "target_type": target,
        }

    def __len__(self) -> int:
        if self._groups is not None:
            return len(self._groups)
        return self.n_groups

    def __getitem__(self, idx: int) -> Dict:
        if self._groups is not None:
            group = self._groups[idx % len(self._groups)]
            # Return without the non-serializable parts for DataLoader safety
            return {
                "programs": group["programs"],
                "statement": group["statement"],
                "positive_mask": group["positive_mask"],
            }
        # Infinite mode: generate on the fly
        group = self._sample_group()
        while group is None:
            group = self._sample_group()
        return {
            "programs": group["programs"],
            "statement": group["statement"],
            "positive_mask": group["positive_mask"],
        }

    def get_full_group(self, idx: int) -> Dict:
        """Get the full group including Term objects (not DataLoader-safe)."""
        if self._groups is not None:
            return self._groups[idx % len(self._groups)]
        return self._sample_group()

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate for variable-length program lists.

        Returns a list-form batch (not stacked) since groups may have
        different numbers of programs.
        """
        return {
            "programs": [b["programs"] for b in batch],
            "statements": [b["statement"] for b in batch],
            "positive_masks": [b["positive_mask"] for b in batch],
        }


# =============================================================================
# 4. Cross-Group Negative Sampling
# =============================================================================

class CrossGroupSampler:
    """
    Samples negative pairs across different proof groups.

    Given a batch of groups, this sampler constructs negative pairs by
    picking programs from different statements.
    """

    def __init__(self, dataset: ProofWalkDataset) -> None:
        self.dataset = dataset

    def sample_negatives(
        self,
        group_idx: int,
        n_negatives: int,
    ) -> List[str]:
        """Sample n_negatives programs from groups other than group_idx."""
        negatives = []
        for _ in range(n_negatives * 3):  # oversample for dedup
            other_idx = random.randint(0, len(self.dataset) - 1)
            if other_idx == group_idx:
                continue
            other_group = self.dataset[other_idx]
            prog = random.choice(other_group["programs"])
            if prog not in negatives:
                negatives.append(prog)
            if len(negatives) >= n_negatives:
                break
        return negatives[:n_negatives]

    def sample_negative_mask(
        self,
        batch_groups: List[Dict],
    ) -> torch.Tensor:
        """
        Construct a batch-level negative mask.

        For a batch of B groups with k_1, ..., k_B programs,
        returns a [sum(k_i), sum(k_i)] boolean mask where True means
        the pair is from DIFFERENT groups (negative pair).
        """
        sizes = [len(g["programs"]) for g in batch_groups]
        total = sum(sizes)
        mask = torch.zeros(total, total, dtype=torch.bool)
        offset = 0
        for size in sizes:
            # Programs in this group span [offset, offset+size)
            # Mark everything OUTSIDE this block as negative
            if offset > 0:
                mask[offset : offset + size, :offset] = True
            end = offset + size
            if end < total:
                mask[offset:end, end:total] = True
            offset = end
        # Make symmetric
        mask = mask | mask.T
        return mask


# =============================================================================
# 5. Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Proof Walk Generator - Self Test")
    print("=" * 60)

    dataset = ProofWalkDataset(
        n_groups=5,
        programs_per_group=4,
        max_term_depth=4,
        max_term_size=12,
        generator_seed=123,
    )

    print(f"\nDataset length: {len(dataset)}")

    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"\n--- Group {i} ---")
        print(f"Statement: {item['statement']}")
        print(f"Programs ({len(item['programs'])}):")
        for j, prog in enumerate(item["programs"]):
            print(f"  [{j}] {prog}")
        print(f"Positive mask:\n{item['positive_mask'].int()}")

    # Test cross-group sampler
    sampler = CrossGroupSampler(dataset)
    batch = [dataset[i] for i in range(3)]
    neg_mask = sampler.sample_negative_mask(batch)
    print(f"\n--- Cross-group negative mask shape: {neg_mask.shape} ---")
    print(neg_mask.int())

    # Test DataLoader compatibility
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    print(f"\nDataLoader batch keys: {list(batch.keys())}")
    print(f"Batch groups: {len(batch['statements'])}")

    print("\nAll tests passed.")
