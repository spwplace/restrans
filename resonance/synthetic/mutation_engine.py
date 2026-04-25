"""
Semantic-Preserving Mutation Engine
====================================

Implements bisimilar program mutations for the Resonance Transformer:

> "regular semantic-preserving mutation of programs probably good in midtraining,
> to learn higher structures ('shortcuts'), can vary the number of mutations
> applied over training eg later in training doing updates on a program that
> has accumulated 10000 edits (still bisimilar)"

The core insight: two programs are bisimilar if they have the same normal form.
By applying semantic-preserving mutations during training, the model learns
that syntactic distance is irrelevant to semantic identity -- it discovers
"shortcuts" in the space of programs.

Each mutation operation preserves both:
  1. Static semantics (typing)
  2. Dynamic semantics (beta-eta normal form)

The mutation engine tracks edit history per program, enabling training on
programs that have accumulated thousands of edits while remaining bisimilar
to their origin.

Operations
----------
  β-expansion      : e ~~> (λx. e[x]) arg          (introduce redex)
  β-reduction      : (λx. body) arg ~~> body[x:=arg]  (eliminate redex)
  η-expansion      : f ~~> λx. f x                    (when f has arrow type)
  η-reduction      : λx. f x ~~> f                    (when x not free in f)
  let-intro        : e ~~> (λx. e') arg              (simulate let binding)
  let-elim         : (λx. e') arg ~~> e'[x:=arg]    (beta reduce)
  dead-code-intro  : e ~~> (λ_. e) ()               (wrap with unit)
  dead-code-elim   : (λ_. e) () ~~> e               (reduce unit redex)
  var-rename       : α-conversion (trivial with de Bruijn)
  identity-wrap    : e ~~> id e                      (apply identity function)
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .lambda_generator import (
    Abs,
    App,
    ArrowType,
    BaseType,
    Context,
    T_UNIT,
    Term,
    Var,
    shift,
    substitute,
    typecheck,
    is_well_typed,
    normal_form,
)


# =============================================================================
# 1. Mutation Record
# =============================================================================

@dataclass
class MutationRecord:
    """Record of a single mutation applied to a program."""

    operation: str
    before: str
    after: str
    preserved_nf: bool


@dataclass
class MutatedProgram:
    """
    A program with full mutation history.

    The origin is the original well-typed term. Each edit preserves the
    beta-eta normal form, so `current` and `origin` are bisimilar.
    """

    origin: Term
    current: Term
    history: List[MutationRecord] = field(default_factory=list)
    ctx: Context = field(default_factory=list)
    target_type: Optional[object] = None

    @property
    def edit_count(self) -> int:
        return len(self.history)

    @property
    def current_str(self) -> str:
        return self.current.to_string()

    @property
    def origin_str(self) -> str:
        return self.origin.to_string()

    def verify_bisimilarity(self) -> bool:
        """Check that origin and current have the same normal form."""
        try:
            nf_origin = normal_form(self.origin)
            nf_current = normal_form(self.current)
            return nf_origin == nf_current
        except Exception:
            return False


# =============================================================================
# 2. Mutation Operations
# =============================================================================

MutationFn = Callable[[Term, Context, random.Random], Optional[Term]]


def _beta_expand(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    β-expansion: e ~~> (λx:T. e') arg

    Strategy: pick a subterm and wrap it in a redex.
    We pick a random position and wrap with an identity abstraction.
    """
    positions = _collect_positions(term)
    if not positions:
        return None
    pos = rng.choice(positions)
    subterm = _get_at_position(term, pos)
    if subterm is None:
        return None

    # Wrap the subterm: (λx:ty. x) subterm  -- this is a trivial beta-expansion
    # But we want something more interesting: pick a type and create a binder
    # For simplicity, use the type of the subterm if we can infer it
    try:
        subty = typecheck(subterm, _ctx_at_position(ctx, term, pos))
    except Exception:
        subty = T_UNIT  # fallback

    id_abs = Abs(subty, Var(0))
    wrapped = App(id_abs, subterm)
    new_term = _set_at_position(term, pos, wrapped)
    return new_term


def _beta_reduce(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    β-reduction: find a redex (λx. body) arg and reduce it.
    """
    redexes = _find_redexes(term)
    if not redexes:
        return None
    pos = rng.choice(redexes)
    return _beta_reduce_at(term, pos)


def _eta_expand(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    η-expansion: f ~~> λx. (shift(f, 1)) x

    Only valid when f has arrow type.
    """
    try:
        ty = typecheck(term, ctx)
    except Exception:
        return None
    if not isinstance(ty, ArrowType):
        return None
    # λx:ty.arg. (shift(term,1) x)
    shifted = shift(term, 1, 0)
    body = App(shifted, Var(0))
    return Abs(ty.arg, body)


def _eta_reduce(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    η-reduction: λx. f x ~~> f  when x not free in f.

    This is exactly the condition: in λx. body, if body = App(f, Var(0))
    and Var(0) is not free in f (after shifting f down by 1).
    """
    if isinstance(term, Abs):
        body = term.body
        if isinstance(body, App) and isinstance(body.arg, Var) and body.arg.index == 0:
            # Check if Var(0) is free in body.func (at depth 1)
            # body.func uses indices >= 1 for free vars (since depth=1)
            # We need Var(0) not to appear in body.func
            free = body.func.free_vars(1)
            if 0 not in free:
                # f is body.func, shift down by 1
                return shift(body.func, -1, 0)
    return None


def _let_intro(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    Let-binding introduction: e ~~> (λx:T. e[x/0]) subterm

    Pick a subterm, abstract over it, and apply the abstraction to it.
    This creates a let-like binding.
    """
    positions = _collect_positions(term)
    if not positions:
        return None
    # Don't pick the root -- we need a proper subterm
    valid_positions = [p for p in positions if p]
    if not valid_positions:
        return None
    pos = rng.choice(valid_positions)
    subterm = _get_at_position(term, pos)
    if subterm is None:
        return None
    try:
        subty = typecheck(subterm, _ctx_at_position(ctx, term, pos))
    except Exception:
        return None

    # Replace subterm with Var(0) (at the new depth after abstraction)
    # But we need to shift the rest of the term up by 1 first
    # Then replace the subterm position with Var(0)
    # Simpler approach: wrap the entire term at the root
    # Actually, let's do: wrap at the parent of the chosen subterm
    if len(pos) == 1:
        # Subterm is direct child of root
        parent = term
        child_idx = pos[0]
    else:
        parent = _get_at_position(term, pos[:-1])
        child_idx = pos[-1]

    if parent is None or not isinstance(parent, (App, Abs)):
        return None

    # New abstraction body: replace the subterm with Var(0) in the parent
    if isinstance(parent, App):
        if child_idx == 0:
            new_body = App(Var(0), shift(parent.arg, 1, 0))
        else:
            new_body = App(shift(parent.func, 1, 0), Var(0))
    elif isinstance(parent, Abs):
        new_body = Abs(parent.ty, substitute(parent.body, Var(0), 0))
        # This gets messy with de Bruijn -- simpler to just do a root-level let
        return None
    else:
        return None

    # Wrap: (λx:subty. new_body) (shift subterm up)
    shifted_sub = shift(subterm, 1, 0)
    wrapped = App(Abs(subty, new_body), shifted_sub)

    # Replace parent with wrapped in the root term
    if len(pos) == 1:
        return wrapped
    return _set_at_position(term, pos[:-1], wrapped)


def _dead_code_intro(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    Dead code introduction: e ~~> (λ_:Unit. e) ().

    Since we don't have a unit literal, we simulate it with a variable
    of unit type if available in context, or use a trivial identity.
    """
    # Check if there's a unit variable in context
    unit_vars = [i for i, ty in enumerate(ctx) if ty == T_UNIT]
    if unit_vars:
        unit_arg = Var(rng.choice(unit_vars))
    else:
        # No unit in context -- create a trivial identity application
        # (λx:Int. x) is semantically () for Int? No, not quite.
        # Better: just wrap with a dummy abstraction that ignores its arg
        # We'll need to invent a unit-like term. Use a fresh base variable.
        # Fallback: just identity-wrap
        return _identity_wrap(term, ctx, rng)

    wrapped = Abs(T_UNIT, shift(term, 1, 0))
    return App(wrapped, unit_arg)


def _dead_code_elim(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    Dead code elimination: (λ_:Unit. e) arg ~~> e

    This is just a specific form of beta reduction where the bound variable
    is not used in the body.
    """
    # Find redexes where the bound variable doesn't appear free in the body
    redexes = _find_redexes(term)
    for pos in redexes:
        redex = _get_at_position(term, pos)
        if redex is None or not isinstance(redex, App) or not isinstance(redex.func, Abs):
            continue
        abs_term = redex.func
        body = abs_term.body
        # Check if Var(0) is free in body (at depth 1)
        free = body.free_vars(1)
        if 0 not in free:
            # Safe to reduce
            result = substitute(body, shift(redex.arg, 1, 0), 0)
            result = shift(result, -1, 0)
            if len(pos) == 0:
                return result
            return _set_at_position(term, pos, result)
    return None


def _identity_wrap(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """Wrap term with identity: e ~~> (λx:T. x) e"""
    try:
        ty = typecheck(term, ctx)
    except Exception:
        return None
    id_fn = Abs(ty, Var(0))
    return App(id_fn, term)


def _commute_app(term: Term, ctx: Context, rng: random.Random) -> Optional[Term]:
    """
    Commutativity of certain applications -- limited to specific patterns.
    (λf. λg. λx. f (g x)) is function composition -- we can introduce or
    eliminate composition patterns.
    """
    # For now, this is a no-op placeholder
    return None


# =============================================================================
# 3. BisimilarMutation Engine
# =============================================================================

MUTATION_REGISTRY: Dict[str, MutationFn] = {
    "beta_expand": _beta_expand,
    "beta_reduce": _beta_reduce,
    "eta_expand": _eta_expand,
    "eta_reduce": _eta_reduce,
    "let_intro": _let_intro,
    "dead_code_intro": _dead_code_intro,
    "dead_code_elim": _dead_code_elim,
    "identity_wrap": _identity_wrap,
}


class BisimilarMutation:
    """
    Engine for applying semantic-preserving mutations to lambda calculus terms.

    All mutations preserve the beta-eta normal form, ensuring that the mutated
    program is bisimilar to the original. The engine tracks edit history and
    supports training schedules that ramp up mutation count over epochs.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible mutations.
    verify : bool
        If True, verify bisimilarity after each mutation.
    max_retries : int
        Maximum attempts to find a valid mutation before giving up.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        verify: bool = True,
        max_retries: int = 10,
    ) -> None:
        self.rng = random.Random(seed)
        self.verify = verify
        self.max_retries = max_retries

    def mutate(
        self,
        program: MutatedProgram,
        n_mutations: int = 1,
    ) -> MutatedProgram:
        """
        Apply n_mutations random semantic-preserving edits to a program.

        Returns a new MutatedProgram with updated history.
        """
        current = copy.deepcopy(program.current)
        ctx = program.ctx
        new_history = list(program.history)

        for _ in range(n_mutations):
            mutated = self._try_mutate_once(current, ctx)
            if mutated is not None:
                record = MutationRecord(
                    operation=mutated[1],
                    before=current.to_string(),
                    after=mutated[0].to_string(),
                    preserved_nf=True,  # verified below
                )
                if self.verify:
                    record.preserved_nf = self._verify_nf(
                        program.origin, mutated[0], ctx
                    )
                current = mutated[0]
                new_history.append(record)

        return MutatedProgram(
            origin=program.origin,
            current=current,
            history=new_history,
            ctx=ctx,
            target_type=program.target_type,
        )

    def mutate_program_str(
        self,
        program_str: str,
        n_mutations: int = 1,
        ctx: Optional[Context] = None,
    ) -> Tuple[str, List[MutationRecord]]:
        """
        Parse a program string, apply mutations, return the new string + history.

        Note: This is a simplified interface that requires the program to be
        passed as a Term object for full functionality. For string-based
        usage, the caller should manage parsing externally.
        """
        # This is a stub -- the real workflow uses MutatedProgram objects
        # which carry Term references. String parsing is not implemented here
        # because it requires a full parser.
        raise NotImplementedError(
            "mutate_program_str requires a parser. "
            "Use MutatedProgram objects with Term references instead."
        )

    def _try_mutate_once(
        self, term: Term, ctx: Context
    ) -> Optional[Tuple[Term, str]]:
        """Attempt one random mutation. Returns (new_term, operation_name)."""
        ops = list(MUTATION_REGISTRY.items())
        self.rng.shuffle(ops)
        for op_name, op_fn in ops:
            for _ in range(self.max_retries // len(ops) + 1):
                candidate = op_fn(term, ctx, self.rng)
                if candidate is not None:
                    # Verify well-typed
                    try:
                        typecheck(candidate, ctx)
                        return candidate, op_name
                    except Exception:
                        continue
        return None

    def _verify_nf(self, origin: Term, current: Term, ctx: Context) -> bool:
        """Check that origin and current have the same normal form."""
        try:
            nf1 = normal_form(origin)
            nf2 = normal_form(current)
            return nf1 == nf2
        except Exception:
            return False

    def mutate_term(
        self,
        term: Term,
        ctx: Context,
        n_mutations: int = 1,
    ) -> Optional[Term]:
        """
        Directly mutate a term without tracking history.

        Returns the mutated term or None if all attempts fail.
        """
        current = term
        for _ in range(n_mutations):
            result = self._try_mutate_once(current, ctx)
            if result is None:
                return current if _ > 0 else None
            current = result[0]
        return current


# =============================================================================
# 4. Training Schedule
# =============================================================================


def training_schedule(
    epoch: int,
    max_epochs: int,
    schedule: str = "linear",
    min_mutations: int = 0,
    max_mutations: int = 10000,
) -> int:
    """
    Compute the recommended number of mutations for a given training epoch.

    As training progresses, programs accumulate more edits. The model must
    learn to recognize semantic identity even across programs with thousands
    of accumulated syntactic differences -- this forces it to discover
    higher-level structural invariants ("shortcuts").

    Parameters
    ----------
    epoch : int
        Current epoch (0-indexed).
    max_epochs : int
        Total number of epochs.
    schedule : str
        "linear", "quadratic", "exponential", or "sigmoid".
    min_mutations : int
        Minimum mutations at epoch 0.
    max_mutations : int
        Maximum mutations at final epoch.

    Returns
    -------
    int
        Recommended number of mutations to apply this epoch.
    """
    if max_epochs <= 1:
        return max_mutations

    progress = epoch / (max_epochs - 1)

    if schedule == "linear":
        ratio = progress
    elif schedule == "quadratic":
        ratio = progress ** 2
    elif schedule == "exponential":
        ratio = (2 ** (progress * 10) - 1) / (2 ** 10 - 1)
        ratio = min(1.0, ratio)
    elif schedule == "sigmoid":
        import math
        ratio = 1 / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return int(min_mutations + ratio * (max_mutations - min_mutations))


# =============================================================================
# 5. Position Utilities (for mutation targeting)
# =============================================================================

Position = Tuple[int, ...]


def _collect_positions(term: Term) -> List[Position]:
    """Collect all positions (as index paths) in a term."""
    positions: List[Position] = [()]
    if isinstance(term, App):
        for p in _collect_positions(term.func):
            positions.append((0,) + p)
        for p in _collect_positions(term.arg):
            positions.append((1,) + p)
    elif isinstance(term, Abs):
        for p in _collect_positions(term.body):
            positions.append((0,) + p)
    return positions


def _get_at_position(term: Term, pos: Position) -> Optional[Term]:
    """Get the subterm at a given position."""
    if not pos:
        return term
    idx, *rest = pos
    if isinstance(term, App):
        child = term.func if idx == 0 else term.arg
        return _get_at_position(child, tuple(rest))
    if isinstance(term, Abs) and idx == 0:
        return _get_at_position(term.body, tuple(rest))
    return None


def _set_at_position(term: Term, pos: Position, new_sub: Term) -> Optional[Term]:
    """Replace the subterm at a given position with new_sub."""
    if not pos:
        return new_sub
    idx, *rest = pos
    if isinstance(term, App):
        if idx == 0:
            new_func = _set_at_position(term.func, tuple(rest), new_sub)
            if new_func is None:
                return None
            return App(new_func, term.arg)
        else:
            new_arg = _set_at_position(term.arg, tuple(rest), new_sub)
            if new_arg is None:
                return None
            return App(term.func, new_arg)
    if isinstance(term, Abs) and idx == 0:
        new_body = _set_at_position(term.body, tuple(rest), new_sub)
        if new_body is None:
            return None
        return Abs(term.ty, new_body)
    return None


def _find_redexes(term: Term) -> List[Position]:
    """Find all beta-redex positions in a term."""
    redexes: List[Position] = []
    if isinstance(term, App) and isinstance(term.func, Abs):
        redexes.append(())
    if isinstance(term, App):
        for p in _find_redexes(term.func):
            redexes.append((0,) + p)
        for p in _find_redexes(term.arg):
            redexes.append((1,) + p)
    if isinstance(term, Abs):
        for p in _find_redexes(term.body):
            redexes.append((0,) + p)
    return redexes


def _beta_reduce_at(term: Term, pos: Position) -> Optional[Term]:
    """Beta reduce the redex at the given position."""
    sub = _get_at_position(term, pos)
    if sub is None or not isinstance(sub, App) or not isinstance(sub.func, Abs):
        return None
    abs_term = sub.func
    arg = sub.arg
    # Substitute
    body = abs_term.body
    shifted_arg = shift(arg, 1, 0)
    result = substitute(body, shifted_arg, 0)
    result = shift(result, -1, 0)
    if not pos:
        return result
    return _set_at_position(term, pos, result)


def _ctx_at_position(ctx: Context, term: Term, pos: Position) -> Context:
    """
    Compute the typing context at a given position.

    As we descend into abstractions, we extend the context.
    """
    current_ctx = list(ctx)
    current = term
    for idx in pos:
        if isinstance(current, Abs):
            current_ctx.insert(0, current.ty)
            current = current.body
        elif isinstance(current, App):
            current = current.func if idx == 0 else current.arg
        else:
            break
    return current_ctx


# =============================================================================
# 6. Self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mutation Engine - Self Test")
    print("=" * 60)

    from .lambda_generator import LambdaGenerator, T_INT, ArrowType

    gen = LambdaGenerator(seed=42)
    engine = BisimilarMutation(seed=42, verify=True)

    # Generate a base term
    base = gen.random_term(max_depth=4, max_size=10, target_type=T_INT >> T_INT)
    ctx = []
    print(f"\nBase term: {base.to_string()}")
    print(f"Type: {typecheck(base, ctx)}")

    # Create a mutated program
    prog = MutatedProgram(origin=base, current=base, ctx=ctx, target_type=T_INT >> T_INT)

    # Apply mutations
    for n in [1, 5, 10, 50]:
        mutated = engine.mutate(prog, n_mutations=n)
        print(f"\n--- After {mutated.edit_count} mutations ---")
        print(f"Current: {mutated.current.to_string()}")
        print(f"Size: {mutated.current.size()} (origin: {mutated.origin.size()})")
        bisim = mutated.verify_bisimilarity()
        print(f"Bisimilar: {bisim}")
        assert bisim, "Bisimilarity violated!"

    # Test training schedule
    print("\n--- Training Schedule (linear, max=10000) ---")
    for epoch in [0, 10, 50, 99]:
        n_mut = training_schedule(epoch, 100, schedule="linear", max_mutations=10000)
        print(f"  Epoch {epoch}: {n_mut} mutations")

    print("\n--- Training Schedule (quadratic, max=10000) ---")
    for epoch in [0, 10, 50, 99]:
        n_mut = training_schedule(epoch, 100, schedule="quadratic", max_mutations=10000)
        print(f"  Epoch {epoch}: {n_mut} mutations")

    print("\n--- Training Schedule (sigmoid, max=10000) ---")
    for epoch in [0, 10, 50, 99]:
        n_mut = training_schedule(epoch, 100, schedule="sigmoid", max_mutations=10000)
        print(f"  Epoch {epoch}: {n_mut} mutations")

    # Test individual mutations
    print("\n--- Individual mutation tests ---")
    test_term = App(Abs(T_INT, Var(0)), Var(0))  # (λx.x) y
    print(f"Test term: {test_term.to_string()}")

    # Beta reduce
    reduced = _beta_reduce(test_term, [], random.Random(1))
    if reduced:
        print(f"Beta reduced: {reduced.to_string()}")

    # Eta expand
    id_fn = Abs(T_INT, Var(0))
    expanded = _eta_expand(id_fn, [], random.Random(1))
    if expanded:
        print(f"Eta expanded: {expanded.to_string()}")

    # Identity wrap
    wrapped = _identity_wrap(Var(0), [T_INT], random.Random(1))
    if wrapped:
        print(f"Identity wrapped: {wrapped.to_string()}")

    print("\nAll tests passed.")
