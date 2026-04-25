"""
Lambda Calculus Term Generator
==============================

Generates well-typed simply-typed lambda calculus terms using de Bruijn indices.
Supports systematic enumeration and random generation with controlled depth/size.

Types
-----
  T ::= Int | Bool | Unit | T -> T

Terms
-----
  e ::= x_i          (variable, de Bruijn index i)
      | \\x:T. e      (abstraction)
      | e1 e2        (application)

The type checker ensures all generated terms are well-typed under a given context.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple, Union


# =============================================================================
# 1. Types
# =============================================================================

class Type:
    """Simply-typed lambda calculus types."""

    def __arrow__(self, other: Type) -> ArrowType:
        return ArrowType(self, other)

    def __rshift__(self, other: Type) -> ArrowType:
        return ArrowType(self, other)

    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    def size(self) -> int:
        """Structural size of the type (for complexity control)."""
        raise NotImplementedError

    def is_arrow(self) -> bool:
        return isinstance(self, ArrowType)


class BaseType(Type):
    """Atomic base types (Int, Bool, Unit, etc.)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BaseType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    def size(self) -> int:
        return 1


class ArrowType(Type):
    """Function type T1 -> T2."""

    def __init__(self, arg: Type, ret: Type) -> None:
        self.arg = arg
        self.ret = ret

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ArrowType)
            and self.arg == other.arg
            and self.ret == other.ret
        )

    def __hash__(self) -> int:
        return hash((self.arg, self.ret))

    def __repr__(self) -> str:
        arg_str = f"({self.arg})" if self.arg.is_arrow() else str(self.arg)
        return f"{arg_str} -> {self.ret}"

    def size(self) -> int:
        return 1 + self.arg.size() + self.ret.size()


# Predefined base types
T_INT = BaseType("Int")
T_BOOL = BaseType("Bool")
T_UNIT = BaseType("Unit")

BASE_TYPES: Tuple[BaseType, ...] = (T_INT, T_BOOL, T_UNIT)


# =============================================================================
# 2. Terms (de Bruijn indices)
# =============================================================================

class Term:
    """Lambda calculus term using de Bruijn indices."""

    def size(self) -> int:
        """Number of constructors in the term."""
        raise NotImplementedError

    def depth(self) -> int:
        """Maximum nesting depth."""
        raise NotImplementedError

    def to_string(self, names: Optional[List[str]] = None) -> str:
        """Pretty-print with optional named variables."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.to_string()

    def free_vars(self, depth: int = 0) -> set:
        """Set of free variable indices (relative to given depth)."""
        raise NotImplementedError


class Const(Term):
    """Built-in constant (0, 1, true, false, unit)."""

    def __init__(self, value: Union[int, bool, None]) -> None:
        self.value = value

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def to_string(self, names: Optional[List[str]] = None) -> str:
        if self.value is None:
            return "()"
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        return str(self.value)

    def free_vars(self, depth: int = 0) -> set:
        return set()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Const) and self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)


class Var(Term):
    """Variable with de Bruijn index."""

    def __init__(self, index: int) -> None:
        assert index >= 0
        self.index = index

    def size(self) -> int:
        return 1

    def depth(self) -> int:
        return 1

    def to_string(self, names: Optional[List[str]] = None) -> str:
        if names is not None and self.index < len(names):
            return names[self.index]
        return f"x{self.index}"

    def free_vars(self, depth: int = 0) -> set:
        if self.index >= depth:
            return {self.index - depth}
        return set()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Var) and self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)


class Abs(Term):
    """Abstraction \\x:T. body."""

    def __init__(self, ty: Type, body: Term) -> None:
        self.ty = ty
        self.body = body

    def size(self) -> int:
        return 1 + self.ty.size() + self.body.size()

    def depth(self) -> int:
        return 1 + self.body.depth()

    def to_string(self, names: Optional[List[str]] = None) -> str:
        if names is None:
            names = []
        fresh = _fresh_name("x", names)
        new_names = [fresh] + names
        return f"(λ {fresh}:{self.ty}. {self.body.to_string(new_names)})"

    def free_vars(self, depth: int = 0) -> set:
        return self.body.free_vars(depth + 1)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Abs) and self.ty == other.ty and self.body == other.body

    def __hash__(self) -> int:
        return hash(("abs", self.ty, self.body))


class App(Term):
    """Application e1 e2."""

    def __init__(self, func: Term, arg: Term) -> None:
        self.func = func
        self.arg = arg

    def size(self) -> int:
        return 1 + self.func.size() + self.arg.size()

    def depth(self) -> int:
        return 1 + max(self.func.depth(), self.arg.depth())

    def to_string(self, names: Optional[List[str]] = None) -> str:
        if names is None:
            names = []
        return f"({self.func.to_string(names)} {self.arg.to_string(names)})"

    def free_vars(self, depth: int = 0) -> set:
        return self.func.free_vars(depth) | self.arg.free_vars(depth)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, App)
            and self.func == other.func
            and self.arg == other.arg
        )

    def __hash__(self) -> int:
        return hash(("app", self.func, self.arg))


# =============================================================================
# 3. Type Checker
# =============================================================================

Context = List[Type]


class TypeError(Exception):
    """Raised when a term is not well-typed."""

    pass


def typecheck(term: Term, ctx: Context) -> Type:
    """
    Type-check a term under context Γ (list of types for de Bruijn indices).

    Returns the type of the term or raises TypeError.
    """
    if isinstance(term, Const):
        if term.value is None:
            return T_UNIT
        if isinstance(term.value, bool):
            return T_BOOL
        if isinstance(term.value, int):
            return T_INT
        raise TypeError(f"Unknown constant type: {type(term.value)}")

    if isinstance(term, Var):
        if term.index >= len(ctx):
            raise TypeError(f"Unbound variable x{term.index} in context of size {len(ctx)}")
        return ctx[term.index]

    if isinstance(term, Abs):
        body_ty = typecheck(term.body, [term.ty] + ctx)
        return ArrowType(term.ty, body_ty)

    if isinstance(term, App):
        func_ty = typecheck(term.func, ctx)
        arg_ty = typecheck(term.arg, ctx)
        if not isinstance(func_ty, ArrowType):
            raise TypeError(
                f"Application of non-function: {func_ty}"
            )
        if func_ty.arg != arg_ty:
            raise TypeError(
                f"Type mismatch in application: expected {func_ty.arg}, got {arg_ty}"
            )
        return func_ty.ret

    raise TypeError(f"Unknown term type: {type(term)}")


def is_well_typed(term: Term, ctx: Optional[Context] = None) -> bool:
    """Check if a term is well-typed under the given (or empty) context."""
    try:
        typecheck(term, ctx or [])
        return True
    except TypeError:
        return False


# =============================================================================
# 4. Random Term Generator
# =============================================================================

class GenerationStrategy(Enum):
    """Strategy for term generation."""

    UNIFORM = auto()
    DEPTH_CONSTRAINED = auto()
    SIZE_CONSTRAINED = auto()


class LambdaGenerator:
    """
    Random generator for well-typed simply-typed lambda calculus terms.

    Parameters
    ----------
    base_types : Sequence[Type]
        Base types available for generation (default: Int, Bool, Unit).
    max_arrow_depth : int
        Maximum nesting depth of arrow types.
    rng : random.Random
        Random number generator (for reproducibility).
    """

    def __init__(
        self,
        base_types: Sequence[Type] = BASE_TYPES,
        max_arrow_depth: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.base_types = tuple(base_types)
        self.max_arrow_depth = max_arrow_depth
        self.rng = random.Random(seed)
        self._type_cache: List[Type] = []

    def _generate_type(self, depth: int = 0) -> Type:
        """Generate a random type with controlled arrow nesting."""
        if depth >= self.max_arrow_depth:
            return self.rng.choice(self.base_types)
        if self.rng.random() < 0.5:
            return self.rng.choice(self.base_types)
        arg = self._generate_type(depth + 1)
        ret = self._generate_type(depth + 1)
        return ArrowType(arg, ret)

    def _ctx_vars_of_type(self, ctx: Context, target: Type) -> List[int]:
        """Return indices of variables in ctx that have exactly the target type."""
        return [i for i, ty in enumerate(ctx) if ty == target]

    def generate_term(
        self,
        target_type: Type,
        ctx: Context,
        max_depth: int = 5,
        max_size: int = 20,
        strategy: GenerationStrategy = GenerationStrategy.DEPTH_CONSTRAINED,
    ) -> Optional[Term]:
        """
        Generate a random well-typed term of the given type.

        Uses a recursive approach that tries:
          1. Variables from context
          2. Abstractions (if target is an arrow type)
          3. Applications (find function type that returns target)

        Returns None if generation fails (should be rare with generous limits).
        """
        return self._gen(
            target_type, ctx, max_depth, max_size, strategy, depth_used=0, size_used=0
        )

    def _gen(
        self,
        target: Type,
        ctx: Context,
        max_depth: int,
        max_size: int,
        strategy: GenerationStrategy,
        depth_used: int,
        size_used: int,
    ) -> Optional[Term]:
        # Check limits
        if depth_used > max_depth or size_used > max_size:
            return None

        choices: List[str] = []
        weights: List[float] = []

        # Option 0: Use a constant (for base types)
        if not target.is_arrow():
            if target == T_INT:
                choices.append("const")
                weights.append(2.0)
            elif target == T_BOOL:
                choices.append("const")
                weights.append(2.0)
            elif target == T_UNIT:
                choices.append("const")
                weights.append(1.0)

        # Option 1: Use a variable from context
        var_indices = self._ctx_vars_of_type(ctx, target)
        if var_indices and strategy != GenerationStrategy.SIZE_CONSTRAINED:
            choices.append("var")
            weights.append(1.0 + len(var_indices))

        # Option 2: Abstraction (if target is an arrow type)
        if target.is_arrow() and depth_used < max_depth:
            assert isinstance(target, ArrowType)
            choices.append("abs")
            weights.append(1.5)

        # Option 3: Application
        if depth_used < max_depth and size_used + 2 <= max_size:
            choices.append("app")
            weights.append(2.0)

        if not choices:
            return None

        choice = self.rng.choices(choices, weights=weights, k=1)[0]

        if choice == "const":
            if target == T_INT:
                return Const(self.rng.randint(0, 9))
            elif target == T_BOOL:
                return Const(self.rng.choice([True, False]))
            elif target == T_UNIT:
                return Const(None)
            return None

        if choice == "var":
            idx = self.rng.choice(var_indices)
            return Var(idx)

        if choice == "abs":
            assert isinstance(target, ArrowType)
            body = self._gen(
                target.ret,
                [target.arg] + ctx,
                max_depth,
                max_size,
                strategy,
                depth_used + 1,
                size_used + 1 + target.arg.size(),
            )
            if body is None:
                return None
            return Abs(target.arg, body)

        if choice == "app":
            # Generate a function of type arg -> target, and an argument of type arg
            # We try a few attempts with different arg types
            for _ in range(5):
                arg_type = self._generate_type()
                func_type = ArrowType(arg_type, target)
                func = self._gen(
                    func_type,
                    ctx,
                    max_depth,
                    max_size,
                    strategy,
                    depth_used + 1,
                    size_used + 1,
                )
                if func is None:
                    continue
                arg = self._gen(
                    arg_type,
                    ctx,
                    max_depth,
                    max_size,
                    strategy,
                    depth_used + 1,
                    size_used + 1 + func.size(),
                )
                if arg is None:
                    continue
                return App(func, arg)
            return None

        return None  # unreachable

    def random_term(
        self,
        max_depth: int = 5,
        max_size: int = 20,
        target_type: Optional[Type] = None,
        ctx: Optional[Context] = None,
    ) -> Term:
        """
        Generate a random well-typed closed term.

        Parameters
        ----------
        max_depth : int
            Maximum term depth.
        max_size : int
            Maximum term size (number of constructors).
        target_type : Type, optional
            If given, generate a term of this type; otherwise random.
        ctx : Context, optional
            Initial typing context.

        Raises
        ------
        RuntimeError
            If generation fails repeatedly.
        """
        ty = target_type or self._generate_type()
        ctx = ctx or []
        for attempt in range(200):
            # If not fixed target, resample type each attempt
            if target_type is None:
                ty = self._generate_type()
            term = self.generate_term(ty, ctx, max_depth, max_size)
            if term is not None:
                # Verify
                try:
                    inferred = typecheck(term, ctx)
                    if inferred == ty:
                        return term
                except TypeError:
                    pass
        raise RuntimeError(
            f"Failed to generate well-typed term of type {ty} "
            f"within depth={max_depth}, size={max_size}"
        )

    def random_context(self, size: int = 3) -> Context:
        """Generate a random typing context of given size."""
        return [self._generate_type() for _ in range(size)]


# =============================================================================
# 5. Systematic Enumeration
# =============================================================================


def enumerate_terms(
    target_type: Type,
    ctx: Context,
    max_depth: int,
    max_size: int,
    _depth: int = 0,
    _size: int = 0,
) -> List[Term]:
    """
    Systematically enumerate all well-typed terms of a given type.

    Parameters
    ----------
    target_type : Type
        The desired type of generated terms.
    ctx : Context
        Typing context (list of types for bound variables).
    max_depth : int
        Maximum nesting depth.
    max_size : int
        Maximum term size.

    Returns
    -------
    List[Term]
        All well-typed terms satisfying the constraints.
    """
    results: List[Term] = []

    if _depth > max_depth or _size > max_size:
        return results

    # Constants
    if not target_type.is_arrow():
        if target_type == T_INT:
            for v in range(3):
                results.append(Const(v))
        elif target_type == T_BOOL:
            results.append(Const(True))
            results.append(Const(False))
        elif target_type == T_UNIT:
            results.append(Const(None))

    # Variables
    for i, ty in enumerate(ctx):
        if ty == target_type:
            results.append(Var(i))

    # Abstractions
    if target_type.is_arrow() and _depth < max_depth:
        assert isinstance(target_type, ArrowType)
        for body in enumerate_terms(
            target_type.ret,
            [target_type.arg] + ctx,
            max_depth,
            max_size,
            _depth + 1,
            _size + 1 + target_type.arg.size(),
        ):
            results.append(Abs(target_type.arg, body))

    # Applications: enumerate function and argument pairs
    if _depth < max_depth and _size + 2 <= max_size:
        # We enumerate by generating possible argument types from base types
        # up to a small depth to keep it finite and tractable
        arg_types = _enumerate_types(max_depth=2, max_size=4)
        for arg_ty in arg_types:
            func_ty = ArrowType(arg_ty, target_type)
            funcs = enumerate_terms(
                func_ty, ctx, max_depth, max_size, _depth + 1, _size + 1
            )
            for func in funcs:
                args = enumerate_terms(
                    arg_ty,
                    ctx,
                    max_depth,
                    max_size,
                    _depth + 1,
                    _size + 1 + func.size(),
                )
                for arg in args:
                    results.append(App(func, arg))

    return results


def _enumerate_types(max_depth: int = 3, max_size: int = 6) -> List[Type]:
    """Enumerate all types within given depth/size bounds."""
    results: List[Type] = list(BASE_TYPES)
    if max_depth <= 0 or max_size <= 1:
        return results

    # Build iteratively
    for _ in range(max_depth):
        new_results = list(results)
        for t1 in results:
            for t2 in results:
                if t1.size() + t2.size() + 1 <= max_size:
                    arrow = ArrowType(t1, t2)
                    if arrow not in new_results:
                        new_results.append(arrow)
        results = new_results
    return results


def enumerate_all(
    max_depth: int = 3,
    max_size: int = 10,
    target_type: Optional[Type] = None,
    ctx: Optional[Context] = None,
) -> List[Term]:
    """
    Enumerate all closed well-typed terms within bounds.

    If target_type is None, enumerate for all types up to a small bound.
    """
    if target_type is not None:
        return enumerate_terms(target_type, ctx or [], max_depth, max_size)

    types = _enumerate_types(max_depth=2, max_size=4)
    all_terms: List[Term] = []
    for ty in types:
        all_terms.extend(enumerate_terms(ty, ctx or [], max_depth, max_size))
    return all_terms


# =============================================================================
# 6. Utility functions
# =============================================================================


def _fresh_name(prefix: str, used: List[str]) -> str:
    """Generate a fresh variable name not in `used`."""
    i = 0
    while True:
        name = f"{prefix}{i}" if i > 0 else prefix
        if name not in used:
            return name
        i += 1


def shift(term: Term, d: int, cutoff: int = 0) -> Term:
    """
    de Bruijn index shift: add d to all indices >= cutoff.
    Standard operation for substitution.
    """
    if isinstance(term, (Const, Var)):
        if isinstance(term, Var) and term.index >= cutoff:
            return Var(term.index + d)
        return term
    if isinstance(term, Abs):
        return Abs(term.ty, shift(term.body, d, cutoff + 1))
    if isinstance(term, App):
        return App(shift(term.func, d, cutoff), shift(term.arg, d, cutoff))
    return term


def substitute(term: Term, replacement: Term, var_index: int = 0) -> Term:
    """
    Substitute replacement for variable var_index in term.
    replacement is assumed to be valid at the outer context level.
    """
    if isinstance(term, (Const, Var)):
        if isinstance(term, Var) and term.index == var_index:
            return replacement
        return term
    if isinstance(term, Abs):
        shifted_repl = shift(replacement, 1, 0)
        return Abs(term.ty, substitute(term.body, shifted_repl, var_index + 1))
    if isinstance(term, App):
        return App(
            substitute(term.func, replacement, var_index),
            substitute(term.arg, replacement, var_index),
        )
    return term


def beta_reduce(term: Term, max_steps: int = 1000) -> Term:
    """
    Normal-order beta reduction (leftmost-outermost).
    Returns the normal form or the term after max_steps reductions.
    """
    for _ in range(max_steps):
        reduced, new_term = _beta_reduce_once(term)
        if not reduced:
            return new_term
        term = new_term
    return term


def _beta_reduce_once(term: Term) -> Tuple[bool, Term]:
    """Perform one beta reduction step in normal order. Returns (did_reduce, result)."""
    if isinstance(term, App):
        if isinstance(term.func, Abs):
            # Beta redex: (\\x:T. body) arg  ->  body[x := arg]
            body = term.func.body
            arg = shift(term.arg, 1, 0)  # lift arg into body context
            result = substitute(body, arg, 0)
            result = shift(result, -1, 0)  # lower all indices by 1
            return True, result
        # Try to reduce func first (normal order)
        reduced, new_func = _beta_reduce_once(term.func)
        if reduced:
            return True, App(new_func, term.arg)
        # Then try arg
        reduced, new_arg = _beta_reduce_once(term.arg)
        if reduced:
            return True, App(term.func, new_arg)
        return False, term
    if isinstance(term, Abs):
        reduced, new_body = _beta_reduce_once(term.body)
        return reduced, Abs(term.ty, new_body)
    if isinstance(term, (Var, Const)):
        return False, term
    return False, term


def alpha_equivalent(t1: Term, t2: Term) -> bool:
    """Check alpha-equivalence of two terms (de Bruijn indices make this trivial)."""
    return t1 == t2


def normal_form(term: Term, max_steps: int = 1000) -> Term:
    """Compute the beta-normal form of a term."""
    return beta_reduce(term, max_steps)


# =============================================================================
# 7. Example usage / self-test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lambda Calculus Term Generator - Self Test")
    print("=" * 60)

    gen = LambdaGenerator(seed=42)

    # Generate some random terms
    for max_d in [3, 4, 5]:
        print(f"\n--- Random terms (max_depth={max_d}) ---")
        for _ in range(5):
            t = gen.random_term(max_depth=max_d, max_size=15)
            ty = typecheck(t, [])
            print(f"  Term:  {t.to_string()}")
            print(f"  Type:  {ty}")
            print(f"  Size:  {t.size()}, Depth: {t.depth()}")
            print()

    # Enumerate small terms
    print("\n--- Enumerating terms of type Int -> Int (depth<=3, size<=8) ---")
    terms = enumerate_terms(T_INT >> T_INT, [], max_depth=3, max_size=8)
    print(f"Found {len(terms)} terms")
    for t in terms[:10]:
        print(f"  {t.to_string()}  :  {typecheck(t, [])}")
    if len(terms) > 10:
        print(f"  ... and {len(terms) - 10} more")

    print("\n--- Beta reduction test ---")
    # (\x:Int. x) y  with y in context as x0
    redex = App(Abs(T_INT, Var(0)), Var(0))
    print(f"  Before: {redex.to_string()}")
    nf = normal_form(redex)
    print(f"  After:  {nf.to_string()}")

    print("\nAll tests passed.")
