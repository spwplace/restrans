"""Template-equivalence task with controlled lexical overlap.

The latent object is a small expression template.  Each example renders two
surface instantiations with separate vocabularies and asks whether they share
the same template shape.  This is meant to test structural matching without
entity-name overlap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


UNARY = ("not", "box", "lift", "mark")
BINARY = ("and", "or", "then", "because")
ATOM_MARK = "atom"


@dataclass(frozen=True)
class Template:
    op: str
    children: tuple["Template", ...] = ()

    def signature(self) -> str:
        if not self.children:
            return ATOM_MARK
        return f"{self.op}({','.join(child.signature() for child in self.children)})"


@dataclass(frozen=True)
class TemplateEquivalenceExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    same_template: bool
    left_signature: str
    right_signature: str


def generate_template(rng: random.Random, depth: int, p_leaf: float = 0.25) -> Template:
    if depth <= 0 or rng.random() < p_leaf:
        return Template(ATOM_MARK)
    if rng.random() < 0.35:
        return Template(rng.choice(UNARY), (generate_template(rng, depth - 1, p_leaf),))
    return Template(
        rng.choice(BINARY),
        (
            generate_template(rng, depth - 1, p_leaf),
            generate_template(rng, depth - 1, p_leaf),
        ),
    )


def template_bank(*, seed: int, n_templates: int, depth: int) -> list[Template]:
    rng = random.Random(seed)
    bank: list[Template] = []
    seen: set[str] = set()
    tries = 0
    while len(bank) < n_templates and tries < n_templates * 200:
        tries += 1
        template = generate_template(rng, depth=depth)
        sig = template.signature()
        if sig in seen or sig == ATOM_MARK:
            continue
        seen.add(sig)
        bank.append(template)
    if len(bank) < n_templates:
        raise RuntimeError("failed to generate enough distinct templates")
    return bank


def _fresh_words(rng: random.Random, prefix: str, n: int) -> list[str]:
    return [f"{prefix}{idx}_{rng.randrange(10_000, 99_999)}" for idx in range(n)]


def render_template(template: Template, rng: random.Random, side: str, atom_counter: list[int]) -> str:
    op_map = {op: f"{side}_{op}_{rng.randrange(100, 999)}" for op in (*UNARY, *BINARY)}
    atom_words = _fresh_words(rng, f"{side}_x", 64)

    def go(node: Template) -> str:
        if not node.children:
            word = atom_words[atom_counter[0] % len(atom_words)]
            atom_counter[0] += 1
            return word
        rendered = [go(child) for child in node.children]
        if len(rendered) == 1:
            return f"( {op_map[node.op]} {rendered[0]} )"
        return f"( {rendered[0]} {op_map[node.op]} {rendered[1]} )"

    return go(template)


def render_template_equivalence_example(
    *,
    seed: int,
    graph_id: str,
    templates: list[Template],
    positive: bool,
) -> TemplateEquivalenceExample:
    rng = random.Random(seed)
    left = rng.choice(templates)
    if positive:
        right = left
    else:
        candidates = [template for template in templates if template.signature() != left.signature()]
        right = rng.choice(candidates)

    left_text = render_template(left, rng, "left", [0])
    right_text = render_template(right, rng, "right", [0])
    prefix = f"Expression one: {left_text}. Expression two: {right_text}. Same template? Answer"
    answer = "yes" if positive else "no"
    return TemplateEquivalenceExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        same_template=positive,
        left_signature=left.signature(),
        right_signature=right.signature(),
    )


class TemplateEquivalenceDataset(Dataset):
    """Balanced yes/no latent-template equivalence examples."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        n_templates: int = 50,
        depth: int = 4,
    ) -> None:
        templates = template_bank(seed=seed + 50_000, n_templates=n_templates, depth=depth)
        self.examples = [
            render_template_equivalence_example(
                seed=seed + idx,
                graph_id=f"tpl{idx}",
                templates=templates,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TemplateEquivalenceExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[TemplateEquivalenceExample]) -> list[TemplateEquivalenceExample]:
    return batch
