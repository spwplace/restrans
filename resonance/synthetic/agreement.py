"""Subject-verb agreement examples with controlled attractors.

This is a local, reproducible Linzen-style generator.  It is not a replacement
for the original Linzen or BLiMP data; it gives the same probe harness a syntax
task whose dependency length and attractor count are controlled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from torch.utils.data import Dataset


SINGULAR_NOUNS = [
    "key",
    "letter",
    "doctor",
    "artist",
    "teacher",
    "student",
    "judge",
    "captain",
    "author",
    "robot",
]

PLURAL_NOUNS = [
    "keys",
    "letters",
    "doctors",
    "artists",
    "teachers",
    "students",
    "judges",
    "captains",
    "authors",
    "robots",
]

PREPOSITIONS = ["near", "beside", "behind", "with", "around", "above"]
MODIFIERS = ["old", "careful", "bright", "quiet", "strange", "local"]


@dataclass(frozen=True)
class AgreementExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    subject_number: str
    attractor_count: int
    candidate_verb: str
    correct: bool


def _noun(rng: random.Random, number: str) -> str:
    return rng.choice(SINGULAR_NOUNS if number == "singular" else PLURAL_NOUNS)


def render_agreement_example(
    *,
    seed: int,
    graph_id: str,
    max_attractors: int = 4,
    force_attractors: int | None = None,
    positive: bool = True,
) -> AgreementExample:
    rng = random.Random(seed)
    subject_number = rng.choice(["singular", "plural"])
    subject = _noun(rng, subject_number)
    attractor_count = force_attractors if force_attractors is not None else rng.randrange(max_attractors + 1)

    phrases = [f"The {rng.choice(MODIFIERS)} {subject}"]
    for idx in range(attractor_count):
        attractor_number = "plural" if subject_number == "singular" else "singular"
        if rng.random() < 0.25:
            attractor_number = subject_number
        attractor = _noun(rng, attractor_number)
        phrases.append(f"{rng.choice(PREPOSITIONS)} the {rng.choice(MODIFIERS)} {attractor}")

    correct_verb = "is" if subject_number == "singular" else "are"
    wrong_verb = "are" if correct_verb == "is" else "is"
    candidate_verb = correct_verb if positive else wrong_verb
    prompt = " ".join(phrases)
    prefix = f"Sentence prefix: {prompt}. Candidate verb: {candidate_verb}. Does the verb agree? Answer"
    answer = "yes" if positive else "no"
    full_text = f"{prefix} {answer}."
    return AgreementExample(
        prefix=prefix,
        answer=answer,
        full_text=full_text,
        graph_id=graph_id,
        subject_number=subject_number,
        attractor_count=attractor_count,
        candidate_verb=candidate_verb,
        correct=positive,
    )


class AgreementDataset(Dataset):
    """Balanced yes/no subject-verb agreement classification."""

    def __init__(
        self,
        *,
        n_examples: int = 512,
        seed: int = 0,
        max_attractors: int = 4,
        force_attractors: int | None = None,
    ) -> None:
        self.examples = [
            render_agreement_example(
                seed=seed + idx,
                graph_id=f"agr{idx}",
                max_attractors=max_attractors,
                force_attractors=force_attractors,
                positive=idx % 2 == 0,
            )
            for idx in range(n_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> AgreementExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[AgreementExample]) -> list[AgreementExample]:
    return batch
