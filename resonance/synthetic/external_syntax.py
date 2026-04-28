"""Local-file loaders for established syntax datasets.

These loaders do not download data.  They adapt already-downloaded Linzen-style
agreement rows or BLiMP JSONL minimal pairs to the same yes/no probe interface
used by the synthetic tasks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntaxPairExample:
    prefix: str
    answer: str
    full_text: str
    graph_id: str
    source: str


def _example(prefix: str, answer: str, graph_id: str, source: str) -> SyntaxPairExample:
    return SyntaxPairExample(
        prefix=prefix,
        answer=answer,
        full_text=f"{prefix} {answer}.",
        graph_id=graph_id,
        source=source,
    )


class BlimpMinimalPairDataset(Dataset):
    """BLiMP JSONL loader.

    Expected fields are the standard `sentence_good` and `sentence_bad`.
    The order is alternated so the answer is balanced.
    """

    def __init__(self, *, path: str | Path, n_examples: int | None = None, offset: int = 0) -> None:
        rows = []
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        if n_examples is not None:
            rows = rows[offset : offset + n_examples]
        else:
            rows = rows[offset:]
        self.examples = []
        for idx, row in enumerate(rows):
            good = row["sentence_good"]
            bad = row["sentence_bad"]
            if idx % 2 == 0:
                prefix = f"Sentence one: {good} Sentence two: {bad} Is sentence one more grammatical? Answer"
                answer = "yes"
            else:
                prefix = f"Sentence one: {bad} Sentence two: {good} Is sentence one more grammatical? Answer"
                answer = "no"
            uid = str(row.get("UID", row.get("uid", f"blimp{idx}")))
            self.examples.append(_example(prefix, answer, uid, "blimp"))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SyntaxPairExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class AgreementFileDataset(Dataset):
    """Flexible loader for local agreement data.

    Supported JSONL/CSV/TSV fields:
      - `prefix`, `correct`, `wrong`; or
      - `sentence_good`, `sentence_bad`.
      - Linzen simple `numpred.*` rows: `VBZ|VBP<TAB>prefix`.
      - Linzen dependency TSV rows with `orig_sentence` and `verb_pos`.
    """

    def __init__(self, *, path: str | Path, n_examples: int | None = None, offset: int = 0) -> None:
        path = Path(path)
        if path.name.startswith("numpred."):
            rows = []
            with path.open() as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) != 2:
                        continue
                    rows.append({"linzen_number": parts[0], "prefix": parts[1]})
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            dialect = "excel-tab" if path.suffix in {".tsv", ".tab"} else "excel"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle, dialect=dialect))
        if n_examples is not None:
            rows = rows[offset : offset + n_examples]
        else:
            rows = rows[offset:]
        self.examples = []
        for idx, row in enumerate(rows):
            if "prefix" in row and "correct" in row and "wrong" in row:
                candidate = row["correct"] if idx % 2 == 0 else row["wrong"]
                answer = "yes" if idx % 2 == 0 else "no"
                prefix = f"Sentence prefix: {row['prefix']} Candidate word: {candidate}. Does the word agree? Answer"
            elif "sentence_good" in row and "sentence_bad" in row:
                good = row["sentence_good"]
                bad = row["sentence_bad"]
                if idx % 2 == 0:
                    prefix = f"Sentence: {good} Is this sentence grammatical? Answer"
                    answer = "yes"
                else:
                    prefix = f"Sentence: {bad} Is this sentence grammatical? Answer"
                    answer = "no"
            elif "linzen_number" in row and "prefix" in row:
                correct = "singular" if row["linzen_number"] == "VBZ" else "plural"
                wrong = "plural" if correct == "singular" else "singular"
                candidate = correct if idx % 2 == 0 else wrong
                answer = "yes" if idx % 2 == 0 else "no"
                prefix = (
                    f"Sentence prefix: {row['prefix']}. Candidate verb number: {candidate}. "
                    "Does the candidate number agree with the subject? Answer"
                )
            elif "orig_sentence" in row and "verb_pos" in row:
                correct = "singular" if row["verb_pos"] == "VBZ" else "plural"
                wrong = "plural" if correct == "singular" else "singular"
                candidate = correct if idx % 2 == 0 else wrong
                answer = "yes" if idx % 2 == 0 else "no"
                sentence = row["orig_sentence"]
                prefix = (
                    f"Sentence: {sentence}. Subject: {row.get('subj', 'unknown')}. "
                    f"Candidate verb number: {candidate}. Does the candidate number agree? Answer"
                )
            else:
                raise ValueError(
                    "agreement rows need fields prefix/correct/wrong, sentence_good/sentence_bad, "
                    "Linzen numpred rows, or Linzen dependency TSV fields"
                )
            self.examples.append(_example(prefix, answer, str(row.get("id", f"agrfile{idx}")), "agreement_file"))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SyntaxPairExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


class GenericProbeDataset(Dataset):
    """JSONL yes/no probe loader.

    Expected fields:
      - `prefix`: prompt ending before the yes/no answer.
      - `answer`: `yes` or `no`.

    Optional fields `id`, `source`, and `metadata` are preserved only through
    `graph_id`/`source` naming for now.  This keeps external converted datasets
    compatible with the shared structural-task harness.
    """

    def __init__(self, *, path: str | Path, n_examples: int | None = None, offset: int = 0) -> None:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if n_examples is not None:
            rows = rows[offset : offset + n_examples]
        else:
            rows = rows[offset:]
        self.examples = []
        for idx, row in enumerate(rows):
            answer = str(row["answer"]).strip().lower()
            if answer not in {"yes", "no"}:
                raise ValueError(f"generic probe answer must be yes/no, got {answer!r}")
            prefix = str(row["prefix"]).strip()
            graph_id = str(row.get("id", f"generic{idx}"))
            source = str(row.get("source", "generic_probe"))
            self.examples.append(_example(prefix, answer, graph_id, source))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SyntaxPairExample:
        return self.examples[idx]

    def all_texts(self) -> list[str]:
        return [example.full_text for example in self.examples]


def collate_examples(batch: list[SyntaxPairExample]) -> list[SyntaxPairExample]:
    return batch
