#!/usr/bin/env python3
"""Download web-scale datasets to the repo-local HF cache for fast training."""
import os
from pathlib import Path

from datasets import load_dataset

repo_root = Path(__file__).resolve().parents[1]
data_dir = Path(os.environ.get("DATA_DIR", repo_root / "data")).expanduser().resolve()
hf_home = data_dir / "hf_home"
hf_datasets = data_dir / "hf_datasets"
hf_home.mkdir(parents=True, exist_ok=True)
hf_datasets.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_home)
os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
os.environ["HF_DATASETS_CACHE"] = str(hf_datasets)

print(f"HF_HOME={os.environ['HF_HOME']}")
print(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")

datasets_to_download = [
    # (name, config, split)
    ("Skylion007/openwebtext", None, "train"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-51", "train"),
    ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-46", "train"),
    ("HuggingFaceFW/fineweb-edu", "CC-MAIN-2024-42", "train"),
]

for name, config, split in datasets_to_download:
    print(f"\n{'='*60}")
    print(f"Downloading: {name} | config={config} | split={split}")
    print(f"{'='*60}")
    try:
        ds = load_dataset(name, config, split=split, streaming=False, cache_dir=os.environ["HF_DATASETS_CACHE"])
        print(f"  -> Loaded {len(ds)} examples")
    except Exception as e:
        print(f"  -> ERROR: {e}")

print("\nAll downloads complete!")
