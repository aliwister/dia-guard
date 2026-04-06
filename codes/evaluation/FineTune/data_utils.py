"""
Shared data loading utilities with disk caching for DIA-GUARD training scripts.

Caches formatted (chat-templated) datasets per model to avoid redundant
tokenization across training waves. Cache is stored at:
    /data/vibe_exp/dia-guard/dataset/formatted_cache/{model_slug}/{split}/

Usage:
    from data_utils import load_and_format_dataset

    train_ds = load_and_format_dataset(
        jsonl_path="/data/.../train.jsonl",
        tokenizer=tokenizer,
        model_name="google/gemma-3-270m-it",
        system_prompt=SYSTEM_PROMPT,
        split="train",
    )
"""

import hashlib
import json
import os
from pathlib import Path

from datasets import Dataset, load_from_disk


CACHE_ROOT = "/data/vibe_exp/dia-guard/dataset/formatted_cache"


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(".", "_")


def _cache_key(jsonl_path: str, model_name: str, system_prompt: str) -> str:
    """Hash the data file + model + prompt to detect stale caches."""
    h = hashlib.md5()
    h.update(model_name.encode())
    h.update(system_prompt.encode())
    # Use file size + mtime as a fast proxy for content changes
    stat = os.stat(jsonl_path)
    h.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def load_jsonl(path: str) -> Dataset:
    """Load a JSONL file into a HuggingFace Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def load_and_format_dataset(
    jsonl_path: str,
    tokenizer,
    model_name: str,
    system_prompt: str,
    split: str = "train",
) -> Dataset:
    """
    Load JSONL data, apply chat template formatting, and cache to disk.

    On subsequent calls with the same model + data, loads from cache instantly
    instead of re-processing ~800K+ records.
    """
    slug = _model_slug(model_name)
    cache_dir = os.path.join(CACHE_ROOT, slug, split)
    key_file = os.path.join(cache_dir, ".cache_key")
    current_key = _cache_key(jsonl_path, model_name, system_prompt)

    # Check if valid cache exists
    if os.path.isdir(cache_dir) and os.path.isfile(key_file):
        with open(key_file) as f:
            cached_key = f.read().strip()
        if cached_key == current_key:
            print(f"Loading cached formatted {split} dataset from {cache_dir}")
            return load_from_disk(cache_dir)
        else:
            print(f"Cache key mismatch for {split}, regenerating...")

    # Load and format
    print(f"Loading {split} data: {jsonl_path}")
    dataset = load_jsonl(jsonl_path)

    def format_example(example):
        prompt_text = example.get("text", example.get("prompt", ""))
        response_text = "unsafe" if example.get("label", 0) == 1 else "safe"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": response_text},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            example["text"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            example["text"] = (
                f"<|system|>{system_prompt}</s>\n"
                f"<|user|>{prompt_text}</s>\n"
                f"<|assistant|>{response_text}</s>"
            )
        return example

    print(f"Formatting {split} dataset ({len(dataset)} records)...")
    dataset = dataset.map(format_example)

    # Save to cache
    print(f"Caching formatted {split} dataset to {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(cache_dir)
    with open(key_file, "w") as f:
        f.write(current_key)

    return dataset


def load_jsonl_records(path: str, model_name: str, split: str = "train"):
    """
    Load JSONL records with caching for contrastive training scripts.
    Returns list of dicts (not a HF Dataset) since contrastive scripts
    use custom Dataset classes.
    """
    slug = _model_slug(model_name)
    cache_path = os.path.join(CACHE_ROOT, slug, f"{split}_records.json")

    if os.path.isfile(cache_path):
        stat_orig = os.stat(path)
        stat_cache = os.stat(cache_path)
        if stat_cache.st_mtime > stat_orig.st_mtime:
            print(f"Loading cached {split} records from {cache_path}")
            with open(cache_path) as f:
                return json.load(f)

    print(f"Loading {split} records: {path}")
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"Caching {split} records to {cache_path}")
    with open(cache_path, "w") as f:
        json.dump(records, f)

    return records
