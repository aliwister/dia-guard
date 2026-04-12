"""
DIA-GUARD Splits Generator
==========================
Generates stratified train/val/test splits from DIA-GUARD LLM data.

Data source: per-dialect CSV files under LLM_Data/{dialect}/{dataset}_zeroshot_...csv
  - 48 dialects × 15 datasets = up to 720 CSV files
  - Each row expands into up to 8 training records:

    TEXT VARIANTS (label = 1, harmful)
    -----------------------------------
    original_input       → text_type="original"         (SAE, Standard American English)
    transformed_input    → text_type="transformed"       (full dialect transform)
    basic_transform      → text_type="basic_transform"   (basic dialect features)
    coi_transform        → text_type="coi_transform"     (coarse-grained dialect features)

    COUNTERHARM VARIANTS (label = 0, safe/benign)
    -----------------------------------------------
    counterharm_original    → text_type="counterharm_original"    (benign ↔ original_input)
    counterharm_transformed → text_type="counterharm_transformed" (benign ↔ transformed_input)
    counterharm_basic       → text_type="counterharm_basic"       (benign ↔ basic_transform)
    counterharm_coi         → text_type="counterharm_coi"         (benign ↔ coi_transform)

    NOT USED: counterharm_score, counterharm_validated, counterharm_model (metadata only)

Global sample IDs are constructed as:
    {dataset}__{dialect}__{original_row_id}__{text_type}
ensuring every record is uniquely traceable back to its source row and column.

For contrastive training, each harmful record carries a neg_text field pointing
to its paired counterharm text (e.g. original → counterharm_original).

Stratification is performed by (dialect × label) stratum so that every
dialect and both label classes are represented proportionally in each split.

Usage
-----
  # Full run (all dialects, all datasets)
  python generate_splits.py \\
      --llm_data_dir ../LLM_Data \\
      --output_dir ../DIA_Splits \\
      --train_ratio 0.70 --val_ratio 0.15 --test_ratio 0.15 --seed 42

  # Specific dialects only
  python generate_splits.py \\
      --llm_data_dir ../LLM_Data \\
      --dialects aboriginal_english australian_english \\
      --output_dir ../DIA_Splits

  # Dry run (stats only, no files written)
  python generate_splits.py --llm_data_dir ../LLM_Data --dry_run
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

# Columns that produce harmful (label=1) training records, with their text_type tag
HARMFUL_COLS = [
    ("original_input",    "original"),
    ("transformed_input", "transformed"),
    ("basic_transform",   "basic_transform"),
    ("coi_transform",     "coi_transform"),
]

# Counterharm (label=0) columns, paired with their harmful counterpart text_type
COUNTERHARM_COLS = [
    ("counterharm_original",    "counterharm_original",    "original"),
    ("counterharm_transformed", "counterharm_transformed", "transformed"),
    ("counterharm_basic",       "counterharm_basic",       "basic_transform"),
    ("counterharm_coi",         "counterharm_coi",         "coi_transform"),
]

# Fields to carry through from each row (metadata only, not used for labels)
PASSTHROUGH_FIELDS = [
    "dataset", "dialect", "model", "generation_model",
    "original_harmfulness", "transformed_harmfulness",
]

CSV_SUFFIX = "_zeroshot_harmfulness_results_with_transforms.csv"

# Refusal / generation-failure strings to filter out.
# Any text field whose stripped, lowercased value starts with or exactly matches
# one of these prefixes/strings is considered a failed generation and the
# corresponding training record is omitted.
REFUSAL_EXACT = {
    "guardrail policy violation",
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "as a language model",
}

REFUSAL_PREFIXES = (
    "guardrail policy violation",
    "i cannot ",
    "i can't ",
    "i am unable",
    "i'm unable",
    "as an ai",
    "as a language model",
    "i apologize, but i",
    "sorry, but i cannot",
    "sorry, i cannot",
    "i'm sorry, i can",
    "i'm sorry, but i",
    "i must refuse",
    "i will not",
    "this request violates",
    "this goes against",
    "i'm not able to",
    "i am not able to",
)


# ─── Data loading ─────────────────────────────────────────────────────────────

def _is_refusal(val: str) -> bool:
    """
    Return True if the text is a model refusal or generation failure.
    Matches exact strings and common refusal prefixes (case-insensitive).
    """
    v = val.strip().lower()
    if v in REFUSAL_EXACT:
        return True
    return any(v.startswith(p) for p in REFUSAL_PREFIXES)


def _is_valid_text(val: str) -> bool:
    """Return True if a text field is non-empty, not a placeholder, and not a refusal."""
    v = val.strip() if val else ""
    if not v or v.lower() in {"nan", "none", "null", "n/a"}:
        return False
    return not _is_refusal(v)


def build_records_from_row(row: dict) -> list[dict]:
    """
    Expand a single CSV row into up to 8 training records.

    Global sample_id format:
        {dataset}__{dialect}__{row_sample_id}__{text_type}
    This ensures every record is uniquely traceable to its origin.

    Each record schema:
    {
      "sample_id":        str   — globally unique, traceable ID
      "source_sample_id": str   — original sample_id from the CSV row
      "dataset":          str   — source benchmark name
      "dialect":          str   — English dialect
      "text":             str   — training text for this variant
      "text_type":        str   — one of: original / transformed / basic_transform /
                                          coi_transform / counterharm_original /
                                          counterharm_transformed / counterharm_basic /
                                          counterharm_coi
      "label":            int   — 1 = harmful, 0 = safe/benign
      "label_str":        str   — "unsafe" or "safe"
      "neg_text":         str   — paired counterharm text (for contrastive; "" if N/A)
      "original_input":   str   — SAE original text (reference)
      "transformed_input":str   — full dialect transform (reference)
      "basic_transform":  str   — basic dialect transform (reference)
      "coi_transform":    str   — coi dialect transform (reference)
      "model":            str   — generation model
    }
    """
    dataset       = row.get("dataset", "").strip()
    dialect       = row.get("dialect", "").strip()
    row_sample_id = str(row.get("sample_id", "")).strip()

    # Build a lookup of counterharm texts keyed by text_type of the harmful twin
    # e.g. "original" → counterharm_original text
    counterharm_by_harmful_type: dict[str, str] = {}
    for col, _text_type_tag, harmful_text_type in COUNTERHARM_COLS:
        val = row.get(col, "")
        if _is_valid_text(val):
            counterharm_by_harmful_type[harmful_text_type] = val.strip()

    records = []
    # Track skipped fields for caller statistics (returned via side-effect list)
    # We use the module-level counter so load_llm_data can report totals.
    # (Simple approach: just filter silently; caller sees final counts.)

    # ── Harmful records (label = 1) ───────────────────────────────────────────
    for col, text_type in HARMFUL_COLS:
        text = row.get(col, "")
        if not _is_valid_text(text):
            continue

        global_id = f"{dataset}__{dialect}__{row_sample_id}__{text_type}"
        neg_text = counterharm_by_harmful_type.get(text_type, "")

        records.append({
            "sample_id":        global_id,
            "source_sample_id": row_sample_id,
            "dataset":          dataset,
            "dialect":          dialect,
            "text":             text.strip(),
            "text_type":        text_type,
            "label":            1,
            "label_str":        "unsafe",
            "neg_text":         neg_text,
            "original_input":   row.get("original_input", "").strip(),
            "transformed_input":row.get("transformed_input", "").strip(),
            "basic_transform":  row.get("basic_transform", "").strip(),
            "coi_transform":    row.get("coi_transform", "").strip(),
            "model":            row.get("generation_model", row.get("model", "")).strip(),
        })

    # ── Safe / benign records (label = 0) ─────────────────────────────────────
    for col, text_type, harmful_twin_type in COUNTERHARM_COLS:
        text = row.get(col, "")
        if not _is_valid_text(text):
            continue

        global_id = f"{dataset}__{dialect}__{row_sample_id}__{text_type}"
        # neg_text for a safe record = the corresponding harmful text
        harm_text = row.get(
            # map back from counterharm_col to its harmful column name
            {"counterharm_original":    "original_input",
             "counterharm_transformed": "transformed_input",
             "counterharm_basic":       "basic_transform",
             "counterharm_coi":         "coi_transform"}.get(col, ""),
            ""
        )

        records.append({
            "sample_id":        global_id,
            "source_sample_id": row_sample_id,
            "dataset":          dataset,
            "dialect":          dialect,
            "text":             text.strip(),
            "text_type":        text_type,
            "label":            0,
            "label_str":        "safe",
            "neg_text":         harm_text.strip() if _is_valid_text(harm_text) else "",
            "original_input":   row.get("original_input", "").strip(),
            "transformed_input":row.get("transformed_input", "").strip(),
            "basic_transform":  row.get("basic_transform", "").strip(),
            "coi_transform":    row.get("coi_transform", "").strip(),
            "model":            row.get("generation_model", row.get("model", "")).strip(),
        })

    return records


def load_llm_data(
    data_dir: str,
    dialects: list[str] | None = None,
) -> list[dict]:
    """
    Walk data_dir/{dialect}/{dataset}_zeroshot_...csv and expand each row.
    If dialects is specified, only those are loaded.
    """
    data_path = Path(data_dir)
    all_records: list[dict] = []
    n_rows_seen = 0
    n_rows_skipped = 0     # whole rows skipped (original_input is a refusal)
    n_fields_before = 0    # potential records before per-column refusal filter
    n_fields_after = 0     # records accepted after per-column refusal filter

    dialect_dirs = sorted(
        d for d in data_path.iterdir()
        if d.is_dir() and (dialects is None or d.name in dialects)
    )

    for dialect_dir in dialect_dirs:
        csv_files = sorted(dialect_dir.glob(f"*{CSV_SUFFIX}"))
        for csv_file in csv_files:
            dataset_name = csv_file.name.replace(CSV_SUFFIX, "")
            try:
                with open(csv_file, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        n_rows_seen += 1
                        # Skip entire row if original_input is a refusal
                        # (whole row is unusable in this case)
                        if _is_refusal(row.get("original_input", "")):
                            n_rows_skipped += 1
                            continue
                        # Normalise: ensure dataset / dialect come from file path
                        # in case the column values differ
                        row["dataset"] = row.get("dataset") or dataset_name
                        row["dialect"] = row.get("dialect") or dialect_dir.name
                        # Count potential fields (8 per row) vs accepted
                        n_fields_before += 8
                        records = build_records_from_row(row)
                        n_fields_after += len(records)
                        all_records.extend(records)
            except Exception as exc:
                print(f"  Warning: could not read {csv_file}: {exc}")

    n_field_refusals = n_fields_before - n_fields_after
    if n_rows_skipped or n_field_refusals:
        print(
            f"  Refusal filter: skipped {n_rows_skipped:,} whole rows "
            f"+ {n_field_refusals:,} individual text fields "
            f"({(n_rows_skipped*8 + n_field_refusals) / max(n_fields_before + n_rows_skipped*8, 1)*100:.1f}% of potential records removed)"
        )
    return all_records


def load_rule_data(rule_data_dir: str) -> list[dict]:
    """
    Load rule-based (multi-value) transformation data.
    Expected structure: {rule_data_dir}/{dialect}/*.csv
    Each CSV row has: prompt, prompt_transformed, transformation_tool,
    counterharm_prompt, counterharm_prompt_transformed, counterharm_score,
    counterharm_validated, counterharm_model.

    Produces up to 3 records per row:
      1. mv_transform          (label=1) from prompt_transformed
      2. mv_counterharm        (label=0) from counterharm_prompt        (if present)
      3. mv_counterharm_trans  (label=0) from counterharm_prompt_transformed (if present)

    Harmful records carry neg_text pointing to their benign counterpart and vice versa,
    enabling contrastive training.
    """
    rule_path = Path(rule_data_dir)
    if not rule_path.exists():
        print(f"  Warning: rule_data_dir not found: {rule_data_dir}")
        return []

    records = []
    n_benign_original = 0
    n_benign_transformed = 0
    for csv_file in sorted(rule_path.rglob("*.csv")):
        try:
            with open(csv_file, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row_idx, row in enumerate(reader):
                    text = row.get("prompt_transformed", row.get("mv_transform", row.get("transformed_text", "")))
                    if not _is_valid_text(text):
                        continue
                    dialect = row.get("dialect", csv_file.parent.name)
                    raw_stem = csv_file.stem
                    dataset = row.get("dataset", raw_stem.replace(f"_{dialect}", "", 1))
                    row_id = row.get("sample_id", row.get("id", str(row_idx)))

                    prompt_orig = row.get("prompt", row.get("original_input", ""))
                    prompt_trans = row.get("prompt_transformed", row.get("transformed_input", ""))
                    ch_prompt = row.get("counterharm_prompt", "").strip()
                    ch_prompt_trans = row.get("counterharm_prompt_transformed", "").strip()
                    model = row.get("model", "")

                    # Determine best benign text for neg_text on the harmful record
                    benign_for_neg = ch_prompt_trans if _is_valid_text(ch_prompt_trans) else (
                        ch_prompt if _is_valid_text(ch_prompt) else ""
                    )

                    # ── Harmful record (label=1): prompt_transformed ──────────
                    records.append({
                        "sample_id":         f"{dataset}__{dialect}__{row_id}__mv_transform",
                        "source_sample_id":  str(row_id),
                        "dataset":           dataset,
                        "dialect":           dialect,
                        "text":              text.strip(),
                        "text_type":         "mv_transform",
                        "label":             1,
                        "label_str":         "unsafe",
                        "neg_text":          benign_for_neg,
                        "original_input":    prompt_orig,
                        "transformed_input": prompt_trans,
                        "basic_transform":   "",
                        "coi_transform":     "",
                        "model":             model,
                    })

                    # ── Benign record (label=0): counterharm_prompt ───────────
                    if _is_valid_text(ch_prompt):
                        n_benign_original += 1
                        records.append({
                            "sample_id":         f"{dataset}__{dialect}__{row_id}__mv_counterharm",
                            "source_sample_id":  str(row_id),
                            "dataset":           dataset,
                            "dialect":           dialect,
                            "text":              ch_prompt,
                            "text_type":         "mv_counterharm",
                            "label":             0,
                            "label_str":         "safe",
                            "neg_text":          prompt_orig.strip() if _is_valid_text(prompt_orig) else text.strip(),
                            "original_input":    prompt_orig,
                            "transformed_input": prompt_trans,
                            "basic_transform":   "",
                            "coi_transform":     "",
                            "model":             model,
                        })

                    # ── Benign record (label=0): counterharm_prompt_transformed
                    if _is_valid_text(ch_prompt_trans):
                        n_benign_transformed += 1
                        records.append({
                            "sample_id":         f"{dataset}__{dialect}__{row_id}__mv_counterharm_trans",
                            "source_sample_id":  str(row_id),
                            "dataset":           dataset,
                            "dialect":           dialect,
                            "text":              ch_prompt_trans,
                            "text_type":         "mv_counterharm_trans",
                            "label":             0,
                            "label_str":         "safe",
                            "neg_text":          text.strip(),
                            "original_input":    prompt_orig,
                            "transformed_input": prompt_trans,
                            "basic_transform":   "",
                            "coi_transform":     "",
                            "model":             model,
                        })
        except Exception as exc:
            print(f"  Warning: could not read {csv_file}: {exc}")

    if n_benign_original or n_benign_transformed:
        print(
            f"  Multi-value benign records: {n_benign_original:,} counterharm_prompt "
            f"+ {n_benign_transformed:,} counterharm_prompt_transformed"
        )

    return records


# ─── Stratified splitting ──────────────────────────────────────────────────────

def stratified_split(
    records: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Contamination-safe stratified split.

    Groups records by (dataset, source_sample_id) — the unique source prompt key —
    so that ALL variants of a source row (every dialect, every text_type) are
    confined to exactly one split.  This prevents two forms of leakage:

      1. Within-dialect: different text_type variants of the same row (original,
         basic_transform, counterharm_*, …) landing in different splits.
      2. Cross-dialect: the same underlying harmful prompt appearing in both
         train and test under different dialect transformations.

    Stratification is performed at the source-group level by (dataset) stratum,
    preserving benchmark balance across splits.

    Assigns the 'split' field in-place on every record.
    Returns (train, val, test) lists.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    rng = random.Random(seed)

    # ── Step 1: group all records by their source prompt key ──────────────────
    # Key = (dataset, source_sample_id) uniquely identifies one original prompt
    # across all its dialect × text_type expansions.
    source_groups: dict[tuple, list[dict]] = defaultdict(list)
    for rec in records:
        key = (rec["dataset"], rec["source_sample_id"])
        source_groups[key].append(rec)

    # ── Step 2: stratify source-group keys by dataset ─────────────────────────
    # Stratifying by dataset keeps each benchmark proportionally represented
    # in every split regardless of how many rows it contributes.
    dataset_strata: dict[str, list[tuple]] = defaultdict(list)
    for key in source_groups:
        dataset_strata[key[0]].append(key)

    train_keys: set[tuple] = set()
    val_keys:   set[tuple] = set()
    test_keys:  set[tuple] = set()

    for dataset, keys in sorted(dataset_strata.items()):
        rng.shuffle(keys)
        n = len(keys)
        n_train = max(1, round(n * train_ratio))
        n_val   = max(1, round(n * val_ratio))
        n_test  = n - n_train - n_val
        if n_test < 0:
            n_train = max(1, n - 2)
            n_val   = 1 if n >= 2 else 0
            n_test  = max(0, n - n_train - n_val)

        for k in keys[:n_train]:
            train_keys.add(k)
        for k in keys[n_train : n_train + n_val]:
            val_keys.add(k)
        for k in keys[n_train + n_val :]:
            test_keys.add(k)

    # ── Step 3: assign every record to its split via the group key ────────────
    train_set, val_set, test_set = [], [], []

    for key, group in source_groups.items():
        if key in train_keys:
            split_name = "train"
            target = train_set
        elif key in val_keys:
            split_name = "val"
            target = val_set
        else:
            split_name = "test"
            target = test_set

        for r in group:
            r["split"] = split_name
        target.extend(group)

    # Shuffle within each final set
    rng.shuffle(train_set)
    rng.shuffle(val_set)
    rng.shuffle(test_set)

    return train_set, val_set, test_set


# ─── Output writers ───────────────────────────────────────────────────────────

# Rows whose text matches any of these criteria are moved to train-removed.jsonl
# instead of new_train.jsonl.  These patterns produce degenerate attention
# distributions (especially with flash_attention_2) that cause NaN loss.
_MAX_TRAIN_CHARS   = 2048 * 3   # ~2048 tokens at 3 chars/token (conservative for multilingual)
_SPACE_RATIO_LIMIT = 0.70       # markdown table padding rows
_WORD_DUP_LIMIT    = 0.60       # single word repeated ad nauseam


def _is_degenerate_text(text: str) -> tuple[bool, str]:
    """Return (True, reason) if text should be excluded from training."""
    if not text or not text.strip():
        return True, "empty"
    if len(text) > _MAX_TRAIN_CHARS:
        return True, f"too_long={len(text)}"
    if len(text) > 50 and text.count(" ") / len(text) > _SPACE_RATIO_LIMIT:
        return True, f"space_ratio={text.count(' ') / len(text):.2f}"
    words = text.split()
    if len(words) > 20:
        top_word = max(set(words), key=words.count)
        ratio = words.count(top_word) / len(words)
        if ratio > _WORD_DUP_LIMIT:
            return True, f"word_repeat={top_word!r}_{ratio:.2f}"
    return False, ""


def write_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sample_stratified(records: list[dict], ratio: float, seed: int = 42) -> list[dict]:
    """Return a stratified sample of `ratio` from records, preserving label balance."""
    rng = random.Random(seed)
    by_label: dict = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    sampled = []
    for group in by_label.values():
        k = max(1, round(len(group) * ratio))
        sampled.extend(rng.sample(group, k))
    rng.shuffle(sampled)
    return sampled


def write_splits(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: str,
    val_sample_ratio: float = 0.10,
    seed: int = 42,
) -> None:
    """Write global JSONL files and per-dialect JSONL files.

    val is downsampled to val_sample_ratio for use as the eval set during
    training. The remaining val rows are merged into test so no data is lost.
    The split field on those rows is updated to 'test' in-place.
    """
    out = Path(output_dir)

    # Original splits written unchanged
    write_jsonl(train, str(out / "train.jsonl"))
    write_jsonl(val,   str(out / "val.jsonl"))
    write_jsonl(test,  str(out / "test.jsonl"))

    # new_train: train minus degenerate rows; train-removed: the removed rows
    new_train, removed_train = [], []
    for rec in train:
        bad, reason = _is_degenerate_text(rec.get("text", ""))
        if bad:
            removed_train.append({**rec, "_removed_reason": reason})
        else:
            new_train.append(rec)
    write_jsonl(new_train,     str(out / "new_train.jsonl"))
    write_jsonl(removed_train, str(out / "train-removed.jsonl"))
    print(
        f"  new_train: {len(new_train):,} "
        f"(removed {len(removed_train):,} degenerate rows → train-removed.jsonl)"
    )

    # new_val: 10% stratified sample of val
    # new_test: original test + remaining 90% of val
    val_sample = sample_stratified(val, val_sample_ratio, seed=seed)
    val_sample_ids = {r["sample_id"] for r in val_sample}
    val_remainder = [r for r in val if r["sample_id"] not in val_sample_ids]
    extended_test = test + val_remainder

    write_jsonl(val_sample,    str(out / "new_val.jsonl"))
    write_jsonl(extended_test, str(out / "new_test.jsonl"))

    pct = int(val_sample_ratio * 100)
    print(
        f"  Global  — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}"
    )
    print(
        f"  New     — new_val: {len(val_sample):,} ({pct}% of val)  "
        f"new_test: {len(extended_test):,} (test + {len(val_remainder):,} from val)"
    )

    # Per-dialect splits use the original split assignments
    by_dialect_split: dict[str, dict[str, list[dict]]] = defaultdict(
        lambda: {"train": [], "val": [], "test": []}
    )
    for split_name, recs in [("train", train), ("val", val), ("test", test)]:
        for r in recs:
            by_dialect_split[r["dialect"]][split_name].append(r)

    for dialect, splits in sorted(by_dialect_split.items()):
        d_dir = out / "by_dialect" / dialect
        for split_name, recs in splits.items():
            write_jsonl(recs, str(d_dir / f"{split_name}.jsonl"))

    print(f"  Per-dialect splits written for {len(by_dialect_split)} dialects")
    return val_sample, extended_test


# ─── Metadata ─────────────────────────────────────────────────────────────────

def compute_metadata(
    train: list[dict],
    val: list[dict],
    test: list[dict],
) -> dict:
    """Compute split statistics and return as a dict."""
    def dist(recs, field):
        counts: dict = defaultdict(int)
        for r in recs:
            counts[str(r.get(field, "unknown"))] += 1
        return dict(sorted(counts.items()))

    total = len(train) + len(val) + len(test)
    return {
        "total_records": total,
        "splits": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
        },
        "label_distribution": {
            "train": dist(train, "label_str"),
            "val":   dist(val,   "label_str"),
            "test":  dist(test,  "label_str"),
        },
        "dialect_distribution": {
            "train": dist(train, "dialect"),
            "val":   dist(val,   "dialect"),
            "test":  dist(test,  "dialect"),
        },
        "dataset_distribution": {
            "train": dist(train, "dataset"),
            "val":   dist(val,   "dataset"),
            "test":  dist(test,  "dataset"),
        },
        "text_type_distribution": {
            "train": dist(train, "text_type"),
            "val":   dist(val,   "text_type"),
            "test":  dist(test,  "text_type"),
        },
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate stratified DIA-GUARD train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--llm_data_dir",
        type=str,
        required=True,
        help="Root directory containing per-dialect CSV folders (LLM_Data/)",
    )
    parser.add_argument(
        "--rule_data_dir",
        type=str,
        default=None,
        help="Root directory for rule-based multi-value transformation CSVs (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../DIA_Splits",
        help="Destination for generated JSONL splits (default: ../DIA_Splits)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.70, help="Train split ratio (default 0.70)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Val split ratio (default 0.15)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15, help="Test split ratio (default 0.15)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=None,
        help="Subset of dialects to process (default: all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print statistics only; do not write any files",
    )
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    # ── Load LLM data ─────────────────────────────────────────────────────────
    print(f"\nLoading LLM data from: {args.llm_data_dir}")
    records = load_llm_data(args.llm_data_dir, dialects=args.dialects)
    print(f"  {len(records):,} records loaded from LLM data")

    # ── Load rule-based data (optional) ───────────────────────────────────────
    if args.rule_data_dir:
        print(f"\nLoading rule-based data from: {args.rule_data_dir}")
        rule_records = load_rule_data(args.rule_data_dir)
        print(f"  {len(rule_records):,} records loaded from rule-based data")
        records.extend(rule_records)

    if not records:
        print("Error: no records loaded. Check --llm_data_dir.")
        return

    print(f"\nTotal records before split: {len(records):,}")

    # ── Label summary ─────────────────────────────────────────────────────────
    label_counts: dict[str, int] = defaultdict(int)
    text_type_counts: dict[str, int] = defaultdict(int)
    dialect_counts: dict[str, int] = defaultdict(int)
    for r in records:
        label_counts[r["label_str"]] += 1
        text_type_counts[r["text_type"]] += 1
        dialect_counts[r["dialect"]] += 1

    print(f"\nLabel distribution:")
    for k, v in sorted(label_counts.items()):
        print(f"  {k}: {v:,} ({100*v/len(records):.1f}%)")

    print(f"\nText type distribution:")
    for k, v in sorted(text_type_counts.items()):
        print(f"  {k}: {v:,}")

    print(f"\nDialects: {len(dialect_counts)}")

    # ── Stratified split ──────────────────────────────────────────────────────
    print(f"\nSplitting (train={args.train_ratio:.0%} / val={args.val_ratio:.0%} / test={args.test_ratio:.0%}, seed={args.seed})...")
    train, val, test = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"  train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    # ── Dry run: stop here ────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        meta = compute_metadata(train, val, test)
        print(json.dumps(meta, indent=2))
        return

    # ── Write files ───────────────────────────────────────────────────────────
    print(f"\nWriting splits to: {args.output_dir}")
    val_out, test_out = write_splits(train, val, test, args.output_dir, seed=args.seed)

    meta = compute_metadata(train, val_out, test_out)
    meta_path = os.path.join(args.output_dir, "splits_metadata.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata written: {meta_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
