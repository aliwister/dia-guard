#!/usr/bin/env python3
"""
Fix invalid/failed CounterHarm-SHIELD rows using a single repair prompt.

For each invalid cell, provides the model with:
  - The original harmful text (reference)
  - The bad generation (what went wrong)
  - The dialect
  - The validation criteria
And asks for a corrected benign version in one shot.

Much faster than re-running the full 6-chain pipeline.

Part of DIA-GUARD → Dia-LLM → CounterHarm-SHIELD.
"""

import sys
import os
import csv
import json
import time
import threading
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure counterharm_shield dir is first in path so we import from here, not harm_shield
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_harm_shield_dir = os.path.join(_this_dir, "..", "harm_shield")
if _harm_shield_dir not in sys.path:
    sys.path.insert(1, _harm_shield_dir)

from full_generation_parallel import (
    DIALECT_MAPPING, POLICY_VIOLATION,
    COUNTERHARM_COLUMNS, SOURCE_TO_OUTPUT, MODEL_CONFIGS, create_llm,
)

import logging
logger = logging.getLogger("counterharm_shield")

MAX_REPAIR_RETRIES = 3


def is_row_invalid(row: Dict[str, Any]) -> bool:
    """Check if a row needs fixing."""
    model = row.get("counterharm_model", "").strip()
    if not model:
        return False

    if row.get("counterharm_validated", "") != "True":
        return True

    for col in ["counterharm_original", "counterharm_transformed", "counterharm_basic", "counterharm_coi"]:
        val = row.get(col, "").strip()
        if val == "[FAILED]":
            return True

    return False


REPAIR_SYSTEM = "You are a repair agent for CounterHarm-SHIELD (DIA-GUARD). You fix failed benign counterexamples. Return ONLY valid JSON."

VALIDATE_SYSTEM = "You are a safety validator for CounterHarm-SHIELD. You validate repaired outputs. Return ONLY valid JSON."


def _parse_json(text: str) -> dict:
    """Parse JSON from model response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return json.loads(text)


def repair_cell(llm, original_text: str, bad_output: str, dialect: str) -> str:
    """Single-shot repair: fix one bad cell given the original reference."""
    user_prompt = f"""TASK: Generate a corrected benign version of the harmful source text below.

SOURCE TEXT (harmful — do NOT reproduce harmful content):
"{original_text}"

PREVIOUS ATTEMPT (failed validation):
"{bad_output}"

DIALECT: {dialect}

VALIDATION CRITERIA — your output MUST satisfy ALL of these:
1. BENIGN: Completely harmless — no harmful, offensive, or dangerous content, even implicitly
2. LABEL FLIP: Genuinely benign, not just cosmetic rewording of harmful content
3. DIALECT: Preserve authentic {dialect} dialect features (vocabulary, grammar, phrasing)
4. STRUCTURE: Match the format of the source (if source is a question, output a question; if imperative, use imperative; if narrative, write narrative)
5. LENGTH: Similar length to the source text (within ~30%)
6. TOPIC SHIFT: Replace the harmful topic with a constructive/positive one while keeping the linguistic style

Return ONLY valid JSON:
{{
  "repaired_text": "<your corrected benign text>",
  "fixes_applied": "<brief note on what you fixed>"
}}"""

    for attempt in range(MAX_REPAIR_RETRIES):
        try:
            response = llm.generate(REPAIR_SYSTEM, user_prompt, max_tokens=1024, temperature=0.3)
            result = _parse_json(response)
            repaired = result.get("repaired_text", "").strip()
            if repaired:
                return repaired
        except Exception as e:
            logger.error("[Repair] Attempt %d failed: %s", attempt + 1, str(e)[:80])
            time.sleep(1)

    return ""


def validate_repair(llm, original_text: str, repaired_text: str, dialect: str) -> bool:
    """Quick validation check on the repaired output."""
    user_prompt = f"""Source (harmful): "{original_text}"
Repaired (should be benign): "{repaired_text}"
Dialect: {dialect}

Check:
1. Is the repaired text genuinely benign? (no harm, even implicit)
2. Does it preserve {dialect} dialect features?
3. Does it preserve the structural format of the source?
4. Is it similar length to the source?

Return ONLY JSON:
{{
  "valid": true/false,
  "reason": "<brief reason if invalid>"
}}"""

    try:
        response = llm.generate(VALIDATE_SYSTEM, user_prompt, max_tokens=256, temperature=0.1)
        result = _parse_json(response)
        return result.get("valid", False)
    except:
        return False


def fix_row(row: Dict[str, Any], dialect_key: str, llm) -> Dict[str, Any]:
    """Fix all bad cells in a single row."""
    result = dict(row)
    scores = []
    all_validated = True

    for src_col, out_col in SOURCE_TO_OUTPUT.items():
        src_text = row.get(src_col, "").strip()
        out_text = row.get(out_col, "").strip()

        # Skip empty/policy violation sources
        if not src_text or src_text == POLICY_VIOLATION:
            continue

        # Only fix bad cells: [FAILED], empty, or if row is invalid
        needs_fix = (not out_text or out_text == "[FAILED]" or
                     row.get("counterharm_validated", "") != "True")

        if not needs_fix:
            continue

        # Repair the cell
        repaired = repair_cell(llm, src_text, out_text, dialect_key)

        if not repaired:
            result[out_col] = "[FAILED]"
            all_validated = False
            continue

        # Validate the repair
        is_valid = validate_repair(llm, src_text, repaired, dialect_key)

        if is_valid:
            result[out_col] = repaired
            scores.append(1.0)
        else:
            # Try one more repair with feedback
            repaired2 = repair_cell(llm, src_text, repaired, dialect_key)
            if repaired2:
                result[out_col] = repaired2
                scores.append(0.8)
            else:
                result[out_col] = repaired  # keep first attempt
                scores.append(0.5)
                all_validated = False

    avg_score = sum(scores) / len(scores) if scores else float(row.get("counterharm_score", 0))
    result["counterharm_score"] = f"{avg_score:.4f}"
    result["counterharm_validated"] = str(all_validated)
    result["counterharm_model"] = f"{llm.name} [repaired]"

    return result


class FixProcessor:
    """Process invalid rows in parallel with thread-safe saving."""

    def __init__(self, output_path: str, fieldnames: list, all_rows: list, max_workers: int = 4):
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.all_rows = all_rows
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()
        self.processed_count = 0
        self.fixed_count = 0
        self.start_time = time.time()

    def save_results(self):
        with self.lock:
            output_rows = []
            for i, row in enumerate(self.all_rows):
                if i in self.results:
                    output_rows.append(self.results[i])
                else:
                    output_rows.append(row)

            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

    def process_batch(self, rows_to_fix: List[tuple], dialect_key: str, llm):
        def worker(idx_row_tuple):
            idx, row = idx_row_tuple
            try:
                result = fix_row(row, dialect_key, llm)
                return (idx, result, None)
            except Exception as e:
                return (idx, row, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(worker, item): item for item in rows_to_fix}

            for future in as_completed(futures):
                idx, result, error = future.result()

                with self.lock:
                    self.results[idx] = result
                    self.processed_count += 1

                    new_valid = result.get("counterharm_validated", "") == "True"
                    if new_valid:
                        self.fixed_count += 1

                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / (elapsed / 60) if elapsed > 0 else 0

                    sample_id = result.get("sample_id", "N/A")[:15]
                    status = "FIXED" if new_valid else "STILL_INVALID"
                    score = result.get("counterharm_score", "?")

                    if error:
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} ERROR: {error[:50]}")
                    else:
                        print(f"    [{self.processed_count}] idx={idx} sample={sample_id} "
                              f"score={score} {status} ({rate:.1f} rows/min)")

                self.save_results()


def fix_csv(csv_path: str, dialect_key: str, llm, max_workers: int = 4):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    rows_to_fix = [(i, row) for i, row in enumerate(all_rows) if is_row_invalid(row)]

    if not rows_to_fix:
        print(f"    No invalid rows to fix")
        return 0, 0

    print(f"    Fixing {len(rows_to_fix)} invalid rows with {max_workers} workers")

    processor = FixProcessor(csv_path, fieldnames, all_rows, max_workers)

    # Pre-populate valid rows
    fix_indices = {idx for idx, _ in rows_to_fix}
    for i, row in enumerate(all_rows):
        if i not in fix_indices:
            processor.results[i] = row

    processor.process_batch(rows_to_fix, dialect_key, llm)

    return len(rows_to_fix), processor.fixed_count


def main(
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"),
    max_workers: int = 4,
    specific_dialect: Optional[str] = None,
    model_key: str = "bedrock-mistral-large3",
):
    print("=" * 100)
    print("CounterHarm-SHIELD: REPAIR INVALID ROWS")
    print(f"Model: {model_key}")
    print(f"Workers: {max_workers}")
    print(f"Method: Single-shot repair + validation (2 API calls per cell)")
    print("=" * 100)

    llm = create_llm(model_key)

    dialect_folders = sorted(os.listdir(data_dir))
    dialect_folders = [d for d in dialect_folders if os.path.isdir(os.path.join(data_dir, d))]

    if specific_dialect:
        dialect_folders = [d for d in dialect_folders if d == specific_dialect]

    total_attempted = 0
    total_fixed = 0
    start_time = time.time()

    for dialect_folder in dialect_folders:
        if dialect_folder not in DIALECT_MAPPING:
            continue

        dialect_key = DIALECT_MAPPING[dialect_folder]
        dialect_path = os.path.join(data_dir, dialect_folder)

        csv_files = [f for f in os.listdir(dialect_path) if f.endswith('.csv')]
        invalid_count = 0
        for csv_file in csv_files:
            csv_path = os.path.join(dialect_path, csv_file)
            with open(csv_path, 'r', encoding='utf-8') as f:
                invalid_count += sum(1 for r in csv.DictReader(f) if is_row_invalid(r))

        if invalid_count == 0:
            continue

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_folder} -> {dialect_key} ({invalid_count} invalid rows)")
        print(f"{'=' * 80}")

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(dialect_path, csv_file)
            print(f"\n  Dataset: {csv_file}")

            try:
                attempted, fixed = fix_csv(csv_path, dialect_key, llm, max_workers=max_workers)
                total_attempted += attempted
                total_fixed += fixed
            except Exception as e:
                print(f"    ERROR: {str(e)[:60]}...")

    elapsed = time.time() - start_time
    rate = total_attempted / (elapsed / 60) if elapsed > 0 else 0

    print("\n" + "=" * 100)
    print("REPAIR COMPLETE")
    print(f"Total attempted: {total_attempted}")
    print(f"Total fixed: {total_fixed}")
    print(f"Fix rate: {total_fixed}/{total_attempted} ({int(total_fixed/total_attempted*100) if total_attempted else 0}%)")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {rate:.1f} rows/minute")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Repair invalid CounterHarm-SHIELD rows with single-shot fix"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dialect", type=str, help="Specific dialect to fix")
    parser.add_argument("--model", type=str, default="bedrock-mistral-large3",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model for repair (default: bedrock-mistral-large3)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "dia_llm"))

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        max_workers=args.workers,
        specific_dialect=args.dialect,
        model_key=args.model,
    )
