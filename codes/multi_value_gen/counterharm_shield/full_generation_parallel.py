#!/usr/bin/env python3
"""
Parallel generation script for CounterHarm-SHIELD benign sample generation
on Multi-VALUE transformed datasets.

Processes multiple rows concurrently, running the 6-chain CoI pipeline on
2 text columns per row: the original column and its _transformed counterpart.

Part of DIA-GUARD → Multi-VALUE → CounterHarm-SHIELD.
"""

import sys
import os
import csv
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add dia_llm counterharm_shield so we can import pipeline & models
_this_dir = os.path.dirname(os.path.abspath(__file__))
_dia_llm_cs = os.path.join(_this_dir, "..", "..", "dia_llm", "counterharm_shield")
_harm_shield = os.path.join(_this_dir, "..", "..", "dia_llm", "harm_shield")
sys.path.insert(0, _dia_llm_cs)
sys.path.insert(0, _harm_shield)

from models import AzureOpenAIBackend, BedrockBackend
from counterharm_pipeline import CounterHarmSHIELD

# ── Dataset → text column pairs ──
# Each entry maps a dataset keyword to a list of (original_col, transformed_col) pairs.
# CounterHarm-SHIELD will generate benign counterparts for both columns in each pair.
DATASET_TEXT_COLUMNS = {
    "advbench": [("prompt", "prompt_transformed")],
    "bipia": [("context", "context_transformed"), ("user_intent", "user_intent_transformed")],
    "cyberseceval": [("prompt", "prompt_transformed")],
    "do_not_answer": [("question", "question_transformed")],
    "forbiddent_questions": [("prompt", "prompt_transformed")],
    "harmBench": [("Behavior", "Behavior_transformed")],
    "injecagent": [("user_instruction", "user_instruction_transformed")],
    "jailbreakbench": [("goal", "goal_transformed")],
    "llmseceval": [
        ("llm_generated_nl_prompt", "llm_generated_nl_prompt_transformed"),
        ("manually_fixed_nl_prompt", "manually_fixed_nl_prompt_transformed"),
    ],
    "Salad_Bench": [("prompt", "prompt_transformed")],
    "securityeval": [("prompt", "prompt_transformed")],
    "Simple_Safety_Tests": [("prompt", "prompt_transformed")],
    "sorry_bench": [("turns", "turns_transformed")],
    "Toxic_Text": [("goal", "goal_transformed")],
    "Toxicity_Jigsaw": [("comment_text", "comment_text_transformed")],
}

# Same dialect mapping as Harm_Shield / dia_llm
DIALECT_MAPPING = {
    # North America
    "appalachian_english": "appalachian",
    "chicano_english": "chicano",
    "colloquial_american_english": "colloquial_american",
    "earlier_african_american_vernacular_english": "earlier_aave",
    "newfoundland_english": "newfoundland",
    "ozark_english": "ozark",
    "rural_african_american_vernacular_english": "rural_aave",
    "southeast_american_enclave_dialects": "southeast_enclave",
    "urban_african_american_vernacular_english": "urban_aave",
    # British Isles
    "channel_islands_english": "channel_islands",
    "east_anglian_english": "east_anglian",
    "english_dialects_in_the_north_of_england": "northern_england",
    "english_dialects_in_the_southeast_of_england": "southeast_england",
    "english_dialects_in_the_southwest_of_england": "southwest_england",
    "irish_english": "irish",
    "maltese_english": "maltese_english",
    "manx_english": "manx",
    "orkney_and_shetland_english": "orkney_shetland",
    "scottish_english": "scottish",
    "welsh_english": "welsh",
    # Caribbean
    "bahamian_english": "bahamian",
    "jamaican_english": "jamaican",
    # Africa
    "black_south_african_english": "south_african_black",
    "cameroon_english": "cameroon",
    "cape_flats_english": "cape_flats",
    "ghanaian_english": "ghanaian",
    "indian_south_african_english": "south_african_indian",
    "kenyan_english": "kenyan",
    "liberian_settler_english": "liberian_settler",
    "nigerian_english": "nigerian",
    "tanzanian_english": "tanzanian",
    "ugandan_english": "ugandan",
    "white_south_african_english": "south_african_white",
    # South/Southeast Asia
    "colloquial_singapore_english_singlish": "singapore",
    "hong_kong_english": "hong_kong",
    "indian_english": "indian",
    "malaysian_english": "malaysian",
    "pakistani_english": "pakistani",
    "philippine_english": "philippine",
    "sri_lankan_english": "sri_lankan",
    # Australia & Pacific
    "aboriginal_english": "aboriginal",
    "acrolectal_fiji_english": "fiji_acrolectal",
    "australian_english": "australian",
    "australian_vernacular_english": "australian_vernacular",
    "new_zealand_english": "new_zealand",
    "pure_fiji_english_basilectal_fijie": "fiji_basilectal",
    # Atlantic Islands
    "falkland_islands_english": "falkland",
    "st_helena_english": "st_helena",
    "tristan_da_cunha_english": "tristan",
}

POLICY_VIOLATION = "guardrail policy violation"

# ── Model configs (shared with dia_llm) ──
MODEL_CONFIGS = {
    "gpt4.1": {
        "deployment_name": "gpt-4.1",
        "base_url": "https://jsl-diaguard.openai.azure.com/openai/v1",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    },
    "deepseek": {
        "deployment_name": "DeepSeek-V3.2",
        "base_url": "https://ai-jsl57102192ai462716477695.services.ai.azure.com/openai/v1/",
        "api_key": os.getenv("AZURE_AI_API_KEY", ""),
    },
    "kimi": {
        "deployment_name": "Kimi-K2.5",
        "base_url": "https://ai-jsl57102192ai462716477695.services.ai.azure.com/openai/v1/",
        "api_key": os.getenv("AZURE_AI_API_KEY", ""),
    },
    "gemini": {
        "deployment_name": "gemini-2.5-flash-lite",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    "gemini2.5": {
        "deployment_name": "gemini-2.5-flash-lite",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    "gemini2.5flash": {
        "deployment_name": "gemini-2.5-flash",
        "project_id": "diaguard-new-project",
        "location": "us-central1",
    },
    # ── AWS Bedrock models ──
    "bedrock-deepseek": {
        "model_id": "deepseek.v3.2",
        "region": "us-east-1",
    },
    "bedrock-llama3-8b": {
        "model_id": "meta.llama3-8b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama3.3-70b": {
        "model_id": "us.meta.llama3-3-70b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama4-maverick": {
        "model_id": "us.meta.llama4-maverick-17b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-llama4-scout": {
        "model_id": "meta.llama4-scout-17b-instruct-v1:0",
        "region": "us-east-1",
    },
    "bedrock-mistral-large3": {
        "model_id": "mistral.mistral-large-3-675b-instruct",
        "region": "us-east-1",
    },
    "bedrock-claude-opus-4.6": {
        "model_id": "us.anthropic.claude-opus-4-6-v1",
        "region": "us-east-1",
    },
    "bedrock-claude-sonnet-4.6": {
        "model_id": "us.anthropic.claude-sonnet-4-6",
        "region": "us-east-1",
    },
    "bedrock-safeguard-120b": {
        "model_id": "openai.gpt-oss-safeguard-120b",
        "region": "us-east-1",
    },
    "bedrock-safeguard-20b": {
        "model_id": "openai.gpt-oss-safeguard-20b",
        "region": "us-east-1",
    },
    "bedrock-gpt-oss-120b": {
        "model_id": "openai.gpt-oss-120b-1:0",
        "region": "us-east-1",
    },
    "bedrock-gpt-oss-20b": {
        "model_id": "openai.gpt-oss-20b-1:0",
        "region": "us-east-1",
    },
    "bedrock-qwen3-32b": {
        "model_id": "qwen.qwen3-32b-v1:0",
        "region": "us-east-1",
    },
}


def create_llm(model_key: str = "gpt4.1"):
    """Create LLM backend for the specified model."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[model_key]
    if model_key in ("gemini", "gemini2.5", "gemini2.5flash"):
        from models import GeminiBackend
        return GeminiBackend(
            model=config["deployment_name"],
            use_vertex=True,
            project_id=config["project_id"],
            location=config["location"],
        )
    if model_key.startswith("bedrock-"):
        return BedrockBackend(
            model_id=config["model_id"],
            region=config["region"],
        )
    return AzureOpenAIBackend(
        deployment_name=config["deployment_name"],
        base_url=config["base_url"],
        api_key=config["api_key"],
    )


def detect_dataset_key(csv_filename: str) -> Optional[str]:
    """Match CSV filename to a DATASET_TEXT_COLUMNS key."""
    base = csv_filename.lower()
    for key in DATASET_TEXT_COLUMNS:
        if key.lower() in base:
            return key
    return None


def get_counterharm_col(original_col: str) -> str:
    """Build counterharm output column name from the source column name."""
    return f"counterharm_{original_col}"


def get_counterharm_columns(dataset_key: str) -> List[str]:
    """Return all counterharm output column names for a dataset, plus meta columns."""
    cols = []
    for orig_col, trans_col in DATASET_TEXT_COLUMNS[dataset_key]:
        cols.append(get_counterharm_col(orig_col))
        cols.append(get_counterharm_col(trans_col))
    cols.extend(["counterharm_score", "counterharm_validated", "counterharm_model"])
    return cols


def row_already_processed(row: Dict[str, Any], dataset_key: str) -> bool:
    """Check if row already has all CounterHarm outputs filled."""
    for orig_col, trans_col in DATASET_TEXT_COLUMNS[dataset_key]:
        for src_col in (orig_col, trans_col):
            out_col = get_counterharm_col(src_col)
            src_val = row.get(src_col, "").strip()
            out_val = row.get(out_col, "").strip()
            if not src_val or src_val == POLICY_VIOLATION:
                continue
            if out_val == "[FAILED]":
                continue
            if not out_val:
                return False
    return True


DIALECT_ADAPT_SYSTEM = (
    "You are a dialect adaptation agent for CounterHarm-SHIELD (DIA-GUARD). "
    "You convert standard benign text into dialectal variants. Return ONLY valid JSON."
)


def dialect_adapt(llm, benign_text: str, original_text: str, transformed_text: str, dialect: str) -> str:
    """Adapt a benign SAE text into the target dialect, mirroring how the
    original was transformed into the transformed version.

    Instead of running the full 6-chain pipeline independently on the
    transformed column, we take the already-generated benign text and
    apply the same dialect transformation pattern.
    """
    prompt = f"""TASK: Apply dialect transformation to this benign text so it mirrors the dialect style of the reference.

REFERENCE — original (SAE):
"{original_text}"

REFERENCE — dialect-transformed version:
"{transformed_text}"

BENIGN TEXT to transform into {dialect} dialect:
"{benign_text}"

RULES:
1. Apply the SAME dialect features visible in the reference transformed version (grammar, vocabulary, spelling, syntax)
2. Keep the SAME meaning, structure, and format as the benign text — only change the dialect
3. The output must be the same length (±2 words) as the benign text
4. Do NOT change the topic or content — only apply dialect features
5. If the original starts with "Write a..." the output must also start with the dialect equivalent (e.g., "Write one..." or similar)

Return ONLY JSON:
{{
  "dialect_text": "<the benign text rewritten in {dialect} dialect>"
}}"""

    try:
        raw = llm.generate(DIALECT_ADAPT_SYSTEM, prompt, max_tokens=1024, temperature=0.4)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        import json
        result = json.loads(raw)
        return result.get("dialect_text", "").strip()
    except Exception:
        return ""


def process_row(
    row: Dict[str, Any],
    row_idx: int,
    dialect_key: str,
    pipeline: CounterHarmSHIELD,
    dataset_key: str,
) -> Dict[str, Any]:
    """Process a single row: run 6-chain pipeline on the original column,
    then dialect-adapt the result for the _transformed column."""
    result = dict(row)

    scores = []
    all_validated = True

    for orig_col, trans_col in DATASET_TEXT_COLUMNS[dataset_key]:
        orig_out = get_counterharm_col(orig_col)
        trans_out = get_counterharm_col(trans_col)

        orig_text = row.get(orig_col, "").strip()
        trans_text = row.get(trans_col, "").strip()

        # ── Step 1: Generate benign for original column via full pipeline ──
        benign_original = row.get(orig_out, "").strip()
        if not benign_original:
            if not orig_text or orig_text == POLICY_VIOLATION:
                result[orig_out] = ""
            else:
                try:
                    state = pipeline.run(harmful_seed=orig_text, dialect=dialect_key)
                    output = state.refined_text or ""
                    if not output:
                        result[orig_out] = "[FAILED]"
                        all_validated = False
                    else:
                        result[orig_out] = output
                        benign_original = output
                        scores.append(state.harmlessness_score)
                        if not state.validated:
                            all_validated = False
                except Exception:
                    result[orig_out] = "[FAILED]"
                    all_validated = False

        # ── Step 2: Dialect-adapt benign original for the _transformed column ──
        existing_trans = row.get(trans_out, "").strip()
        if not existing_trans:
            if not trans_text or trans_text == POLICY_VIOLATION:
                result[trans_out] = ""
            elif not benign_original or benign_original == "[FAILED]":
                result[trans_out] = "[FAILED]"
                all_validated = False
            else:
                adapted = dialect_adapt(
                    pipeline.llm, benign_original, orig_text, trans_text, dialect_key
                )
                if adapted:
                    result[trans_out] = adapted
                else:
                    result[trans_out] = "[FAILED]"
                    all_validated = False

    avg_score = sum(scores) / len(scores) if scores else 0.0
    result["counterharm_score"] = f"{avg_score:.4f}"
    result["counterharm_validated"] = str(all_validated)
    result["counterharm_model"] = pipeline.llm.name

    return result


class ParallelProcessor:
    """Process rows in parallel with thread-safe saving."""

    def __init__(self, output_path: str, fieldnames: list, counterharm_cols: list, max_workers: int = 4):
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.counterharm_cols = counterharm_cols
        self.max_workers = max_workers
        self.results = {}
        self.lock = threading.Lock()
        self.processed_count = 0
        self.start_time = time.time()

    def save_results(self, all_rows: list):
        """Save all results in order, preserving unprocessed rows."""
        with self.lock:
            output_rows = []
            for i, row in enumerate(all_rows):
                if i in self.results:
                    output_rows.append(self.results[i])
                else:
                    row_copy = dict(row)
                    for col in self.counterharm_cols:
                        if col not in row_copy:
                            row_copy[col] = ""
                    output_rows.append(row_copy)

            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(output_rows)

    def process_batch(
        self,
        rows_to_process: List[tuple],
        all_rows: list,
        dialect_key: str,
        llm,
        dataset_key: str,
    ):
        """Process a batch of rows in parallel."""
        def worker(idx_row_tuple):
            idx, row = idx_row_tuple
            pipeline = CounterHarmSHIELD(llm=llm, verbose=False)
            try:
                result = process_row(row, idx, dialect_key, pipeline, dataset_key)
                return (idx, result, None)
            except Exception as e:
                row_copy = dict(row)
                for col in self.counterharm_cols:
                    row_copy[col] = ""
                return (idx, row_copy, str(e))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(worker, item): item for item in rows_to_process}

            for future in as_completed(futures):
                idx, result, error = future.result()

                with self.lock:
                    self.results[idx] = result
                    self.processed_count += 1

                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / (elapsed / 60) if elapsed > 0 else 0

                    if error:
                        print(f"    [{self.processed_count}] idx={idx} ERROR: {error[:50]}")
                    else:
                        validated = result.get("counterharm_validated", "?")
                        score = result.get("counterharm_score", "?")
                        print(f"    [{self.processed_count}] idx={idx} "
                              f"score={score} valid={validated} ({rate:.1f} rows/min)")

                self.save_results(all_rows)


def process_csv(
    input_path: str,
    output_path: str,
    dialect_key: str,
    llm,
    dataset_key: str,
    max_workers: int = 4,
    test_mode: bool = False,
    test_rows: int = 1,
):
    """Process a single CSV file with parallel workers."""
    counterharm_cols = get_counterharm_columns(dataset_key)
    text_pairs = DATASET_TEXT_COLUMNS[dataset_key]
    # Use first original column as the resume key
    first_orig_col = text_pairs[0][0]

    # Load existing output if resuming
    existing_rows = {}
    if os.path.exists(output_path) and output_path != input_path:
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get(first_orig_col, "")
                existing_rows[key] = row
    elif os.path.exists(output_path):
        # In-place mode: check if counterharm columns already exist
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and any(c in reader.fieldnames for c in counterharm_cols):
                for row in reader:
                    key = row.get(first_orig_col, "")
                    existing_rows[key] = row

    # Load input CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fieldnames = list(reader.fieldnames)
        input_rows = list(reader)

    # Determine output fieldnames
    output_fieldnames = list(input_fieldnames)
    for col in counterharm_cols:
        if col not in output_fieldnames:
            output_fieldnames.append(col)

    # Determine rows to process
    rows_to_process = []
    skipped_count = 0

    for i, row in enumerate(input_rows):
        key = row.get(first_orig_col, "")

        if key in existing_rows and row_already_processed(existing_rows[key], dataset_key):
            skipped_count += 1
            continue

        if test_mode and len(rows_to_process) >= test_rows:
            continue

        rows_to_process.append((i, row))

    if not rows_to_process:
        print(f"    All {skipped_count} rows already processed")
        return 0, skipped_count

    print(f"    Processing {len(rows_to_process)} rows ({skipped_count} skipped) with {max_workers} workers")
    print(f"    Text columns: {[(o, t) for o, t in text_pairs]}")

    processor = ParallelProcessor(output_path, output_fieldnames, counterharm_cols, max_workers)

    # Pre-populate with existing results
    for i, row in enumerate(input_rows):
        key = row.get(first_orig_col, "")
        if key in existing_rows and row_already_processed(existing_rows[key], dataset_key):
            processor.results[i] = existing_rows[key]

    processor.process_batch(rows_to_process, input_rows, dialect_key, llm, dataset_key)

    return len(rows_to_process), skipped_count


def main(
    data_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "multi_value"),
    max_workers: int = 4,
    test_mode: bool = False,
    test_rows: int = 1,
    specific_dialect: Optional[str] = None,
    specific_dataset: Optional[str] = None,
    model_key: str = "gpt4.1",
):
    """Main function to run CounterHarm-SHIELD benign generation on Multi-VALUE data.

    Reads Multi-VALUE CSVs and writes output with counterharm columns appended.
    For each row, generates benign counterparts for the original and _transformed
    text columns using the 6-chain CoI pipeline (ToxiCraft + PromptSafe + FIZLE).

    ~12 API calls per row for single-pair datasets (6 chains x 2 columns).
    ~24 API calls per row for dual-pair datasets (6 chains x 4 columns).
    """
    print("=" * 100)
    print("CounterHarm-SHIELD: Multi-VALUE Benign Sample Generation")
    print("DIA-GUARD -> Multi-VALUE -> CounterHarm-SHIELD")
    print(f"Data dir (in-place): {data_dir}")
    print(f"Model: {model_key}")
    print(f"Workers: {max_workers}")
    print(f"Test mode: {test_mode} (rows: {test_rows})")
    print(f"Datasets: {list(DATASET_TEXT_COLUMNS.keys())}")
    print(f"Pipeline: 6-chain CoI (ToxiCraft + PromptSafe + FIZLE)")
    print("=" * 100)

    llm = create_llm(model_key)

    dialect_folders = sorted(os.listdir(data_dir))
    dialect_folders = [d for d in dialect_folders if os.path.isdir(os.path.join(data_dir, d))]

    if specific_dialect:
        dialect_folders = [d for d in dialect_folders if d == specific_dialect]

    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    for dialect_folder in dialect_folders:
        if dialect_folder not in DIALECT_MAPPING:
            print(f"\n[SKIP] {dialect_folder}: No dialect mapping")
            continue

        dialect_key = DIALECT_MAPPING[dialect_folder]
        dialect_path = os.path.join(data_dir, dialect_folder)

        print(f"\n{'=' * 80}")
        print(f"DIALECT: {dialect_folder} -> {dialect_key}")
        print(f"{'=' * 80}")

        csv_files = [f for f in os.listdir(dialect_path) if f.endswith('.csv')]

        if specific_dataset:
            csv_files = [f for f in csv_files if specific_dataset in f]

        for csv_file in sorted(csv_files):
            csv_path = os.path.join(dialect_path, csv_file)

            dataset_key = detect_dataset_key(csv_file)
            if not dataset_key:
                print(f"\n  [SKIP] {csv_file}: No column mapping found")
                continue

            print(f"\n  Dataset: {csv_file} (key={dataset_key})")

            try:
                processed, skipped = process_csv(
                    csv_path, csv_path, dialect_key, llm,
                    dataset_key=dataset_key,
                    max_workers=max_workers,
                    test_mode=test_mode, test_rows=test_rows,
                )
                total_processed += processed
                total_skipped += skipped
            except Exception as e:
                print(f"    ERROR: {str(e)[:60]}...")

            if test_mode:
                break

        if test_mode:
            break

    elapsed = time.time() - start_time
    rate = total_processed / (elapsed / 60) if elapsed > 0 else 0

    print("\n" + "=" * 100)
    print("CounterHarm-SHIELD (Multi-VALUE) GENERATION COMPLETE")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped (already done): {total_skipped}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Average rate: {rate:.1f} rows/minute")
    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CounterHarm-SHIELD: Parallel benign sample generation for Multi-VALUE (in-place)"
    )
    parser.add_argument("--test", action="store_true", help="Test mode (limited rows)")
    parser.add_argument("--test-rows", type=int, default=1, help="Number of rows in test mode")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dialect", type=str, help="Specific dialect folder to process")
    parser.add_argument("--dataset", type=str, help="Specific dataset name pattern to process")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "dataset", "multi_value"))
    parser.add_argument("--model", type=str, default="gpt4.1",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use for generation (default: gpt4.1)")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        max_workers=args.workers,
        test_mode=args.test,
        test_rows=args.test_rows,
        specific_dialect=args.dialect,
        specific_dataset=args.dataset,
        model_key=args.model,
    )
