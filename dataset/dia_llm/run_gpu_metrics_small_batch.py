#!/usr/bin/env python3
"""
Sequential GPU Metrics Evaluation for LLM_Data - Small Batch Version
For files that failed with larger batch sizes
"""

import pandas as pd
import torch
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import traceback
import sys

# Smaller batch sizes to avoid OOM
BATCH_SIZES = {
    "BERTScore": 32,  # Reduced from 128
    "BARTScore": 16,  # Reduced from 32
}

def compute_bertscore(references, candidates):
    """Compute BERTScore"""
    print(f"      Loading BERTScore model...")
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type='microsoft/deberta-xlarge-mnli',
        device='cuda',
        batch_size=BATCH_SIZES['BERTScore'],
        rescale_with_baseline=True,
        lang='en'
    )

    print(f"      Computing BERTScore (batch_size={BATCH_SIZES['BERTScore']})...")
    P, R, F1 = scorer.score(candidates, references)
    scores = F1.cpu().numpy().tolist()

    del scorer
    torch.cuda.empty_cache()
    return scores

def compute_bartscore(references, candidates):
    """Compute BARTScore"""
    print(f"      Loading BARTScore model...")
    from transformers import BartForConditionalGeneration, BartTokenizer

    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to('cuda')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model.eval()

    batch_size = BATCH_SIZES['BARTScore']
    scores = []

    print(f"      Computing BARTScore (batch_size={batch_size})...")
    for i in tqdm(range(0, len(references), batch_size), desc="      BARTScore batches", leave=False):
        batch_refs = references[i:i+batch_size]
        batch_cands = candidates[i:i+batch_size]

        for ref, cand in zip(batch_refs, batch_cands):
            inputs = tokenizer(
                [ref], [cand],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to('cuda')

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                score = -outputs.loss.item()
                scores.append(score)

    del model, tokenizer
    torch.cuda.empty_cache()
    return scores

def process_csv_file(csv_path, progress_data):
    """Process a single CSV file"""
    try:
        df = pd.read_csv(csv_path)
        file_key = str(csv_path)

        if file_key in progress_data.get('completed_files', []):
            print(f"  ✓ Skip (completed): {csv_path.name}")
            return True

        print(f"\n  Processing: {csv_path.name} ({len(df)} rows)")

        # Check required columns
        if 'original_input' not in df.columns or 'basic_transform' not in df.columns or 'coi_transform' not in df.columns:
            print(f"    ✗ Missing required columns")
            return False

        # Filter valid rows
        df = df[df['basic_transform'].notna() & df['coi_transform'].notna()].copy()
        if len(df) == 0:
            print(f"    ✗ No valid transforms")
            return False

        references = df['original_input'].fillna('').astype(str).tolist()
        basic = df['basic_transform'].fillna('').astype(str).tolist()
        coi = df['coi_transform'].fillna('').astype(str).tolist()

        # Compute BERTScore
        if 'basic_bertscore' not in df.columns:
            print(f"    Computing BERTScore...")
            df['basic_bertscore'] = compute_bertscore(references, basic)
            df['coi_bertscore'] = compute_bertscore(references, coi)
            df.to_csv(csv_path, index=False)
            print(f"    ✓ BERTScore saved")

        # Compute BARTScore
        if 'basic_bartscore' not in df.columns:
            print(f"    Computing BARTScore...")
            df['basic_bartscore'] = compute_bartscore(references, basic)
            df['coi_bartscore'] = compute_bartscore(references, coi)
            df.to_csv(csv_path, index=False)
            print(f"    ✓ BARTScore saved")

        progress_data['completed_files'].append(file_key)
        progress_data['last_updated'] = datetime.now().isoformat()

        print(f"    ✓ Complete: {csv_path.name}")
        return True

    except Exception as e:
        print(f"    ✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.', help='LLM_Data directory')
    parser.add_argument('--files', nargs='+', help='Specific files to process')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    progress_file = data_dir / '.llm_data_gpu_progress.json'

    # Load progress
    if progress_file.exists():
        with open(progress_file) as f:
            progress_data = json.load(f)
    else:
        progress_data = {'completed_files': [], 'last_updated': None}

    # Get files to process
    if args.files:
        csv_files = [Path(f) for f in args.files]
    else:
        # Find incomplete files
        dialects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        all_files = []
        for dialect in dialects:
            all_files.extend(sorted((data_dir / dialect).glob('*_zeroshot_harmfulness_results_with_transforms.csv')))

        completed = set(progress_data.get('completed_files', []))
        csv_files = [f for f in all_files if str(f) not in completed]

    print(f"\n{'='*80}")
    print(f"GPU METRICS EVALUATION (SMALL BATCH) FOR LLM_DATA")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Data Directory: {data_dir}")
    print(f"CSV Files to Process: {len(csv_files)}")
    print(f"Batch Sizes: {BATCH_SIZES}")
    print(f"{'='*80}\n")

    # Process files
    successful = 0
    failed = 0

    for csv_file in csv_files:
        result = process_csv_file(csv_file, progress_data)
        if result:
            successful += 1
        else:
            failed += 1

        # Save progress every file
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total Processed: {successful + failed}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
