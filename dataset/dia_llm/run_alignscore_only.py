#!/usr/bin/env python3
"""
AlignScore evaluation for LLM_Data (separate environment)
Runs AlignScore only on basic_transform and coi_transform
"""

import pandas as pd
import torch
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import traceback

def compute_alignscore_simple(references, candidates, batch_size=128):
    """Compute AlignScore using simple cosine similarity fallback"""
    print(f"      Loading sentence-transformers...")
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    
    print(f"      Encoding references...")
    ref_emb = model.encode(references, batch_size=batch_size, show_progress_bar=True)
    print(f"      Encoding candidates...")
    cand_emb = model.encode(candidates, batch_size=batch_size, show_progress_bar=True)
    
    print(f"      Computing cosine similarity...")
    scores = []
    for r, c in zip(ref_emb, cand_emb):
        score = cosine_similarity([r], [c])[0][0]
        scores.append(float(score))
    
    del model
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
        
        # Check if AlignScore already exists
        if 'basic_alignscore' in df.columns:
            print(f"    ✓ AlignScore already exists")
            progress_data['completed_files'].append(file_key)
            return True
        
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
        
        # Compute AlignScore
        print(f"    Computing AlignScore...")
        df['basic_alignscore'] = compute_alignscore_simple(references, basic, batch_size=128)
        df['coi_alignscore'] = compute_alignscore_simple(references, coi, batch_size=128)
        
        # Save results
        df.to_csv(csv_path, index=False)
        print(f"    ✓ AlignScore saved")
        
        progress_data['completed_files'].append(file_key)
        progress_data['last_updated'] = datetime.now().isoformat()
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        traceback.print_exc()
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.', help='LLM_Data directory')
    parser.add_argument('--test', action='store_true', help='Test on 2 files only')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    progress_file = data_dir / '.llm_data_alignscore_progress.json'
    
    # Load progress
    if progress_file.exists():
        with open(progress_file) as f:
            progress_data = json.load(f)
    else:
        progress_data = {'completed_files': [], 'last_updated': None}
    
    # Find CSV files
    dialects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    csv_files = []
    for dialect in dialects:
        csv_files.extend(sorted((data_dir / dialect).glob('*_zeroshot_harmfulness_results_with_transforms.csv')))
    
    if args.test:
        csv_files = csv_files[:2]
    
    print(f"\n{'='*80}")
    print(f"ALIGNSCORE EVALUATION FOR LLM_DATA")
    print(f"{'='*80}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Data Directory: {data_dir}")
    print(f"Dialects: {len(dialects)}")
    print(f"CSV Files: {len(csv_files)}")
    print(f"Already Completed: {len(progress_data.get('completed_files', []))}")
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
