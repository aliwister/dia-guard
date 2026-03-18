#!/usr/bin/env python3
"""
Parallel GPU Metrics Evaluation for LLM_Data
Runs 2 metrics simultaneously on different GPU memory regions
Maximizes T4 16GB utilization
"""

import pandas as pd
import torch
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import traceback
from multiprocessing import Process, Queue
import time

# Memory allocation for parallel execution
# BERTScore + BARTScore OR AlignScore + METEOR can run together
PARALLEL_CONFIGS = [
    {
        'name': 'bert_bart',
        'metrics': ['bertscore', 'bartscore'],
        'batch_sizes': {'bertscore': 64, 'bartscore': 16}  # Conservative for parallel
    },
    {
        'name': 'align_only',
        'metrics': ['alignscore'],
        'batch_sizes': {'alignscore': 128}  # Can use more memory alone
    }
]

def compute_bertscore_batch(references, candidates, batch_size=64, device='cuda:0'):
    """Compute BERTScore"""
    from bert_score import BERTScorer
    scorer = BERTScorer(
        model_type='microsoft/deberta-xlarge-mnli',
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=True
    )
    P, R, F1 = scorer.score(candidates, references)
    return F1.cpu().numpy().tolist()

def compute_bartscore_batch(references, candidates, batch_size=16, device='cuda:0'):
    """Compute BARTScore"""
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large-cnn'
    ).to(device)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model.eval()
    
    scores = []
    for i in range(0, len(references), batch_size):
        batch_refs = references[i:i+batch_size]
        batch_cands = candidates[i:i+batch_size]
        
        batch_scores = []
        for ref, cand in zip(batch_refs, batch_cands):
            inputs = tokenizer(
                [ref], [cand],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs['input_ids'])
                score = -outputs.loss.item()
                batch_scores.append(score)
        
        scores.extend(batch_scores)
    
    del model
    torch.cuda.empty_cache()
    return scores

def compute_alignscore_batch(references, candidates, batch_size=128, device='cuda:0'):
    """Compute AlignScore"""
    try:
        from alignscore import AlignScore
        scorer = AlignScore(
            model='roberta-large',
            batch_size=batch_size,
            device=device,
            ckpt_path=None,
            evaluation_mode='nli_sp'
        )
        scores = scorer.score(contexts=references, claims=candidates)
        return scores
    except Exception as e:
        print(f"AlignScore failed, using fallback: {e}")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        ref_emb = model.encode(references, batch_size=batch_size, show_progress_bar=False)
        cand_emb = model.encode(candidates, batch_size=batch_size, show_progress_bar=False)
        
        scores = []
        for r, c in zip(ref_emb, cand_emb):
            score = cosine_similarity([r], [c])[0][0]
            scores.append(float(score))
        
        del model
        torch.cuda.empty_cache()
        return scores

def process_file_with_metric(csv_path, metric_name, references, basic, coi, batch_size, device, result_queue):
    """Process a single metric for a file"""
    try:
        print(f"    [{metric_name}] Starting on {device}...")
        
        if metric_name == 'bertscore':
            basic_scores = compute_bertscore_batch(references, basic, batch_size, device)
            coi_scores = compute_bertscore_batch(references, coi, batch_size, device)
        elif metric_name == 'bartscore':
            basic_scores = compute_bartscore_batch(references, basic, batch_size, device)
            coi_scores = compute_bartscore_batch(references, coi, batch_size, device)
        elif metric_name == 'alignscore':
            basic_scores = compute_alignscore_batch(references, basic, batch_size, device)
            coi_scores = compute_alignscore_batch(references, coi, batch_size, device)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        result_queue.put({
            'metric': metric_name,
            'basic_scores': basic_scores,
            'coi_scores': coi_scores,
            'success': True
        })
        print(f"    [{metric_name}] ✓ Complete")
        
    except Exception as e:
        print(f"    [{metric_name}] ✗ Error: {e}")
        traceback.print_exc()
        result_queue.put({'metric': metric_name, 'success': False, 'error': str(e)})

def process_csv_file(csv_path, progress_data):
    """Process a single CSV file with parallel metrics"""
    try:
        df = pd.read_csv(csv_path)
        file_key = str(csv_path)
        
        if file_key in progress_data.get('completed_files', []):
            print(f"  ✓ Already processed: {csv_path.name}")
            return True
        
        print(f"\n  Processing: {csv_path.name} ({len(df)} rows)")
        
        # Check required columns
        required_cols = ['original_input', 'basic_transform', 'coi_transform']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"    ✗ Missing columns: {missing}")
            return False
        
        # Filter valid rows
        df = df[df['basic_transform'].notna() & df['coi_transform'].notna()].copy()
        if len(df) == 0:
            print(f"    ✗ No valid transforms")
            return False
        
        references = df['original_input'].fillna('').astype(str).tolist()
        basic = df['basic_transform'].fillna('').astype(str).tolist()
        coi = df['coi_transform'].fillna('').astype(str).tolist()
        
        # Check which metrics are missing
        metrics_needed = []
        if 'basic_bertscore' not in df.columns:
            metrics_needed.append('bertscore')
        if 'basic_bartscore' not in df.columns:
            metrics_needed.append('bartscore')
        if 'basic_alignscore' not in df.columns:
            metrics_needed.append('alignscore')
        
        if not metrics_needed:
            print(f"    ✓ All metrics already computed")
            progress_data['completed_files'].append(file_key)
            return True
        
        print(f"    Metrics needed: {metrics_needed}")
        
        # Strategy: Run 2 at a time if possible
        # BERTScore (large) + BARTScore (large) can run together with reduced batch sizes
        # AlignScore can run alone with larger batch size
        
        if 'bertscore' in metrics_needed and 'bartscore' in metrics_needed:
            # Run both in parallel
            print("    Running BERTScore + BARTScore in parallel...")
            result_queue = Queue()
            
            p1 = Process(target=process_file_with_metric, args=(
                csv_path, 'bertscore', references, basic, coi, 64, 'cuda:0', result_queue
            ))
            p2 = Process(target=process_file_with_metric, args=(
                csv_path, 'bartscore', references, basic, coi, 16, 'cuda:0', result_queue
            ))
            
            p1.start()
            time.sleep(2)  # Stagger start
            p2.start()
            
            p1.join()
            p2.join()
            
            # Collect results
            for _ in range(2):
                result = result_queue.get()
                if result['success']:
                    metric = result['metric']
                    df[f'basic_{metric}'] = result['basic_scores']
                    df[f'coi_{metric}'] = result['coi_scores']
            
            metrics_needed.remove('bertscore')
            metrics_needed.remove('bartscore')
        
        # Run remaining metrics sequentially with full batch size
        for metric in metrics_needed:
            print(f"    Running {metric}...")
            if metric == 'bertscore':
                df['basic_bertscore'] = compute_bertscore_batch(references, basic, 128, 'cuda:0')
                df['coi_bertscore'] = compute_bertscore_batch(references, coi, 128, 'cuda:0')
            elif metric == 'bartscore':
                df['basic_bartscore'] = compute_bartscore_batch(references, basic, 32, 'cuda:0')
                df['coi_bartscore'] = compute_bartscore_batch(references, coi, 32, 'cuda:0')
            elif metric == 'alignscore':
                df['basic_alignscore'] = compute_alignscore_batch(references, basic, 128, 'cuda:0')
                df['coi_alignscore'] = compute_alignscore_batch(references, coi, 128, 'cuda:0')
        
        # Save results
        df.to_csv(csv_path, index=False)
        print(f"    ✓ Saved: {csv_path.name}")
        
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
    parser.add_argument('--dialects', nargs='+', default=None, help='Specific dialects')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    progress_file = data_dir / '.llm_data_gpu_progress.json'
    
    # Load progress
    if progress_file.exists():
        with open(progress_file) as f:
            progress_data = json.load(f)
    else:
        progress_data = {'completed_files': [], 'last_updated': None}
    
    # Find CSV files
    if args.dialects:
        dialects = args.dialects
    else:
        dialects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    csv_files = []
    for dialect in dialects:
        dialect_dir = data_dir / dialect
        if dialect_dir.exists():
            csv_files.extend(sorted(dialect_dir.glob('*.csv')))
    
    print(f"\n{'='*80}")
    print(f"PARALLEL GPU METRICS EVALUATION FOR LLM_DATA")
    print(f"{'='*80}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Data Directory: {data_dir}")
    print(f"Dialects: {len(dialects)}")
    print(f"CSV Files: {len(csv_files)}")
    print(f"Metrics: BERTScore, BARTScore, AlignScore (parallel execution)")
    print(f"{'='*80}\n")
    
    # Process files
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        if str(csv_file) in progress_data.get('completed_files', []):
            continue
        
        result = process_csv_file(csv_file, progress_data)
        if result:
            successful += 1
        else:
            failed += 1
        
        # Save progress periodically
        if (successful + failed) % 3 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
    
    # Save final progress
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(csv_files)}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
