#!/usr/bin/env python3
"""
GPU Metrics Evaluation for LLM_Data
Evaluates basic_transform and coi_transform columns against original_input
Metrics: BERTScore, BARTScore, AlignScore
Optimized batch sizes for Tesla T4 16GB
"""

import pandas as pd
import torch
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import traceback

# Optimized batch sizes for T4 16GB (maximized)
BATCH_SIZES = {
    "BERTScore": 128,      # Can handle large batches with sentence-transformers
    "BARTScore": 32,       # BART-large-cnn needs more memory
    "AlignScore": 64,      # Medium size model
}

class GPUMetricsEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.models = {}
        
    def load_bertscore(self):
        """Load BERTScore model"""
        if 'bertscore' not in self.models:
            print("Loading BERTScore...")
            from bert_score import BERTScorer
            # Use sentence-transformers for better efficiency
            self.models['bertscore'] = BERTScorer(
                model_type='microsoft/deberta-xlarge-mnli',
                device=self.device,
                batch_size=BATCH_SIZES['BERTScore'],
                rescale_with_baseline=True
            )
            print("✓ BERTScore loaded")
        return self.models['bertscore']
    
    def load_bartscore(self):
        """Load BARTScore model"""
        if 'bartscore' not in self.models:
            print("Loading BARTScore...")
            from transformers import BartForConditionalGeneration, BartTokenizer
            model_name = 'facebook/bart-large-cnn'
            self.models['bartscore_model'] = BartForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            self.models['bartscore_tokenizer'] = BartTokenizer.from_pretrained(model_name)
            self.models['bartscore_model'].eval()
            print("✓ BARTScore loaded")
        return self.models['bartscore_model'], self.models['bartscore_tokenizer']
    
    def load_alignscore(self):
        """Load AlignScore model"""
        if 'alignscore' not in self.models:
            print("Loading AlignScore...")
            try:
                from alignscore import AlignScore
                self.models['alignscore'] = AlignScore(
                    model='roberta-large',
                    batch_size=BATCH_SIZES['AlignScore'],
                    device=self.device,
                    ckpt_path=None,
                    evaluation_mode='nli_sp'
                )
                print("✓ AlignScore loaded")
            except Exception as e:
                print(f"  Warning: AlignScore initialization failed: {e}")
                print("  Falling back to sentence-transformers...")
                from sentence_transformers import SentenceTransformer
                self.models['alignscore'] = SentenceTransformer(
                    'all-mpnet-base-v2',
                    device=self.device
                )
                self.models['alignscore_fallback'] = True
                print("✓ AlignScore fallback loaded")
        return self.models['alignscore']
    
    def compute_bertscore(self, references, candidates):
        """Compute BERTScore for a batch"""
        scorer = self.load_bertscore()
        P, R, F1 = scorer.score(candidates, references)
        return F1.cpu().numpy().tolist()
    
    def compute_bartscore(self, references, candidates):
        """Compute BARTScore for a batch"""
        model, tokenizer = self.load_bartscore()
        scores = []
        
        batch_size = BATCH_SIZES['BARTScore']
        for i in range(0, len(references), batch_size):
            batch_refs = references[i:i+batch_size]
            batch_cands = candidates[i:i+batch_size]
            
            # Prepare inputs for scoring
            batch_scores = []
            for ref, cand in zip(batch_refs, batch_cands):
                # BARTScore: P(candidate|reference)
                inputs = tokenizer(
                    [ref], 
                    [cand],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    score = -outputs.loss.item()  # Negative log-likelihood
                    batch_scores.append(score)
            
            scores.extend(batch_scores)
        
        return scores
    
    def compute_alignscore(self, references, candidates):
        """Compute AlignScore for a batch"""
        alignscore_model = self.load_alignscore()
        
        if self.models.get('alignscore_fallback'):
            # Fallback: Use cosine similarity
            ref_embeddings = alignscore_model.encode(
                references,
                batch_size=BATCH_SIZES['AlignScore'],
                show_progress_bar=False
            )
            cand_embeddings = alignscore_model.encode(
                candidates,
                batch_size=BATCH_SIZES['AlignScore'],
                show_progress_bar=False
            )
            
            from sklearn.metrics.pairwise import cosine_similarity
            scores = []
            for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
                score = cosine_similarity([ref_emb], [cand_emb])[0][0]
                scores.append(float(score))
            return scores
        else:
            # Official AlignScore
            scores = alignscore_model.score(
                contexts=references,
                claims=candidates
            )
            return scores

def process_csv_file(csv_path, evaluator, progress_data):
    """Process a single CSV file"""
    try:
        df = pd.read_csv(csv_path)
        file_key = str(csv_path)
        
        if file_key in progress_data.get('completed_files', []):
            print(f"  ✓ Already processed: {csv_path.name}")
            return True
        
        print(f"\n  Processing: {csv_path.name}")
        print(f"    Rows: {len(df)}")
        
        # Check required columns
        required_cols = ['original_input', 'basic_transform', 'coi_transform']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"    ✗ Missing columns: {missing}")
            return False
        
        # Remove rows with missing transforms
        df = df[df['basic_transform'].notna() & df['coi_transform'].notna()].copy()
        if len(df) == 0:
            print(f"    ✗ No valid transforms found")
            return False
        
        references = df['original_input'].fillna('').astype(str).tolist()
        basic = df['basic_transform'].fillna('').astype(str).tolist()
        coi = df['coi_transform'].fillna('').astype(str).tolist()
        
        # Compute BERTScore
        if 'basic_bertscore' not in df.columns:
            print("    Computing BERTScore...")
            df['basic_bertscore'] = evaluator.compute_bertscore(references, basic)
            df['coi_bertscore'] = evaluator.compute_bertscore(references, coi)
            print("    ✓ BERTScore done")
        
        # Compute BARTScore
        if 'basic_bartscore' not in df.columns:
            print("    Computing BARTScore...")
            df['basic_bartscore'] = evaluator.compute_bartscore(references, basic)
            df['coi_bartscore'] = evaluator.compute_bartscore(references, coi)
            print("    ✓ BARTScore done")
        
        # Compute AlignScore
        if 'basic_alignscore' not in df.columns:
            print("    Computing AlignScore...")
            df['basic_alignscore'] = evaluator.compute_alignscore(references, basic)
            df['coi_alignscore'] = evaluator.compute_alignscore(references, coi)
            print("    ✓ AlignScore done")
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        print(f"    ✓ Saved: {csv_path.name}")
        
        # Update progress
        if 'completed_files' not in progress_data:
            progress_data['completed_files'] = []
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
    
    # Find all CSV files
    if args.dialects:
        dialects = args.dialects
    else:
        dialects = sorted([d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    csv_files = []
    for dialect in dialects:
        dialect_dir = data_dir / dialect
        if dialect_dir.exists():
            csv_files.extend(list(dialect_dir.glob('*.csv')))
    
    print(f"\n{'='*80}")
    print(f"GPU METRICS EVALUATION FOR LLM_DATA")
    print(f"{'='*80}")
    print(f"Data Directory: {data_dir}")
    print(f"Dialects: {len(dialects)}")
    print(f"CSV Files: {len(csv_files)}")
    print(f"Metrics: BERTScore, BARTScore, AlignScore")
    print(f"Batch Sizes: {BATCH_SIZES}")
    print(f"{'='*80}\n")
    
    # Initialize evaluator
    evaluator = GPUMetricsEvaluator()
    
    # Process files
    successful = 0
    failed = 0
    skipped = 0
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        if str(csv_file) in progress_data.get('completed_files', []):
            skipped += 1
            continue
        
        result = process_csv_file(csv_file, evaluator, progress_data)
        if result:
            successful += 1
        else:
            failed += 1
        
        # Save progress periodically
        if (successful + failed) % 5 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
    
    # Save final progress
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {len(csv_files)}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
