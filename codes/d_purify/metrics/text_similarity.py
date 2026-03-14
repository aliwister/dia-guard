#!/usr/bin/env python3
"""
Text Similarity Metrics (CPU-based)
D-PURiFY | DIA-GUARD -> Dia-LLM

Computes BLEU, METEOR, ROUGE-L, and DiffLib sequence matching
between original and dialect-transformed text pairs.
"""

import difflib
import pandas as pd
from typing import List, Tuple


class TextSimilarityMetrics:
    """CPU-based text similarity metrics for dialect transformation evaluation."""

    def __init__(self):
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """Download required NLTK data."""
        import nltk
        for resource, path in [
            ("punkt", "tokenizers/punkt"),
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("wordnet", "corpora/wordnet"),
            ("omw-1.4", "corpora/omw-1.4"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)

    def compute_bleu(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute sentence-level BLEU scores."""
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        smooth = SmoothingFunction().method1
        scores = []
        for ref, cand in zip(references, candidates):
            try:
                ref_str = str(ref) if pd.notna(ref) else ""
                cand_str = str(cand) if pd.notna(cand) else ""
                if not ref_str or not cand_str:
                    scores.append(0.0)
                    continue
                ref_tokens = word_tokenize(ref_str.lower())
                cand_tokens = word_tokenize(cand_str.lower())
                score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smooth)
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        return scores

    def compute_meteor(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute METEOR scores."""
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize

        scores = []
        for ref, cand in zip(references, candidates):
            try:
                ref_str = str(ref) if pd.notna(ref) else ""
                cand_str = str(cand) if pd.notna(cand) else ""
                if not ref_str or not cand_str:
                    scores.append(0.0)
                    continue
                ref_tokens = word_tokenize(ref_str.lower())
                cand_tokens = word_tokenize(cand_str.lower())
                score = meteor_score([ref_tokens], cand_tokens)
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        return scores

    def compute_rouge_l(self, references: List[str], candidates: List[str]) -> List[Tuple[float, float, float]]:
        """Compute ROUGE-L (F1, Precision, Recall) scores."""
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        results = []
        for ref, cand in zip(references, candidates):
            try:
                ref_str = str(ref) if pd.notna(ref) else ""
                cand_str = str(cand) if pd.notna(cand) else ""
                if not ref_str or not cand_str:
                    results.append((0.0, 0.0, 0.0))
                    continue
                score = scorer.score(ref_str, cand_str)["rougeL"]
                results.append((score.fmeasure, score.precision, score.recall))
            except Exception:
                results.append((0.0, 0.0, 0.0))
        return results

    def compute_difflib(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute DiffLib sequence matching ratio."""
        scores = []
        for ref, cand in zip(references, candidates):
            try:
                ref_str = str(ref) if pd.notna(ref) else ""
                cand_str = str(cand) if pd.notna(cand) else ""
                if not ref_str or not cand_str:
                    scores.append(0.0)
                    continue
                score = difflib.SequenceMatcher(None, ref_str, cand_str).ratio()
                scores.append(float(score))
            except Exception:
                scores.append(0.0)
        return scores

    def compute_all(self, references: List[str], candidates: List[str], prefix: str = "basic") -> dict:
        """Compute all text similarity metrics and return as a dict of column_name -> scores."""
        results = {}
        results[f"{prefix}_bleu"] = self.compute_bleu(references, candidates)
        results[f"{prefix}_meteor"] = self.compute_meteor(references, candidates)

        rouge_scores = self.compute_rouge_l(references, candidates)
        results[f"{prefix}_rouge_l"] = [s[0] for s in rouge_scores]
        results[f"{prefix}_rouge_l_precision"] = [s[1] for s in rouge_scores]
        results[f"{prefix}_rouge_l_recall"] = [s[2] for s in rouge_scores]

        results[f"{prefix}_difflib"] = self.compute_difflib(references, candidates)
        return results
