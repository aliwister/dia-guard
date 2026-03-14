#!/usr/bin/env python3
"""
Neural/Semantic Metrics (GPU-based)
D-PURiFY | DIA-GUARD -> Dia-LLM

Computes BERTScore, BARTScore, and AlignScore for evaluating
semantic preservation in dialect transformations.
"""

import torch
import pandas as pd
from typing import List, Optional


# Optimized batch sizes for T4 16GB
BATCH_SIZES = {
    "BERTScore": 128,
    "BARTScore": 32,
    "AlignScore": 64,
}


class NeuralMetrics:
    """GPU-based neural evaluation metrics."""

    def __init__(self, device: Optional[str] = None, batch_sizes: Optional[dict] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_sizes = batch_sizes or BATCH_SIZES
        self._models = {}
        print(f"[D-PURiFY] NeuralMetrics using device: {self.device}")
        if torch.cuda.is_available():
            print(f"[D-PURiFY] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[D-PURiFY] GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.2f} GB")

    def _load_bertscore(self):
        if "bertscore" not in self._models:
            from bert_score import BERTScorer
            self._models["bertscore"] = BERTScorer(
                model_type="microsoft/deberta-xlarge-mnli",
                device=self.device,
                batch_size=self.batch_sizes["BERTScore"],
                rescale_with_baseline=True,
                lang="en",
            )
        return self._models["bertscore"]

    def _load_bartscore(self):
        if "bartscore" not in self._models:
            from transformers import BartForConditionalGeneration, BartTokenizer
            model_name = "facebook/bart-large-cnn"
            self._models["bartscore_model"] = BartForConditionalGeneration.from_pretrained(
                model_name
            ).to(self.device)
            self._models["bartscore_tokenizer"] = BartTokenizer.from_pretrained(model_name)
            self._models["bartscore_model"].eval()
        return self._models["bartscore_model"], self._models["bartscore_tokenizer"]

    def _load_alignscore(self):
        if "alignscore" not in self._models:
            try:
                from alignscore import AlignScore
                self._models["alignscore"] = AlignScore(
                    model="roberta-large",
                    batch_size=self.batch_sizes["AlignScore"],
                    device=self.device,
                    ckpt_path=None,
                    evaluation_mode="nli_sp",
                )
                self._models["alignscore_fallback"] = False
            except Exception:
                from sentence_transformers import SentenceTransformer
                self._models["alignscore"] = SentenceTransformer(
                    "all-mpnet-base-v2", device=self.device
                )
                self._models["alignscore_fallback"] = True
        return self._models["alignscore"]

    def compute_bertscore(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute BERTScore F1 (DeBERTa-XLarge-MNLI)."""
        scorer = self._load_bertscore()
        P, R, F1 = scorer.score(candidates, references)
        return F1.cpu().numpy().tolist()

    def compute_bartscore(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute BARTScore (BART-Large-CNN, negative log-likelihood)."""
        model, tokenizer = self._load_bartscore()
        batch_size = self.batch_sizes["BARTScore"]
        scores = []

        for i in range(0, len(references), batch_size):
            batch_refs = references[i : i + batch_size]
            batch_cands = candidates[i : i + batch_size]

            for ref, cand in zip(batch_refs, batch_cands):
                inputs = tokenizer(
                    [ref], [cand],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                ).to(self.device)

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    score = -outputs.loss.item()
                    scores.append(score)

        return scores

    def compute_alignscore(self, references: List[str], candidates: List[str]) -> List[float]:
        """Compute AlignScore (RoBERTa-Large NLI, fallback: MPNet cosine)."""
        alignscore_model = self._load_alignscore()

        if self._models.get("alignscore_fallback"):
            from sklearn.metrics.pairwise import cosine_similarity
            ref_emb = alignscore_model.encode(
                references, batch_size=self.batch_sizes["AlignScore"], show_progress_bar=False
            )
            cand_emb = alignscore_model.encode(
                candidates, batch_size=self.batch_sizes["AlignScore"], show_progress_bar=False
            )
            scores = [
                float(cosine_similarity([r], [c])[0][0])
                for r, c in zip(ref_emb, cand_emb)
            ]
            return scores
        else:
            return alignscore_model.score(contexts=references, claims=candidates)

    def compute_all(self, references: List[str], candidates: List[str], prefix: str = "basic") -> dict:
        """Compute all neural metrics and return as a dict of column_name -> scores."""
        results = {}
        print(f"  Computing BERTScore ({prefix})...")
        results[f"{prefix}_bertscore"] = self.compute_bertscore(references, candidates)

        print(f"  Computing BARTScore ({prefix})...")
        results[f"{prefix}_bartscore"] = self.compute_bartscore(references, candidates)

        print(f"  Computing AlignScore ({prefix})...")
        results[f"{prefix}_alignscore"] = self.compute_alignscore(references, candidates)
        return results

    def cleanup(self):
        """Free GPU memory."""
        self._models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
