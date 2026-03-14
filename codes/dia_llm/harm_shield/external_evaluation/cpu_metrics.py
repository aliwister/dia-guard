"""
CPU-based evaluation metrics for dialect transformation.

Metrics:
- ROUGE-L: Longest common subsequence based overlap
- DiffLib: Python's built-in sequence matching (SequenceMatcher)
- BLEU: Bilingual Evaluation Understudy score
"""

import difflib
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""
    metric_name: str
    score: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ROUGELEvaluator:
    """
    ROUGE-L evaluator using longest common subsequence.

    Measures overlap between reference and candidate texts.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize ROUGE-L evaluator.

        Args:
            verbose: Print progress
        """
        self.verbose = verbose
        self._scorer = None

    def _init_scorer(self):
        """Initialize ROUGE scorer."""
        if self._scorer is None:
            try:
                from rouge_score import rouge_scorer
                self._scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            except ImportError:
                raise ImportError("rouge-score not installed. Run: pip install rouge-score")

    def score(
        self,
        references: List[str],
        candidates: List[str]
    ) -> List[MetricResult]:
        """
        Compute ROUGE-L scores.

        Args:
            references: Reference texts
            candidates: Candidate texts

        Returns:
            List of MetricResult objects
        """
        self._init_scorer()

        results = []
        iterator = zip(references, candidates)

        if self.verbose:
            iterator = tqdm(list(iterator), desc="ROUGE-L")

        for ref, cand in iterator:
            try:
                scores = self._scorer.score(ref, cand)
                rouge_l = scores['rougeL']

                results.append(MetricResult(
                    metric_name="ROUGE-L",
                    score=rouge_l.fmeasure,
                    precision=rouge_l.precision,
                    recall=rouge_l.recall,
                    f1=rouge_l.fmeasure
                ))
            except Exception as e:
                if self.verbose:
                    print(f"ROUGE-L error: {e}")
                results.append(MetricResult(
                    metric_name="ROUGE-L",
                    score=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0
                ))

        return results

    def score_batch(
        self,
        references: List[str],
        candidates: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Batch scoring.

        Returns:
            List of F1 scores
        """
        self.verbose = show_progress
        results = self.score(references, candidates)
        return [r.f1 for r in results]

    def score_detailed(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, List[float]]:
        """
        Get detailed scores with precision, recall, and F1.

        Returns:
            Dictionary with 'precision', 'recall', 'f1' keys
        """
        results = self.score(references, candidates)
        return {
            'precision': [r.precision for r in results],
            'recall': [r.recall for r in results],
            'f1': [r.f1 for r in results]
        }

    def cleanup(self):
        """No cleanup needed for ROUGE."""
        pass


class DiffLibEvaluator:
    """
    DiffLib-based similarity evaluator using Python's SequenceMatcher.

    Measures similarity ratio between reference and candidate texts.
    Fast and requires no external dependencies.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize DiffLib evaluator.

        Args:
            verbose: Print progress
        """
        self.verbose = verbose

    def score(
        self,
        references: List[str],
        candidates: List[str]
    ) -> List[MetricResult]:
        """
        Compute DiffLib similarity scores.

        Args:
            references: Reference texts
            candidates: Candidate texts

        Returns:
            List of MetricResult objects
        """
        results = []
        iterator = zip(references, candidates)

        if self.verbose:
            iterator = tqdm(list(iterator), desc="DiffLib")

        for ref, cand in iterator:
            try:
                # Use SequenceMatcher to compute similarity ratio
                matcher = difflib.SequenceMatcher(None, ref.lower(), cand.lower())
                ratio = matcher.ratio()

                # Also compute quick_ratio (faster approximation)
                quick_ratio = matcher.quick_ratio()

                # Get matching blocks for detailed analysis
                matching_blocks = matcher.get_matching_blocks()
                total_matched = sum(block.size for block in matching_blocks)

                results.append(MetricResult(
                    metric_name="DiffLib",
                    score=ratio,
                    details={
                        "quick_ratio": quick_ratio,
                        "matched_chars": total_matched,
                        "ref_len": len(ref),
                        "cand_len": len(cand)
                    }
                ))
            except Exception as e:
                if self.verbose:
                    print(f"DiffLib error: {e}")
                results.append(MetricResult(
                    metric_name="DiffLib",
                    score=0.0,
                    details={"error": str(e)}
                ))

        return results

    def score_batch(
        self,
        references: List[str],
        candidates: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Batch scoring.

        Returns:
            List of similarity ratios
        """
        self.verbose = show_progress
        results = self.score(references, candidates)
        return [r.score for r in results]

    def score_detailed(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, List[float]]:
        """
        Get detailed scores.

        Returns:
            Dictionary with 'ratio' and 'quick_ratio' keys
        """
        results = self.score(references, candidates)
        return {
            'ratio': [r.score for r in results],
            'quick_ratio': [r.details.get('quick_ratio', 0.0) if r.details else 0.0 for r in results]
        }

    def get_diff_ops(
        self,
        reference: str,
        candidate: str
    ) -> List[str]:
        """
        Get human-readable diff operations.

        Returns:
            List of diff operation strings
        """
        matcher = difflib.SequenceMatcher(None, reference, candidate)
        ops = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                ops.append(f"REPLACE '{reference[i1:i2]}' -> '{candidate[j1:j2]}'")
            elif tag == 'delete':
                ops.append(f"DELETE '{reference[i1:i2]}'")
            elif tag == 'insert':
                ops.append(f"INSERT '{candidate[j1:j2]}'")
            # 'equal' operations are skipped

        return ops

    def cleanup(self):
        """No cleanup needed for DiffLib."""
        pass


class BLEUEvaluator:
    """
    BLEU score evaluator using NLTK.

    Measures n-gram overlap between reference and candidate texts.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize BLEU evaluator.

        Args:
            verbose: Print progress
        """
        self.verbose = verbose
        self._initialized = False

    def _init_nltk(self):
        """Initialize NLTK resources."""
        if not self._initialized:
            try:
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                except LookupError:
                    nltk.download('punkt_tab', quiet=True)
                self._initialized = True
            except ImportError:
                raise ImportError("nltk not installed. Run: pip install nltk")

    def score(
        self,
        references: List[str],
        candidates: List[str]
    ) -> List[MetricResult]:
        """
        Compute BLEU scores.

        Args:
            references: Reference texts
            candidates: Candidate texts

        Returns:
            List of MetricResult objects
        """
        self._init_nltk()
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.tokenize import word_tokenize

        smoothing = SmoothingFunction().method1

        results = []
        iterator = zip(references, candidates)

        if self.verbose:
            iterator = tqdm(list(iterator), desc="BLEU")

        for ref, cand in iterator:
            try:
                ref_tokens = word_tokenize(ref.lower())
                cand_tokens = word_tokenize(cand.lower())

                # BLEU expects list of references
                bleu_score = sentence_bleu(
                    [ref_tokens],
                    cand_tokens,
                    smoothing_function=smoothing
                )

                results.append(MetricResult(
                    metric_name="BLEU",
                    score=bleu_score
                ))
            except Exception as e:
                if self.verbose:
                    print(f"BLEU error: {e}")
                results.append(MetricResult(
                    metric_name="BLEU",
                    score=0.0
                ))

        return results

    def score_batch(
        self,
        references: List[str],
        candidates: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Batch scoring.

        Returns:
            List of BLEU scores
        """
        self.verbose = show_progress
        results = self.score(references, candidates)
        return [r.score for r in results]

    def cleanup(self):
        """No cleanup needed for BLEU."""
        pass
