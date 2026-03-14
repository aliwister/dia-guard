"""
External Evaluation Module for Dialect Transformer

GPU-optimized metrics:
- BERTScore: Semantic similarity using BERT embeddings
- BARTScore: Generation quality using BART
- AlignScore: Factual consistency alignment
- METEOR: Translation quality metric

CPU metrics:
- ROUGE-L: Longest common subsequence overlap

LLM-based evaluation (via dialect_transformer):
- LLM-as-a-Judge: fluency, faithfulness, authenticity, coherence, readability
- Feature Accuracy: eWAVE feature validation
"""

from .gpu_metrics import (
    BERTScoreEvaluator,
    BARTScoreEvaluator,
    AlignScoreEvaluator,
    METEORScoreEvaluator,
    GPUMetricsEvaluator,
)

from .cpu_metrics import ROUGELEvaluator, DiffLibEvaluator

from .run_evaluation import (
    run_full_evaluation,
    run_gpu_evaluation,
    run_cpu_evaluation,
    run_llm_evaluation,
)

__all__ = [
    "BERTScoreEvaluator",
    "BARTScoreEvaluator",
    "AlignScoreEvaluator",
    "METEORScoreEvaluator",
    "GPUMetricsEvaluator",
    "ROUGELEvaluator",
    "DiffLibEvaluator",
    "run_full_evaluation",
    "run_gpu_evaluation",
    "run_cpu_evaluation",
    "run_llm_evaluation",
]
