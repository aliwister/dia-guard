#!/usr/bin/env python3
"""
Dialect Validation Metrics (eWAVE-based)
D-PURiFY | DIA-GUARD -> Dia-LLM

Validates dialect transformations against the eWAVE (Electronic World Atlas
of Varieties of English) linguistic feature specifications. Computes eWAVE
accuracy, valid feature counts, and transformation error detection.

Also includes LLM-as-a-Judge evaluation across 6 quality dimensions:
Fluency, Faithfulness, Dialect Authenticity, Feature Accuracy, Coherence, Readability.
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Add dialect_transformer path for eWAVE data
DIALECT_TRANSFORMER_DIR = Path(__file__).parent.parent.parent / "dialect_transformer"
if str(DIALECT_TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(DIALECT_TRANSFORMER_DIR))


@dataclass
class DialectValidationResult:
    """Validation result for a single dialect transformation."""
    ewave_accuracy: float = 0.0
    valid_features: int = 0
    errors: int = 0
    features_found: List[str] = field(default_factory=list)
    features_missing: List[str] = field(default_factory=list)


class DialectValidator:
    """Validates dialect transformations against eWAVE specifications."""

    def __init__(self, dialect_key: str):
        self.dialect_key = dialect_key
        self._validator = None
        self._load_validator()

    def _load_validator(self):
        """Load the feature validator for this dialect."""
        try:
            from feature_validator import FeatureValidator
            self._validator = FeatureValidator(self.dialect_key)
        except ImportError:
            print(f"[D-PURiFY] Warning: feature_validator not available for {self.dialect_key}")

    def validate(self, original: str, transformed: str) -> DialectValidationResult:
        """Validate a single transformation against eWAVE features."""
        if not self._validator:
            return DialectValidationResult()

        try:
            result = self._validator.validate(original=original, transformed=transformed)
            return DialectValidationResult(
                ewave_accuracy=result.accuracy if hasattr(result, "accuracy") else 0.0,
                valid_features=result.valid_count if hasattr(result, "valid_count") else 0,
                errors=result.error_count if hasattr(result, "error_count") else 0,
                features_found=result.features_found if hasattr(result, "features_found") else [],
                features_missing=result.features_missing if hasattr(result, "features_missing") else [],
            )
        except Exception as e:
            print(f"[D-PURiFY] Validation error: {e}")
            return DialectValidationResult()

    def validate_batch(
        self, originals: List[str], transformeds: List[str], prefix: str = "basic"
    ) -> dict:
        """Validate a batch of transformations. Returns dict of column_name -> values."""
        accuracies = []
        valid_features = []
        errors = []

        for orig, trans in zip(originals, transformeds):
            result = self.validate(orig, trans)
            accuracies.append(result.ewave_accuracy)
            valid_features.append(result.valid_features)
            errors.append(result.errors)

        return {
            f"{prefix}_ewave_accuracy": accuracies,
            f"{prefix}_valid_features": valid_features,
            f"{prefix}_errors": errors,
        }


class LLMJudgeEvaluator:
    """LLM-as-a-Judge evaluation across 6 quality dimensions.

    Dimensions (1-7 scale):
        - Fluency: Grammar, syntax, naturalness in target dialect
        - Faithfulness: Meaning preservation from original
        - Dialect Authenticity: Accuracy of dialect features
        - Feature Accuracy: Alignment with eWAVE specifications
        - Coherence: Logical flow and consistency
        - Readability: Ease of comprehension
    """

    DIMENSIONS = [
        "fluency",
        "faithfulness",
        "dialect_authenticity",
        "feature_accuracy",
        "coherence",
        "readability",
    ]

    def __init__(self, llm_backend=None):
        self.llm = llm_backend

    def evaluate(self, original: str, transformed: str, dialect_name: str) -> Dict[str, Any]:
        """Evaluate a single transformation across all 6 dimensions."""
        if not self.llm:
            raise ValueError("LLM backend required for LLM-as-a-Judge evaluation")

        prompt = self._build_prompt(original, transformed, dialect_name)
        response = self.llm.generate(
            system="You are an expert linguist evaluating dialect transformations. "
                   "Return your evaluation as valid JSON only.",
            user=prompt,
        )

        return self._parse_response(response)

    def _build_prompt(self, original: str, transformed: str, dialect_name: str) -> str:
        return f"""Evaluate this dialect transformation on 6 dimensions (score 1-7 each):

ORIGINAL (SAE): {original}
TRANSFORMED ({dialect_name}): {transformed}

Evaluate:
1. fluency - Grammar, syntax, naturalness in target dialect
2. faithfulness - Meaning preservation from original
3. dialect_authenticity - Accuracy of dialect features
4. feature_accuracy - Alignment with known linguistic features
5. coherence - Logical flow and consistency
6. readability - Ease of comprehension

Return JSON:
{{"fluency": <1-7>, "faithfulness": <1-7>, "dialect_authenticity": <1-7>, "feature_accuracy": <1-7>, "coherence": <1-7>, "readability": <1-7>, "overall": <1-7>}}"""

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM judge response into scores."""
        try:
            match = re.search(r"\{[^}]+\}", response)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {d: 0 for d in self.DIMENSIONS + ["overall"]}
