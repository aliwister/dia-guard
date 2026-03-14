"""
LLM-as-a-Judge Evaluation Module.

Uses dialect_transformer's evaluation module for:
- Fluency (1-7 scale)
- Faithfulness (meaning preservation)
- Dialect Authenticity
- Coherence
- Readability

Also includes Feature Accuracy using eWAVE validation.
"""

import os
import sys
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

# Add parent directory to path for dialect_transformer imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


@dataclass
class LLMEvaluationResult:
    """Result from LLM-as-a-Judge evaluation."""
    fluency: float = 0.0
    faithfulness: float = 0.0
    dialect_authenticity: float = 0.0
    coherence: float = 0.0
    readability: float = 0.0
    overall: float = 0.0
    reasoning: Dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureAccuracyResult:
    """Result from eWAVE feature validation."""
    accuracy: float = 0.0
    valid_features: int = 0
    wrong_dialect: int = 0
    errors: int = 0
    total_changes: int = 0
    features_applied: List[str] = field(default_factory=list)


class LLMJudgeEvaluator:
    """
    LLM-as-a-Judge evaluator using dialect_transformer's evaluation module.

    Evaluates transformations on 5 dimensions using a 1-7 Likert scale.
    """

    def __init__(
        self,
        backend: str = "azure",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize LLM Judge evaluator.

        Args:
            backend: LLM backend ("azure", "openai", "anthropic", "ollama")
            model: Model name/deployment
            api_key: API key (uses env var if None)
            endpoint: API endpoint (for Azure)
            verbose: Print progress
        """
        self.backend_name = backend
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint
        self.verbose = verbose

        self._evaluator = None
        self._llm = None

    def _init_evaluator(self):
        """Lazy initialization of evaluator."""
        if self._evaluator is None:
            try:
                from evaluation import DialectEvaluator

                # Initialize LLM backend
                self._llm = self._create_llm_backend()

                self._evaluator = DialectEvaluator(
                    llm=self._llm,
                    verbose=self.verbose
                )
            except ImportError as e:
                raise ImportError(f"Could not import dialect_transformer modules: {e}")

    def _create_llm_backend(self):
        """Create LLM backend based on configuration."""
        from models import (
            OpenAIBackend,
            AzureOpenAIBackend,
            AnthropicBackend,
            OllamaBackend
        )

        if self.backend_name == "azure":
            return AzureOpenAIBackend(
                deployment_name=self.model or os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
                endpoint=self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
            )
        elif self.backend_name == "openai":
            return OpenAIBackend(
                model=self.model or "gpt-4",
                api_key=self.api_key or os.getenv("OPENAI_API_KEY")
            )
        elif self.backend_name == "anthropic":
            return AnthropicBackend(
                model=self.model or "claude-3-sonnet-20240229",
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.backend_name == "ollama":
            return OllamaBackend(
                model=self.model or "llama3.1"
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")

    def evaluate(
        self,
        original_text: str,
        transformed_text: str,
        dialect_name: str
    ) -> LLMEvaluationResult:
        """
        Evaluate a single transformation.

        Args:
            original_text: Original SAE text
            transformed_text: Dialect-transformed text
            dialect_name: Name of target dialect

        Returns:
            LLMEvaluationResult with scores and reasoning
        """
        self._init_evaluator()

        try:
            result = self._evaluator.evaluate(
                original_text=original_text,
                dialect_text=transformed_text,
                dialect_name=dialect_name,
                comprehensive=True
            )

            return LLMEvaluationResult(
                fluency=result.dimension_scores.get('fluency', {}).score if hasattr(result.dimension_scores.get('fluency', {}), 'score') else 0.0,
                faithfulness=result.dimension_scores.get('faithfulness', {}).score if hasattr(result.dimension_scores.get('faithfulness', {}), 'score') else 0.0,
                dialect_authenticity=result.dimension_scores.get('dialect_authenticity', {}).score if hasattr(result.dimension_scores.get('dialect_authenticity', {}), 'score') else 0.0,
                coherence=result.dimension_scores.get('coherence', {}).score if hasattr(result.dimension_scores.get('coherence', {}), 'score') else 0.0,
                readability=result.dimension_scores.get('readability', {}).score if hasattr(result.dimension_scores.get('readability', {}), 'score') else 0.0,
                overall=result.overall_score,
                reasoning={
                    dim: score.reasoning if hasattr(score, 'reasoning') else ""
                    for dim, score in result.dimension_scores.items()
                }
            )
        except Exception as e:
            if self.verbose:
                print(f"LLM evaluation error: {e}")
            return LLMEvaluationResult()

    def evaluate_batch(
        self,
        originals: List[str],
        transformed: List[str],
        dialect_name: str,
        show_progress: bool = True
    ) -> List[LLMEvaluationResult]:
        """
        Evaluate multiple transformations.

        Args:
            originals: List of original texts
            transformed: List of transformed texts
            dialect_name: Target dialect name
            show_progress: Show progress bar

        Returns:
            List of LLMEvaluationResult objects
        """
        results = []
        iterator = zip(originals, transformed)

        if show_progress:
            iterator = tqdm(list(iterator), desc=f"LLM Judge ({dialect_name})")

        for orig, trans in iterator:
            result = self.evaluate(orig, trans, dialect_name)
            results.append(result)

        return results

    def get_scores_dict(
        self,
        results: List[LLMEvaluationResult]
    ) -> Dict[str, List[float]]:
        """
        Convert results to dictionary of score lists.

        Returns:
            Dictionary with dimension names as keys
        """
        return {
            'llm_fluency': [r.fluency for r in results],
            'llm_faithfulness': [r.faithfulness for r in results],
            'llm_authenticity': [r.dialect_authenticity for r in results],
            'llm_coherence': [r.coherence for r in results],
            'llm_readability': [r.readability for r in results],
            'llm_overall': [r.overall for r in results]
        }


class FeatureAccuracyEvaluator:
    """
    eWAVE Feature Accuracy evaluator.

    Validates which linguistic features were correctly applied
    based on the eWAVE database.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize Feature Accuracy evaluator.

        Args:
            verbose: Print progress
        """
        self.verbose = verbose
        self._validators = {}  # Cache validators per dialect

    def _get_validator(self, dialect_key: str):
        """Get or create validator for a specific dialect."""
        if dialect_key not in self._validators:
            try:
                from feature_validator import FeatureValidator
                self._validators[dialect_key] = FeatureValidator(dialect_key=dialect_key, verbose=self.verbose)
            except ImportError as e:
                raise ImportError(f"Could not import feature_validator: {e}")
        return self._validators[dialect_key]

    def evaluate(
        self,
        original_text: str,
        transformed_text: str,
        dialect_key: str
    ) -> FeatureAccuracyResult:
        """
        Evaluate feature accuracy for a single transformation.

        Args:
            original_text: Original SAE text
            transformed_text: Dialect-transformed text
            dialect_key: Dialect key (e.g., "aboriginal", "urban_aave")

        Returns:
            FeatureAccuracyResult with accuracy metrics
        """
        validator = self._get_validator(dialect_key)

        try:
            result = validator.validate(
                original=original_text,
                transformed=transformed_text
            )

            return FeatureAccuracyResult(
                accuracy=result.ewave_accuracy if hasattr(result, 'ewave_accuracy') else result.accuracy,
                valid_features=result.valid_count if hasattr(result, 'valid_count') else result.correct_count,
                wrong_dialect=result.wrong_dialect_count if hasattr(result, 'wrong_dialect_count') else 0,
                errors=result.error_count if hasattr(result, 'error_count') else 0,
                total_changes=result.total_changes if hasattr(result, 'total_changes') else 0,
                features_applied=[
                    f.feature_key for f in result.features_validated
                    if hasattr(f, 'status') and f.status.value == 'correct'
                ] if hasattr(result, 'features_validated') else []
            )
        except Exception as e:
            if self.verbose:
                print(f"Feature validation error: {e}")
            return FeatureAccuracyResult()

    def evaluate_batch(
        self,
        originals: List[str],
        transformed: List[str],
        dialect_key: str,
        show_progress: bool = True
    ) -> List[FeatureAccuracyResult]:
        """
        Evaluate multiple transformations.

        Args:
            originals: List of original texts
            transformed: List of transformed texts
            dialect_key: Dialect key
            show_progress: Show progress bar

        Returns:
            List of FeatureAccuracyResult objects
        """
        results = []
        iterator = zip(originals, transformed)

        if show_progress:
            iterator = tqdm(list(iterator), desc=f"Feature Acc ({dialect_key})")

        for orig, trans in iterator:
            result = self.evaluate(orig, trans, dialect_key)
            results.append(result)

        return results

    def get_scores_dict(
        self,
        results: List[FeatureAccuracyResult]
    ) -> Dict[str, List[float]]:
        """
        Convert results to dictionary of score lists.

        Returns:
            Dictionary with metric names as keys
        """
        return {
            'feature_accuracy': [r.accuracy for r in results],
            'feature_valid_count': [r.valid_features for r in results],
            'feature_wrong_dialect': [r.wrong_dialect for r in results],
            'feature_errors': [r.errors for r in results]
        }


def convert_dialect_folder_to_key(folder_name: str) -> str:
    """
    Convert dialect folder name to dialect key.

    Examples:
        "aboriginal_english" -> "aboriginal"
        "urban_african_american_vernacular_english" -> "urban_aave"
    """
    # Common mappings
    mappings = {
        "aboriginal_english": "aboriginal",
        "urban_african_american_vernacular_english": "urban_aave",
        "rural_african_american_vernacular_english": "rural_aave",
        "earlier_african_american_vernacular_english": "earlier_aave",
        "colloquial_singapore_english_singlish": "singapore",
        "jamaican_english": "jamaican",
        "nigerian_english": "nigerian",
        "indian_english": "indian",
        "australian_english": "australian",
        "scottish_english": "scottish",
        "irish_english": "irish",
        "welsh_english": "welsh",
        "appalachian_english": "appalachian",
        "chicano_english": "chicano",
        "hong_kong_english": "hong_kong",
        "malaysian_english": "malaysian",
        "philippine_english": "philippine",
        "pakistani_english": "pakistani",
        "sri_lankan_english": "sri_lankan",
        "kenyan_english": "kenyan",
        "tanzanian_english": "tanzanian",
        "ugandan_english": "ugandan",
        "ghanaian_english": "ghanaian",
        "cameroon_english": "cameroon",
        "new_zealand_english": "new_zealand",
        "bahamian_english": "bahamian",
        "black_south_african_english": "south_african_black",
        "indian_south_african_english": "south_african_indian",
        "white_south_african_english": "south_african_white",
        "cape_flats_english": "cape_flats",
        "falkland_islands_english": "falkland",
        "st_helena_english": "st_helena",
        "tristan_da_cunha_english": "tristan",
        "maltese_english": "maltese_english",
        "manx_english": "manx",
        "channel_islands_english": "channel_islands",
        "orkney_and_shetland_english": "orkney_shetland",
        "east_anglian_english": "east_anglian",
        "english_dialects_in_the_north_of_england": "northern_england",
        "english_dialects_in_the_southeast_of_england": "southeast_england",
        "english_dialects_in_the_southwest_of_england": "southwest_england",
        "ozark_english": "ozark",
        "newfoundland_english": "newfoundland",
        "colloquial_american_english": "colloquial_american",
        "southeast_american_enclave_dialects": "southeast_enclave",
        "liberian_settler_english": "liberian_settler",
        "acrolectal_fiji_english": "fiji",
        "pure_fiji_english_basilectal_fijie": "fiji",
        "australian_vernacular_english": "australian_vernacular",
        "white_zimbabwean_english": "zimbabwean_white",
    }

    # Try direct mapping first
    if folder_name in mappings:
        return mappings[folder_name]

    # Try removing "_english" suffix
    if folder_name.endswith("_english"):
        base = folder_name[:-8]
        if base in mappings:
            return mappings[base]
        return base

    return folder_name
