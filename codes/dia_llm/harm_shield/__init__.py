# Dialect Transformer - Transform SAE to 67+ English Dialects
#
# Quick usage:
#   from dialect_transformer import DialectTransformer
#   transformer = DialectTransformer(backend="auto")
#   result = transformer.transform("She is working", dialect="urban_aave")
#
# CoI Dialect Transformation (4-chain agentic approach):
#   from dialect_transformer import CoIDialectTransformer
#   transformer = CoIDialectTransformer(llm=llm)
#   result = transformer.transform("She is working", dialect="urban_aave")
#
# CoI Token Reintegration:
#   from dialect_transformer import CoIDialectReintegrator
#   reintegrator = CoIDialectReintegrator(llm=llm)
#   result = reintegrator.reintegrate(original_text, multivalue_text)
#
# LLM-as-a-Judge Evaluation:
#   from dialect_transformer import DialectEvaluator
#   evaluator = DialectEvaluator(llm=llm)
#   result = evaluator.evaluate(original, dialect_text, "urban_aave")

from .transformer import DialectTransformer, TransformResult, transform_to_dialect
from .models import (
    BaseLLM,
    ModelFactory,
    OpenAIBackend,
    AzureOpenAIBackend,
    OllamaBackend,
    HuggingFaceBackend,
    AnthropicBackend
)
from .dialects import (
    DIALECT_REGISTRY,
    REGIONS,
    get_dialect,
    list_dialects_by_region,
    get_dialect_count,
    validate_dialect_features
)
from .features import (
    FEATURE_LIBRARY,
    CATEGORIES,
    get_features_by_category,
    get_feature_by_id,
    list_all_features,
    get_feature_count
)
from .coi_reintegration import (
    CoIDialectReintegrator,
    CoIResult,
    SensitiveToken,
    ChainOutput,
    reintegrate_tokens,
    build_coi_prompt,
    SENSITIVE_TOKEN_PATTERNS
)
from .evaluation import (
    DialectEvaluator,
    EvaluationResult,
    DimensionScore,
    EvaluationDimension,
    evaluate_transformation,
    quick_score,
    build_evaluation_prompt,
    aggregate_results,
    format_results_report
)
from .coi_transformation import (
    CoIDialectTransformer,
    CoITransformResult,
    coi_transform,
    build_coi_transform_prompt,
    get_dialect_specs
)
from .feature_validator import (
    # Rule-based validation
    FeatureValidator,
    ValidationResult,
    FeatureValidation,
    ValidationStatus,
    validate_transformation,
    compare_transformations,
    print_comparison_report,
    # LLM-enhanced validation (existing)
    LLMFeatureValidator,
    validate_with_llm,
    # LLM-based comprehensive validation
    LLMComprehensiveValidator,
    ComprehensiveValidationResult,
    ChangeAnalysis,
    ChangeCategory,  # NEW: Three-category model enum
    comprehensive_validate,
    compare_transformations_llm
)

__version__ = "2.4.0"
__all__ = [
    # Main classes
    "DialectTransformer",
    "TransformResult",
    "transform_to_dialect",

    # LLM backends
    "BaseLLM",
    "ModelFactory",
    "OpenAIBackend",
    "AzureOpenAIBackend",
    "OllamaBackend",
    "HuggingFaceBackend",
    "AnthropicBackend",

    # Dialect data
    "DIALECT_REGISTRY",
    "REGIONS",
    "get_dialect",
    "list_dialects_by_region",
    "get_dialect_count",
    "validate_dialect_features",

    # Feature data
    "FEATURE_LIBRARY",
    "CATEGORIES",
    "get_features_by_category",
    "get_feature_by_id",
    "list_all_features",
    "get_feature_count",

    # Chain of Interaction (CoI) Token Reintegration
    "CoIDialectReintegrator",
    "CoIResult",
    "SensitiveToken",
    "ChainOutput",
    "reintegrate_tokens",
    "build_coi_prompt",
    "SENSITIVE_TOKEN_PATTERNS",

    # LLM-as-a-Judge Evaluation
    "DialectEvaluator",
    "EvaluationResult",
    "DimensionScore",
    "EvaluationDimension",
    "evaluate_transformation",
    "quick_score",
    "build_evaluation_prompt",
    "aggregate_results",
    "format_results_report",

    # Chain of Interaction (CoI) Dialect Transformation (Dual Attention)
    "CoIDialectTransformer",
    "CoITransformResult",
    "coi_transform",
    "build_coi_transform_prompt",
    "get_dialect_specs",

    # Feature Validation (Rule-based)
    "FeatureValidator",
    "ValidationResult",
    "FeatureValidation",
    "ValidationStatus",
    "validate_transformation",
    "compare_transformations",
    "print_comparison_report",

    # Feature Validation (LLM-enhanced)
    "LLMFeatureValidator",
    "validate_with_llm",

    # Feature Validation (LLM Comprehensive)
    "LLMComprehensiveValidator",
    "ComprehensiveValidationResult",
    "ChangeAnalysis",
    "ChangeCategory",  # Three-category model: VALID, WRONG_DIALECT, NON_EWAVE_ERROR
    "comprehensive_validate",
    "compare_transformations_llm",
]
