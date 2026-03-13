"""
Configuration registry for all 9 safety/security benchmark datasets.
Each config specifies how to load the dataset, which columns to transform,
and which columns to preserve verbatim.
"""

DATASET_CONFIGS = {
    "securityeval": {
        "loader": "huggingface",
        "hf_path": "s2e-lab/SecurityEval",
        "hf_config": None,
        "hf_split": "train",
        "text_columns": ["Prompt"],
        "code_columns": ["Insecure_code"],
        "preserve_columns": ["ID", "Insecure_code"],
        "contains_code_mixed": True,
        "estimated_size": 121,
    },
    "llmseceval": {
        "loader": "github_csv",
        "github_url": "https://raw.githubusercontent.com/tuhh-softsec/LLMSecEval/main/Dataset/LLMSecEval-Prompts_dataset.csv",
        "text_columns": ["LLM-generated NL Prompt", "Manually-fixed NL Prompt"],
        "code_columns": [],
        "preserve_columns": [],  # All non-text columns preserved by default
        "estimated_size": 150,
    },
    "jailbreakbench": {
        "loader": "huggingface",
        "hf_path": "JailbreakBench/JBB-Behaviors",
        "hf_config": "behaviors",
        "hf_split": None,  # Load all splits (harmful + benign)
        "text_columns": ["Goal", "Target"],
        "code_columns": [],
        "preserve_columns": ["Index", "Behavior", "Category", "Source"],
        "estimated_size": 200,
    },
    "sorry_bench": {
        "loader": "huggingface",
        "hf_path": "sorry-bench/sorry-bench-202503",
        "hf_config": None,
        "hf_split": None,
        "text_columns": ["turns"],  # list of strings
        "text_column_types": {"turns": "list_of_strings"},
        "code_columns": [],
        "preserve_columns": ["question_id", "category"],
        "estimated_size": 450,
    },
    "advbench": {
        "loader": "huggingface",
        "hf_path": "walledai/AdvBench",
        "hf_config": None,
        "hf_split": "train",
        "text_columns": ["prompt"],
        "code_columns": [],
        "preserve_columns": ["target"],
        "estimated_size": 500,
    },
    "do_not_answer": {
        "loader": "huggingface",
        "hf_path": "LibrAI/do-not-answer",
        "hf_config": None,
        "hf_split": "train",
        "text_columns": ["question"],
        "code_columns": [],
        "preserve_columns": [],  # Preserve all non-text columns
        "estimated_size": 939,
    },
    "injecagent": {
        "loader": "github_json",
        "github_files": {
            "test_cases_dh_base": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
            "test_cases_ds_base": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json",
        },
        "text_columns": ["User Instruction"],
        "code_columns": [],
        "preserve_columns": [],  # Preserve all non-text columns
        "estimated_size": 1054,
    },
    "cyberseceval": {
        "loader": "huggingface",
        "hf_path": "walledai/CyberSecEval",
        "hf_config": ["autocomplete", "instruct"],
        "hf_split": "all",
        "text_columns": ["prompt"],
        "code_columns": ["origin_code", "line_text"],
        "contains_code_mixed": True,  # prompt may mix NL and code
        "preserve_columns": [],
        "estimated_size": 3840,
    },
    "bipia": {
        "loader": "huggingface",
        "hf_path": "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT",
        "hf_config": None,
        "hf_split": None,
        "text_columns": ["context", "user_intent"],
        "code_columns": [],
        "preserve_columns": ["label", "source"],
        "estimated_size": 70000,
        "sample_size": 5000,  # Stratified sample
        "sample_stratify_col": "label",
    },
}

# Processing order: smallest datasets first for fast validation
PROCESSING_ORDER = [
    "securityeval",
    "llmseceval",
    "jailbreakbench",
    "sorry_bench",
    "advbench",
    "do_not_answer",
    "injecagent",
    "cyberseceval",
    "bipia",
]
