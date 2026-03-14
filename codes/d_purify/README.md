# D-PURiFY: Dataset Purity Evaluation Framework

**DIA-GUARD -> Dia-LLM -> D-PURiFY**

Automatic evaluation framework for assessing the quality of dialect transformations (Harm-SHIELD) and benign counterexample generation (CounterHarm-SHIELD) in the DIA-LLM pipeline.

## Metrics Overview

### Dialect Transformation Metrics

Evaluate how well `basic_transform` and `coi_transform` preserve meaning while applying dialect features from `original_input`.

#### Text Similarity (CPU)

| Metric | Description | Range |
|--------|-------------|-------|
| **BLEU** | N-gram precision overlap | 0-1 (higher = more similar) |
| **METEOR** | Unigram matching with stemming, synonyms, word order | 0-1 (higher = more similar) |
| **ROUGE-L** | Longest common subsequence (F1, Precision, Recall) | 0-1 (higher = more similar) |
| **DiffLib** | Sequence matching ratio (edit similarity) | 0-1 (higher = more similar) |

#### Neural/Semantic (GPU)

| Metric | Model | Description | Range |
|--------|-------|-------------|-------|
| **BERTScore** | DeBERTa-XLarge-MNLI | Contextual embedding similarity | 0-1 |
| **BARTScore** | BART-Large-CNN | Generation log-likelihood | negative (higher = better) |
| **AlignScore** | RoBERTa-Large NLI | NLI-based semantic alignment | 0-1 |

#### Dialect Validation (eWAVE)

| Metric | Description |
|--------|-------------|
| **eWAVE Accuracy** | Proportion of expected dialect features correctly applied |
| **Valid Features** | Count of correctly applied eWAVE features |
| **Errors** | Count of transformation errors detected |

#### LLM-as-a-Judge (6 Dimensions, 1-7 Scale)

| Dimension | What It Measures |
|-----------|------------------|
| **Fluency** | Grammar, syntax, naturalness in target dialect |
| **Faithfulness** | Meaning preservation from original |
| **Dialect Authenticity** | Accuracy of dialect features |
| **Feature Accuracy** | Alignment with eWAVE specifications |
| **Coherence** | Logical flow and consistency |
| **Readability** | Ease of comprehension |

### CounterHarm Metrics (New)

Evaluate the quality of benign counterexamples by measuring the difference between harmful source text and its benign counterharm output.

| Metric | Model | Description | Expected Pattern |
|--------|-------|-------------|------------------|
| **BERTScore** | DeBERTa-XLarge-MNLI | Surface/structural similarity | Moderate (0.3-0.6): similar structure, different content |
| **Semantic Similarity** | all-mpnet-base-v2 | Meaning relatedness (cosine) | Moderate (0.3-0.6): related topic, different intent |
| **NLI Contradiction** | NLI-DeBERTa-v3-base | Contradiction probability | High (>0.5): benign contradicts harmful intent |
| **NLI Entailment** | NLI-DeBERTa-v3-base | Entailment probability | Low (<0.3): benign does NOT follow from harmful |
| **NLI Neutral** | NLI-DeBERTa-v3-base | Neutral probability | Variable |

Column pairs evaluated:
- `counterharm_original` vs `original_input`
- `counterharm_transformed` vs `transformed_input`
- `counterharm_basic` vs `basic_transform`
- `counterharm_coi` vs `coi_transform`

## Usage

```bash
# Evaluate all dialects (all metrics)
python evaluate.py --data_dir ../LLM_Data

# Evaluate specific dialect
python evaluate.py --data_dir ../LLM_Data --dialect aboriginal_english

# Evaluate only counterharm metrics
python evaluate.py --data_dir ../LLM_Data --counterharm-only

# Evaluate only dialect transform metrics
python evaluate.py --data_dir ../LLM_Data --transform-only

# Test on first 2 files
python evaluate.py --data_dir ../LLM_Data --test
```

## Project Structure

```
D-PURiFY/
├── README.md
├── evaluate.py                      # Main evaluation runner
└── metrics/
    ├── __init__.py
    ├── text_similarity.py           # BLEU, METEOR, ROUGE-L, DiffLib (CPU)
    ├── neural_metrics.py            # BERTScore, BARTScore, AlignScore (GPU)
    ├── dialect_validation.py        # eWAVE validation + LLM-as-a-Judge
    └── counterharm_metrics.py       # BERTScore, Semantic Sim, NLI (GPU)
```

## Requirements

```
torch
transformers
bert-score
sentence-transformers
rouge-score
nltk
pandas
scikit-learn
```

## Output Columns

### Transform Evaluation Columns
`basic_bleu`, `basic_meteor`, `basic_rouge_l`, `basic_rouge_l_precision`, `basic_rouge_l_recall`, `basic_difflib`, `basic_bertscore`, `basic_bartscore`, `basic_alignscore`
(Same with `coi_` prefix for CoI transform)

### CounterHarm Evaluation Columns
`ch_original_bertscore`, `ch_original_semantic_sim`, `ch_original_nli_contradiction`, `ch_original_nli_entailment`, `ch_original_nli_neutral`
(Same with `ch_transformed_`, `ch_basic_`, `ch_coi_` prefixes)
