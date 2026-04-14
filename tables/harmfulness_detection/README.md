# Harmfulness Detection — ICL Results

LaTeX tables and CSV summaries for the dialectal harmfulness detection experiments
across zero-shot and few-shot in-context learning (ICL) settings.

## Files

| File | Description |
|------|-------------|
| `table_combined.tex` | Main table (Dia-LLM CoI + Multi-VALUE combined detection rates) |
| `table_dia_llm_coi.tex` | Dia-LLM CoI-transformed dialectal data only |
| `table_multi_value.tex` | Multi-VALUE rule-transformed dialectal data only |
| `detection_rates.csv` | Machine-readable per-model × per-mode rates |

## Metrics

For each (model, shot count) cell:
- **Acc$_d$** (Dialectal Accuracy): % of harmful **dialect-transformed** prompts the
  model correctly flags as `unsafe`.
- **$\Delta$** (Gap): difference between dialectal and SAE accuracy
  (`Acc_d − Acc_s`). Negative values indicate the model performs **worse** on
  dialectal text than on the SAE original — a measure of dialectal robustness loss.

## Source data

Raw per-sample results live under `codes/inference/results/{shot}/harmfulness_detection/`:
- `summary_by_model.csv` — aggregated per model (with coi, mv, and combined splits)
- `by_dialect.csv` — per (model × dialect)
- `by_dataset.csv` — per (model × dataset)

## Models we ran on Bedrock API

15 models across 3 categories (Safety-Grade Guards, General Purpose LLMs,
Proprietary LLMs). Models requiring local GPU inference (e.g., ShieldGemma,
Llama-3.1-8B, Gemma-3-1B) and the proposed defense (`\diaguard`) are still
marked `[XX]` and to be filled in from a separate GPU run.

## Sample size

- 50 dialects × 30 dataset configs (15 dia_llm_coi + 15 multi_value)
  × 50 samples per dataset = **75,000 samples per model per shot count**.
- Each sample contributes 2 inference calls (SAE original + dialectal transform).
