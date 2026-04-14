# 2-SHOT (= --num_shots 2) Harmfulness Detection Results

Aggregated detection rates per model for the **2-SHOT (= --num_shots 2)** evaluation
(`--num_shots 0`).

## Files

| File | Granularity | Rows |
|------|-------------|-----:|
| `summary_by_model.csv` | one row per model with coi/mv/combined splits | ~15 |
| `by_dialect.csv` | one row per (model × dialect) | ~750 |
| `by_dataset.csv` | one row per (model × dataset) | ~225 |

## Columns

For each (model, breakdown) the CSVs report:
- `coi_acc_sae` / `coi_acc_dial` / `coi_delta` / `coi_n` — metrics on the
  **Dia-LLM CoI** transformed slice.
- `mv_acc_sae` / `mv_acc_dial` / `mv_delta` / `mv_n` — metrics on the
  **Multi-VALUE** rule-transformed slice.
- `combined_*` — pooled across both slices (in summary file only).

## Metric definitions

- `acc_sae`: detection rate on Standard American English (SAE) original.
  All test inputs are *harmful*, so a higher rate = more correct flags.
- `acc_dial`: detection rate on the dialect-transformed input.
- `delta`: `acc_dial − acc_sae`. Negative = dialect transformation hurt
  detection; positive = dialect transformation helped.

## Companion folders

Same structure with the corresponding shot count is mirrored at:
- `../../2-shot/harmfulness_detection/`
- `../../4-shot/harmfulness_detection/`
- `../../8-shot/harmfulness_detection/`
