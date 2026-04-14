# Shield — LaTeX Tables

Auto-generated LaTeX tables summarising the DIA-GUARD Shield fine-tuning and knowledge-distillation experiments.

## Regenerating

All tables are built from the committed `metrics.json` files under
`codes/evaluation/results/Shield/`. To regenerate after new results land:

```bash
python tables/shield/build_tables.py
```

## Files

| File | Contents |
|------|----------|
| `shield_ft_holdout.tex`       | Shield fine-tuning (Baseline, PEFT-CE, Full-FT-CE) × 7 models on the dialect Holdout test set (181,874 samples) |
| `shield_ft_sae.tex`           | Same on the SAE test set (36,050 samples) |
| `shield_ft_by_dataset.tex`    | Compact Holdout vs SAE side-by-side view |
| `kd_scenario1.tex`            | Scenario 1 KD — off-the-shelf teacher → off-the-shelf student (12 cells) |
| `kd_scenario2.tex`            | Scenario 2 KD — fine-tuned teacher → FT/base student (36 cells) |
| `shield_per_dialect_top5.tex` | Per-dialect accuracy for the top-5 models across all 50 dialects |

## Required LaTeX packages

```latex
\usepackage{booktabs}      % \toprule, \midrule, \bottomrule
\usepackage{multirow}      % multi-row cells in shield_ft_by_dataset
\usepackage{longtable}     % multi-page per-dialect table
```

## Column legend

| Column | Meaning |
|--------|---------|
| Acc | Accuracy on the test split |
| Prec | Macro-averaged precision |
| Rec | Macro-averaged recall |
| ASR | Attack Success Rate = TP / (TP + FN) — fraction of unsafe content correctly caught |
| F1 | Macro-averaged F1 |
