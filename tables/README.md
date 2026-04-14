# Tables

LaTeX source for all result tables in the paper. Each `.tex` file is a self-contained table that compiles standalone and links back (via a header comment) to the JSON data it was built from.

## Folder layout

```
tables/
├── README.md                                   # (this file)
├── harmfulness_detection/
│   ├── guard_classifier_accuracy.tex           # Table 1 — Overall Acc on SAE vs dialect, MV / DL columns
│   ├── category_accuracy.tex                   # Table 2 — Accuracy disaggregated by threat category
│   └── per_dialect_guards.tex                  # Table (appendix) — Safety Guard detection rate by dialect & region
└── attack_success/
    └── end_to_end_asr.tex                      # Table 3 — End-to-end ASR grouped by defense setting
```

## Table-to-data mapping

Every cell in every table is sourced from a single JSON file under `codes/inference/results/`.

### `harmfulness_detection/guard_classifier_accuracy.tex`
Overall harmfulness-detection accuracy per guard model on Standard American English (SAE) vs. dialectal inputs, broken down by prompt template (`MV` = Multi-Value, `DL` = Dia-LLM).

| Column | Source |
|---|---|
| `Acc_SAE` (MV) | `zero-shot/harmfulness_detection/multi_value/<model>/metrics.json` → `overall.accuracy_sae` |
| `Acc_Dial` (MV) | same file → `overall.accuracy_dialect` |
| `Δ` (MV) | `accuracy_dialect − accuracy_sae` |
| `Acc_SAE` (DL) | aggregate over `llm_basic/<model>/metrics.json` and `llm_coi/<model>/metrics.json` |
| `Acc_Dial` (DL) | same aggregate → `accuracy_dialect` |

### `harmfulness_detection/per_dialect_guards.tex`
Safety-guard harmfulness detection rate (%) for every one of the 50 dialects, grouped by region (US, UK, Africa, Asia-Pacific, Caribbean/Atlantic, Other). Column headers use compact model abbreviations (LG-1 = LlamaGuard-1, QG-4B = Qwen3Guard-4B, PG-86M = PromptGuard-86M, etc.).

| Column | Source |
|---|---|
| `<Model>` for a dialect row | `zero-shot/harmfulness_detection/multi_value/<Model>/per_dialect.json` → `<dialect>.accuracy_dialect × 100` |
| `SAE (baseline)` row | same file → `<dialect>.accuracy_sae × 100` (identical across dialects for the same model) |
| `Dialect Avg` row | support-weighted mean of the column across all dialects |
| `Δ (Dia − SAE)` row | `Dialect Avg − SAE baseline` |

Markers: $^\blacktriangle$ = best dialect for that model, $^\blacktriangledown$ = worst dialect for that model.

### `harmfulness_detection/category_accuracy.tex`
Same accuracy broken down by **threat category** (Toxicity/Safety, Prompt Injection, Code Generation).

| Column | Source |
|---|---|
| `Acc_d` | `zero-shot/harmfulness_detection/multi_value/<model>/categories/<category>/metrics.json` → `overall.accuracy_dialect` |
| `Δ` | same file: `accuracy_dialect − accuracy_sae` |

Category-to-dataset mapping:
- **Toxicity/Safety** (10 datasets): `Toxicity_Jigsaw`, `Salad_Bench`, `harmBench`, `advbench`, `jailbreakbench`, `sorry_bench`, `do_not_answer`, `Simple_Safety_Tests`, `Toxic_Text`, `forbiddent_questions`
- **Prompt Injection** (2 datasets): `bipia`, `injecagent`
- **Code Generation** (3 datasets): `cyberseceval`, `llmseceval`, `securityeval`

### `attack_success/end_to_end_asr.tex`
End-to-end attack success rate for production models under four guard defense configurations.

| Column | Source |
|---|---|
| `GBR / PBR / ASR / IARR` | `zero-shot/attack_success/multi_value/<defense_config>/<production_model>/metrics.json` → `overall.{gbr,pbr,asr,iarr}_{sae,dialect}` |
| `Δ` (GBR/PBR/ASR) | `sae − dialect` (negative = degraded on dialects) |
| `Δ` (IARR) | `dialect − sae` (negative = degraded on dialects) |

Defense configurations (`<defense_config>`):
- `best_guard` — **Qwen3Guard-4B** (highest individual accuracy)
- `worst_guard` — **HarmBench-Llama** (lowest individual accuracy)
- `majority_vote` — top-5 guards, block if ≥3 flag unsafe (low-stakes)
- `any_guard` — top-5 guards, block if any flag unsafe (high-stakes)

Top-5 guards in the ensemble: Qwen3Guard-4B, Qwen3Guard-8B, PolyGuard, LlamaGuard-3, DuoGuard-1B. **Production models are not used as guards.**

## Conventions

- **Bold** within each section = best value per column (lowest for GBR/PBR/ASR/error metrics, highest for IARR/accuracy, most positive Δ).
- **\textcolor{blue}{[XX]}** = pending experiment, to be filled when data is available.
- All values in percent unless noted otherwise.
- Rounded to one or two decimals for display; underlying JSON retains four.

## Regenerating values

The values are computed from raw inference outputs in `codes/inference/results/` by the scripts referenced in the main project README. After adding a new model or dataset, rerun those scripts to regenerate the JSONs, then copy updated values into the `.tex` cells (or write a small replacer — the `.tex` files are plain text).
