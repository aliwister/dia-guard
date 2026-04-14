#!/bin/bash
# Organize Shield results into a clean download package and zip it.
set -euo pipefail

STAGE=/tmp/dia-guard-shield-results
ZIP_OUT=/data/vibe_exp/dia-guard-shield-results.zip
SRC=/data/vibe_exp/dia-guard

rm -rf "$STAGE" "$ZIP_OUT"
mkdir -p "$STAGE"

# ─── 1. Top-level docs ───
cp "$SRC/tables/shield/SCENARIOS.md"  "$STAGE/SCENARIOS.md"

cat > "$STAGE/README.md" <<'EOF'
# DIA-GUARD Shield Results Package

Complete reproducible results for the Shield safety-classifier experiments
on the DIA-GUARD benchmark (50 English dialects, 181K Holdout + 36K SAE).

## Directory layout

```
dia-guard-shield-results/
├── README.md                    <- you are here
├── SCENARIOS.md                 <- what each scenario means (Baseline / PEFT / Full-FT / KD-S1 / KD-S2 / Teachers)
├── tables/                      <- LaTeX tables ready to paste into the paper
│   ├── main/                    <-    7 summary tables (2x2 style: method x split)
│   └── breakdowns/              <-   26 per-dialect and per-source-dataset long-tables
└── metrics/                     <- raw JSON results that power the tables
    ├── baseline/                <- off-the-shelf 7 students (no training)
    ├── shield_peft_ce/          <- Shield LoRA-CE students (14 cells)
    ├── shield_full_ft_ce/       <- Shield Full-FT-CE students (14 cells)
    ├── teachers_oob/            <- Qwen3-4B-SafeRL + Qwen3Guard-8B off-the-shelf
    ├── teachers_ft/             <- same teachers LoRA-CE finetuned on DIA-GUARD
    ├── kd_scenario1/            <- OOB teacher x OOB student (12 cells × 2 splits)
    └── kd_scenario2/            <- FT teacher x FT/base student (36 cells × 1 split)
```

Each `*.json` contains:
  - `overall` (accuracy, precision, recall, f1, support)
  - `confusion_matrix` (tp, fn, fp, tn)
  - `per_class` (safe / unsafe)
  - `per_dialect` (accuracy for each of 50 dialects)
  - `per_dataset` (accuracy for each of 15 source datasets)

## Required LaTeX preamble for the tables

```latex
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{longtable}
\usepackage{lscape}        % for landscape mode on per-dialect tables
```

## Headline results

| Rank | Model | Pipeline | Accuracy | F1 |
|------|-------|----------|----------|----|
| 1 | SmolLM2-1.7B | Shield PEFT-CE | 0.9742 | 0.9741 |
| 2 | Qwen3-4B-SafeRL | Teacher FT PEFT-CE | 0.9692 | 0.9688 |
| 3 | Gemma-3-1B | Shield Full-FT-CE | 0.9670 | 0.9669 |
| 4 | Qwen3-1.7B | KD-GKD (8B-ft teacher → PEFT student) | 0.9577 | 0.9572 |
| 5 | Gemma-3-270m | Shield Full-FT-CE | 0.9654 | 0.9650 |

## Data splits

| Split | Samples | Notes |
|-------|---------|-------|
| Holdout | 181,874 | 50 dialect-transformed variants of the 15 source datasets |
| SAE | 36,050 | Same content in Standard American English (no dialect transform) |

## Source code

All code that produced these results lives in the public GitHub repo:
<https://github.com/jsl5710/dia-guard>

## Reproducing the tables

```bash
# Re-compute per-dataset breakdowns from raw predictions
python codes/evaluation/compute_per_dataset.py

# Regenerate all main + breakdown tables
python tables/shield/build_tables.py
python tables/shield/build_breakdown_tables.py
```

## Model checkpoints

All trained models (62 total) are hosted on HuggingFace:
<https://huggingface.co/jsl5710>

Naming convention:
  - `Shield-<student>-<PEFT-CE|Full-FT-CE>`  (Shield students)
  - `Shield-<teacher>-FT-PEFT-CE`             (FT teachers)
  - `Shield-<student>-KD-<METHOD>-<teacher>-OOB`          (Scenario 1)
  - `Shield-<student>-KD-<METHOD>-<teacher>-S2-<start>`   (Scenario 2)
EOF

# ─── 2. Main tables ───
mkdir -p "$STAGE/tables/main"
for f in shield_ft_holdout shield_ft_sae shield_ft_by_dataset kd_scenario1 kd_scenario2 teachers shield_per_dialect_top5; do
  cp "$SRC/tables/shield/$f.tex" "$STAGE/tables/main/"
done
cp "$SRC/tables/shield/README.md" "$STAGE/tables/main/README.md"

# ─── 3. Breakdown tables ───
mkdir -p "$STAGE/tables/breakdowns"
cp "$SRC/tables/shield/breakdowns"/*.tex "$STAGE/tables/breakdowns/"

# ─── 4. Copy metrics organized by scenario ───
copy_scenario() {
  local scenario_label=$1; shift
  local targets="$@"
  for pair in $targets; do
    IFS=':' read -r src_dir_name dest_split <<< "$pair"
    src="$SRC/codes/evaluation/results/Shield/$src_dir_name"
    if [ ! -d "$src" ]; then
      continue
    fi
    for cell_dir in "$src"/*/; do
      cell=$(basename "$cell_dir")
      mkdir -p "$STAGE/metrics/$scenario_label/$dest_split"
      if [ -f "$cell_dir/metrics.json" ]; then
        cp "$cell_dir/metrics.json" "$STAGE/metrics/$scenario_label/$dest_split/${cell}.json"
      fi
    done
  done
}

# Students — Baseline / PEFT-CE / Full-FT-CE
copy_scenario "baseline"           "Baseline:Holdout"      "Baseline-SAE:SAE"
copy_scenario "shield_peft_ce"     "PEFT-CE:Holdout"       "PEFT-CE-SAE:SAE"
copy_scenario "shield_full_ft_ce"  "Full-FT-CE:Holdout"    "Full-FT-CE-SAE:SAE"

# Teachers — OOB vs FT (the OOB teachers ARE inside the Baseline dir alongside students)
# Separate them out for clarity
mkdir -p "$STAGE/metrics/teachers_oob/Holdout" "$STAGE/metrics/teachers_oob/SAE"
for t in Qwen3-4B-SafeRL Qwen3Guard-Gen-8B; do
  for d in Baseline:Holdout Baseline-SAE:SAE; do
    IFS=':' read -r sd ss <<< "$d"
    src="$SRC/codes/evaluation/results/Shield/$sd/$t/metrics.json"
    [ -f "$src" ] && cp "$src" "$STAGE/metrics/teachers_oob/$ss/${t}.json"
  done
done
copy_scenario "teachers_ft"        "Teacher-FT-PEFT-CE:Holdout"   "Teacher-FT-PEFT-CE-SAE:SAE"

# Remove teacher models from the `baseline` dir (keep only student Baselines there)
for t in Qwen3-4B-SafeRL Qwen3Guard-Gen-8B; do
  rm -f "$STAGE/metrics/baseline/Holdout/${t}.json"
  rm -f "$STAGE/metrics/baseline/SAE/${t}.json"
done

# KD Scenario 1 — 12 cells × (Holdout, SAE)
mkdir -p "$STAGE/metrics/kd_scenario1/Holdout" "$STAGE/metrics/kd_scenario1/SAE"
for dir in "$SRC/codes/evaluation/results/Shield"/KD-*-OOB; do
  [ -d "$dir" ] || continue
  dname=$(basename "$dir")
  for cell in "$dir"/*/; do
    student=$(basename "$cell")
    [ -f "$cell/metrics.json" ] && cp "$cell/metrics.json" "$STAGE/metrics/kd_scenario1/Holdout/${dname#KD-}_${student}.json"
  done
done
for dir in "$SRC/codes/evaluation/results/Shield"/KD-*-OOB-SAE; do
  [ -d "$dir" ] || continue
  dname=$(basename "$dir"); dname=${dname%-SAE}
  for cell in "$dir"/*/; do
    student=$(basename "$cell")
    [ -f "$cell/metrics.json" ] && cp "$cell/metrics.json" "$STAGE/metrics/kd_scenario1/SAE/${dname#KD-}_${student}.json"
  done
done

# KD Scenario 2 — 36 cells (Holdout only, no SAE runs)
mkdir -p "$STAGE/metrics/kd_scenario2/Holdout"
for dir in "$SRC/codes/evaluation/results/Shield"/KD-*-FT-S2-*; do
  [ -d "$dir" ] || continue
  dname=$(basename "$dir")
  for cell in "$dir"/*/; do
    if [ -f "$cell/metrics.json" ]; then
      student=$(basename "$cell")
      cp "$cell/metrics.json" "$STAGE/metrics/kd_scenario2/Holdout/${dname}_${student}.json"
    fi
  done
  # Some S2 cells had the metrics.json directly in dir (not a subdir)
  [ -f "$dir/metrics.json" ] && cp "$dir/metrics.json" "$STAGE/metrics/kd_scenario2/Holdout/${dname}.json"
done

# ─── 5. Summary stats ───
cd "$STAGE"
TOTAL_JSON=$(find metrics -name "*.json" | wc -l)
TOTAL_TEX=$(find tables -name "*.tex" | wc -l)
cat >> README.md <<EOF

---

## Package contents (this zip)

- **${TOTAL_JSON}** JSON files with full evaluation metrics (accuracy, precision,
  recall, F1, per-class, per-dialect, per-dataset, confusion matrix)
- **${TOTAL_TEX}** LaTeX tables ready to include in the paper
EOF

# ─── 6. Zip ───
cd /tmp
zip -qr "$ZIP_OUT" dia-guard-shield-results
SIZE=$(du -h "$ZIP_OUT" | cut -f1)
echo ""
echo "============================================"
echo "Built: $ZIP_OUT"
echo "Size:  $SIZE"
echo "JSON:  $TOTAL_JSON files"
echo "TeX:   $TOTAL_TEX files"
echo "============================================"
