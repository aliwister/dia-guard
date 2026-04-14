#!/usr/bin/env python3
"""
Build per-dialect and per-dataset accuracy tables for each scenario.

Produces 4 scenarios × 2 axes (dialect/dataset) × 2 splits (Holdout/SAE) =
up to 16 LaTeX tables in tables/shield/breakdowns/.

Scenarios:
  T1. OOB teachers     (Qwen3-4B-SafeRL, Qwen3Guard-Gen-8B, off-the-shelf)
  T2. FT teachers      (same, LoRA-CE fine-tuned on DIA-GUARD)
  S1. KD Scenario 1    (OOB teacher + OOB student)
  S2. KD Scenario 2    (FT teacher + {PEFT, Full, Base} student)

Students for Shield FT already covered by shield_per_dialect_top5.tex;
this script generates tables for the *teacher* and *KD* scenarios.
"""
import json
import os
from pathlib import Path

ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield")
OUT  = Path("/data/vibe_exp/dia-guard/tables/shield/breakdowns")
OUT.mkdir(parents=True, exist_ok=True)


def load_metrics(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Scenario catalogs
# ─────────────────────────────────────────────────────────────────────────────
def cell(name, holdout_dir, sae_dir):
    """Represents one eval cell with its Holdout and SAE paths."""
    return {
        "name": name,
        "holdout": ROOT / holdout_dir if holdout_dir else None,
        "sae":     ROOT / sae_dir     if sae_dir     else None,
    }


def oob_teacher_cells():
    return [
        cell("Qwen3-4B-SafeRL",  "Baseline/Qwen3-4B-SafeRL",      "Baseline-SAE/Qwen3-4B-SafeRL"),
        cell("Qwen3Guard-Gen-8B","Baseline/Qwen3Guard-Gen-8B",    "Baseline-SAE/Qwen3Guard-Gen-8B"),
    ]

def ft_teacher_cells():
    return [
        cell("Qwen3-4B-SafeRL FT",  "Teacher-FT-PEFT-CE/Qwen3-4B-SafeRL",      "Teacher-FT-PEFT-CE-SAE/Qwen3-4B-SafeRL"),
        cell("Qwen3Guard-Gen-8B FT","Teacher-FT-PEFT-CE/Qwen3Guard-Gen-8B",    "Teacher-FT-PEFT-CE-SAE/Qwen3Guard-Gen-8B"),
    ]


def kd_s1_cells():
    cells = []
    for method in ["MINILLM","GKD","TED"]:
        for t_full,t_short in [("Qwen3-4B-SafeRL","4B"),("Qwen3Guard-Gen-8B","8B")]:
            for student in ["Qwen3Guard-Gen-0.6B","Qwen3-1.7B"]:
                hdir = f"KD-{method}-{t_full}-OOB/{student}"
                sdir = f"KD-{method}-{t_full}-OOB-SAE/{student}"
                cells.append(cell(f"{method} {t_short}→{student}", hdir, sdir))
    return cells


def kd_s2_cells():
    cells = []
    for method in ["MINILLM","GKD","TED"]:
        for t_full,t_short in [("Qwen3-4B-SafeRL-FT","4B-ft"),("Qwen3Guard-Gen-8B-FT","8B-ft")]:
            for start in ["QG-PEFT","Q17-PEFT","QG-FULL","Q17-FULL","QG-BASE","Q17-BASE"]:
                hdir = None
                for root,dirs,files in os.walk(ROOT):
                    if f"KD-{method}-{t_full}-S2-{start}" in root and "metrics.json" in files:
                        hdir = Path(root).relative_to(ROOT)
                        break
                if hdir:
                    cells.append(cell(f"{method} {t_short}→{start}", str(hdir), None))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# Table builders
# ─────────────────────────────────────────────────────────────────────────────
def collect_axis_values(cells, axis_key, split="holdout"):
    """axis_key = 'per_dialect' or 'per_dataset'. Returns {cell_name: {axis_value: acc}}."""
    out = {}
    all_axes = set()
    for c in cells:
        path = c[split]
        if path is None:
            continue
        data = None
        if axis_key == "per_dialect":
            # Separate file
            pdf = path / "per_dialect.json"
            if pdf.exists():
                data = json.load(open(pdf))
        else:
            # Inside metrics.json
            m = load_metrics(path / "metrics.json")
            if m and axis_key in m:
                data = m[axis_key]
        if not data:
            continue
        accs = {k: v["accuracy"] for k,v in data.items()}
        out[c["name"]] = accs
        all_axes.update(accs.keys())
    return out, sorted(all_axes)


def latex_long(cell_results, all_axes, caption, label, filename):
    if not cell_results:
        print(f"  [skip] {filename} — no data")
        return
    col_models = list(cell_results.keys())
    lines = [
        "\\begin{landscape}",
        "\\begin{longtable}{l" + "c"*len(col_models) + "}",
        f"\\caption{{{caption}}} \\\\",
        f"\\label{{{label}}}",
        "\\toprule",
        "Key & " + " & ".join(c.replace("_","\\_") for c in col_models) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{" + str(1+len(col_models)) + "}{c}{\\tablename\\ \\thetable{} -- continued} \\\\",
        "\\toprule",
        "Key & " + " & ".join(c.replace("_","\\_") for c in col_models) + " \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for axis in all_axes:
        cells = [axis.replace("_","\\_")]
        for mn in col_models:
            v = cell_results[mn].get(axis)
            cells.append(f"{v:.4f}" if v is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
        "\\end{landscape}",
    ]
    (OUT / filename).write_text("\n".join(lines))
    print(f"  wrote {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Run it
# ─────────────────────────────────────────────────────────────────────────────
def main():
    scenarios = {
        "oob_teachers": ("OOB teachers", oob_teacher_cells()),
        "ft_teachers":  ("FT teachers",  ft_teacher_cells()),
        "kd_s1":        ("KD Scenario 1 (OOB teacher $\\to$ OOB student)", kd_s1_cells()),
        "kd_s2":        ("KD Scenario 2 (FT teacher $\\to$ FT/base student)", kd_s2_cells()),
    }
    for key, (title, cells) in scenarios.items():
        # Per-dialect, Holdout
        cr, axes = collect_axis_values(cells, "per_dialect", "holdout")
        latex_long(cr, axes,
                   f"Per-dialect accuracy — {title} (Holdout split, 181{{,}}874 samples).",
                   f"tab:{key}_per_dialect_holdout",
                   f"{key}_per_dialect_holdout.tex")

        # Per-dialect, SAE
        cr, axes = collect_axis_values(cells, "per_dialect", "sae")
        if cr:
            latex_long(cr, axes,
                       f"Per-dialect accuracy — {title} (SAE split, 36{{,}}050 samples).",
                       f"tab:{key}_per_dialect_sae",
                       f"{key}_per_dialect_sae.tex")

        # Per-dataset, Holdout
        cr, axes = collect_axis_values(cells, "per_dataset", "holdout")
        latex_long(cr, axes,
                   f"Per-source-dataset accuracy — {title} (Holdout split).",
                   f"tab:{key}_per_dataset_holdout",
                   f"{key}_per_dataset_holdout.tex")

        # Per-dataset, SAE
        cr, axes = collect_axis_values(cells, "per_dataset", "sae")
        if cr:
            latex_long(cr, axes,
                       f"Per-source-dataset accuracy — {title} (SAE split).",
                       f"tab:{key}_per_dataset_sae",
                       f"{key}_per_dataset_sae.tex")

    # Also do the student models (Baseline/PEFT-CE/Full-FT-CE)
    student_names = ["Gemma-3-270m","Qwen3Guard-Gen-0.6B","Qwen3.5-0.8B",
                     "Gemma-3-1B","Llama-3.2-1B","Qwen3-1.7B","SmolLM2-1.7B"]
    for method in ["Baseline","PEFT-CE","Full-FT-CE"]:
        key = f"students_{method.lower().replace('-','_')}"
        cells = []
        for sn in student_names:
            cells.append(cell(sn, f"{method}/{sn}", f"{method}-SAE/{sn}"))
        title = f"Shield {method} students"
        # Per-dialect, Holdout
        cr, axes = collect_axis_values(cells, "per_dialect", "holdout")
        latex_long(cr, axes,
                   f"Per-dialect accuracy — {title} (Holdout).",
                   f"tab:{key}_per_dialect_holdout",
                   f"{key}_per_dialect_holdout.tex")
        # Per-dialect, SAE
        cr, axes = collect_axis_values(cells, "per_dialect", "sae")
        latex_long(cr, axes,
                   f"Per-dialect accuracy — {title} (SAE).",
                   f"tab:{key}_per_dialect_sae",
                   f"{key}_per_dialect_sae.tex")
        # Per-dataset, Holdout
        cr, axes = collect_axis_values(cells, "per_dataset", "holdout")
        latex_long(cr, axes,
                   f"Per-source-dataset accuracy — {title} (Holdout).",
                   f"tab:{key}_per_dataset_holdout",
                   f"{key}_per_dataset_holdout.tex")
        # Per-dataset, SAE
        cr, axes = collect_axis_values(cells, "per_dataset", "sae")
        latex_long(cr, axes,
                   f"Per-source-dataset accuracy — {title} (SAE).",
                   f"tab:{key}_per_dataset_sae",
                   f"{key}_per_dataset_sae.tex")

    print(f"\nAll breakdown tables in {OUT}")


if __name__ == "__main__":
    main()
