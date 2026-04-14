#!/usr/bin/env python3
"""
Build all Shield LaTeX tables from committed metrics.json files.

Writes six .tex files into tables/shield/:
  1. shield_ft_holdout.tex       — Baseline + PEFT-CE + Full-FT-CE × 7 models on Holdout
  2. shield_ft_sae.tex            — same on SAE
  3. shield_ft_by_dataset.tex    — compact Holdout vs SAE side-by-side
  4. kd_scenario1.tex             — Scenario 1 OOB KD (12 cells)
  5. kd_scenario2.tex             — Scenario 2 FT KD (36 cells)
  6. shield_per_dialect_top5.tex — top-5 models × 50 dialects
"""
import json
import os
from pathlib import Path

ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield")
OUT  = Path("/data/vibe_exp/dia-guard/tables/shield")

MODELS_7 = [
    "Gemma-3-270m", "Qwen3Guard-Gen-0.6B", "Qwen3.5-0.8B",
    "Gemma-3-1B", "Llama-3.2-1B", "Qwen3-1.7B", "SmolLM2-1.7B",
]

def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v, prec=4, na="--"):
    return f"{v:.{prec}f}" if v is not None else na


def row(model, metrics, extra=""):
    if metrics is None:
        return f"{model} & -- & -- & -- & -- & -- \\\\{extra}"
    o = metrics["overall"]; cm = metrics["confusion_matrix"]
    asr = cm["tp"] / (cm["tp"] + cm["fn"]) * 100 if (cm["tp"] + cm["fn"]) else 0
    return (f"{model} & {o['accuracy']:.4f} & {o['precision']:.4f} & {o['recall']:.4f} "
            f"& {asr:.2f}\\% & {o['f1']:.4f} \\\\{extra}")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Shield FT on Holdout
# ─────────────────────────────────────────────────────────────────────────────
def table_shield_ft(split, method_dirs, filename, caption):
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "ccccc" * len(method_dirs) + "}",
        "\\toprule",
    ]
    # Multi-col header
    hdr1 = "Model"
    for _ in method_dirs:
        hdr1 += " & \\multicolumn{5}{c}{" + _["name"] + "}"
    lines.append(hdr1 + " \\\\")
    hdr2 = ""
    for _ in method_dirs:
        hdr2 += " & Acc & Prec & Rec & ASR & F1"
    lines.append(hdr2 + " \\\\")
    lines.append("\\midrule")

    for model in MODELS_7:
        row_parts = [model]
        for md in method_dirs:
            m = load(ROOT / md["dir"] / model / "metrics.json")
            if m is None:
                row_parts.append("-- & -- & -- & -- & --")
            else:
                o = m["overall"]; cm = m["confusion_matrix"]
                asr = cm["tp"] / (cm["tp"] + cm["fn"]) * 100 if (cm["tp"] + cm["fn"]) else 0
                row_parts.append(f"{o['accuracy']:.4f} & {o['precision']:.4f} & {o['recall']:.4f} & {asr:.2f}\\% & {o['f1']:.4f}")
        lines.append(" & ".join(row_parts) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:shield_{split}}}",
        "\\end{table*}",
    ]
    (OUT / filename).write_text("\n".join(lines))
    print(f"  wrote {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# Table: KD Scenario 1
# ─────────────────────────────────────────────────────────────────────────────
def table_kd_s1():
    students = ["Qwen3Guard-Gen-0.6B", "Qwen3-1.7B"]
    methods  = ["MINILLM", "GKD", "TED"]
    teachers = [("Qwen3-4B-SafeRL", "4B"), ("Qwen3Guard-Gen-8B", "8B")]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Student & Method & Teacher & Accuracy & Precision & Recall & ASR & F1 \\\\",
        "\\midrule",
    ]
    for student in students:
        for method in methods:
            for t_full, t_short in teachers:
                dname = f"KD-{method}-{t_full}-OOB"
                m = load(ROOT / dname / student / "metrics.json")
                if m is None:
                    continue
                o = m["overall"]; cm = m["confusion_matrix"]
                asr = cm["tp"]/(cm["tp"]+cm["fn"])*100 if (cm["tp"]+cm["fn"]) else 0
                lines.append(f"{student} & {method} & {t_short} & {o['accuracy']:.4f} & {o['precision']:.4f} & {o['recall']:.4f} & {asr:.2f}\\% & {o['f1']:.4f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Scenario 1 KD (off-the-shelf teacher and student) results on the DIA-GUARD holdout test set.}",
        "\\label{tab:kd_scenario1}",
        "\\end{table*}",
    ]
    (OUT / "kd_scenario1.tex").write_text("\n".join(lines))
    print("  wrote kd_scenario1.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table: KD Scenario 2
# ─────────────────────────────────────────────────────────────────────────────
def table_kd_s2():
    student_pretty = {"qg-peft": "QwenGuard-0.6B PEFT", "q17-peft": "Qwen3-1.7B PEFT",
                      "qg-full": "QwenGuard-0.6B Full-FT", "q17-full": "Qwen3-1.7B Full-FT",
                      "qg-base": "QwenGuard-0.6B Base", "q17-base": "Qwen3-1.7B Base"}
    starts = ["qg-peft","q17-peft","qg-full","q17-full","qg-base","q17-base"]
    methods = ["MINILLM", "GKD", "TED"]
    teachers = [("Qwen3-4B-SafeRL-FT", "4B-ft"), ("Qwen3Guard-Gen-8B-FT", "8B-ft")]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllccccc}",
        "\\toprule",
        "Method & Teacher & Student & Starting & Accuracy & Precision & Recall & ASR & F1 \\\\",
        "\\midrule",
    ]
    for method in methods:
        for t_full, t_short in teachers:
            for start in starts:
                dname = f"KD-{method}-{t_full}-S2-{start.upper()}"
                results_dir = ROOT / dname
                if not results_dir.exists():
                    continue
                # Find metrics.json anywhere in there
                metrics_file = None
                for root,dirs,files in os.walk(results_dir):
                    if "metrics.json" in files:
                        metrics_file = os.path.join(root,"metrics.json")
                        break
                if not metrics_file: continue
                m = json.load(open(metrics_file))
                o = m["overall"]; cm = m["confusion_matrix"]
                asr = cm["tp"]/(cm["tp"]+cm["fn"])*100 if (cm["tp"]+cm["fn"]) else 0
                student_label = "QwenGuard-0.6B" if start.startswith("qg") else "Qwen3-1.7B"
                start_label = start.split("-")[1].upper()
                lines.append(f"{method} & {t_short} & {student_label} & {start_label} & {o['accuracy']:.4f} & {o['precision']:.4f} & {o['recall']:.4f} & {asr:.2f}\\% & {o['f1']:.4f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Scenario 2 KD (FT teacher \\& FT student) results on the DIA-GUARD holdout test set. `PEFT', `FULL', `BASE' indicate the student starting point.}",
        "\\label{tab:kd_scenario2}",
        "\\end{table*}",
    ]
    (OUT / "kd_scenario2.tex").write_text("\n".join(lines))
    print("  wrote kd_scenario2.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table: Per-dialect for top models (Holdout)
# ─────────────────────────────────────────────────────────────────────────────
def table_teachers():
    """Teacher models — off-the-shelf (OOB) vs DIA-GUARD LoRA-CE fine-tuned (FT), on Holdout and SAE."""
    teachers = ["Qwen3-4B-SafeRL", "Qwen3Guard-Gen-8B"]
    dirs = [
        ("OOB",  "Holdout", "Baseline"),
        ("OOB",  "SAE",     "Baseline-SAE"),
        ("FT",   "Holdout", "Teacher-FT-PEFT-CE"),
        ("FT",   "SAE",     "Teacher-FT-PEFT-CE-SAE"),
    ]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Teacher & Variant & Split & Accuracy & Precision & Recall & ASR & F1 \\\\",
        "\\midrule",
    ]
    for teacher in teachers:
        for variant, split, dirn in dirs:
            m = load(ROOT / dirn / teacher / "metrics.json")
            if m is None:
                continue
            o = m["overall"]; cm = m["confusion_matrix"]
            asr = cm["tp"]/(cm["tp"]+cm["fn"])*100 if (cm["tp"]+cm["fn"]) else 0
            lines.append(f"{teacher} & {variant} & {split} & {o['accuracy']:.4f} & {o['precision']:.4f} & {o['recall']:.4f} & {asr:.2f}\\% & {o['f1']:.4f} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Teacher models used in Scenarios 1 \\& 2 KD, evaluated on the dialect Holdout (181{,}874) and SAE (36{,}050) test sets. `OOB' = off-the-shelf, `FT' = LoRA-CE fine-tuned on DIA-GUARD.}",
        "\\label{tab:teachers}",
        "\\end{table*}",
    ]
    (OUT / "teachers.tex").write_text("\n".join(lines))
    print("  wrote teachers.tex")


def table_per_dialect():
    # Pick the top 5 by holdout accuracy
    candidates = [
        ("SmolLM2-1.7B PEFT-CE", "PEFT-CE", "SmolLM2-1.7B"),
        ("Gemma-3-1B Full-FT-CE", "Full-FT-CE", "Gemma-3-1B"),
        ("Gemma-3-270m Full-FT-CE", "Full-FT-CE", "Gemma-3-270m"),
        ("Llama-3.2-1B Full-FT-CE", "Full-FT-CE", "Llama-3.2-1B"),
        ("Qwen3.5-0.8B Full-FT-CE", "Full-FT-CE", "Qwen3.5-0.8B"),
    ]
    per_dialect_data = {}
    for label, method_dir, model in candidates:
        f = ROOT / method_dir / model / "per_dialect.json"
        if f.exists():
            per_dialect_data[label] = json.load(open(f))

    if not per_dialect_data:
        print("  per_dialect: no data, skipping")
        return

    # All dialects
    all_dialects = sorted(set().union(*[d.keys() for d in per_dialect_data.values()]))

    lines = [
        "\\begin{longtable}{l" + "c" * len(per_dialect_data) + "}",
        "\\caption{Per-dialect accuracy for top 5 Shield models on the DIA-GUARD holdout test set.} \\\\",
        "\\label{tab:shield_per_dialect}",
        "\\toprule",
        "Dialect & " + " & ".join(per_dialect_data.keys()) + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\multicolumn{" + str(1 + len(per_dialect_data)) + "}{c}{\\tablename\\ \\thetable{} -- continued} \\\\",
        "\\toprule",
        "Dialect & " + " & ".join(per_dialect_data.keys()) + " \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for d in all_dialects:
        cells = [d.replace("_", "\\_")]
        for k in per_dialect_data:
            v = per_dialect_data[k].get(d, {}).get("accuracy")
            cells.append(f"{v:.4f}" if v is not None else "--")
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{longtable}",
    ]
    (OUT / "shield_per_dialect_top5.tex").write_text("\n".join(lines))
    print("  wrote shield_per_dialect_top5.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table: Holdout vs SAE side-by-side
# ─────────────────────────────────────────────────────────────────────────────
def table_by_dataset():
    methods = [("Baseline", "Baseline", "Baseline-SAE"),
               ("PEFT-CE",  "PEFT-CE",  "PEFT-CE-SAE"),
               ("Full-FT-CE", "Full-FT-CE", "Full-FT-CE-SAE")]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\begin{tabular}{ll" + "cc" * 3 + "}",
        "\\toprule",
        "\\multirow{2}{*}{Model} & \\multirow{2}{*}{Method} & \\multicolumn{2}{c}{Accuracy} & \\multicolumn{2}{c}{F1} & \\multicolumn{2}{c}{ASR} \\\\",
        "& & Holdout & SAE & Holdout & SAE & Holdout & SAE \\\\",
        "\\midrule",
    ]
    for model in MODELS_7:
        for mname, hdir, sdir in methods:
            mh = load(ROOT / hdir / model / "metrics.json")
            ms = load(ROOT / sdir / model / "metrics.json")
            def stats(m):
                if m is None: return ("--","--","--")
                o = m["overall"]; cm = m["confusion_matrix"]
                asr = cm["tp"]/(cm["tp"]+cm["fn"])*100 if (cm["tp"]+cm["fn"]) else 0
                return (f"{o['accuracy']:.4f}", f"{o['f1']:.4f}", f"{asr:.2f}\\%")
            ah, fh, rh = stats(mh)
            asa, fsa, rsa = stats(ms)
            lines.append(f"{model} & {mname} & {ah} & {asa} & {fh} & {fsa} & {rh} & {rsa} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Shield fine-tuning results by model and method, comparing dialect Holdout vs SAE test sets.}",
        "\\label{tab:shield_by_dataset}",
        "\\end{table*}",
    ]
    (OUT / "shield_ft_by_dataset.tex").write_text("\n".join(lines))
    print("  wrote shield_ft_by_dataset.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Table 1: FT Holdout
    table_shield_ft(
        "holdout",
        [{"name":"Baseline", "dir":"Baseline"},
         {"name":"PEFT-CE",  "dir":"PEFT-CE"},
         {"name":"Full-FT-CE","dir":"Full-FT-CE"}],
        "shield_ft_holdout.tex",
        "Shield fine-tuning results on the dialect Holdout test set (181{,}874 samples).",
    )
    # Table 2: FT SAE
    table_shield_ft(
        "sae",
        [{"name":"Baseline", "dir":"Baseline-SAE"},
         {"name":"PEFT-CE",  "dir":"PEFT-CE-SAE"},
         {"name":"Full-FT-CE","dir":"Full-FT-CE-SAE"}],
        "shield_ft_sae.tex",
        "Shield fine-tuning results on the SAE test set (36{,}050 samples).",
    )
    # Table 3: By-dataset compact
    table_by_dataset()
    # Table 4 + 5: KD Scenarios
    table_kd_s1()
    table_kd_s2()
    # Table 6: Per-dialect
    table_per_dialect()
    # Table 7: Teachers (OOB vs FT × Holdout vs SAE)
    table_teachers()

    print("\nAll tables written to", OUT)


if __name__ == "__main__":
    main()
