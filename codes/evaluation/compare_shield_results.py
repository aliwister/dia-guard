#!/usr/bin/env python3
"""
Compare Shield model holdout test results across:
  - Baseline (un-fine-tuned base models)
  - PEFT/CE (LoRA fine-tune)
  - Full FT/CE (full fine-tune)

Walks codes/evaluation/results/Shield/{Baseline,PEFT-CE,Full-FT-CE}/<Model>/metrics.json
and produces:
  - A wide comparison table (terminal + markdown)
  - Δ improvement of FT methods over baseline
  - Optional per-dialect comparison if --per-dialect

Usage:
  python compare_shield_results.py
  python compare_shield_results.py --markdown > leaderboard.md
  python compare_shield_results.py --per-dialect Gemma-3-1B
  python compare_shield_results.py --csv > comparison.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path("/data/vibe_exp/dia-guard/codes/evaluation/results/Shield")

# Display order — small to big
MODELS = [
    "Gemma-3-270m",
    "Qwen3Guard-Gen-0.6B",
    "Qwen3.5-0.8B",
    "Gemma-3-1B",
    "Llama-3.2-1B",
    "Qwen3-1.7B",
    "SmolLM2-1.7B",
]

METHODS = [
    ("Baseline",   "baseline"),
    ("PEFT-CE",    "peft"),
    ("Full-FT-CE", "full_ft"),
]


def load_metrics(method_dir, model):
    f = ROOT / method_dir / model / "metrics.json"
    if not f.exists():
        return None
    return json.load(open(f))


def fmt(v, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def fmt_delta(curr, base):
    """Format improvement as +X.XX pp (percentage points)"""
    if curr is None or base is None:
        return "—"
    delta = (curr - base) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}pp"


def get_metric(metrics, key):
    if metrics is None:
        return None
    return metrics["overall"].get(key)


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def print_terminal_table(metric_key="accuracy"):
    """Print a wide comparison table to the terminal."""
    print(f"\n=== Shield {metric_key.upper()} Comparison ===\n")
    header = f"{'Model':<22}"
    for label, _ in METHODS:
        header += f" {label:>12}"
    header += f" {'Δ PEFT':>10} {'Δ Full FT':>12}"
    print(header)
    print("-" * len(header))

    for model in MODELS:
        row = f"{model:<22}"
        results = {}
        for label, method_dir_label in METHODS:
            method_dir = label  # PEFT-CE / Full-FT-CE / Baseline
            m = load_metrics(method_dir, model)
            results[label] = get_metric(m, metric_key)
            row += f" {fmt(results[label]):>12}"

        # Deltas vs baseline
        base = results.get("Baseline")
        peft = results.get("PEFT-CE")
        full = results.get("Full-FT-CE")
        row += f" {fmt_delta(peft, base):>10} {fmt_delta(full, base):>12}"
        print(row)


def print_markdown_table(metric_key="accuracy"):
    """Print a markdown comparison table."""
    print(f"\n## Shield Models — Holdout Test {metric_key.title()}\n")
    print(f"| Model | Baseline | PEFT/CE | Full FT/CE | Δ PEFT | Δ Full FT |")
    print(f"|-------|---------:|--------:|-----------:|-------:|----------:|")

    for model in MODELS:
        results = {}
        for label, _ in METHODS:
            m = load_metrics(label, model)
            results[label] = get_metric(m, metric_key)

        base = results.get("Baseline")
        peft = results.get("PEFT-CE")
        full = results.get("Full-FT-CE")

        # Bold the best method per row
        best = max((v for v in [base, peft, full] if v is not None), default=None)
        def cell(v):
            if v is None:
                return "—"
            s = f"{v:.4f}"
            return f"**{s}**" if v == best else s

        print(f"| {model} | {cell(base)} | {cell(peft)} | {cell(full)} | "
              f"{fmt_delta(peft, base)} | {fmt_delta(full, base)} |")


def print_full_metrics():
    """Print all 4 metrics (acc, prec, rec, f1) for each model × method."""
    for metric in ["accuracy", "precision", "recall", "f1"]:
        print_terminal_table(metric)
        print()


def print_csv():
    writer = csv.writer(sys.stdout)
    writer.writerow(["model", "method", "accuracy", "precision", "recall", "f1", "support"])
    for model in MODELS:
        for label, method_short in METHODS:
            m = load_metrics(label, model)
            if m is None:
                writer.writerow([model, method_short, "", "", "", "", ""])
            else:
                o = m["overall"]
                writer.writerow([model, method_short, o["accuracy"], o["precision"],
                                 o["recall"], o["f1"], o["support"]])


def print_per_dialect(model_name):
    """Compare per-dialect accuracy for a specific model across methods."""
    data = {}
    for label, _ in METHODS:
        f = ROOT / label / model_name / "per_dialect.json"
        if f.exists():
            data[label] = json.load(open(f))

    if not data:
        print(f"No per-dialect results found for {model_name}")
        return

    all_dialects = sorted(set().union(*[d.keys() for d in data.values()]))

    print(f"\n=== {model_name} — Per-Dialect Accuracy ===\n")
    header = f"{'Dialect':<48}"
    for label in data:
        header += f" {label:>14}"
    print(header)
    print("-" * len(header))

    # Sort by best method's accuracy descending
    def avg_acc(dialect):
        accs = [data[lbl].get(dialect, {}).get("accuracy", 0) for lbl in data]
        return -sum(accs) / max(len(accs), 1)

    for dialect in sorted(all_dialects, key=avg_acc):
        row = f"{dialect:<48}"
        for label in data:
            m = data[label].get(dialect, {})
            row += f" {fmt(m.get('accuracy')):>14}"
        print(row)


def print_summary():
    """Print all 4 metrics + a count of how many models we have for each method."""
    print(f"\n=== Coverage ===\n")
    print(f"{'Method':<14} {'Models with results':>22}")
    for label, _ in METHODS:
        n = sum(1 for m in MODELS if (ROOT / label / m / "metrics.json").exists())
        print(f"{label:<14} {n}/{len(MODELS):>20}")

    print_terminal_table("accuracy")
    print_terminal_table("f1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="accuracy",
                        choices=["accuracy", "precision", "recall", "f1"])
    parser.add_argument("--markdown", action="store_true",
                        help="Output markdown table")
    parser.add_argument("--csv", action="store_true",
                        help="Output CSV")
    parser.add_argument("--full", action="store_true",
                        help="Print all 4 metrics")
    parser.add_argument("--per-dialect", type=str, default=None, metavar="MODEL",
                        help="Show per-dialect accuracy for one model")
    parser.add_argument("--summary", action="store_true",
                        help="Print coverage + accuracy + F1 (default)")
    args = parser.parse_args()

    if args.csv:
        print_csv()
    elif args.markdown:
        print_markdown_table(args.metric)
    elif args.per_dialect:
        print_per_dialect(args.per_dialect)
    elif args.full:
        print_full_metrics()
    else:
        print_summary()


if __name__ == "__main__":
    main()
