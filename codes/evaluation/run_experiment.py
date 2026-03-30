"""
run_experiment.py — DIA-GUARD Experiment Orchestrator

Runs fine-tuning and/or knowledge distillation experiments with full
resumeability. Checkpoints are detected automatically so interrupted
runs continue from where they left off.

Experiment Groups
-----------------
  Group 1  → Teacher FT   (fine-tune Qwen3-4B-SafeRL or tiny-aya-global)
  Group 2  → KD Students  (distil fine-tuned teacher into a student; implicit with --stage kd/full)
  Group 3  → Student FT Baseline (fine-tune student directly, no teacher)

Pipeline Stages
---------------
  full  → Group 1 FT then Group 2 KD  (--group not needed; always teacher → student)
  ft    → FT only; requires --group 1 (teacher) or --group 3 (student baseline)
  kd    → Group 2 KD only (teacher FT model must already exist)

Output layout
-------------
  models/
    group1_teacher_ft/{full_ft|peft}/{model}/
    group2_kd_students/{minillm|gkd|ted}/{teacher}_to_{student}/
    group3_student_ft_baseline/{full_ft|peft}/{model}/

  results/
    group1_teacher_ft/{full_ft|peft}/{ce|contrastive}/{model}/
    group2_kd_students/{minillm|gkd|ted}/{teacher}_to_{student}/
    group3_student_ft_baseline/{full_ft|peft}/{ce|contrastive}/{model}/

Usage Examples
--------------
# Group 1 — Teacher full fine-tune, CE loss
python run_experiment.py \\
    --stage ft --group 1 \\
    --ft_method full_ft \\
    --teacher_model Qwen/Qwen3-4B-SafeRL \\
    --loss ce

# Group 1 → Group 2 — Teacher FT then distil to student (full pipeline)
python run_experiment.py \\
    --stage full \\
    --ft_method full_ft --kd_method minillm \\
    --teacher_model Qwen/Qwen3-4B-SafeRL \\
    --student_model HuggingFaceTB/SmolLM2-1.7B-Instruct \\
    --loss ce

# Group 2 — KD only (teacher must already be in group1_teacher_ft/)
python run_experiment.py \\
    --stage kd \\
    --kd_method ted \\
    --teacher_model Qwen/Qwen3-4B-SafeRL \\
    --student_model google/gemma-3-1b-it \\
    --ft_method full_ft

# Group 3 — Student FT baseline, contrastive loss
python run_experiment.py \\
    --stage ft --group 3 \\
    --ft_method peft \\
    --teacher_model meta-llama/Llama-3.2-1B-Instruct \\
    --loss contrastive

# Dry run (print plan, do nothing)
python run_experiment.py \\
    --stage full --ft_method full_ft --kd_method minillm \\
    --teacher_model Qwen/Qwen3-4B-SafeRL \\
    --student_model HuggingFaceTB/SmolLM2-1.7B-Instruct \\
    --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ─── Path constants ───────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent  # DIA-LLM/
EVAL_ROOT = Path(__file__).resolve().parent          # DIA-LLM/Evaluation/

MODELS_DIR   = REPO_ROOT / "models"
SPLITS_DIR   = REPO_ROOT / "DIA_Splits"
RESULTS_DIR  = EVAL_ROOT / "results"

FT_SCRIPTS = {
    ("full_ft", "ce"):          EVAL_ROOT / "FineTune/full_ft/train_ce.py",
    ("full_ft", "contrastive"): EVAL_ROOT / "FineTune/full_ft/train_contrastive.py",
    ("peft",    "ce"):          EVAL_ROOT / "FineTune/peft/train_ce_lora.py",
    ("peft",    "contrastive"): EVAL_ROOT / "FineTune/peft/train_contrastive_lora.py",
}

KD_SCRIPTS = {
    "minillm": EVAL_ROOT / "Knowledge_Distillation/minillm/train_minillm.py",
    "gkd":     EVAL_ROOT / "Knowledge_Distillation/gkd/train_gkd.py",
    "ted":     EVAL_ROOT / "Knowledge_Distillation/ted/train_ted.py",
}

FT_CONFIGS = {
    # ── Teacher models (Group 1) ──────────────────────────────────────────────
    ("full_ft", "Qwen/Qwen3-4B-SafeRL"):                    EVAL_ROOT / "FineTune/full_ft/configs/qwen3_4b.yaml",
    ("full_ft", "CohereLabs/tiny-aya-global"):               EVAL_ROOT / "FineTune/full_ft/configs/aya_3b.yaml",
    ("peft",    "Qwen/Qwen3-4B-SafeRL"):                    EVAL_ROOT / "FineTune/peft/configs/qwen3_4b_lora.yaml",
    ("peft",    "CohereLabs/tiny-aya-global"):               EVAL_ROOT / "FineTune/peft/configs/aya_3b_lora.yaml",
    # ── Student models — FT baseline (Group 3) ───────────────────────────────
    ("full_ft", "meta-llama/Llama-3.2-1B-Instruct"):        EVAL_ROOT / "FineTune/full_ft/configs/llama_1b.yaml",
    ("full_ft", "google/gemma-3-1b-it"):                    EVAL_ROOT / "FineTune/full_ft/configs/gemma_1b.yaml",
    ("full_ft", "Qwen/Qwen3Guard-Gen-0.6B"):                EVAL_ROOT / "FineTune/full_ft/configs/qwen_guard_0.6b.yaml",
    ("full_ft", "Qwen/Qwen3.5-0.8B"):                       EVAL_ROOT / "FineTune/full_ft/configs/qwen_0.8b.yaml",
    ("full_ft", "google/gemma-3-270m-it"):                  EVAL_ROOT / "FineTune/full_ft/configs/gemma_270m.yaml",
    ("full_ft", "HuggingFaceTB/SmolLM2-1.7B-Instruct"):    EVAL_ROOT / "FineTune/full_ft/configs/smollm_1.7b.yaml",
    ("full_ft", "Qwen/Qwen3-1.7B"):                         EVAL_ROOT / "FineTune/full_ft/configs/qwen_1.7b.yaml",
    ("peft",    "meta-llama/Llama-3.2-1B-Instruct"):        EVAL_ROOT / "FineTune/peft/configs/llama_1b_lora.yaml",
    ("peft",    "google/gemma-3-1b-it"):                    EVAL_ROOT / "FineTune/peft/configs/gemma_1b_lora.yaml",
    ("peft",    "Qwen/Qwen3Guard-Gen-0.6B"):                EVAL_ROOT / "FineTune/peft/configs/qwen_guard_0.6b_lora.yaml",
    ("peft",    "Qwen/Qwen3.5-0.8B"):                       EVAL_ROOT / "FineTune/peft/configs/qwen_0.8b_lora.yaml",
    ("peft",    "google/gemma-3-270m-it"):                  EVAL_ROOT / "FineTune/peft/configs/gemma_270m_lora.yaml",
    ("peft",    "HuggingFaceTB/SmolLM2-1.7B-Instruct"):    EVAL_ROOT / "FineTune/peft/configs/smollm_1.7b_lora.yaml",
    ("peft",    "Qwen/Qwen3-1.7B"):                         EVAL_ROOT / "FineTune/peft/configs/qwen_1.7b_lora.yaml",
}

EVALUATE_SCRIPT = EVAL_ROOT / "evaluate.py"


# ─── Model name utilities ─────────────────────────────────────────────────────

def model_shortname(model_id: str) -> str:
    """Convert HuggingFace model ID to a short filesystem-safe name."""
    name = model_id.split("/")[-1].lower()
    for ch in ["-", ".", " "]:
        name = name.replace(ch, "_")
    return name


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ─── Path resolution ──────────────────────────────────────────────────────────

def ft_model_dir(ft_method: str, model_id: str, group: int = 1) -> Path:
    """Expected save path for a fine-tuned model.

    group=1 → group1_teacher_ft/
    group=3 → group3_student_ft_baseline/
    """
    sub = "full_ft" if ft_method == "full_ft" else "peft"
    if group == 3:
        return MODELS_DIR / "group3_student_ft_baseline" / sub / model_shortname(model_id)
    return MODELS_DIR / "group1_teacher_ft" / sub / model_shortname(model_id)


def kd_model_dir(kd_method: str, teacher_id: str, student_id: str) -> Path:
    """Expected save path for a distilled model (always Group 2)."""
    name = f"{model_shortname(teacher_id)}_to_{model_shortname(student_id)}"
    return MODELS_DIR / "group2_kd_students" / kd_method / name


def results_dir_ft(ft_method: str, loss: str, model_id: str, group: int = 1) -> Path:
    """Results path for FT experiments.

    Structure: results/group{N}_{label}/{ft_method}/{loss}/{model}/
    No timestamp in path — reruns overwrite cleanly; timestamp is inside metrics.json.
    """
    if group == 3:
        group_dir = "group3_student_ft_baseline"
    else:
        group_dir = "group1_teacher_ft"
    return RESULTS_DIR / group_dir / ft_method / loss / model_shortname(model_id)


def results_dir_kd(kd_method: str, teacher_id: str, student_id: str) -> Path:
    """Results path for KD experiments (always Group 2).

    Structure: results/group2_kd_students/{kd_method}/{teacher}_to_{student}/
    """
    name = f"{model_shortname(teacher_id)}_to_{model_shortname(student_id)}"
    return RESULTS_DIR / "group2_kd_students" / kd_method / name


# ─── Checkpoint detection (resumeability) ─────────────────────────────────────

def is_ft_complete(ft_method: str, model_id: str, group: int = 1) -> bool:
    """
    A fine-tuning run is considered complete when the final model directory
    contains config.json (standard HuggingFace save).
    """
    model_path = ft_model_dir(ft_method, model_id, group)
    return (model_path / "config.json").exists()


def is_kd_complete(kd_method: str, teacher_id: str, student_id: str) -> bool:
    """A KD run is complete when the distilled model has config.json."""
    model_path = kd_model_dir(kd_method, teacher_id, student_id)
    return (model_path / "config.json").exists()


def find_latest_ft_checkpoint(ft_method: str, model_id: str, group: int = 1) -> Path | None:
    """
    Look for the latest 'checkpoint-*' subdirectory in the model save dir.
    Used to resume interrupted fine-tuning.
    """
    model_path = ft_model_dir(ft_method, model_id, group)
    if not model_path.exists():
        return None
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


def find_latest_kd_checkpoint(
    kd_method: str, teacher_id: str, student_id: str
) -> Path | None:
    model_path = kd_model_dir(kd_method, teacher_id, student_id)
    if not model_path.exists():
        return None
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None


# ─── State persistence ────────────────────────────────────────────────────────

def state_file(experiment_key: str) -> Path:
    return RESULTS_DIR / ".state" / f"{experiment_key}.json"


def load_state(experiment_key: str) -> dict:
    path = state_file(experiment_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"ft_done": False, "kd_done": False, "eval_ft_done": False, "eval_kd_done": False}


def save_state(experiment_key: str, state: dict) -> None:
    path = state_file(experiment_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def experiment_key(args) -> str:
    group = getattr(args, "group", None) or (2 if args.kd_method else 1)
    parts = [f"g{group}", args.stage, args.ft_method or "noft"]
    if args.kd_method:
        parts.append(args.kd_method)
    parts.append(model_shortname(args.teacher_model))
    if args.student_model:
        parts.append(model_shortname(args.student_model))
    parts.append(args.loss)
    return "_".join(parts)


# ─── Subprocess runner ────────────────────────────────────────────────────────

def run_command(cmd: list[str], dry_run: bool = False, env: dict | None = None) -> int:
    """
    Run a subprocess command. Streams output live. Returns exit code.
    If dry_run=True, prints the command instead.
    """
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n$ {cmd_str}\n")

    if dry_run:
        print("  [DRY RUN — command not executed]")
        return 0

    result = subprocess.run(cmd, env=env)
    return result.returncode


# ─── FT stage ─────────────────────────────────────────────────────────────────

def run_ft(args, state: dict, dry_run: bool = False) -> bool:
    """
    Run fine-tuning. Returns True on success.
    Skips automatically if already complete.
    """
    group = args.group
    if state.get("ft_done") or is_ft_complete(args.ft_method, args.teacher_model, group):
        print(f"[FT] Already complete: {ft_model_dir(args.ft_method, args.teacher_model, group)}")
        state["ft_done"] = True
        return True

    script_key = (args.ft_method, args.loss)
    if script_key not in FT_SCRIPTS:
        print(
            f"[FT] Error: no script for ft_method={args.ft_method!r}, loss={args.loss!r}",
            file=sys.stderr,
        )
        return False

    script = FT_SCRIPTS[script_key]
    out_dir = ft_model_dir(args.ft_method, args.teacher_model, group)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config file
    cfg_key = (args.ft_method, args.teacher_model)
    config_arg = []
    if cfg_key in FT_CONFIGS:
        config_arg = ["--config", str(FT_CONFIGS[cfg_key])]

    # Resume from checkpoint if one exists
    resume_arg = []
    ckpt = find_latest_ft_checkpoint(args.ft_method, args.teacher_model, group)
    if ckpt:
        print(f"[FT] Resuming from checkpoint: {ckpt}")
        resume_arg = ["--resume_from_checkpoint", str(ckpt)]

    group_label = "Teacher FT" if group == 1 else "Student FT Baseline"
    cmd = [
        sys.executable,
        str(script),
        "--model_name", args.teacher_model,
        "--output_dir", str(out_dir),
        "--train_data", str(SPLITS_DIR / "train.jsonl"),
        "--val_data",   str(SPLITS_DIR / "val.jsonl"),
        *config_arg,
        *resume_arg,
    ]

    print(f"\n{'='*60}")
    print(f"[FT] Group {group}: {group_label}")
    print(f"[FT] Method: {args.ft_method.upper()} | Loss: {args.loss}")
    print(f"[FT] Model: {args.teacher_model}")
    print(f"[FT] Output: {out_dir}")
    print(f"{'='*60}")

    rc = run_command(cmd, dry_run=dry_run)
    if rc != 0:
        print(f"[FT] Error: training exited with code {rc}", file=sys.stderr)
        return False

    # Handle PEFT: merge adapters before using as teacher
    if args.ft_method == "peft":
        merge_script = EVAL_ROOT / "FineTune/peft/merge_lora.py"
        merged_dir = out_dir / "merged"
        merge_cmd = [
            sys.executable,
            str(merge_script),
            "--base_model", args.teacher_model,
            "--adapter_dir", str(out_dir),
            "--output_dir", str(merged_dir),
        ]
        print(f"\n[FT] Merging LoRA adapters → {merged_dir}")
        rc2 = run_command(merge_cmd, dry_run=dry_run)
        if rc2 != 0:
            print(f"[FT] Warning: merge exited with code {rc2}", file=sys.stderr)

    state["ft_done"] = True
    return True


# ─── KD stage ─────────────────────────────────────────────────────────────────

def get_teacher_path(args) -> Path:
    """
    Resolve the teacher model path for KD.
    Always reads from group1_teacher_ft (KD depends on Group 1 output).
    Uses merged PEFT model if ft_method=peft, otherwise full_ft save.
    """
    base = ft_model_dir(args.ft_method, args.teacher_model, group=1)
    if args.ft_method == "peft":
        merged = base / "merged"
        if merged.exists():
            return merged
    return base


def run_kd(args, state: dict, dry_run: bool = False) -> bool:
    """
    Run knowledge distillation. Returns True on success.
    Requires fine-tuned teacher model to exist.
    """
    if state.get("kd_done") or is_kd_complete(
        args.kd_method, args.teacher_model, args.student_model
    ):
        print(
            f"[KD] Already complete: "
            f"{kd_model_dir(args.kd_method, args.teacher_model, args.student_model)}"
        )
        state["kd_done"] = True
        return True

    # Dependency check: teacher FT model must exist
    teacher_path = get_teacher_path(args)
    if not (teacher_path / "config.json").exists():
        print(
            f"[KD] Error: teacher model not found at {teacher_path}\n"
            f"       Run --stage ft first, or use --stage full.",
            file=sys.stderr,
        )
        return False

    if args.kd_method not in KD_SCRIPTS:
        print(f"[KD] Error: unknown kd_method={args.kd_method!r}", file=sys.stderr)
        return False

    script = KD_SCRIPTS[args.kd_method]
    out_dir = kd_model_dir(args.kd_method, args.teacher_model, args.student_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if one exists
    resume_arg = []
    ckpt = find_latest_kd_checkpoint(args.kd_method, args.teacher_model, args.student_model)
    if ckpt:
        print(f"[KD] Resuming from checkpoint: {ckpt}")
        resume_arg = ["--resume_from_checkpoint", str(ckpt)]

    cmd = [
        sys.executable,
        str(script),
        "--teacher_model", str(teacher_path),
        "--student_model", args.student_model,
        "--output_dir",   str(out_dir),
        "--train_data",   str(SPLITS_DIR / "train.jsonl"),
        "--val_data",     str(SPLITS_DIR / "val.jsonl"),
        *resume_arg,
    ]

    print(f"\n{'='*60}")
    print(f"[KD] Method: {args.kd_method.upper()}")
    print(f"[KD] Teacher: {teacher_path}")
    print(f"[KD] Student: {args.student_model}")
    print(f"[KD] Output: {out_dir}")
    print(f"{'='*60}")

    rc = run_command(cmd, dry_run=dry_run)
    if rc != 0:
        print(f"[KD] Error: distillation exited with code {rc}", file=sys.stderr)
        return False

    state["kd_done"] = True
    return True


# ─── Evaluation ───────────────────────────────────────────────────────────────

def run_eval_ft(args, state: dict, dry_run: bool = False) -> bool:
    if state.get("eval_ft_done"):
        print("[EVAL-FT] Already done.")
        return True

    group = args.group
    model_path = ft_model_dir(args.ft_method, args.teacher_model, group)
    if args.ft_method == "peft" and (model_path / "merged").exists():
        model_path = model_path / "merged"

    if not (model_path / "config.json").exists():
        print(f"[EVAL-FT] Model not found at {model_path}, skipping evaluation.")
        return False

    out_dir = results_dir_ft(args.ft_method, args.loss, args.teacher_model, group)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--model_dir",  str(model_path),
        "--test_data",  str(SPLITS_DIR / "test.jsonl"),
        "--output_dir", str(out_dir),
        "--stage",  "FT",
        "--method", args.ft_method,
        "--loss",   args.loss,
    ]

    print(f"\n[EVAL-FT] Evaluating FT model → {out_dir}")
    rc = run_command(cmd, dry_run=dry_run)
    if rc != 0:
        print(f"[EVAL-FT] Warning: evaluate.py exited with code {rc}", file=sys.stderr)
        return False

    state["eval_ft_done"] = True
    state["eval_ft_dir"] = str(out_dir)
    return True


def run_eval_kd(args, state: dict, dry_run: bool = False) -> bool:
    if state.get("eval_kd_done"):
        print("[EVAL-KD] Already done.")
        return True

    model_path = kd_model_dir(args.kd_method, args.teacher_model, args.student_model)

    if not (model_path / "config.json").exists():
        print(f"[EVAL-KD] Model not found at {model_path}, skipping evaluation.")
        return False

    out_dir = results_dir_kd(args.kd_method, args.teacher_model, args.student_model)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--model_dir",     str(model_path),
        "--test_data",     str(SPLITS_DIR / "test.jsonl"),
        "--output_dir",    str(out_dir),
        "--stage",         "KD",
        "--method",        args.kd_method,
        "--loss",          "ce",
        "--teacher_model", args.teacher_model,
        "--student_model", args.student_model,
    ]

    print(f"\n[EVAL-KD] Evaluating KD model → {out_dir}")
    rc = run_command(cmd, dry_run=dry_run)
    if rc != 0:
        print(f"[EVAL-KD] Warning: evaluate.py exited with code {rc}", file=sys.stderr)
        return False

    state["eval_kd_done"] = True
    state["eval_kd_dir"] = str(out_dir)
    return True


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(state: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for step, label in [
        ("ft_done",      "Fine-Tuning"),
        ("kd_done",      "Knowledge Distillation"),
        ("eval_ft_done", "Evaluation (FT model)"),
        ("eval_kd_done", "Evaluation (KD model)"),
    ]:
        status = "✓ Done" if state.get(step) else "— Skipped"
        print(f"  {label:<35} {status}")

    for key in ("eval_ft_dir", "eval_kd_dir"):
        if state.get(key):
            print(f"\n  Results: {state[key]}")

    print()


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DIA-GUARD experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--stage",
        choices=["full", "ft", "kd"],
        default="full",
        help="Experiment stage: 'full' (FT+KD), 'ft' (FT only), 'kd' (KD only)",
    )
    parser.add_argument(
        "--ft_method",
        choices=["full_ft", "peft"],
        default="full_ft",
        help="Fine-tuning method",
    )
    parser.add_argument(
        "--kd_method",
        choices=["minillm", "gkd", "ted"],
        default=None,
        help="Knowledge distillation method (required for stage=kd or stage=full)",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen3-4B-SafeRL",
        help="HuggingFace model ID for the teacher (>2B param) model",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default=None,
        help="HuggingFace model ID for the student (<2B param) model (required for KD)",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "contrastive"],
        default="ce",
        help="Loss type for fine-tuning",
    )
    parser.add_argument(
        "--group",
        type=int,
        choices=[1, 3],
        default=None,
        help=(
            "Experiment group for FT stage: "
            "1=Teacher FT (group1_teacher_ft/), "
            "3=Student FT Baseline (group3_student_ft_baseline/). "
            "Required when --stage is 'ft'. "
            "Ignored for --stage 'full' (always group 1→2) and --stage 'kd' (always group 2)."
        ),
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore saved state and re-run all steps",
    )

    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate arguments ────────────────────────────────────────────────────
    if args.stage in ("full", "kd"):
        if not args.kd_method:
            print(
                "Error: --kd_method is required when --stage is 'full' or 'kd'",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.student_model:
            print(
                "Error: --student_model is required when --stage is 'full' or 'kd'",
                file=sys.stderr,
            )
            sys.exit(1)

    # Resolve effective group:
    #   --stage full  → group 1 for FT portion, group 2 implicit for KD
    #   --stage ft    → --group required (1 or 3)
    #   --stage kd    → group 2 (no --group needed)
    if args.stage == "ft" and args.group is None:
        print(
            "Error: --group is required when --stage is 'ft'. "
            "Use --group 1 (Teacher FT) or --group 3 (Student FT Baseline).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.stage == "full":
        args.group = 1   # full pipeline always fine-tunes the teacher first
    elif args.stage == "kd":
        args.group = 1   # KD reads teacher from group1; kd_model_dir always → group2

    exp_key = experiment_key(args)

    # Load or reset state
    if args.reset:
        state = {
            "ft_done": False,
            "kd_done": False,
            "eval_ft_done": False,
            "eval_kd_done": False,
        }
        print(f"[*] State reset for experiment: {exp_key}")
    else:
        state = load_state(exp_key)

    group_labels = {1: "Group 1 — Teacher FT", 2: "Group 2 — KD Students", 3: "Group 3 — Student FT Baseline"}
    effective_group = args.group if args.stage != "kd" else 2
    print(f"\n{'='*60}")
    print(f"DIA-GUARD Experiment: {exp_key}")
    print(f"Group     : {group_labels.get(effective_group, '')}")
    print(f"Stage     : {args.stage}")
    print(f"FT method : {args.ft_method} | Loss: {args.loss}")
    if args.kd_method:
        print(f"KD method : {args.kd_method}")
    print(f"Teacher   : {args.teacher_model}")
    if args.student_model:
        print(f"Student   : {args.student_model}")
    print(f"Dry run   : {args.dry_run}")
    print(f"{'='*60}\n")

    # ── Fine-Tuning ───────────────────────────────────────────────────────────
    if args.stage in ("full", "ft"):
        ok = run_ft(args, state, dry_run=args.dry_run)
        save_state(exp_key, state)
        if not ok:
            print("[*] Fine-tuning failed. Aborting.", file=sys.stderr)
            print_summary(state)
            sys.exit(1)

        if not args.skip_eval:
            run_eval_ft(args, state, dry_run=args.dry_run)
            save_state(exp_key, state)

    # ── Knowledge Distillation ────────────────────────────────────────────────
    if args.stage in ("full", "kd"):
        ok = run_kd(args, state, dry_run=args.dry_run)
        save_state(exp_key, state)
        if not ok:
            print("[*] Knowledge distillation failed. Aborting.", file=sys.stderr)
            print_summary(state)
            sys.exit(1)

        if not args.skip_eval:
            run_eval_kd(args, state, dry_run=args.dry_run)
            save_state(exp_key, state)

    print_summary(state)


if __name__ == "__main__":
    main()
