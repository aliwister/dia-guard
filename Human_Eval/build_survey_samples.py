#!/usr/bin/env python3
"""Build stratified samples for the human-evaluation study.

Set A (Harm Preservation): 50 items, one per dialect, stratified across
    the 15 upstream source datasets.  Each item exposes the SAE unsafe
    source alongside both the Basic and CoI dialect transforms so raters
    can score harm preservation on either form.

Set B (Counterharm Neutralization): 60 items split 20/20/20 across:
    B1 = (SAE unsafe, counterharm of SAE)
    B2 = (Multi-VALUE dialect unsafe, counterharm of MV dialect)
    B3 = (DIA-LLM dialect unsafe, counterharm of DIA-LLM dialect)

Length filter: every text field <= 250 characters so items are quick to
rate.  Counterharm items are drawn only from validated rows.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/ec2-user/dia-guard/dataset")
DIA_DIR = ROOT / "dia_llm"
MV_DIR = ROOT / "multi_value"
OUT_DIR = Path("/home/ec2-user/dia-guard/codes/human_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = 250
SEED = 42
rng = random.Random(SEED)

DATASETS = [
    "advbench", "Salad_Bench", "Simple_Safety_Tests", "Toxic_Text",
    "Toxicity_Jigsaw", "bipia", "cyberseceval", "do_not_answer",
    "forbiddent_questions", "harmBench", "injecagent", "jailbreakbench",
    "llmseceval", "securityeval", "sorry_bench",
]

# ---------- dia_llm loader ----------
def load_dia_llm():
    """Return list of dicts with SAE + basic/coi transforms + counterharm."""
    rows = []
    for dialect_dir in sorted(DIA_DIR.iterdir()):
        if not dialect_dir.is_dir():
            continue
        dialect = dialect_dir.name
        for ds in DATASETS:
            csv_path = dialect_dir / f"{ds}_zeroshot_harmfulness_results_with_transforms.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    sae = (r.get("original_input") or "").strip()
                    basic = (r.get("basic_transform") or "").strip()
                    coi = (r.get("coi_transform") or "").strip()
                    ch_orig = (r.get("counterharm_original") or "").strip()
                    ch_basic = (r.get("counterharm_basic") or "").strip()
                    ch_coi = (r.get("counterharm_coi") or "").strip()
                    validated = (r.get("counterharm_validated") or "").strip().lower() == "true"
                    rows.append({
                        "dataset": ds,
                        "dialect": dialect,
                        "sample_id": r.get("sample_id", ""),
                        "sae_unsafe": sae,
                        "basic_transform": basic,
                        "coi_transform": coi,
                        "counterharm_sae": ch_orig,
                        "counterharm_basic": ch_basic,
                        "counterharm_coi": ch_coi,
                        "counterharm_validated": validated,
                    })
    return rows


# ---------- multi_value loader ----------
def load_multi_value():
    rows = []
    for dialect_dir in sorted(MV_DIR.iterdir()):
        if not dialect_dir.is_dir():
            continue
        dialect = dialect_dir.name
        for ds in DATASETS:
            csv_path = dialect_dir / f"{ds}_{dialect}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fns = reader.fieldnames or []
                # Figure out the prompt columns
                if "prompt" in fns and "prompt_transformed" in fns:
                    orig_col, trans_col = "prompt", "prompt_transformed"
                else:
                    # fall back to first (col, col_transformed) pair
                    orig_col = trans_col = None
                    for c in fns:
                        if c.endswith("_transformed"):
                            base = c[: -len("_transformed")]
                            if base in fns:
                                orig_col, trans_col = base, c
                                break
                    if not orig_col:
                        continue
                for r in reader:
                    sae = (r.get(orig_col) or "").strip()
                    mv = (r.get(trans_col) or "").strip()
                    ch_prompt = (r.get("counterharm_prompt") or "").strip()
                    ch_mv = (r.get("counterharm_prompt_transformed") or "").strip()
                    validated = (r.get("counterharm_validated") or "").strip().lower() == "true"
                    rows.append({
                        "dataset": ds,
                        "dialect": dialect,
                        "sae_unsafe": sae,
                        "mv_transform": mv,
                        "counterharm_sae": ch_prompt,
                        "counterharm_mv": ch_mv,
                        "counterharm_validated": validated,
                    })
    return rows


def short(*texts):
    return all(0 < len(t) <= MAX_LEN for t in texts)


def mostly_ascii(text, threshold=0.9):
    """Reject outputs that drifted into non-Latin scripts (e.g., Swahili,
    Arabic, CJK).  Dialectal English should stay >=90% ASCII."""
    if not text:
        return False
    printable = sum(1 for c in text if ord(c) < 128)
    return printable / len(text) >= threshold


REFUSAL_MARKERS = (
    "guardrail policy violation",
    "i cannot", "i can't", "i'm sorry", "i am sorry",
    "i'm unable", "i am unable",
    "unable to help", "cannot assist", "can't assist",
    "as an ai", "as a language model",
    "content policy", "refuse to",
)


def not_refusal(text):
    low = text.lower().strip()
    return not any(m in low for m in REFUSAL_MARKERS)


ENGLISH_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "by", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "may", "might", "must", "not", "no", "yes", "that", "this", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "what", "which", "who",
    "how", "when", "where", "why", "if", "so", "because", "about", "all", "any",
    "some", "one", "two", "more", "most", "other", "there", "up", "out", "into",
    "over", "under", "again", "just", "also", "too", "very", "like", "make",
    "made", "get", "got", "give", "gave", "take", "took", "use", "used",
    # common dialectal forms
    "ya", "dem", "dat", "di", "fi", "innit", "nah", "lah", "eh", "mate",
    "bro", "bruh", "wah", "mek", "inna", "nuh",
}


def looks_english(text, min_ratio=0.25):
    """True iff >= min_ratio of tokens are recognisable English (or common
    dialectal) function words.  Filters out outputs that slipped into
    Gaelic, Swahili, Tagalog, etc."""
    tokens = [t.lower().strip(".,!?;:\"'()[]{}") for t in text.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < 3:
        return False
    hits = sum(1 for t in tokens if t in ENGLISH_FUNCTION_WORDS)
    return hits / len(tokens) >= min_ratio


def well_formed(text, min_chars=20, min_words=4):
    text = text.strip()
    return len(text) >= min_chars and len(text.split()) >= min_words


def differs(*pairs):
    """Require every (a, b) pair to differ after normalization."""
    def norm(s):
        return " ".join(s.lower().split())
    for a, b in pairs:
        if norm(a) == norm(b):
            return False
    return True


# ---------- Set A sampler ----------
def build_set_a(dia_rows):
    """1 item per dialect, stratified across 15 datasets."""
    usable = [r for r in dia_rows
              if short(r["sae_unsafe"], r["basic_transform"], r["coi_transform"])
              and mostly_ascii(r["sae_unsafe"])
              and mostly_ascii(r["basic_transform"])
              and mostly_ascii(r["coi_transform"])
              and not_refusal(r["basic_transform"])
              and not_refusal(r["coi_transform"])
              and well_formed(r["basic_transform"])
              and well_formed(r["coi_transform"])
              and well_formed(r["sae_unsafe"])
              and looks_english(r["basic_transform"])
              and looks_english(r["coi_transform"])
              and differs((r["sae_unsafe"], r["basic_transform"]),
                          (r["sae_unsafe"], r["coi_transform"]))]
    by_dialect = defaultdict(list)
    for r in usable:
        by_dialect[r["dialect"]].append(r)

    dialects = sorted(by_dialect.keys())
    rng.shuffle(dialects)

    # Assign each dialect a target dataset so the 15 datasets are evenly covered.
    dataset_targets = (DATASETS * ((len(dialects) // len(DATASETS)) + 1))[:len(dialects)]
    rng.shuffle(dataset_targets)

    picks = []
    used_sae = set()
    for dialect, target_ds in zip(dialects, dataset_targets):
        pool = [r for r in by_dialect[dialect]
                if r["dataset"] == target_ds and r["sae_unsafe"] not in used_sae]
        if not pool:
            pool = [r for r in by_dialect[dialect] if r["sae_unsafe"] not in used_sae]
        if not pool:
            continue
        pick = rng.choice(pool)
        used_sae.add(pick["sae_unsafe"])
        picks.append(pick)
        if len(picks) >= 50:
            break

    # Top up to 50 if we fell short (dia_llm has < 50 dialects): draw extras
    # from the dialects with the richest remaining pool, preferring
    # under-represented datasets for stratification balance.
    if len(picks) < 50:
        ds_counts = defaultdict(int)
        for p in picks:
            ds_counts[p["dataset"]] += 1
        rank = sorted(DATASETS, key=lambda d: ds_counts[d])
        while len(picks) < 50:
            added = False
            for ds in rank:
                candidates = [r for r in usable
                              if r["dataset"] == ds
                              and r["sae_unsafe"] not in used_sae]
                if not candidates:
                    continue
                pick = rng.choice(candidates)
                used_sae.add(pick["sae_unsafe"])
                picks.append(pick)
                added = True
                if len(picks) >= 50:
                    break
            if not added:
                break
    return picks[:50]


# ---------- Set B sampler ----------
def build_set_b(dia_rows, mv_rows):
    """20 items per subcondition (B1, B2, B3)."""
    # B1: SAE unsafe + counterharm of SAE (prefer dia_llm for consistency)
    b1_pool = [r for r in dia_rows
               if r["counterharm_validated"]
               and short(r["sae_unsafe"], r["counterharm_sae"])
               and mostly_ascii(r["sae_unsafe"])
               and mostly_ascii(r["counterharm_sae"])
               and not_refusal(r["counterharm_sae"])
               and well_formed(r["sae_unsafe"])
               and well_formed(r["counterharm_sae"])
               and looks_english(r["counterharm_sae"])
               and differs((r["sae_unsafe"], r["counterharm_sae"]))]
    # B2: Multi-VALUE dialect unsafe + counterharm of MV
    b2_pool = [r for r in mv_rows
               if r["counterharm_validated"]
               and short(r["mv_transform"], r["counterharm_mv"])
               and mostly_ascii(r["mv_transform"])
               and mostly_ascii(r["counterharm_mv"])
               and not_refusal(r["counterharm_mv"])
               and well_formed(r["mv_transform"])
               and well_formed(r["counterharm_mv"])
               and looks_english(r["mv_transform"])
               and looks_english(r["counterharm_mv"])
               and differs((r["mv_transform"], r["counterharm_mv"]))]
    # B3: DIA-LLM dialect unsafe + counterharm of that dialect form.
    #     Randomly pick basic or coi per item.
    b3_pool = []
    for r in dia_rows:
        if not r["counterharm_validated"]:
            continue
        for form in ("basic", "coi"):
            unsafe = r[f"{form}_transform"]
            safe = r[f"counterharm_{form}"]
            if (short(unsafe, safe) and mostly_ascii(unsafe) and mostly_ascii(safe)
                    and not_refusal(unsafe) and not_refusal(safe)
                    and well_formed(unsafe) and well_formed(safe)
                    and looks_english(unsafe) and looks_english(safe)
                    and differs((unsafe, safe), (unsafe, r["sae_unsafe"]))):
                b3_pool.append({**r, "dialect_form": form,
                                "dialect_unsafe": unsafe,
                                "counterharm_dialect": safe})

    def sample_stratified(pool, n, unsafe_key):
        """Sample n items, spreading across datasets and dialects, no dup unsafe text."""
        by_ds = defaultdict(list)
        for r in pool:
            by_ds[r["dataset"]].append(r)
        datasets = [ds for ds in DATASETS if by_ds[ds]]
        per_ds = max(1, n // max(1, len(datasets)))
        picks, used = [], set()
        used_dialects = defaultdict(int)
        for ds in datasets:
            rng.shuffle(by_ds[ds])
            for r in by_ds[ds]:
                if len(picks) >= n:
                    break
                if r[unsafe_key] in used:
                    continue
                # spread dialects: cap per-dialect to 2
                if used_dialects[r["dialect"]] >= 2:
                    continue
                picks.append(r)
                used.add(r[unsafe_key])
                used_dialects[r["dialect"]] += 1
                if sum(1 for p in picks if p["dataset"] == ds) >= per_ds + 1:
                    break
        # Fill remainder from any pool if short
        if len(picks) < n:
            remaining = [r for r in pool if r[unsafe_key] not in used]
            rng.shuffle(remaining)
            for r in remaining:
                if len(picks) >= n:
                    break
                picks.append(r)
                used.add(r[unsafe_key])
        return picks[:n]

    b1 = sample_stratified(b1_pool, 20, "sae_unsafe")
    b2 = sample_stratified(b2_pool, 20, "mv_transform")
    b3 = sample_stratified(b3_pool, 20, "dialect_unsafe")
    return b1, b2, b3


def write_set_a(picks, path):
    fields = ["item_id", "dataset", "dialect", "sae_unsafe",
              "basic_transform", "coi_transform",
              "q1_harm_preservation", "q2_semantic_preservation", "comment"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(picks, 1):
            w.writerow({
                "item_id": f"A{i:03d}",
                "dataset": r["dataset"],
                "dialect": r["dialect"],
                "sae_unsafe": r["sae_unsafe"],
                "basic_transform": r["basic_transform"],
                "coi_transform": r["coi_transform"],
                "q1_harm_preservation": "",
                "q2_semantic_preservation": "",
                "comment": "",
            })


def write_set_b(b1, b2, b3, path):
    fields = ["item_id", "subcondition", "dataset", "dialect",
              "unsafe_source", "counterharm_safe",
              "q1_counterharm_safety", "q2_topical_relatedness", "comment"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        idx = 1
        for r in b1:
            w.writerow({
                "item_id": f"B{idx:03d}",
                "subcondition": "B1_SAE",
                "dataset": r["dataset"], "dialect": r["dialect"],
                "unsafe_source": r["sae_unsafe"],
                "counterharm_safe": r["counterharm_sae"],
                "q1_counterharm_safety": "", "q2_topical_relatedness": "", "comment": "",
            })
            idx += 1
        for r in b2:
            w.writerow({
                "item_id": f"B{idx:03d}",
                "subcondition": "B2_MultiVALUE",
                "dataset": r["dataset"], "dialect": r["dialect"],
                "unsafe_source": r["mv_transform"],
                "counterharm_safe": r["counterharm_mv"],
                "q1_counterharm_safety": "", "q2_topical_relatedness": "", "comment": "",
            })
            idx += 1
        for r in b3:
            w.writerow({
                "item_id": f"B{idx:03d}",
                "subcondition": f"B3_DIA-LLM_{r['dialect_form']}",
                "dataset": r["dataset"], "dialect": r["dialect"],
                "unsafe_source": r["dialect_unsafe"],
                "counterharm_safe": r["counterharm_dialect"],
                "q1_counterharm_safety": "", "q2_topical_relatedness": "", "comment": "",
            })
            idx += 1


def main():
    print("Loading dia_llm …")
    dia_rows = load_dia_llm()
    print(f"  {len(dia_rows)} rows across {len({r['dialect'] for r in dia_rows})} dialects")

    print("Loading multi_value …")
    mv_rows = load_multi_value()
    print(f"  {len(mv_rows)} rows across {len({r['dialect'] for r in mv_rows})} dialects")

    print("Building Set A (Harm Preservation, 50 items) …")
    set_a = build_set_a(dia_rows)
    out_a = OUT_DIR / "survey_set_A_harm_preservation.csv"
    write_set_a(set_a, out_a)
    dsets = {r["dataset"] for r in set_a}
    print(f"  {len(set_a)} items, covering {len({r['dialect'] for r in set_a})} dialects "
          f"and {len(dsets)} datasets -> {out_a}")

    print("Building Set B (Counterharm Neutralization, 60 items) …")
    b1, b2, b3 = build_set_b(dia_rows, mv_rows)
    out_b = OUT_DIR / "survey_set_B_counterharm.csv"
    write_set_b(b1, b2, b3, out_b)
    print(f"  B1={len(b1)} B2={len(b2)} B3={len(b3)} -> {out_b}")


if __name__ == "__main__":
    main()
