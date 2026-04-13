# Tables

LaTeX source for all result tables in the paper, organized by experiment.

## Structure

```
tables/
├── harmfulness_detection/
│   └── guard_classifier_accuracy.tex    # Table: Guard accuracy on SAE vs. dialectal inputs
└── attack_success/
    └── end_to_end_asr.tex               # Table: End-to-end ASR by defense setting
```

## Data sources

All tables are populated from the JSON files under:
```
codes/inference/results/zero-shot/
├── harmfulness_detection/{multi_value,llm_basic,llm_coi}/<model>/metrics.json
└── attack_success/multi_value/{best_guard,worst_guard,majority_vote,any_guard}/<model>/metrics.json
```

Each model folder additionally provides `per_dialect.json` and `datasets/<dataset>/{metrics.json,per_dialect.json}` for finer-grained breakdowns.

## Defense configurations (attack_success)

- **best_guard**: Qwen3Guard-4B (highest individual harmfulness-detection accuracy)
- **worst_guard**: HarmBench-Llama (lowest individual accuracy)
- **majority_vote**: Top-5 guards, block if ≥3 flag unsafe (low-stakes)
- **any_guard**: Top-5 guards, block if any flag unsafe (high-stakes)

Top-5 guards: Qwen3Guard-4B, Qwen3Guard-8B, PolyGuard, LlamaGuard-3, DuoGuard-1B. Production models are **not** used as guards.
