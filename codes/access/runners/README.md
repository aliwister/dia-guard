# Bedrock Runner Scripts

7 self-contained bash scripts — one per model. Each clones the repo, installs deps, and runs its assigned dialects **one by one** sequentially.

---

## Quick Start (on any VM)

```bash
# 1. Set your AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"

# 2. Run any script (it clones the repo automatically)
bash run_bedrock_deepseek.sh
```

---

## Model Assignments

| Script | Model | Workers | Dialects | Status |
|--------|-------|---------|----------|--------|
| `run_bedrock_deepseek.sh` | DeepSeek V3.2 | 4 | tanzanian, tristan_da_cunha, ugandan, welsh, urban_aave, southeast_enclave | 5 nearly done + 1 at 52% |
| `run_bedrock_llama4_scout.sh` | Llama 4 Scout 17B | 6 | australian_vernacular, bahamian, black_south_african, cameroon, cape_flats, channel_islands | 1 at 34% + 5 new |
| `run_bedrock_llama4_maverick.sh` | Llama 4 Maverick 17B | 6 | chicano, colloquial_american, singapore_singlish, earlier_aave, east_anglian, north_of_england | 6 new |
| `run_bedrock_mistral_large3.sh` | Mistral Large 3 (675B) | 3 | southeast_england, southwest_england, falkland, ghanaian, hong_kong, indian | 6 new |
| `run_bedrock_llama3_8b.sh` | Llama 3 8B Instruct | 8 | indian_south_african, irish, jamaican, kenyan, liberian_settler, malaysian | 6 new |
| `run_bedrock_gpt_oss_20b.sh` | gpt-oss-20b | 6 | maltese, manx, new_zealand, newfoundland, nigerian, orkney_shetland | 6 new |
| `run_bedrock_qwen3_32b.sh` | Qwen3 32B | 4 | ozark, pakistani, philippine, pure_fiji_basilectal, rural_aave, sri_lankan | 6 new |

**Total: 7 models x 6 dialects = 42 dialects**

---

## How Sequential Execution Works

Each script uses a bash `for` loop over its assigned `DIALECTS` array:

```bash
DIALECTS=(
    "dialect_one"
    "dialect_two"
    ...
)

for i in "${!DIALECTS[@]}"; do
    python full_generation_parallel.py \
        --model "$MODEL" \
        --workers "$WORKERS" \
        --dialect "${DIALECTS[$i]}"
done
```

- Dialects run **one at a time**, in order
- Within each dialect, rows run in parallel (controlled by `--workers`)
- Progress is printed: `[1/6] Starting: tanzanian_english`
- If a dialect finishes, the next one starts automatically

---

## Resumability

**The pipeline is fully resumable.** If a run is interrupted (VM restart, network error, etc.):

- Simply re-run the same script — it picks up where it left off
- Already-completed rows are automatically skipped (checked via CSV output)
- Partially-completed dialects resume from the next unprocessed row
- Rows marked `[FAILED]` are treated as permanently done (not retried)
- Results are saved to disk after **every single row**, so no work is lost

```bash
# Interrupted mid-run? Just re-run the same script:
bash run_bedrock_deepseek.sh
# Output: "2,659 rows skipped, 3 rows to process"
```

**The data lives in `dataset/dia_llm/` inside the cloned repo.** When resuming, the script reads the existing CSV files and skips all rows that already have `counterharm_*` columns filled.

---

## Running on Multiple VMs in Parallel

Each script is independent. To maximize throughput:

```
VM 1:  bash run_bedrock_deepseek.sh
VM 2:  bash run_bedrock_llama4_scout.sh
VM 3:  bash run_bedrock_llama4_maverick.sh
VM 4:  bash run_bedrock_mistral_large3.sh
VM 5:  bash run_bedrock_llama3_8b.sh
VM 6:  bash run_bedrock_gpt_oss_20b.sh
VM 7:  bash run_bedrock_qwen3_32b.sh
```

Since each model processes **different dialects**, there are no conflicts. All 7 can run simultaneously.

---

## After Completion: Push Results Back

```bash
cd dia-guard
git add dataset/dia_llm/
git commit -m "CounterHarm-SHIELD: [model-name] generation complete"
git push origin main
```
