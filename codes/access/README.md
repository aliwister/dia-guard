# DIA-GUARD: Running Guide

This guide covers how to run the **Harm-SHIELD** and **CounterHarm-SHIELD** pipelines across all supported LLM providers. No credentials are stored in this repository — all secrets are passed via environment variables.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Provider Setup](#provider-setup)
  - [1. Azure OpenAI](#1-azure-openai)
  - [2. Google Gemini (Vertex AI)](#2-google-gemini-vertex-ai)
  - [3. AWS Bedrock](#3-aws-bedrock)
- [Available Models](#available-models)
- [CLI Reference](#cli-reference)
- [Running Examples](#running-examples)
  - [Run All Dialects](#run-all-dialects)
  - [Run a Specific Dialect](#run-a-specific-dialect)
  - [Run a Specific Dataset](#run-a-specific-dataset)
  - [Test Mode](#test-mode)
  - [Combine Filters](#combine-filters)
- [Running CounterHarm-SHIELD vs Harm-SHIELD](#running-counterharm-shield-vs-harm-shield)
- [Resumability](#resumability)
- [Running on a New VM](#running-on-a-new-vm)

---

## Prerequisites

```bash
git clone https://github.com/jsl5710/dia-guard.git
cd dia-guard
pip install openai boto3 google-genai
```

---

## Provider Setup

### 1. Azure OpenAI

Used for: `gpt4.1`, `deepseek`, `kimi`

```bash
# For GPT-4.1 (Azure OpenAI Service)
export AZURE_OPENAI_API_KEY="your-azure-openai-key"

# For DeepSeek and Kimi (Azure AI Services)
export AZURE_AI_API_KEY="your-azure-ai-key"
```

**Where to get keys:**
- Azure Portal > your OpenAI resource > Keys and Endpoint
- Azure AI Studio > your deployment > API Keys

### 2. Google Gemini (Vertex AI)

Used for: `gemini`, `gemini2.5`, `gemini2.5flash`

```bash
# Authenticate with Google Cloud (one-time setup)
gcloud auth application-default login

# Ensure the project is set
gcloud config set project diaguard-new-project
```

**Requirements:**
- Google Cloud SDK installed (`gcloud`)
- Vertex AI API enabled on your project
- No API key needed — uses application default credentials

### 3. AWS Bedrock

Used for: all `bedrock-*` models

```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```

**Where to get keys:**
- AWS Console > IAM > Users > your user > Security Credentials > Access Keys
- Ensure your IAM user has `bedrock:InvokeModel` and `bedrock:Converse` permissions
- All models use **us-east-1** region

**Enable model access:**
- AWS Console > Amazon Bedrock > Model access > Request access for desired models

---

## Available Models

### Azure OpenAI (3 models)

| `--model` key | Provider | Model | Deployment |
|---------------|----------|-------|------------|
| `gpt4.1` | Azure OpenAI | GPT-4.1 | `gpt-4.1` |
| `deepseek` | Azure AI | DeepSeek V3.2 | `DeepSeek-V3.2` |
| `kimi` | Azure AI | Moonshot Kimi K2.5 | `Kimi-K2.5` |

### Google Gemini via Vertex AI (3 models)

| `--model` key | Model | Project | Region |
|---------------|-------|---------|--------|
| `gemini` | gemini-2.5-flash-lite | diaguard-new-project | us-central1 |
| `gemini2.5` | gemini-2.5-flash-lite | diaguard-new-project | us-central1 |
| `gemini2.5flash` | gemini-2.5-flash | diaguard-new-project | us-central1 |

### AWS Bedrock (10 models)

| `--model` key | Provider | Model | Bedrock Model ID | Region |
|---------------|----------|-------|-------------------|--------|
| `bedrock-deepseek` | DeepSeek | DeepSeek V3.2 | `deepseek.v3.2` | us-east-1 |
| `bedrock-llama3-8b` | Meta | Llama 3 8B Instruct | `meta.llama3-8b-instruct-v1:0` | us-east-1 |
| `bedrock-llama4-maverick` | Meta | Llama 4 Maverick 17B | `meta.llama4-maverick-17b-instruct-v1:0` | us-east-1 |
| `bedrock-llama4-scout` | Meta | Llama 4 Scout 17B | `meta.llama4-scout-17b-instruct-v1:0` | us-east-1 |
| `bedrock-mistral-large3` | Mistral AI | Mistral Large 3 (675B) | `mistral.mistral-large-3-675b-instruct` | us-east-1 |
| `bedrock-safeguard-120b` | OpenAI | GPT OSS Safeguard 120B | `openai.gpt-oss-safeguard-120b` | us-east-1 |
| `bedrock-safeguard-20b` | OpenAI | GPT OSS Safeguard 20B | `openai.gpt-oss-safeguard-20b` | us-east-1 |
| `bedrock-gpt-oss-120b` | OpenAI | gpt-oss-120b | `openai.gpt-oss-120b-1:0` | us-east-1 |
| `bedrock-gpt-oss-20b` | OpenAI | gpt-oss-20b | `openai.gpt-oss-20b-1:0` | us-east-1 |
| `bedrock-qwen3-32b` | Qwen | Qwen3 32B (dense) | `qwen.qwen3-32b-v1:0` | us-east-1 |

**Total: 16 models across 3 providers**

---

## CLI Reference

Both `harm_shield/full_generation_parallel.py` and `counterharm_shield/full_generation_parallel.py` accept the same arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model MODEL` | string | `gpt4.1` | Model key from the tables above |
| `--workers N` | int | `4` | Number of parallel worker threads |
| `--dialect FOLDER` | string | *(all)* | Process only this dialect folder name |
| `--dataset PATTERN` | string | *(all)* | Process only CSVs matching this pattern |
| `--data-dir PATH` | string | `dataset/dia_llm/` | Override data directory path |
| `--test` | flag | off | Enable test mode (limited rows) |
| `--test-rows N` | int | `1` | Number of rows to process in test mode |

---

## Running Examples

All examples below use `counterharm_shield`. Replace with `harm_shield` for the dialect transformation pipeline.

```bash
cd dia-guard/codes/dia_llm/counterharm_shield
```

### Run All Dialects

Process every dialect with all CSV files:

```bash
# Using Azure OpenAI
python full_generation_parallel.py --model gpt4.1 --workers 4

# Using Gemini
python full_generation_parallel.py --model gemini --workers 4

# Using AWS Bedrock
python full_generation_parallel.py --model bedrock-deepseek --workers 4
python full_generation_parallel.py --model bedrock-llama4-scout --workers 6
python full_generation_parallel.py --model bedrock-qwen3-32b --workers 4
```

### Run a Specific Dialect

Process only one dialect (use the exact folder name):

```bash
# Process only Australian Vernacular English
python full_generation_parallel.py --model bedrock-deepseek --workers 4 \
  --dialect australian_vernacular_english

# Process only Nigerian English
python full_generation_parallel.py --model gemini --workers 4 \
  --dialect nigerian_english

# Process only Urban AAVE
python full_generation_parallel.py --model gpt4.1 --workers 4 \
  --dialect urban_african_american_vernacular_english
```

**Available dialect folder names (48):**

<details>
<summary>Click to expand full list</summary>

**North America:** `appalachian_english`, `chicano_english`, `colloquial_american_english`, `earlier_african_american_vernacular_english`, `newfoundland_english`, `ozark_english`, `rural_african_american_vernacular_english`, `southeast_american_enclave_dialects`, `urban_african_american_vernacular_english`

**British Isles:** `channel_islands_english`, `east_anglian_english`, `english_dialects_in_the_north_of_england`, `english_dialects_in_the_southeast_of_england`, `english_dialects_in_the_southwest_of_england`, `irish_english`, `maltese_english`, `manx_english`, `orkney_and_shetland_english`, `scottish_english`, `welsh_english`

**Caribbean:** `bahamian_english`, `jamaican_english`

**Africa:** `black_south_african_english`, `cameroon_english`, `cape_flats_english`, `ghanaian_english`, `indian_south_african_english`, `kenyan_english`, `liberian_settler_english`, `nigerian_english`, `tanzanian_english`, `ugandan_english`, `white_south_african_english`

**South/Southeast Asia:** `colloquial_singapore_english_singlish`, `hong_kong_english`, `indian_english`, `malaysian_english`, `pakistani_english`, `philippine_english`, `sri_lankan_english`

**Australia & Pacific:** `aboriginal_english`, `acrolectal_fiji_english`, `australian_english`, `australian_vernacular_english`, `new_zealand_english`, `pure_fiji_english_basilectal_fijie`

**Atlantic Islands:** `falkland_islands_english`, `st_helena_english`, `tristan_da_cunha_english`

</details>

### Run a Specific Dataset

Process only CSVs matching a pattern (across all dialects):

```bash
# Only advbench files
python full_generation_parallel.py --model bedrock-llama4-maverick --workers 4 \
  --dataset advbench

# Only sorry_bench files
python full_generation_parallel.py --model gemini2.5flash --workers 4 \
  --dataset sorry_bench

# Only bipia files
python full_generation_parallel.py --model bedrock-mistral-large3 --workers 4 \
  --dataset bipia
```

**Available dataset patterns (15):**
`Salad_Bench`, `Simple_Safety_Tests`, `Toxic_Text`, `Toxicity_Jigsaw`, `advbench`, `bipia`, `cyberseceval`, `do_not_answer`, `forbiddent_questions`, `harmBench`, `injecagent`, `jailbreakbench`, `llmseceval`, `securityeval`, `sorry_bench`

### Test Mode

Run a quick test to verify your setup works before full generation:

```bash
# Test with 1 row (default)
python full_generation_parallel.py --model bedrock-deepseek --test

# Test with 3 rows
python full_generation_parallel.py --model bedrock-qwen3-32b --test --test-rows 3
```

### Combine Filters

```bash
# Specific dialect + specific dataset + specific model
python full_generation_parallel.py \
  --model bedrock-llama4-scout \
  --workers 6 \
  --dialect nigerian_english \
  --dataset advbench

# Test mode on a specific dialect
python full_generation_parallel.py \
  --model bedrock-safeguard-120b \
  --test --test-rows 2 \
  --dialect bahamian_english
```

---

## Running CounterHarm-SHIELD vs Harm-SHIELD

| Pipeline | Purpose | Directory | API Calls/Row |
|----------|---------|-----------|---------------|
| **Harm-SHIELD** | Dialect transformation of harmful text | `codes/dia_llm/harm_shield/` | ~8 |
| **CounterHarm-SHIELD** | Benign counterexample generation | `codes/dia_llm/counterharm_shield/` | ~24 |

```bash
# Harm-SHIELD (dialect transformation)
cd codes/dia_llm/harm_shield
python full_generation_parallel.py --model bedrock-deepseek --workers 6

# CounterHarm-SHIELD (benign generation)
cd codes/dia_llm/counterharm_shield
python full_generation_parallel.py --model bedrock-deepseek --workers 4
```

---

## Resumability

Both pipelines are **fully resumable**. If a run is interrupted:

- Already-processed rows are automatically skipped on restart
- Partial results are saved after each row completes
- Simply re-run the same command to continue where you left off
- Rows marked `[FAILED]` are treated as permanently done (not retried)

```bash
# This will skip all completed rows and continue from where it stopped
python full_generation_parallel.py --model bedrock-deepseek --workers 4
```

---

## Running on a New VM

Complete setup from scratch:

```bash
# 1. Clone the repo
git clone https://github.com/jsl5710/dia-guard.git
cd dia-guard

# 2. Install dependencies
pip install openai boto3 google-genai

# 3. Set credentials for your chosen provider
# -- For AWS Bedrock:
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
# -- For Azure OpenAI:
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_AI_API_KEY="your-key"
# -- For Gemini:
gcloud auth application-default login

# 4. Run (data auto-resolves to dataset/dia_llm/)
cd codes/dia_llm/counterharm_shield
python full_generation_parallel.py --model bedrock-deepseek --workers 4

# 5. When done, push updated data back to the repo
cd ../../..
git add dataset/dia_llm/
git commit -m "Update CounterHarm-SHIELD generation progress"
git push origin main
```

---

## Worker Recommendations

| Model Size | Suggested `--workers` | Notes |
|-----------|----------------------|-------|
| Small (8B-20B) | 6-8 | Fast inference, can handle more concurrency |
| Medium (17B-32B) | 4-6 | Good balance |
| Large (120B-675B) | 2-4 | Slower inference, may hit rate limits |

Adjust based on your provider's rate limits. If you see rate-limit errors, reduce workers.
