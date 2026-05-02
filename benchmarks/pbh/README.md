# PromptBind-Hard (PBH)

A custom T2I reasoning benchmark designed to expose the advantage of the
`prompt_sim` generation order on counting / color-binding / compositional
prompts.

## Categories

| Category | n | What it tests |
|---|---|---|
| `counting_hard` | 75 | 1–5 objects of 1–2 types, often with mixed object classes |
| `color_binding_multi` | 75 | 3+ objects each with a distinct color |
| `count_x_color` | 60 | Joint count + per-instance color (the hardest binding case) |
| `composition_layout` | 35 | Spatial + count |
| `negation_substitution` | 30 | "X but no Y" — global semantic check |
| `relational_reasoning` | 25 | Size / relation comparisons |
| **Total** | **300** | |

To extend further, follow the existing per-category templates (vary objects,
attributes, counts). Always include 2–4 questions per prompt with at least
one negative-answer question (e.g. "are there any bananas?" → "no") to
catch VLM hallucinations. All counts must be ≤ 5.

## Record schema (one JSON object per line in `prompts.jsonl`)

```json
{
  "id": "pbh-counting_hard-007",
  "category": "counting_hard",
  "prompt": "A long table set with four plates, four wine glasses, and four forks.",
  "questions": [
    {"q": "How many plates are on the table?", "type": "count",
     "answer": "4", "accepted": ["4", "four"]},
    {"q": "How many wine glasses are on the table?", "type": "count",
     "answer": "4", "accepted": ["4", "four"]},
    {"q": "How many forks are on the table?", "type": "count",
     "answer": "4", "accepted": ["4", "four"]}
  ],
  "tags": ["counting"],
  "difficulty": 3
}
```

### Fields
- `id`: globally unique, format `pbh-<category>-NNN`.
- `category`: one of the six listed above.
- `prompt`: the T2I prompt sent to the model.
- `questions`: list of dicts, each with:
  - `q`: the question text shown to the VLM.
  - `type`: `count`, `yesno`, or `mc` (no open-ended free text).
  - `answer`: the canonical ground-truth answer.
  - `accepted`: list of acceptable normalized forms (e.g. `["5", "five"]`,
    `["purple", "violet"]`). Match is "any normalized form in this list".
- `tags`: optional, free-form labels for filtering analysis.
- `difficulty`: 1–5 self-rated, only used for downstream slicing.

## How it is used

The eval pipeline (`eval_pbh.py`) follows the same shape as the other
benchmarks in this repo:

1. **Generation** — multi-GPU, generates `--n-samples` images per prompt and
   saves them under
   `/data3/haoyuliu/pbh_eval/maskgen_kl_{model}_{order}_seed{seed}/{idx:05d}/samples/`.
2. **Judging** — `pbh_judge.py` walks the generated images, asks each VLM
   question with structured JSON output, normalizes answers, and compares
   against the `accepted` list. Results cached on disk.
3. **Aggregation** — per-question accuracy, per-prompt strict accuracy
   (best-of-4), and per-prompt mean accuracy by category and overall.

## Running the benchmark

### Prerequisites

```bash
conda activate maskgen
pip install openai google-genai tqdm    # judge SDK deps (one-time)
```

API keys (recommended setup: keep them in `~/.api_keys` with `chmod 600`,
then source from `~/.bashrc`):

```bash
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### Smoke test (~3 min, 1 GPU, < $0.01)

```bash
# 1 prompt, 1 image, 1 VLM call — confirms the whole pipeline works
python eval_pbh.py --model xl --order random \
    --num-samples 1 --n-samples 1 --num-gpus 1
python eval_pbh.py --model xl --order random \
    --num-samples 1 --eval-only --judge-provider openai
```

### Lite subset (30 prompts, ~$0.5/config)

`benchmarks/pbh/prompts_lite.jsonl` is a balanced 30-prompt subset for
quick iteration. Output goes to `/data3/haoyuliu/pbh_eval_lite/` (separate
from the full-bench results).

```bash
python eval_pbh.py --model xl --order prompt_sim \
    --dataset benchmarks/pbh/prompts_lite.jsonl \
    --evaluate --judge-provider both
```

### Full sweep — 4 configs × 3 seeds × both providers

`run_pbh_all.sh` handles the full grid:
- Generation: 4 configs (`{l,xl} × {random,prompt_sim}`) × 3 seeds (42,43,44)
  in parallel on 8 GPUs (2 GPUs/config).
- Judging: sequential over all configs (API-bound; parallel runs trip rate limits).
- Final cross-config summary table.

```bash
# Default: SEEDS="42 43 44" JUDGE_PROVIDER=openai
tmux new-session -d -s pbh_full \
    "JUDGE_PROVIDER=both bash run_pbh_all.sh 2>&1 | tee /tmp/pbh_full.log"
tmux attach -t pbh_full

# Subset overrides
SEEDS=42 JUDGE_PROVIDER=both bash run_pbh_all.sh        # one seed only
JUDGE_PROVIDER=gemini bash run_pbh_all.sh               # gemini only

# Cross-config summary table (mean±std across seeds)
python eval_pbh.py --summary
```

**Estimated cost (full 300-prompt sweep):**
- Generation: ~30 min on 8× A5000 (14,400 images total)
- Judging: ~62k API calls with `JUDGE_PROVIDER=both` → ~$17 GPT-4o + ~$2 Gemini

### Other useful commands

```bash

# Re-aggregate from cached judge responses (no API calls)
python eval_pbh.py --model xl --order random --re-aggregate

# Force re-judge (ignores cache)
python eval_pbh.py --model xl --order random --eval-only --force \
    --judge-provider openai

# Generate only (no judging)
python eval_pbh.py --model xl --order random --num-gpus 4
```

### Output layout

```
/data3/haoyuliu/pbh_eval/
├── maskgen_kl_xl_random_seed42/
│   ├── 00000/{metadata.json, samples/00000.png ... 00003.png}
│   ├── ...
│   ├── judge_cache/{openai,gemini}/...           # per-question cache
│   ├── judge_responses_openai.jsonl              # raw responses (debug)
│   ├── judge_responses_gemini.jsonl
│   └── pbh_summary.json                          # aggregated metrics
└── pbh_summary_table.txt                         # cross-config table
```

`pbh_eval_lite/` mirrors this structure for the lite dataset.
