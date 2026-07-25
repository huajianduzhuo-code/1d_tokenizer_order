# PBH seed=42 hand-inspection cases

Open in IDE markdown preview to see images side-by-side.

Each case shows: prompt, ground-truth questions, all 8 generated images (4 random + 4 prompt_sim), and judge verdicts.

---

## A: prompt_sim wins big — `pbh-count_x_color-055` (difficulty=4)

**Prompt:** Four yellow lemons and one green lime in a wicker basket.

**Questions (ground truth):**

- Q0: How many yellow lemons are in the basket? → **4** (count)
- Q1: How many green limes are in the basket? → **1** (count)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.125 | 0.000 |
| prompt_sim | 0.625 | 0.375 |

Metric used to pick this case: `+0.438`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00236/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00236/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00236/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00236/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00236/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00236/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00236/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00236/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✗ `5` | ✗ `5` | ✗ `3` | ✗ `3` | ✗ `3` | ✗ `6` | ✗ `3` | ✗ `5` |
| Q1 (1) | ✗ `0` | ✗ `0` | ✓ `1` | ✗ `2` | ✗ `0` | ✗ `0` | ✗ `3` | ✗ `3` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✓ `4` | ✗ `5` | ✗ `3` | ✓ `4` | ✗ `3` | ✓ `4` | ✓ `4` | ✗ `5` |
| Q1 (1) | ✓ `1` | ✗ `2` | ✓ `1` | ✓ `1` | ✓ `1` | ✗ `2` | ✗ `0` | ✗ `0` |

---

## A: prompt_sim wins big — `pbh-count_x_color-018` (difficulty=4)

**Prompt:** Four pink macarons and three green macarons stacked on a small plate.

**Questions (ground truth):**

- Q0: How many pink macarons are on the plate? → **4** (count)
- Q1: How many green macarons are on the plate? → **3** (count)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.000 | 0.000 |
| prompt_sim | 0.375 | 0.250 |

Metric used to pick this case: `+0.312`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00067/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00067/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00067/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00067/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00067/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00067/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00067/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00067/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✗ `3` | ✗ `2` | ✗ `3` | ✗ `7` | ✗ `3` | ✗ `2` | ✗ `2` | ✗ `2` |
| Q1 (3) | ✗ `4` | ✗ `4` | ✗ `1` | ✗ `2` | ✗ `2` | ✗ `1` | ✗ `2` | ✗ `0` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✗ `3` | ✗ `3` | ✗ `1` | ✗ `1` | ✗ `3` | ✓ `4` | ✓ `4` | ✗ `5` |
| Q1 (3) | ✗ `1` | ✗ `1` | ✓ `3` | ✗ `2` | ✓ `3` | ✓ `3` | ✗ `2` | ✗ `2` |

---

## A: prompt_sim wins big — `pbh-count_x_color-029` (difficulty=4)

**Prompt:** Four blue marbles and one green marble inside a clear glass jar.

**Questions (ground truth):**

- Q0: How many blue marbles are in the jar? → **4** (count)
- Q1: How many green marbles are in the jar? → **1** (count)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.125 | 0.125 |
| prompt_sim | 0.500 | 0.375 |

Metric used to pick this case: `+0.312`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00210/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00210/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00210/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00210/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00210/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00210/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00210/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00210/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✗ `0` | ✗ `3` | ✗ `3` | ✗ `2` | ✗ `3` | ✗ `1` | ✗ `3` | ✗ `3` |
| Q1 (1) | ✗ `unknown` | ✗ `2` | ✗ `2` | ✗ `0` | ✗ `0` | ✗ `0` | ✓ `1` | ✓ `1` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (4) | ✗ `3` | ✗ `5` | ✓ `4` | ✗ `3` | ✗ `5` | ✓ `4` | ✗ `3` | ✗ `5` |
| Q1 (1) | ✗ `2` | ✗ `5` | ✓ `1` | ✓ `1` | ✓ `1` | ✗ `0` | ✓ `1` | ✓ `1` |

---

## B: judges disagree — `pbh-color_binding_multi-018` (difficulty=3)

**Prompt:** A red kite, a blue kite, and a yellow kite all flying in the sky.

**Questions (ground truth):**

- Q0: How many kites are in the sky? → **3** (count)
- Q1: Is one of the kites red? → **yes** (yesno)
- Q2: Is one of the kites blue? → **yes** (yesno)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.333 | 0.667 |
| prompt_sim | 0.250 | 0.417 |

Metric used to pick this case: `+12.000`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00042/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00042/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00042/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00042/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00042/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00042/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00042/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00042/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (3) | ✗ `unknown` | ✓ `3` | ✓ `3` | ✗ `2` | ✗ `0` | ✗ `0` | ✗ `unknown` | ✗ `0` |
| Q1 (yes) | ✗ `unknown` | ✓ `yes` | ✓ `yes` | ✓ `yes` | ✗ `no` | ✓ `yes` | ✗ `unknown` | ✓ `yes` |
| Q2 (yes) | ✗ `unknown` | ✓ `yes` | ✗ `no` | ✗ `no` | ✓ `yes` | ✓ `yes` | ✓ `yes` | ✓ `yes` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (3) | ✓ `3` | ✗ `0` | ✗ `0` | ✗ `unknown` | ✗ `0` | ✓ `3` | ✗ `2` | ✗ `2` |
| Q1 (yes) | ✗ `no` | ✓ `yes` | ✗ `no` | ✓ `yes` | ✗ `no` | ✗ `no` | ✓ `yes` | ✓ `yes` |
| Q2 (yes) | ✓ `yes` | ✗ `no` | ✗ `unknown` | ✗ `no` | ✗ `no` | ✗ `no` | ✗ `unknown` | ✓ `yes` |

---

## B: judges disagree — `pbh-counting_hard-029` (difficulty=3)

**Prompt:** Five butterflies fluttering above a bed of flowers.

**Questions (ground truth):**

- Q0: How many butterflies are visible? → **5** (count)
- Q1: Is there a bed of flowers in the image? → **yes** (yesno)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.500 | 0.625 |
| prompt_sim | 0.000 | 0.750 |

Metric used to pick this case: `+11.000`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00105/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00105/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00105/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00105/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00105/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00105/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00105/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00105/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (5) | ✓ `5` | ✗ `4` | ✗ `6` | ✓ `5` | ✓ `5` | ✗ `6` | ✗ `4` | ✗ `4` |
| Q1 (yes) | ✗ `no` | ✓ `yes` | ✗ `no` | ✓ `yes` | ✓ `yes` | ✓ `yes` | ✓ `yes` | ✓ `yes` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (5) | ✗ `7` | ✓ `5` | ✗ `2` | ✗ `2` | ✗ `6` | ✓ `5` | ✗ `6` | ✗ `6` |
| Q1 (yes) | ✗ `no` | ✓ `yes` | ✗ `no` | ✓ `yes` | ✗ `no` | ✓ `yes` | ✗ `no` | ✓ `yes` |

---

## C: random wins big — `pbh-count_x_color-057` (difficulty=4)

**Prompt:** Three red kites and two yellow kites flying high in the sky.

**Questions (ground truth):**

- Q0: How many red kites are in the sky? → **3** (count)
- Q1: How many yellow kites are in the sky? → **2** (count)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.375 | 0.375 |
| prompt_sim | 0.000 | 0.000 |

Metric used to pick this case: `-0.375`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00238/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00238/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00238/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00238/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00238/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00238/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00238/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00238/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (3) | ✓ `3` | ✓ `3` | ✓ `3` | ✓ `3` | ✗ `2` | ✗ `unknown` | ✓ `3` | ✓ `3` |
| Q1 (2) | ✗ `3` | ✗ `1` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `3` | ✗ `0` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (3) | ✗ `5` | ✗ `4` | ✗ `2` | ✗ `0` | ✗ `4` | ✗ `unknown` | ✗ `2` | ✗ `1` |
| Q1 (2) | ✗ `1` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `0` | ✗ `0` |

---

## C: random wins big — `pbh-count_x_color-041` (difficulty=4)

**Prompt:** Two white sails and three red sails on a row of boats at sea.

**Questions (ground truth):**

- Q0: How many white sails are visible? → **2** (count)
- Q1: How many red sails are visible? → **3** (count)

**Per-prompt summary (per-question accuracy across 4 samples):**

| Order | OpenAI (gpt-5-mini) | Gemini (2.5-flash) |
|-------|----------------------|---------------------|
| random | 0.625 | 0.250 |
| prompt_sim | 0.125 | 0.125 |

Metric used to pick this case: `-0.312`

### Images

| Sample | Random | Prompt_sim |
|---|---|---|
| 0 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00222/samples/00000.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00222/samples/00000.png) |
| 1 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00222/samples/00001.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00222/samples/00001.png) |
| 2 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00222/samples/00002.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00222/samples/00002.png) |
| 3 | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_random_seed42/00222/samples/00003.png) | ![](/data3/haoyuliu/pbh_eval/maskgen_kl_xl_prompt_sim_seed42/00222/samples/00003.png) |

### Judge verdicts (✓ = correct, ✗ = wrong)

**random:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (2) | ✗ `1` | ✓ `2` | ✓ `2` | ✗ `4` | ✗ `4` | ✗ `4` | ✗ `6` | ✗ `6` |
| Q1 (3) | ✓ `3` | ✗ `4` | ✓ `3` | ✗ `4` | ✓ `3` | ✗ `4` | ✓ `3` | ✓ `3` |

**prompt_sim:**

| Q\Sample | s=0 OAI | s=0 Gem | s=1 OAI | s=1 Gem | s=2 OAI | s=2 Gem | s=3 OAI | s=3 Gem |
|---|---|---|---|---|---|---|---|---|
| Q0 (2) | ✗ `4` | ✗ `4` | ✗ `4` | ✗ `6` | ✗ `4` | ✗ `4` | ✗ `1` | ✓ `2` |
| Q1 (3) | ✗ `2` | ✗ `2` | ✗ `2` | ✗ `2` | ✗ `4` | ✗ `4` | ✓ `3` | ✗ `4` |

