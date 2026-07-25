# 1d-tokenizer / SAR 使用说明（中文）

本仓库是论文 **《Context-Aware Autoregressive Image Generation for Emerging Reasoning Properties》（Saccade AR, 简称 SAR）** 中 **MaskGen 部分** 的实验代码。
论文里所有关于 MaskGen-L / MaskGen-XL 的定量表格（GenEval / DPG-Bench / MJHQ-30K FID / 复杂长 prompt benchmark PBH）和定性图（Fig. 4：object-by-object 的可视化）都在这个仓库里生成。

MAR（Fig. 1 中 "four vases" 例子、附录 B 的 ImageNet class-conditional 结果）由另一个仓库 `~/mar_new_order` 负责，不在本仓库范围内。

---

## 1. 核心概念：`prompt_sim` = SAR

代码里的顺序名称和论文里的名字有一个必须记住的映射：

| 代码里的 order name | 论文里的名字 | 作用 |
|---|---|---|
| `random` | Random | MaskGen 原生的随机顺序基线 |
| **`prompt_sim`** | **SAR (Saccade AR)** | 论文的主方法，按 token 与 pooled text embedding 的 cosine 相似度动态排序，**高相似度先解码** |
| `prompt_sim_rev` | (Anti-SAR) | 反向消融，低相似度先解码 |
| `left_to_right` / `right_to_left` / `center_out` | Sequential / Inside-out 等 | 静态顺序消融（对应论文 Table 5 的 ablation） |

**在可视化图上（compare 模式），下方一行的 label 直接显示为 "SAR"**，而不是 `prompt_sim`（见 `run_maskgen_kl_visualize_steps.py`）。

`SUPPORTED_ORDER_TYPES` 定义在 `modeling/maskgen.py` 里的 `MaskGen_KL` 类中。

---

## 2. 存储约定

按项目根目录的 `CLAUDE.md` 要求，**模型权重、数据集、conda 环境、评测输出全部放在 `/data3/haoyuliu/`**（磁盘大）。仓库里有两个符号链接直接指向那里：

- `1d-tokenizer/data3_lhy/` → `/data3/haoyuliu/`
- `1d-tokenizer/image_outputs/` → `/data3/haoyuliu/image_outputs/`

所有 eval / visualize 脚本头部都设了：

```python
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"
```

所以 HuggingFace 下载的 tokenizer / generator / CLIP 权重都会自动缓存到 `/data3/haoyuliu/huggingface_cache/`，不会占满系统盘。

---

## 3. 环境准备

生成阶段（跑 MaskGen 采样图片）和评测阶段（跑 detector / VQA / FID）需要**分开的 conda 环境**。

### 3.1 生成环境 `maskgen`

用 `requirements.txt` + 一些额外包：

```bash
conda create -n maskgen python=3.10 -y
conda activate maskgen
pip install -r requirements.txt
pip install clean-fid matplotlib pandas   # 评测汇总要用
```

关键依赖：`torch>=2.0`, `open_clip_torch`, `huggingface-hub`, `transformers`, `einops`, `omegaconf`。

### 3.2 各 benchmark 的评测环境

论文里 4 个 benchmark 有各自的评测器，都是独立 conda env，用仓库自带的 setup 脚本装：

| Benchmark | Env 名字 / 路径 | 装法 |
|---|---|---|
| GenEval | `geneval` | `bash setup_geneval_env.sh` |
| DPG-Bench | `/data3/haoyuliu/conda_envs/dpg` | `bash setup_dpg_env.sh` |
| T2I-CompBench++ | `/data3/haoyuliu/conda_envs/t2i_compbench` | `bash setup_t2i_compbench_env.sh` |
| MJHQ-30K FID | 用 `maskgen` 环境即可（`clean-fid`） | — |
| PBH | 用 `maskgen` 环境 + `OPENAI_API_KEY` / `GOOGLE_API_KEY` | — |

### 3.3 外部仓库依赖

有几个评测器代码不在本仓库里，路径写死了：

- `/home/hliu256/geneval` — GenEval 官方仓库
- `/home/hliu256/ELLA` — DPG-Bench 里 mPLUG VQA 评测代码
- `/home/hliu256/T2I-CompBench` — T2I-CompBench++ 的 UniDet / BLIP-VQA 评测

如果换机器需要重新 clone 这三个仓库到同样位置。

### 3.4 数据

- **MJHQ-30K**（做 FID 用的参考数据）：`/data3/haoyuliu/mjhq30k/`
---

## 4. 论文中的图 / 表怎么生成

### 4.1 Fig. 4 — MaskGen-XL 的 object-by-object 可视化

论文里 Fig. 4 展示了三个 prompt（"four vases" / "a child walking to the right of an adult ..." / "a birthday cake with colorful candles ..."），上一行是 random，下一行是 SAR，一列一列显示解码步骤。

对应脚本：**`run_maskgen_kl_visualize_steps.py`**

```bash
conda activate maskgen

# 一次跑完 random 和 prompt_sim(=SAR)，出并排对比图
python run_maskgen_kl_visualize_steps.py --compare --model xl --seed 0

# 用自己的 prompt 文件（一行一个 prompt）
python run_maskgen_kl_visualize_steps.py --compare --prompts-file my_prompts.txt

# 只跑单个 prompt、单个 order（不出对比）
python run_maskgen_kl_visualize_steps.py --prompt "A cat on a table" --order prompt_sim
```

**输出目录**：`image_outputs/run_xl_{时间戳}_seed{seed}/`
- `compare_{idx}_{prompt_slug}.png` — 两行对比图（图 4 用的）
- `random_{...}.png`, `prompt_sim_{...}.png` — 每个 order 单独的多步网格

字体已经全局设成 `Liberation Serif`（Times New Roman 的等价字体，Linux 无 TNR 时的替代）。

另有一个变体脚本 `run_maskgen_kl_visualize_full_predictions.py`：区别在于**每一步会把所有还没提交的位置也预测出来**，可以看到"模型此刻脑子里的完整图像"，而不是只显示已经提交的 token。用法完全一样。

### 4.2 Table 2 — GenEval

553 个 prompt × 4 张图，评测 6 个子类（single_object / two_object / counting / colors / position / color_attr）的 prompt-level pass rate。

```bash
conda activate maskgen

# 生成图片（约 30–60 分钟 per config，用 4 GPU）
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_geneval.py --model xl --order random     --num-gpus 4 --seed 42
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_geneval.py --model xl --order prompt_sim --num-gpus 4 --seed 42

# 切到评测环境跑 Mask2Former 检测
conda activate geneval
python eval_geneval_parallel.py /data3/haoyuliu/geneval_eval/maskgen_kl_xl_prompt_sim_seed42 --num-gpus 8
python eval_geneval_parallel.py /data3/haoyuliu/geneval_eval/maskgen_kl_xl_random_seed42     --num-gpus 8

# 汇总所有 (model, order, seed) 组合成一张表
conda activate maskgen
python eval_geneval.py --summary
```

**一键脚本**：
- `run_geneval_all.sh` — 4 个 config (L/XL × random/prompt_sim) × 多 seed 并行生成
- `run_geneval_evaluate.sh` — 全部 config 的 detector 评测
- `run_geneval_orders.sh` — 支持任意 orders 列表（做静态顺序 ablation 用）

**输出**：`/data3/haoyuliu/geneval_eval/maskgen_kl_{model}_{order}_seed{seed}/`，最终汇总表在 `geneval_summary_table.txt`。

### 4.3 Table 3 — DPG-Bench

1065 个长 prompt，每个 prompt 生成 4 张 512×512 拼成 1024×1024 网格，用 mPLUG-VQA 打分。

```bash
conda activate maskgen
python eval_dpg.py --model xl --order random     --seed 42
python eval_dpg.py --model xl --order prompt_sim --seed 42

# 切到 DPG 环境跑 VQA 评测
conda activate /data3/haoyuliu/conda_envs/dpg
python eval_dpg.py --model xl --order random     --eval-only
python eval_dpg.py --model xl --order prompt_sim --eval-only

# 汇总
conda activate maskgen
python eval_dpg.py --summary
```

**一键脚本**：`run_dpg_all.sh`（生成） + `run_dpg_evaluate.sh`（评测）

**输出**：`/data3/haoyuliu/dpg_bench_eval/maskgen_kl_{model}_{order}_seed{seed}/`

### 4.4 Table 4 — MJHQ-30K FID

从 30K 个 prompt 生成图片，用 `clean-fid` 对 MJHQ 参考图算 FID。

```bash
conda activate maskgen

# 完整 30K，一个 config 大约几小时
python eval_mjhq_fid.py --model xl --order random     --seed 0
python eval_mjhq_fid.py --model xl --order prompt_sim --seed 0

# 汇总
python eval_mjhq_fid.py --summary --summary-suffix 30k
```

**一键脚本**：`run_mjhq_fid_all.sh`（默认跑 seed 0/1/2 三次）

**输出**：`/data3/haoyuliu/mjhq30k_eval/eval_{model}_{order}_30k_seed{seed}/fid_result.txt`

### 4.5 Table 5 — 静态顺序 ablation

对 `left_to_right` / `right_to_left` / `center_out` 三种静态顺序做 GenEval + DPG 消融。

```bash
bash run_new_orders_ablation.sh
```

或手动指定 orders：

```bash
ORDERS="random prompt_sim left_to_right right_to_left center_out" bash run_geneval_orders.sh
ORDERS="random prompt_sim left_to_right right_to_left center_out" bash run_dpg_orders.sh
```

**输出日志**：`/data3/haoyuliu/ablation_logs/`

### 4.6 T2I-CompBench++（附加评测，论文里没有）

论文最终版没放，但仓库里做了：

```bash
conda activate maskgen
python eval_t2i_compbench.py --model xl --order random     --seed 42
python eval_t2i_compbench.py --model xl --order prompt_sim --seed 42

conda activate /data3/haoyuliu/conda_envs/t2i_compbench
python eval_t2i_compbench.py --model xl --order prompt_sim --eval-only

conda activate maskgen
python eval_t2i_compbench.py --summary
```

**一键脚本**：`run_t2i_compbench_all.sh` + `run_t2i_compbench_evaluate.sh`

---

## 5. 一键跑全部（用于复现论文所有 MaskGen 定量结果）

```bash
conda activate maskgen
cd /home/hliu256/1d-tokenizer

# 4 个 benchmark（DPG → GenEval → T2I-CB → MJHQ）× 4 config (L/XL × random/prompt_sim) × 3 seeds
# 生成 + 评测 + 汇总一条龙。日志会写到 /data3/haoyuliu/all_evals_logs/
bash run_all_evals.sh
```

支持的环境变量：
- `SEEDS="0 1 2"` — 想跑几个 seed（论文用 3 seed 平均）
- `GEN_ONLY=1` — 只生成不评测
- `EVAL_ONLY=1` — 只评测（假设图片已生成）
- `SKIP_DPG=1` / `SKIP_GENEVAL=1` / `SKIP_T2I=1` / `SKIP_MJHQ=1` — 跳过某个 benchmark

PBH 因为要调外部 VLM API，没放在 `run_all_evals.sh` 里，单独用 `run_pbh_all.sh`。

---

## 6. 常用的超参（和论文对齐）

所有实验都在这些设定下跑，跟 MaskGen 原论文 recipe 一致：

| 超参 | 取值 | 说明 |
|---|---|---|
| `--num-iter` | 32 | AR 解码步数 T |
| `--cfg` | 3.0 | classifier-free guidance scale（linear schedule）|
| `--aes-score` | 6.5 | 美学打分 micro-conditioning |
| `--model` | `l` / `xl` | MaskGen-L (208M) / MaskGen-XL (479M) |
| CLIP text encoder | `ViT-L-14-336`, `openai` | 冻结 |
| Tokenizer | `turkeyju/tokenizer_tatitok_bl32_vae` | KL 版 TATiTok，N=32 continuous tokens |
| Generator | `turkeyju/generator_maskgen_kl_{l,xl}` | 从 HuggingFace 下载 |

Seed 一般用 `42`（GenEval / DPG / T2I-CB）或 `0/1/2`（MJHQ FID、可视化），论文最终值都是 3 seed 平均。

---

## 7. 关键文件速查

### Python 脚本（仓库根）

| 文件 | 干什么 |
|---|---|
| `run_maskgen_kl_visualize_steps.py` | Fig. 4 那种 random vs SAR 逐步可视化 |
| `run_maskgen_kl_visualize_full_predictions.py` | 同上，但显示每步"模型脑内完整图" |
| `eval_geneval.py` / `eval_geneval_parallel.py` | GenEval 生成 + 评测 |
| `eval_dpg.py` | DPG-Bench 生成 + 评测 |
| `eval_mjhq_fid.py` | MJHQ-30K FID |
| `eval_t2i_compbench.py` | T2I-CompBench++（论文未收） |
| `eval_pbh.py` | PBH 复杂长 prompt benchmark（Table 1） |
| `pbh_judge.py` | PBH 用的 GPT/Gemini VLM judge 模块 |
| `demo_util.py` + `demo.ipynb` | 交互式 demo（上游代码，不涉及 SAR） |

### Shell 一键脚本

| 文件 | 干什么 |
|---|---|
| `run_all_evals.sh` | 4 benchmark 全跑 |
| `run_geneval_all.sh` / `run_geneval_evaluate.sh` / `run_geneval_orders.sh` | GenEval |
| `run_dpg_all.sh` / `run_dpg_evaluate.sh` / `run_dpg_orders.sh` | DPG |
| `run_mjhq_fid_all.sh` | MJHQ FID |
| `run_t2i_compbench_all.sh` / `run_t2i_compbench_evaluate.sh` | T2I-CompBench++ |
| `run_pbh_all.sh` | PBH（含 VLM 评测） |
| `run_new_orders_ablation.sh` | 静态顺序 ablation |
| `setup_geneval_env.sh` / `setup_dpg_env.sh` / `setup_t2i_compbench_env.sh` | 装评测 conda env |

### 目录

| 目录 | 内容 |
|---|---|
| `modeling/` | 模型定义。SAR 逻辑在 `modeling/maskgen.py` 里的 `MaskGen_KL.sample_tokens()`（约 690–830 行）|
| `benchmarks/pbh/` | PBH prompt 集 |
| `configs/` | 训练/推理 YAML 配置（训练用，SAR eval 不需要） |
| `scripts/` | 训练脚本（SAR 不训练，用不到） |
| `assets/` | README 里的示意图 |
| `data3_lhy/` → `/data3/haoyuliu/` | 大盘存储 |
| `image_outputs/` → `/data3/haoyuliu/image_outputs/` | 可视化输出 |

### 已有文档

| 文件 | 内容 |
|---|---|
| `README.md` | 上游 TiTok/MaskGen 项目介绍 |
| `README_MaskGen.md` | 官方 MaskGen 用法（没有 SAR） |
| `README_TiTok.md` / `README_RAR.md` | 上游其他模型 |
| `EVALUATION_HAOYU.md` / `evaluate_orders.md` | Haoyu 自己的评测工作笔记（部分中文）|
| `CLAUDE.md` | 存储规则 |
| `USAGE_CN.md` | 本文档 |

---
