# Evaluating Generation Orders

本 repo 基于 [1d-tokenizer](https://github.com/bytedance/1d-tokenizer)，在
[modeling/maskgen.py](modeling/maskgen.py) 中扩展了 `MaskGen_KL.sample_tokens`，
支持三种 token 生成顺序：

| `--order`        | 含义                                                                 |
| ---------------- | -------------------------------------------------------------------- |
| `random`         | 原 paper 行为：生成前随机采样一次 order                              |
| `prompt_sim`     | 动态顺序：每步优先填充与 pooled text embedding 余弦相似度最高的 token |
| `prompt_sim_rev` | 同上，但优先填相似度最低的 token                                     |

支持的完整列表定义在 [modeling/maskgen.py:642](modeling/maskgen.py#L642) 的
`SUPPORTED_ORDER_TYPES`。

本 repo 提供四套 evaluation 流程：

1. **MJHQ-30K FID**：图像质量指标，衡量生成分布与真实分布的距离
2. **GenEval**：组合能力指标（属性、计数、空间关系等），基于目标检测器判定
3. **DPG-Bench**：dense prompt 理解指标（实体/属性/关系/其它），基于 mPLUG VQA 判定
4. **T2I-CompBench++**：组合性评测（attribute binding / spatial / non-spatial / numeracy / complex），
   BLIP-VQA + UniDet + CLIPScore + 3-in-1 多评估器合评

所有中间产物（生成图、tokenizer 权重、huggingface cache、mask2former 权重等）统一
写到 `/data3/haoyuliu/`（见 [CLAUDE.md](CLAUDE.md)）。仓库根目录有软链接
[data3_lhy](data3_lhy/) 可直接从当前文件夹进入该目录。

---

## 1. MJHQ-30K FID

脚本：[eval_mjhq_fid.py](eval_mjhq_fid.py)

### 数据

MJHQ-30K 数据集应解压到 `/data3/haoyuliu/mjhq30k/`，结构：

```
/data3/haoyuliu/mjhq30k/
├── meta_data.json           # 30K 条 {key: {prompt, category}}
├── animals/                 # 各 category 子目录，原始 jpg
├── art/
├── ...
```

### 运行（生成 + FID）

默认使用所有可见 GPU 并行生成，再用 `clean-fid` 算 FID：

```bash
# MaskGen-XL，random order，全 30K
python eval_mjhq_fid.py --model xl --order random

# MaskGen-XL，prompt_sim order
python eval_mjhq_fid.py --model xl --order prompt_sim

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_mjhq_fid.py --model xl --order prompt_sim

# 快速 smoke test（只跑 1000 张）
python eval_mjhq_fid.py --model xl --order random --num-samples 1000

# 已经生成过图，只算 FID
python eval_mjhq_fid.py --fid-only --gen-dir /path/to/generated
```

常用参数（见 [eval_mjhq_fid.py:219](eval_mjhq_fid.py#L219) 的 argparse）：
- `--model {l,xl}`：MaskGen-KL 模型大小，默认 `xl`
- `--num-iter`：解码步数，默认 32
- `--cfg`：classifier-free guidance scale，默认 3.0
- `--aes-score`：aesthetic micro-condition，默认 6.5
- `--batch-size`：单卡 batch size，默认 512

生成逻辑支持 resume：若 `generated/` 下已有对应 `{key}.png` 会自动跳过。

### 查看结果

每次 run 写到独立目录：

```
/data3/haoyuliu/mjhq30k_eval/eval_{model}_{order}_{30k|nXX}_seed{seed}/
├── generated/           # 生成的 30K 张 png
├── real_resized/        # 指向原图的 symlink（clean-fid 自己做 resize）
└── fid_result.txt       # 最终 FID 分数及所有超参
```

`fid_result.txt` 示例字段：`model / order / num_samples / seed / num_iter / cfg /
aes_score / num_gpus / batch_size_per_gpu / fid`。脚本结尾也会把 FID 打到 stdout，
并对比 paper 中 `xl=6.53 / l=7.24` 的基线。

---

## 2. GenEval

GenEval 分两步，需要**两套 conda 环境**：

- `maskgen`：我们日常用的生成环境（TATiTok + MaskGen + open_clip）
- `geneval`：只用来跑 mmdet + Mask2Former 的检测评估

### 2.1 一次性准备

1. 克隆官方 GenEval 仓库到 `/home/hliu256/geneval`（脚本硬编码了该路径，见
   [eval_geneval.py:53](eval_geneval.py#L53) 和
   [eval_geneval_parallel.py:22](eval_geneval_parallel.py#L22)）。
2. 创建 `geneval` 环境并下载 Mask2Former 权重（只需跑一次）：

```bash
bash setup_geneval_env.sh
```

这会：
- 建 `geneval` conda env（python 3.9 + torch 1.12 + mmcv-full 1.7.1 + mmdet 2.28.2）
- 从源码装 mmdetection 的 `2.x` 分支
- 下载权重到 `/data3/haoyuliu/geneval_eval/mask2former_weights/`

脚本见 [setup_geneval_env.sh](setup_geneval_env.sh)。

### 2.2 生成图像

**要在 `maskgen` 环境下跑**。脚本 [eval_geneval.py](eval_geneval.py) 会读取
`$GENEVAL_ROOT/prompts/evaluation_metadata.jsonl`（553 条 prompt），每 prompt 默认
生成 4 张图，按 GenEval 要求的目录格式写盘：

```bash
conda activate maskgen

# 单个配置
python eval_geneval.py --model xl --order random
python eval_geneval.py --model xl --order prompt_sim
python eval_geneval.py --model l  --order random
python eval_geneval.py --model l  --order prompt_sim

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_geneval.py --model xl --order prompt_sim
```

也可以一次并行跑满 4 个配置（8 卡 A5000，每配置 2 卡）：

```bash
bash run_geneval_all.sh
```

输出布局：

```
/data3/haoyuliu/geneval_eval/maskgen_kl_{model}_{order}/
├── 00000/
│   ├── metadata.jsonl      # 该 prompt 的 metadata
│   └── samples/
│       ├── 00000.png       # n_samples 张图
│       ├── 00001.png
│       └── ...
├── 00001/
└── ...
```

常用参数：
- `--n-samples`：每 prompt 生成多少张图，默认 4（GenEval 标准）
- `--batch-size`：单卡 batch size，默认 256
- `--num-gpus`：默认所有可见卡
- 其它（`--num-iter / --cfg / --aes-score / --seed`）与 MJHQ 脚本一致

### 2.3 目标检测评估

**切换到 `geneval` 环境**。单独跑一个配置，8 卡并行：

```bash
conda activate geneval
python eval_geneval_parallel.py \
    /data3/haoyuliu/geneval_eval/maskgen_kl_xl_prompt_sim \
    --num-gpus 8
```

[eval_geneval_parallel.py](eval_geneval_parallel.py) 会把 prompt 目录 symlink 分到
8 个 shard，各 shard 独立跑 `$GENEVAL_ROOT/evaluation/evaluate_images.py`，最后合并
为单个 `results.jsonl`。

想一键评估全部 4 个配置：

```bash
bash run_geneval_evaluate.sh
```

见 [run_geneval_evaluate.sh](run_geneval_evaluate.sh)。该脚本已经跳过了已有
`results.jsonl` 的目录，所以可以安全重跑。

评估产物：

```
/data3/haoyuliu/geneval_eval/maskgen_kl_{model}_{order}/
├── results.jsonl           # 每张图一行：prompt / tag / correct / reason
└── geneval_summary.json    # 聚合到 prompt 粒度的 tag-level 分数 + overall
```

注意：[eval_geneval.py:299](eval_geneval.py#L299) 的 `_save_parsed_results` 采用
**prompt-level** 指标 —— 只要一个 prompt 的 `n_samples` 张图中**有任意一张通过**，
就算该 prompt 正确。这和 paper 的汇报方式一致。

### 2.4 查看 / 对比结果

```bash
conda activate maskgen
python eval_geneval.py --summary
```

输出一张对比表：

```
Model                        |  S.Obj |  T.Obj |  Count | Colors |   Pos. | C.Attr | Overall
----------------------------------------------------------------------------------------
Paper MaskGen-L (KL)         |   0.99 |   0.57 |   0.36 |   0.80 |   0.11 |   0.29 |   0.52
Paper MaskGen-XL (KL)        |     -- |     -- |     -- |     -- |     -- |     -- |   0.55
----------------------------------------------------------------------------------------
MaskGen-L (random)           |   ...  |   ...  |   ...  |   ...  |   ...  |   ...  |   ...
MaskGen-L (prompt_sim)       |   ...  |   ...  |   ...  |   ...  |   ...  |   ...  |   ...
MaskGen-XL (random)          |   ...
...
```

实现见 [eval_geneval.py:338](eval_geneval.py#L338) 的 `run_summary`：它会扫描
`/data3/haoyuliu/geneval_eval/maskgen_kl_*_*/geneval_summary.json` 并和 paper
的基线并排打印。只要某个配置的 summary json 存在，就会出现在表中。

---

## 3. DPG-Bench

基于官方 repo [ELLA](https://github.com/TencentQQGYLab/ELLA)，已 clone 到
`/home/hliu256/ELLA`。1065 条长 prompt，每 prompt 生成 4 张拼成 2×2 grid，再由
mPLUG VQA 逐 tile 回答多个 yes/no 问题并聚合。

与 GenEval 一样需要**两套 conda 环境**：

- `maskgen`：生成环境
- `dpg`：仅评估用（`accelerate` + `modelscope` + mPLUG）

### 3.1 一次性准备

```bash
bash setup_dpg_env.sh
```

这会：
- 在 `/data3/haoyuliu/conda_envs/dpg` 建 conda env（python 3.10 + torch 2.1 + CUDA 11.8）
  ——因为 `/data1` 已满，env 按 `--prefix` 装到 `/data3`
- `pip install -r /home/hliu256/ELLA/requirements-for-dpg_bench.txt`
- 预下载 mPLUG-large VQA 权重到 `/data3/haoyuliu/modelscope_cache/`（约 8GB）

脚本见 [setup_dpg_env.sh](setup_dpg_env.sh)。

### 3.2 生成图像

**`maskgen` 环境下跑**。脚本 [eval_dpg.py](eval_dpg.py) 读取
`/home/hliu256/ELLA/dpg_bench/prompts/*.txt`（1065 条），每 prompt 生成 4 张
512×512，按顺序拼成 1024×1024 的 2×2 grid 保存：

```bash
conda activate maskgen

python eval_dpg.py --model xl --order random
python eval_dpg.py --model xl --order prompt_sim
python eval_dpg.py --model l  --order random
python eval_dpg.py --model l  --order prompt_sim

# Smoke test（只跑前 10 条）
python eval_dpg.py --model xl --order prompt_sim --num-samples 10

# 并行跑 4 配置（8 卡，每配置 2 卡）
bash run_dpg_all.sh
```

输出布局：

```
/data3/haoyuliu/dpg_bench_eval/maskgen_kl_{model}_{order}/
└── images/
    ├── 0.png               # 1024x1024 (2x2 grid of 512x512 tiles)
    ├── 100.png
    ├── COCOval2014000000191919.png
    └── ...                 # 1065 total, filename = DPG prompt key
```

关键点：DPG 评估器按**固定相对路径** `./dpg_bench/...` 调脚本，所以文件名必须与
`/home/hliu256/ELLA/dpg_bench/prompts/{key}.txt` 的 stem 完全一致。`eval_dpg.py`
已经保证了这点。

Resume 以整个 grid 为单位：若 `{key}.png` 已存在则跳过该 prompt（避免只生成
1-3 块 tile 后没法拼的边界情况）。

### 3.3 VQA 评估

**切换到 `dpg` 环境**（注意 env 在 `/data3` 下，需要用 prefix 路径激活）：

```bash
conda activate /data3/haoyuliu/conda_envs/dpg
python eval_dpg.py --model xl --order prompt_sim --eval-only
```

这一步会：
1. 调 `bash /home/hliu256/ELLA/dpg_bench/dist_eval.sh <images_dir> 512`
   （内部用 `accelerate launch` 多 GPU 并行跑 `compute_dpg_bench.py`）
2. Evaluator 把 `dpg-bench_<timestamp>_results.txt` 写到 images 目录
3. [eval_dpg.py](eval_dpg.py) 再把该 txt 解析成 `dpg_summary.json`

一键评估全部 4 个配置（已做幂等检查）：

```bash
bash run_dpg_evaluate.sh
```

评估产物：

```
/data3/haoyuliu/dpg_bench_eval/maskgen_kl_{model}_{order}/
├── images/
│   ├── dpg-bench_<ts>_results.txt         # evaluator 原始输出
│   └── dpg-bench_<ts>_results_detail.txt  # 每张图每道题的 yes/no
└── dpg_summary.json                        # 结构化 overall + L1/L2 分数
```

`dpg_summary.json` 示例字段：`model / order / overall / l1_categories /
l2_categories / results_txt`。分数范围 0-100，越高越好。

### 3.4 查看 / 对比结果

```bash
conda activate maskgen
python eval_dpg.py --summary
```

输出一张对比表，列 = 所有 L1 类别（通常是 global / entity / attribute /
relation / other）+ overall。见 [eval_dpg.py:run_summary](eval_dpg.py)。

DPG paper 没报 MaskGen baseline，所以表里只有我们自己跑出来的 4 行。

---

## 4. T2I-CompBench++

基于官方 repo [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench)，已 clone 到
`/home/hliu256/T2I-CompBench`。8 个类别 × ~300 val prompts × 10 samples =
每 config ~24k 张图；4 configs 合计 ~96k 张。评估器是按类别切分的：

| Category      | Evaluator                                                 |
| ------------- | --------------------------------------------------------- |
| color / shape / texture   | BLIP-VQA                                      |
| spatial                   | UniDet (2D spatial)                           |
| non_spatial               | CLIPScore                                     |
| complex                   | 3-in-1 (BLIP + UniDet-2D + CLIP with `--complex True`) |
| 3d_spatial                | UniDet (3D spatial, 带 MiDaS 深度)            |
| numeracy                  | UniDet (numeracy)                             |

与 GenEval / DPG 一样需要**两套 conda 环境**：

- `maskgen`：生成环境
- `t2i_compbench`：仅评估用（torch 2.0.1 + detectron2 + diffusers 0.15 等）

### 4.1 一次性准备

```bash
bash setup_t2i_compbench_env.sh
```

这会：
- 在 `/data3/haoyuliu/conda_envs/t2i_compbench` 建 conda env（python 3.10 + torch 2.0.1 + cu118）
- 装 T2I-CompBench 的 pinned deps（transformers 4.30.2 / spacy 3.5.3 / accelerate 0.17.0 等）
- 从 commit pin 编译 detectron2（**最容易炸的一步**，需要 nvcc + gcc 在 PATH）
- 下载 UniDet (RS200 + R50) + MiDaS 权重到
  `/home/hliu256/T2I-CompBench/UniDet_eval/experts/expert_weights/`
- 写默认 accelerate config（fp16, 单 GPU）

脚本见 [setup_t2i_compbench_env.sh](setup_t2i_compbench_env.sh)。

### 4.2 生成图像

**`maskgen` 环境下跑**。脚本 [eval_t2i_compbench.py](eval_t2i_compbench.py) 读取
`/home/hliu256/T2I-CompBench/examples/dataset/{category}_val.txt`，每 prompt 生成
10 张 512×512，按 T2I-CompBench 约定的
`"{prompt}_{global_idx:06d}.png"` 命名（全局索引 = `prompt_idx * 10 + sample_idx`）：

```bash
conda activate maskgen

# 单个 config，全部 8 类
python eval_t2i_compbench.py --model xl --order random
python eval_t2i_compbench.py --model xl --order prompt_sim

# 单类（加快 smoke test 或单独补数）
python eval_t2i_compbench.py --model xl --order prompt_sim --category color
python eval_t2i_compbench.py --model xl --order prompt_sim \
    --category color --num-samples 5 --samples-per-prompt 2 --num-gpus 1

# 并行跑 4 配置（8 卡，每 config 2 卡）
bash run_t2i_compbench_all.sh
```

输出布局：

```
/data3/haoyuliu/t2i_compbench_eval/maskgen_kl_{model}_{order}/
├── color/
│   └── samples/
│       ├── a green bench and a blue bowl_000000.png
│       ├── a green bench and a blue bowl_000001.png
│       └── ...           # 2990 张 (299 prompts × 10 samples)
├── shape/samples/
├── texture/samples/
├── spatial/samples/
├── non_spatial/samples/
├── complex/samples/
├── 3d_spatial/samples/
└── numeracy/samples/
```

Resume 以单张图为单位：若 `{prompt}_{global_idx:06d}.png` 已存在则跳过该 tile。
Seed 与 `--order` 无关，所以 random 和 prompt_sim 两次 run 的 batch 组成一致，
差异只来自 order_type。

常用参数：
- `--category {all,color,shape,texture,spatial,non_spatial,complex,3d_spatial,numeracy}`
- `--samples-per-prompt`：默认 10（**注意：3-in-1 硬编码了 num=10，complex 这一类
  不能随便改**）
- `--num-samples`：限制每类前 N 条 prompt，smoke test 用
- 其它（`--num-iter / --cfg / --aes-score / --seed / --batch-size`）与
  DPG / MJHQ 脚本一致

### 4.3 评估

**切换到 `t2i_compbench` 环境**：

```bash
conda activate /data3/haoyuliu/conda_envs/t2i_compbench
python eval_t2i_compbench.py --model xl --order prompt_sim --eval-only
```

这一步会：
1. 按类别 subprocess 调 T2I-CompBench 的评估脚本
   （`BLIPvqa_eval/BLIP_vqa.py` / `UniDet_eval/{2D_spatial,3D_spatial,numeracy}_eval.py`
   / `CLIPScore_eval/CLIP_similarity.py` / `3_in_1_eval/3_in_1.py`）
2. 每类产出 `{cat_dir}/score.json`（score / n_images / evaluator）
3. 自动 aggregate 到 `t2i_compbench_summary.json`

**一键评估全部 4 配置（每个 category 内部 8 卡切分）**：

```bash
bash run_t2i_compbench_evaluate.sh
```

并行策略（实现见 [eval_t2i_compbench.py](eval_t2i_compbench.py) 里的
`_run_sharded_eval`）：

- **4 个 configs 串行**
- 每个 config 里 **8 个 categories 串行**
- 每个 category 里 **把 samples 图片均匀切成 8 份，8 张 GPU 并行跑同一个 evaluator**

具体做法是在 `{cat_dir}/_shards_{evaluator}/shard_{i}/samples/` 下用 symlink 建
8 份镜像目录（`files[i::8]` 的 stride 分片），每份交给一个 subprocess，通过
`CUDA_VISIBLE_DEVICES=i` 各自绑定 1 卡，等 8 个都 wait 完再把 8 份
`vqa_result.json` 合并回官方路径。合并时两种 qid 约定分别处理：

- BLIP / CLIP：shard 内 `question_id = 本地计数`，合并公式
  `global = local * 8 + shard_idx`（stride 分片保序所以是对的）
- UniDet 2D / 3D / numeracy：`question_id` 从文件名里 parse 得到，
  本来就是 global index，直接 union + dedup + sort

这样每步每张 GPU 处理 ~N/8 张图，几乎无 idle 等待。complex 类虽然要串三步
(BLIP → UniDet-2D → CLIP) 再 3-in-1，但每步都是 8 卡全速。

日志：每个 shard 的 stdout/stderr 写到 `/tmp/t2icb_eval_<pid>/<tag>_shard<i>.log`，
失败会打印 log tail 并 raise。

评估产物：

```
/data3/haoyuliu/t2i_compbench_eval/maskgen_kl_{model}_{order}/
├── color/
│   ├── score.json                           # {score, n_images, evaluator}
│   └── annotation_blip/vqa_result.json      # 逐图原始分
├── spatial/
│   ├── score.json
│   └── labels/annotation_obj_detection_2d/vqa_result.json
├── complex/
│   ├── score.json
│   ├── annotation_blip/ + annotation_clip/ + labels/annotation_obj_detection_2d/
│   └── annotation_3_in_1/vqa_result.json    # 加权合并的最终分
├── ...
└── t2i_compbench_summary.json               # {categories: {...}, overall_avg}
```

### 4.4 查看 / 对比结果

```bash
python eval_t2i_compbench.py --summary
```

输出一张按类别切分的对比表：

```
================================================================================
T2I-CompBench++ Category Comparison (val set, 10 samples/prompt)
================================================================================
Category       |   L-rand    L-sim      ΔL |  XL-rand   XL-sim     ΔXL
--------------------------------------------------------------------------------
color          |   0.XXXX   0.XXXX   +0.0XX |   ...      ...      ...
shape          |   ...
texture        |   ...
spatial        |   ...
non_spatial    |   ...
complex        |   ...
3d_spatial     |   ...
numeracy       |   ...
--------------------------------------------------------------------------------
Overall avg    |   ...
================================================================================
```

结果也写到 `/data3/haoyuliu/t2i_compbench_eval/t2i_compbench_summary_tables.txt`。

T2I-CompBench paper 没报 MaskGen baseline，所以表里只有我们自己的 4 行
（L/XL × random/prompt_sim）。

---

## 典型完整流程

想比较 `random` vs `prompt_sim` 在两个模型大小上的表现：

```bash
# ---- MJHQ-30K FID ----
conda activate maskgen
for m in l xl; do
  for o in random prompt_sim; do
    python eval_mjhq_fid.py --model $m --order $o
  done
done
# 每个 run_dir 下的 fid_result.txt 里就是 FID 数字

# ---- GenEval ----
bash run_geneval_all.sh          # maskgen env，并行生成 4 配置
bash run_geneval_evaluate.sh     # geneval env，并行评估 4 配置
python eval_geneval.py --summary # 打印对比表

# ---- DPG-Bench ----
conda activate maskgen
bash run_dpg_all.sh                                     # maskgen env，并行生成 4 配置
bash run_dpg_evaluate.sh                                # 脚本里已 conda activate /data3/.../dpg
conda activate maskgen
python eval_dpg.py --summary                            # 打印对比表

# ---- T2I-CompBench++ ----
conda activate maskgen
bash run_t2i_compbench_all.sh                           # maskgen env，并行生成 4 配置 (8 类 × 10 samples)
bash run_t2i_compbench_evaluate.sh                      # 脚本里已切到 t2i_compbench env；8 卡并行评估
conda activate maskgen
python eval_t2i_compbench.py --summary                  # 打印对比表
```
