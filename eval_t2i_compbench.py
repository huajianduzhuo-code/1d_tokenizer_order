"""Evaluate MaskGen-KL on T2I-CompBench++.

Generates 10 images/prompt for each of the 8 T2I-CompBench++ val categories
(color, shape, texture, spatial, non_spatial, complex, 3d_spatial, numeracy)
using MaskGen-KL (L/XL), then runs the official per-category evaluators
(BLIP-VQA / UniDet-2D / UniDet-3D / UniDet-numeracy / CLIPScore / 3-in-1).

Usage (generation in `maskgen` env):
    python eval_t2i_compbench.py --model xl --order prompt_sim
    python eval_t2i_compbench.py --model xl --order prompt_sim --category color
    # Smoke test
    python eval_t2i_compbench.py --model xl --order prompt_sim \
        --category color --num-samples 5 --samples-per-prompt 2 --num-gpus 1

Usage (evaluation in `t2i_compbench` env — set up via setup_t2i_compbench_env.sh):
    conda activate /data3/haoyuliu/conda_envs/t2i_compbench
    python eval_t2i_compbench.py --model xl --order prompt_sim --eval-only

Summary (either env):
    python eval_t2i_compbench.py --summary
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"
os.environ.setdefault("MODELSCOPE_CACHE", "/data3/haoyuliu/modelscope_cache")

import argparse
import glob
import json
import math
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------
T2I_ROOT = "/home/hliu256/T2I-CompBench"
T2I_DATASET_DIR = os.path.join(T2I_ROOT, "examples/dataset")
OUTPUT_BASE = "/data3/haoyuliu/t2i_compbench_eval"

RESOLUTION = 512
DEFAULT_SAMPLES_PER_PROMPT = 10   # T2I-CompBench convention (3-in-1 hardcodes num=10)

# Display order + evaluator type per category.
# "needs_complex_flag": pass --complex True to 2D-spatial and CLIP scripts for
# the 'complex' category per Readme.md §2/§3.
CATEGORIES = [
    "color", "shape", "texture",
    "spatial", "non_spatial", "complex",
    "3d_spatial", "numeracy",
]

EVALUATOR = {
    "color":       "blip",
    "shape":       "blip",
    "texture":     "blip",
    "spatial":     "unidet2d",
    "non_spatial": "clip",
    "complex":     "3in1",    # runs blip + unidet2d + clip (--complex) then 3_in_1
    "3d_spatial":  "unidet3d",
    "numeracy":    "numeracy",
}


# -----------------------------------------------------------------
# Lazy imports (generation-only)
# -----------------------------------------------------------------
def _import_generation_stack():
    import torch
    import torch.multiprocessing as mp
    import open_clip
    from modeling.tatitok import TATiTok
    from modeling.maskgen import MaskGen_KL
    return torch, mp, open_clip, TATiTok, MaskGen_KL


# -----------------------------------------------------------------
# Shared helpers (copied from eval_dpg.py to stay self-contained)
# -----------------------------------------------------------------
def set_all_seeds(seed):
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_text_guidance(prompts, clip_tokenizer, clip_encoder, device):
    """Same formula as eval_dpg.py.compute_text_guidance."""
    import torch  # noqa: F401 (ensures torch is imported in the worker)
    text_guidance_ids = clip_tokenizer(prompts).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_guidance = clip_encoder.token_embedding(text_guidance_ids).to(cast_dtype)
    text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.ln_final(text_guidance)
    return text_guidance


# -----------------------------------------------------------------
# Prompt loading
# -----------------------------------------------------------------
def _val_file(category):
    return os.path.join(T2I_DATASET_DIR, f"{category}_val.txt")


def load_prompts(category):
    """Return the list of val prompts for `category`, stripped of CR/LF/whitespace.

    T2I-CompBench val files are CRLF on disk — strip before use.
    """
    path = _val_file(category)
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    # Benchmark convention: filenames are split by '_'. Abort if any prompt
    # contains a literal underscore (would break the eval scripts' prompt
    # recovery: file_name.split('_')[0]).
    for p in prompts:
        if "_" in p:
            raise ValueError(
                f"Prompt in {path} contains '_' which breaks T2I-CompBench's "
                f"filename convention: {p!r}")
    return prompts


def run_dir(model, order, seed):
    return os.path.join(OUTPUT_BASE, f"maskgen_kl_{model}_{order}_seed{seed}")


def category_dir(model, order, category, seed):
    return os.path.join(run_dir(model, order, seed), category)


def samples_dir(model, order, category, seed):
    return os.path.join(category_dir(model, order, category, seed), "samples")


def image_filename(prompt, prompt_idx, sample_idx, samples_per_prompt):
    """Match T2I-CompBench's `inference_eval.py` naming:

    Global image index = prompt_idx * samples_per_prompt + sample_idx. The
    eval scripts sort by this trailing integer; consecutive groups of
    `samples_per_prompt` belong to the same prompt (which is how 3_in_1.py
    slices `[i*num:(i+1)*num]`).
    """
    global_idx = prompt_idx * samples_per_prompt + sample_idx
    return f"{prompt}_{global_idx:06d}.png"


# -----------------------------------------------------------------
# Generation
# -----------------------------------------------------------------
def worker_generate(rank, num_gpus, categories, out_base, args):
    """Each GPU handles a contiguous prompt shard of every requested category.

    Models are loaded once per worker and reused across categories.
    """
    torch, _, open_clip, TATiTok, MaskGen_KL = _import_generation_stack()

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    print(f"[GPU {rank}] Loading models...")
    tatitok = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae")
    tatitok.eval().requires_grad_(False).to(device)

    generator_repo = f"turkeyju/generator_maskgen_kl_{args.model}"
    maskgen = MaskGen_KL.from_pretrained(generator_repo)
    maskgen.eval().requires_grad_(False).to(device)

    clip_encoder, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14-336', pretrained='openai')
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval().requires_grad_(False).to(device)

    for category in categories:
        prompts = load_prompts(category)
        if args.num_samples:
            prompts = prompts[:args.num_samples]

        out_dir = os.path.join(out_base, category, "samples")
        os.makedirs(out_dir, exist_ok=True)

        shard_size = math.ceil(len(prompts) / num_gpus)
        shard_start = rank * shard_size
        shard_end = min(shard_start + shard_size, len(prompts))

        # Build (prompt_idx_in_val, sample_idx, prompt_text, out_path) tasks
        # for this shard, skipping any files already on disk (resume).
        tile_tasks = []
        for prompt_idx in range(shard_start, shard_end):
            prompt_text = prompts[prompt_idx]
            for sample_idx in range(args.samples_per_prompt):
                fname = image_filename(
                    prompt_text, prompt_idx, sample_idx, args.samples_per_prompt)
                path = os.path.join(out_dir, fname)
                if os.path.exists(path):
                    continue
                tile_tasks.append((prompt_idx, sample_idx, prompt_text, path))

        if not tile_tasks:
            if rank == 0:
                print(f"[GPU {rank}] [{category}] all {shard_end - shard_start} "
                      f"prompts already generated, skipping.")
            continue

        total_tiles = len(tile_tasks)
        print(f"[GPU {rank}] [{category}] generating {total_tiles} tiles "
              f"(shard prompts {shard_start}-{shard_end})")

        batch_size = args.batch_size
        num_batches = math.ceil(total_tiles / batch_size)
        start_time = time.time()
        done = 0

        pbar = tqdm(range(num_batches), desc=f"GPU{rank} {category}",
                    position=rank, leave=True, disable=(rank != 0))

        for batch_idx in pbar:
            s = batch_idx * batch_size
            e = min(s + batch_size, total_tiles)
            batch = tile_tasks[s:e]
            bsz = len(batch)
            prompts_batch = [t[2] for t in batch]

            text_guidance = compute_text_guidance(
                prompts_batch, clip_tokenizer, clip_encoder, device)

            # Seed independent of order_type so random and prompt_sim runs share
            # RNG state and differ only because of the order flag.
            set_all_seeds(
                args.seed
                + hash(category) % 1_000_003
                + shard_start * 1_000_003
                + batch_idx)

            with torch.no_grad():
                tokens = maskgen.sample_tokens(
                    bsz=bsz,
                    clip_tokenizer=clip_tokenizer,
                    clip_encoder=clip_encoder,
                    num_iter=args.num_iter,
                    cfg=args.cfg,
                    captions=prompts_batch,
                    aes_scores=args.aes_score,
                    order_type=args.order,
                )

                images = tatitok.decode_tokens(tokens, text_guidance)
                images = torch.clamp(images, 0.0, 1.0)
                images = (images * 255.0).permute(0, 2, 3, 1).to(
                    "cpu", dtype=torch.uint8).numpy()

            for (prompt_idx, sample_idx, prompt_text, out_path), arr in zip(batch, images):
                img = Image.fromarray(arr)
                if img.size != (RESOLUTION, RESOLUTION):
                    img = img.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)
                img.save(out_path)

            done += bsz
            if rank == 0 and (batch_idx + 1) % 5 == 0:
                elapsed = time.time() - start_time
                rate = done / max(elapsed, 1e-6)
                remaining = (total_tiles - done) / max(rate, 1e-6)
                pbar.set_postfix(rate=f"{rate:.1f}/s",
                                 eta=f"{remaining / 60:.1f}min")

        elapsed = time.time() - start_time
        print(f"[GPU {rank}] [{category}] done: {done} in {elapsed/60:.1f}min "
              f"({done/max(elapsed,1):.1f} tiles/s)")


def run_generation(args):
    torch, mp, *_ = _import_generation_stack()

    categories = _resolve_categories(args)
    out_base = run_dir(args.model, args.order, args.seed)
    os.makedirs(out_base, exist_ok=True)

    print(f"Model: MaskGen-KL-{args.model.upper()}")
    print(f"Order: {args.order}")
    print(f"Seed:  {args.seed}")
    print(f"Categories: {categories}")
    print(f"Samples/prompt: {args.samples_per_prompt}")
    print(f"Output base: {out_base}")

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}")

    if num_gpus == 1:
        worker_generate(0, 1, categories, out_base, args)
    else:
        mp.spawn(
            worker_generate,
            args=(num_gpus, categories, out_base, args),
            nprocs=num_gpus,
            join=True,
        )

    for category in categories:
        sdir = samples_dir(args.model, args.order, category, args.seed)
        n = sum(1 for p in os.listdir(sdir) if p.endswith(".png")) if os.path.isdir(sdir) else 0
        n_prompts = len(load_prompts(category))
        if args.num_samples:
            n_prompts = min(n_prompts, args.num_samples)
        expected = n_prompts * args.samples_per_prompt
        print(f"  [{category}] {n}/{expected} images in {sdir}")


# -----------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------
def _resolve_categories(args):
    if args.category == "all":
        return list(CATEGORIES)
    if args.category not in CATEGORIES:
        raise ValueError(f"--category must be 'all' or one of {CATEGORIES}")
    return [args.category]


def _run_subproc(cmd, cwd, tag):
    print(f"\n>>> [{tag}] cwd={cwd}\n    {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


# ---- Image-sharding multi-GPU runner -----------------------------
#
# Each per-category evaluator is single-GPU as written. To use all N GPUs
# for one category, we:
#   1. stride-shard the samples/ dir into N subdirs of symlinks
#      (shard i gets files[i::N], preserving per-shard sort order)
#   2. launch N subprocesses in parallel with CUDA_VISIBLE_DEVICES=i each
#   3. merge the N vqa_result.json files back into the canonical cat_dir path
#
# Safe because: stage outputs land under disjoint shard dirs; merge reads
# them and writes one file; shard dirs are torn down at the end.
# -----------------------------------------------------------------
def _sort_sample_files(samples_dir):
    files = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
    files.sort(key=lambda x: int(x.rsplit("_", 1)[-1].split(".")[0]))
    return files


def _shard_samples(samples_dir, n_shards, shard_base):
    """Create shard_base/shard_{i}/samples/ with symlinks to files[i::n_shards].

    Fresh symlink tree each call (old one is removed) so reruns are clean.
    """
    files = _sort_sample_files(samples_dir)
    if os.path.exists(shard_base):
        shutil.rmtree(shard_base)
    shard_dirs = []
    for i in range(n_shards):
        sd = os.path.join(shard_base, f"shard_{i}")
        s_samples = os.path.join(sd, "samples")
        os.makedirs(s_samples)
        for fname in files[i::n_shards]:
            src = os.path.join(samples_dir, fname)
            dst = os.path.join(s_samples, fname)
            os.symlink(src, dst)
        shard_dirs.append(sd)
    return shard_dirs, files


def _wait_all(procs, tag, logs=None):
    fail = 0
    for i, p in enumerate(procs):
        p.wait()
        if p.returncode != 0:
            fail += 1
            print(f"[{tag}] shard {i} exited with code {p.returncode}")
            if logs and logs[i] and os.path.exists(logs[i]):
                print(f"  log tail: {logs[i]}")
                with open(logs[i]) as f:
                    for line in f.readlines()[-20:]:
                        print(f"    {line.rstrip()}")
    if fail:
        raise RuntimeError(f"[{tag}] {fail}/{len(procs)} shard(s) failed")


def _merge_vqa_json(shard_dirs, n_shards, rel_path, qid_is_global, out_path):
    """Merge per-shard vqa_result.json files into one at out_path.

    qid_is_global=True  : each shard's question_id is already the global
                          image index (UniDet scripts parse it from filename).
                          Just union + dedup + sort by question_id.
    qid_is_global=False : each shard assigns local 0..shard_size-1 based on
                          its sorted file list (BLIP / CLIP). Remap with
                          global_qid = local_qid * n_shards + shard_idx
                          (valid because stride sharding preserves sort order).
    """
    all_entries = []
    for i, sd in enumerate(shard_dirs):
        p = os.path.join(sd, rel_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"shard {i} missing output: {p}")
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            qid = int(entry["question_id"])
            if not qid_is_global:
                qid = qid * n_shards + i
            all_entries.append({"question_id": qid, "answer": entry["answer"]})

    # Dedup (same qid from multiple shards shouldn't happen but guard anyway),
    # keeping the first occurrence, then sort.
    seen = set()
    dedup = []
    for e in all_entries:
        if e["question_id"] in seen:
            continue
        seen.add(e["question_id"])
        dedup.append(e)
    dedup.sort(key=lambda e: e["question_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dedup, f)
    print(f"  merged {len(dedup)} entries -> {out_path}")


def _popen_sharded(cmd_for_shard, shard_dirs, cwd, tag):
    """Launch subprocess per shard with CUDA_VISIBLE_DEVICES=i.

    cmd_for_shard: callable(shard_dir) -> list[str]   (the argv for that shard)
    Returns (procs, log_paths).
    """
    log_dir = os.path.join("/tmp", f"t2icb_eval_{os.getpid()}")
    os.makedirs(log_dir, exist_ok=True)
    procs, logs = [], []
    print(f"\n>>> [{tag}] launching {len(shard_dirs)} shards (cwd={cwd})")
    for i, sd in enumerate(shard_dirs):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        log_path = os.path.join(log_dir, f"{tag.replace('/', '_')}_shard{i}.log")
        log_f = open(log_path, "w")
        p = subprocess.Popen(
            cmd_for_shard(sd), cwd=cwd, env=env,
            stdout=log_f, stderr=subprocess.STDOUT)
        procs.append(p)
        logs.append(log_path)
        print(f"     shard {i} -> GPU {i}, pid {p.pid}, log {log_path}")
    return procs, logs


def _run_sharded_eval(cat_dir, n_gpus, tag, cmd_for_shard, cwd,
                      shard_result_rel, qid_is_global, final_result_path,
                      shard_base_name):
    """End-to-end sharded eval: shard → parallel subprocess → merge → cleanup."""
    samples_dir = os.path.join(cat_dir, "samples")
    shard_base = os.path.join(cat_dir, shard_base_name)
    shard_dirs, _ = _shard_samples(samples_dir, n_gpus, shard_base)

    procs, logs = _popen_sharded(cmd_for_shard, shard_dirs, cwd, tag)
    _wait_all(procs, tag, logs=logs)

    _merge_vqa_json(
        shard_dirs, n_gpus, shard_result_rel, qid_is_global,
        out_path=final_result_path)

    # Tear down shard tree (symlinks + per-shard annotation_* output dirs)
    shutil.rmtree(shard_base, ignore_errors=True)


# ---- Per-evaluator wrappers --------------------------------------
def _eval_blip(cat_dir, tag, n_gpus):
    """BLIP-VQA via image sharding. Each shard runs BLIP_vqa.py on its own
    GPU; merged result lands at {cat_dir}/annotation_blip/vqa_result.json."""
    def cmd(shard_dir):
        return [sys.executable, "BLIP_vqa.py", f"--out_dir={shard_dir}/"]

    _run_sharded_eval(
        cat_dir, n_gpus, tag,
        cmd_for_shard=cmd,
        cwd=os.path.join(T2I_ROOT, "BLIPvqa_eval"),
        shard_result_rel="annotation_blip/vqa_result.json",
        qid_is_global=False,  # BLIP assigns local cnt
        final_result_path=os.path.join(cat_dir, "annotation_blip/vqa_result.json"),
        shard_base_name="_shards_blip",
    )


def _eval_unidet2d(cat_dir, tag, is_complex, n_gpus):
    extra = ["--complex", "True"] if is_complex else []

    def cmd(shard_dir):
        return [sys.executable, "2D_spatial_eval.py",
                f"--outpath={shard_dir}/"] + extra

    _run_sharded_eval(
        cat_dir, n_gpus, tag,
        cmd_for_shard=cmd,
        cwd=os.path.join(T2I_ROOT, "UniDet_eval"),
        shard_result_rel="labels/annotation_obj_detection_2d/vqa_result.json",
        qid_is_global=True,  # parsed from filename
        final_result_path=os.path.join(
            cat_dir, "labels/annotation_obj_detection_2d/vqa_result.json"),
        shard_base_name="_shards_unidet2d",
    )


def _eval_unidet3d(cat_dir, tag, n_gpus):
    # 3D_spatial_eval.py's two stages disagree on how to parse outpath when it
    # has a trailing '/': stage 1 writes depth to <save>/depth/{img_split[-3]}/
    # {img_split[-2]}/<fname>, stage 2 looks under <save>/depth/{data_path.
    # split('/')[-1]}/<fname>. Passing outpath WITHOUT a trailing '/' makes
    # both stages agree on the {shard_N}/samples/ suffix.
    def cmd(shard_dir):
        return [sys.executable, "3D_spatial_eval.py", f"--outpath={shard_dir}"]

    _run_sharded_eval(
        cat_dir, n_gpus, tag,
        cmd_for_shard=cmd,
        cwd=os.path.join(T2I_ROOT, "UniDet_eval"),
        shard_result_rel="labels/annotation_obj_detection_3d/vqa_result.json",
        qid_is_global=True,
        final_result_path=os.path.join(
            cat_dir, "labels/annotation_obj_detection_3d/vqa_result.json"),
        shard_base_name="_shards_unidet3d",
    )


def _eval_numeracy(cat_dir, tag, n_gpus):
    def cmd(shard_dir):
        return [sys.executable, "numeracy_eval.py", f"--outpath={shard_dir}/"]

    _run_sharded_eval(
        cat_dir, n_gpus, tag,
        cmd_for_shard=cmd,
        cwd=os.path.join(T2I_ROOT, "UniDet_eval"),
        shard_result_rel="annotation_num/vqa_result.json",
        qid_is_global=True,
        final_result_path=os.path.join(cat_dir, "annotation_num/vqa_result.json"),
        shard_base_name="_shards_numeracy",
    )


def _eval_clip(cat_dir, tag, is_complex, n_gpus):
    extra = ["--complex", "True"] if is_complex else []

    def cmd(shard_dir):
        return [sys.executable, "CLIPScore_eval/CLIP_similarity.py",
                f"--outpath={shard_dir}/"] + extra

    _run_sharded_eval(
        cat_dir, n_gpus, tag,
        cmd_for_shard=cmd,
        cwd=T2I_ROOT,
        shard_result_rel="annotation_clip/vqa_result.json",
        qid_is_global=False,  # CLIP assigns local i
        final_result_path=os.path.join(cat_dir, "annotation_clip/vqa_result.json"),
        shard_base_name="_shards_clip",
    )


def _eval_3in1(cat_dir, tag):
    """3-in-1 is CPU-only, just merges the three upstream JSONs."""
    cmd = [sys.executable, "3_in_1.py",
           f"--outpath={cat_dir}/",
           f"--data_path={T2I_DATASET_DIR}/"]
    _run_subproc(cmd, cwd=os.path.join(T2I_ROOT, "3_in_1_eval"), tag=tag)


def _mean_from_vqa_json(path):
    """Read a vqa_result.json and return mean of `answer` as float.

    Handles both stringified ("0.6842") and float answers.
    """
    with open(path) as f:
        items = json.load(f)
    vals = []
    for it in items:
        a = it["answer"]
        try:
            vals.append(float(a))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals), len(vals)


def _write_category_score(cat_dir, info):
    """Atomic write of <cat_dir>/score.json."""
    os.makedirs(cat_dir, exist_ok=True)
    path = os.path.join(cat_dir, "score.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(info, f, indent=2)
    os.replace(tmp, path)


def aggregate_config(model, order, seed, samples_per_prompt=None):
    """Scan all <cat_dir>/score.json for a config and write the config-level
    t2i_compbench_summary.json. Idempotent, race-free vs per-category writes
    (atomic rename + reads disjoint files)."""
    config_dir = run_dir(model, order, seed)
    cats = {}
    for cat in CATEGORIES:
        p = os.path.join(config_dir, cat, "score.json")
        if os.path.exists(p):
            with open(p) as f:
                cats[cat] = json.load(f)
    scored = [v["score"] for v in cats.values() if v.get("score") is not None]
    summary = {
        "model": f"maskgen_kl_{model}",
        "order": order,
        "seed": seed,
        "samples_per_prompt": samples_per_prompt,
        "categories": cats,
        "overall_avg": sum(scored) / len(scored) if scored else None,
    }
    out_path = os.path.join(config_dir, "t2i_compbench_summary.json")
    os.makedirs(config_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, out_path)
    return summary


def _score_path_for(category, cat_dir):
    """Where each evaluator writes the per-image JSON."""
    if EVALUATOR[category] == "blip":
        return os.path.join(cat_dir, "annotation_blip/vqa_result.json")
    if EVALUATOR[category] == "unidet2d":
        return os.path.join(cat_dir, "labels/annotation_obj_detection_2d/vqa_result.json")
    if EVALUATOR[category] == "unidet3d":
        return os.path.join(cat_dir, "labels/annotation_obj_detection_3d/vqa_result.json")
    if EVALUATOR[category] == "numeracy":
        return os.path.join(cat_dir, "annotation_num/vqa_result.json")
    if EVALUATOR[category] == "clip":
        return os.path.join(cat_dir, "annotation_clip/vqa_result.json")
    if EVALUATOR[category] == "3in1":
        return os.path.join(cat_dir, "annotation_3_in_1/vqa_result.json")
    raise ValueError(category)


def run_evaluation(args):
    import torch  # for device_count default
    n_gpus = args.num_gpus or torch.cuda.device_count() or 1
    categories = _resolve_categories(args)
    config_dir = run_dir(args.model, args.order, args.seed)
    per_cat_summary = {}

    print(f"Eval with {n_gpus} GPU(s) per category (image sharding). Seed={args.seed}")

    for category in categories:
        cat_dir = category_dir(args.model, args.order, category, args.seed)
        sdir = os.path.join(cat_dir, "samples")
        if not os.path.isdir(sdir):
            print(f"[{category}] samples/ not found: {sdir} — skip")
            continue

        # Resume: if score.json already exists and looks complete, skip.
        # (--force re-runs anyway.)
        existing_score = os.path.join(cat_dir, "score.json")
        if os.path.exists(existing_score) and not args.force:
            try:
                with open(existing_score) as f:
                    prev = json.load(f)
                if prev.get("score") is not None:
                    print(f"[{category}] already evaluated "
                          f"(score={prev['score']:.4f}, evaluator={prev.get('evaluator')}) — skip")
                    per_cat_summary[category] = prev
                    continue
            except (json.JSONDecodeError, KeyError):
                pass  # malformed; re-run

        n_prompts = len(load_prompts(category))
        if args.num_samples:
            n_prompts = min(n_prompts, args.num_samples)
        expected = n_prompts * args.samples_per_prompt
        n_images = sum(1 for p in os.listdir(sdir) if p.endswith(".png"))
        if n_images < expected and not args.force:
            print(f"[{category}] only {n_images}/{expected} images — "
                  f"pass --force to evaluate anyway")
            continue

        tag = f"{args.model}-{args.order}/{category}"
        ev = EVALUATOR[category]
        print(f"\n{'='*60}\nEvaluating {tag}  ({ev}, {n_images} images)\n{'='*60}")

        if ev == "blip":
            _eval_blip(cat_dir, tag, n_gpus)
        elif ev == "unidet2d":
            _eval_unidet2d(cat_dir, tag, is_complex=False, n_gpus=n_gpus)
        elif ev == "unidet3d":
            _eval_unidet3d(cat_dir, tag, n_gpus)
        elif ev == "numeracy":
            _eval_numeracy(cat_dir, tag, n_gpus)
        elif ev == "clip":
            _eval_clip(cat_dir, tag, is_complex=False, n_gpus=n_gpus)
        elif ev == "3in1":
            # 3-in-1 needs BLIP + UniDet-2D + CLIP first (all on complex samples).
            # Run them sequentially, each using all n_gpus via sharding.
            _eval_blip(cat_dir, tag + "/blip", n_gpus)
            _eval_unidet2d(cat_dir, tag + "/unidet2d", is_complex=True, n_gpus=n_gpus)
            _eval_clip(cat_dir, tag + "/clip", is_complex=True, n_gpus=n_gpus)
            _eval_3in1(cat_dir, tag + "/3in1")

        result_json = _score_path_for(category, cat_dir)
        if not os.path.exists(result_json):
            print(f"[{category}] WARN: expected {result_json} not found after eval")
            per_cat_summary[category] = {"score": None, "evaluator": ev}
        else:
            mean_n = _mean_from_vqa_json(result_json)
            if mean_n is None:
                per_cat_summary[category] = {"score": None, "evaluator": ev}
            else:
                mean, n_scored = mean_n
                per_cat_summary[category] = {
                    "score": mean, "n_images": n_scored, "evaluator": ev,
                    "result_json": os.path.relpath(result_json, config_dir),
                }
                print(f"[{category}] {ev} mean score: {mean:.4f} over {n_scored} images")

        # Persist per-category immediately so a later failure doesn't lose
        # the work we just did.
        _write_category_score(cat_dir, per_cat_summary[category])

    # Aggregate after each run. Idempotent: reads whatever score.json files
    # are on disk (including those written by prior invocations).
    summary = aggregate_config(args.model, args.order, args.seed,
                               args.samples_per_prompt)
    print(f"\nAggregate summary: {len(summary['categories'])} categories, "
          f"overall_avg={summary['overall_avg']}")


# -----------------------------------------------------------------
# Cross-config summary table
# -----------------------------------------------------------------
def _seeds_present():
    """Find {(model, order): [seeds...]} of finished t2i_compbench runs."""
    pat = re.compile(r"^maskgen_kl_(l|xl)_(random|prompt_sim|prompt_sim_rev)_seed(\d+)$")
    found = {}
    if not os.path.isdir(OUTPUT_BASE):
        return found
    for name in os.listdir(OUTPUT_BASE):
        m = pat.match(name)
        if not m:
            continue
        found.setdefault((m.group(1), m.group(2)), []).append(int(m.group(3)))
    for k in found:
        found[k].sort()
    return found


def _mean_std(values):
    vs = [v for v in values if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], 0.0
    m = sum(vs) / len(vs)
    var = sum((v - m) ** 2 for v in vs) / len(vs)
    return m, var ** 0.5


def run_summary():
    """Aggregate t2i_compbench_summary.json across seeds; report mean±std."""
    # results[model][order] = list of per-seed summary dicts
    results = {}
    seeds_map = _seeds_present()
    for (model, order), seeds in seeds_map.items():
        seed_summaries = []
        for seed in seeds:
            p = os.path.join(run_dir(model, order, seed), "t2i_compbench_summary.json")
            if os.path.exists(p):
                with open(p) as f:
                    seed_summaries.append(json.load(f))
        if seed_summaries:
            results.setdefault(model, {})[order] = seed_summaries

    if not results:
        print("No evaluation results found yet.")
        print(f"Expected: {OUTPUT_BASE}/maskgen_kl_<model>_<order>_seed<seed>/t2i_compbench_summary.json")
        return

    print("Seeds discovered per (model, order):")
    for model in ["l", "xl"]:
        for order in ["random", "prompt_sim", "prompt_sim_rev"]:
            seeds = sorted(s.get("seed", "?") for s in results.get(model, {}).get(order, []))
            if seeds:
                print(f"  {model:>2} / {order:<14}: {seeds}")
    print()

    models = [m for m in ["l", "xl"] if m in results]
    col_w = 14
    cell_w = 13  # "0.000±0.000"
    header = f"{'Category':<{col_w}}"
    for m in models:
        header += f" | {m.upper()+'-rand':>{cell_w}} {m.upper()+'-sim':>{cell_w}} {'Δ'+m.upper():>7}"
    sep = "-" * len(header)

    def _fmt_ms(mean, std):
        if mean is None:
            return f"{'--':>{cell_w}}"
        return f"{mean:>5.3f}±{std:<5.3f}".rjust(cell_w)

    def _per_cat_values(seeds_list, cat):
        return [(s.get("categories") or {}).get(cat, {}).get("score") for s in seeds_list]

    def _overall_values(seeds_list):
        return [s.get("overall_avg") for s in seeds_list]

    def _cell(rand_seeds, sim_seeds, cat=None):
        rvals = _per_cat_values(rand_seeds, cat) if cat else _overall_values(rand_seeds)
        svals = _per_cat_values(sim_seeds, cat) if cat else _overall_values(sim_seeds)
        rm, rs = _mean_std(rvals)
        sm, ss = _mean_std(svals)
        delta = (sm - rm) if (rm is not None and sm is not None) else None
        d_str = f"{delta:>+7.4f}" if delta is not None else f"{'--':>7}"
        return f"{_fmt_ms(rm, rs)} {_fmt_ms(sm, ss)} {d_str}"

    lines = [
        "=" * len(header),
        "T2I-CompBench++ Category Comparison (mean±std across seeds)",
        "=" * len(header),
        header,
        sep,
    ]
    for cat in CATEGORIES:
        row = f"{cat:<{col_w}}"
        for m in models:
            row += " | " + _cell(
                results[m].get("random", []),
                results[m].get("prompt_sim", []),
                cat=cat)
        lines.append(row)

    lines.append(sep)
    overall_row = f"{'Overall avg':<{col_w}}"
    for m in models:
        overall_row += " | " + _cell(
            results[m].get("random", []),
            results[m].get("prompt_sim", []),
            cat=None)
    lines.append(overall_row)
    lines.append("=" * len(header))

    body = "\n".join(lines)
    print(body)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    out_file = os.path.join(OUTPUT_BASE, "t2i_compbench_summary_tables.txt")
    with open(out_file, "w") as f:
        f.write(body + "\n")
    print(f"\nTable saved to {out_file}")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type=str, default="xl", choices=["l", "xl"])
    parser.add_argument("--order", type=str, default="random")
    parser.add_argument("--category", type=str, default="all",
                        help="'all' or one of: " + ",".join(CATEGORIES))
    parser.add_argument("--samples-per-prompt", type=int,
                        default=DEFAULT_SAMPLES_PER_PROMPT)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit to first N prompts per category (smoke test)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--aes-score", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-gpus", type=int, default=None)

    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation after generation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--summary", action="store_true",
                        help="Print comparison table across all completed runs")
    parser.add_argument("--aggregate", action="store_true",
                        help="Re-aggregate per-category score.json files into "
                             "t2i_compbench_summary.json for this {model, order}.")
    parser.add_argument("--force", action="store_true",
                        help="Run evaluation even if image count is incomplete")

    args = parser.parse_args()

    if not args.summary and not args.eval_only and not args.aggregate:
        from modeling.maskgen import MaskGen_KL
        if args.order not in MaskGen_KL.SUPPORTED_ORDER_TYPES:
            parser.error(
                f"--order must be one of {MaskGen_KL.SUPPORTED_ORDER_TYPES}")

    if args.summary:
        run_summary()
        return

    if args.aggregate:
        s = aggregate_config(args.model, args.order, args.seed,
                             args.samples_per_prompt)
        print(f"Aggregated {len(s['categories'])} categories for "
              f"maskgen_kl_{args.model}/{args.order}/seed{args.seed}: "
              f"overall_avg={s['overall_avg']}")
        return

    if not args.eval_only:
        run_generation(args)

    if args.evaluate or args.eval_only:
        run_evaluation(args)


if __name__ == "__main__":
    main()
