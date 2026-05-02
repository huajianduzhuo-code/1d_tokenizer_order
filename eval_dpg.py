"""Evaluate MaskGen-KL on DPG-Bench.

Generates images for all 1065 DPG-Bench prompts using MaskGen-KL (L/XL),
assembles 4 samples per prompt into a 2x2 grid as DPG-Bench expects,
then runs the official mPLUG-VQA evaluation pipeline.

Usage (generation in `maskgen` env):
    python eval_dpg.py --model xl --order random
    python eval_dpg.py --model xl --order prompt_sim

    # Smoke test with 10 prompts
    python eval_dpg.py --model xl --order prompt_sim --num-samples 10

    # Multi-GPU generation
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_dpg.py --model xl --order prompt_sim

Usage (evaluation in `dpg` env — set up via setup_dpg_env.sh):
    conda activate /data3/haoyuliu/conda_envs/dpg
    python eval_dpg.py --model xl --order prompt_sim --eval-only

Summary (maskgen env):
    python eval_dpg.py --summary
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"
os.environ.setdefault("MODELSCOPE_CACHE", "/data3/haoyuliu/modelscope_cache")

import argparse
import glob
import json
import math
import re
import subprocess
import sys
import time

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
DPG_ROOT = "/home/hliu256/ELLA"
DPG_PROMPTS_DIR = os.path.join(DPG_ROOT, "dpg_bench/prompts")
DPG_EVAL_SCRIPT = os.path.join(DPG_ROOT, "dpg_bench/dist_eval.sh")
DPG_CSV = os.path.join(DPG_ROOT, "dpg_bench/dpg_bench.csv")
OUTPUT_BASE = "/data3/haoyuliu/dpg_bench_eval"

TILE_RESOLUTION = 512          # single-sample resolution (MaskGen default)
GRID_SIZE = 2                  # 2x2 grid => pic_num=4
N_SAMPLES_PER_PROMPT = GRID_SIZE * GRID_SIZE


# -----------------------------------------------------------------
# Lazy imports (only needed for generation; keep import light for eval-only)
# -----------------------------------------------------------------
def _import_generation_stack():
    import torch
    import torch.multiprocessing as mp
    import open_clip
    from modeling.tatitok import TATiTok
    from modeling.maskgen import MaskGen_KL
    return torch, mp, open_clip, TATiTok, MaskGen_KL


# -----------------------------------------------------------------
# Shared helpers (copied from eval_geneval.py to stay self-contained)
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
    """Compute text guidance embeddings for de-tokenization (batched).

    Same formula as eval_geneval.py.compute_text_guidance.
    """
    text_guidance_ids = clip_tokenizer(prompts).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_guidance = clip_encoder.token_embedding(text_guidance_ids).to(cast_dtype)
    text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.ln_final(text_guidance)
    return text_guidance


def load_dpg_prompts():
    """Return a sorted list of (key, prompt_text) tuples."""
    prompt_files = sorted(glob.glob(os.path.join(DPG_PROMPTS_DIR, "*.txt")))
    prompts = []
    for pf in prompt_files:
        key = os.path.splitext(os.path.basename(pf))[0]
        with open(pf) as f:
            text = f.read().strip()
        prompts.append((key, text))
    return prompts


def run_dir_name(model, order, seed):
    return os.path.join(OUTPUT_BASE, f"maskgen_kl_{model}_{order}_seed{seed}")


def images_dir_for(model, order, seed):
    return os.path.join(run_dir_name(model, order, seed), "images")


# -----------------------------------------------------------------
# Generation
# -----------------------------------------------------------------
def worker_generate(rank, num_gpus, prompts, out_dir, args):
    """Each GPU handles a shard of prompts.

    For every prompt we generate N_SAMPLES_PER_PROMPT=4 tiles and stitch
    them into a single 2x2 grid PNG named `{key}.png`. Resume semantics
    operate at the prompt (file) level.
    """
    torch, _, open_clip, TATiTok, MaskGen_KL = _import_generation_stack()

    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    shard_size = math.ceil(len(prompts) / num_gpus)
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, len(prompts))
    shard_prompts = prompts[shard_start:shard_end]

    if not shard_prompts:
        return

    # Filter out prompts whose grid already exists (resume)
    todo = [(key, text) for key, text in shard_prompts
            if not os.path.exists(os.path.join(out_dir, f"{key}.png"))]
    skipped = len(shard_prompts) - len(todo)
    if not todo:
        print(f"[GPU {rank}] All {len(shard_prompts)} grids exist, skipping")
        return

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

    # Expand (prompt) -> N_SAMPLES tile tasks so we can pack varied prompts
    # into a single batch (more GPU utilization than looping per prompt).
    tile_tasks = []   # list of (prompt_idx_in_todo, sample_idx)
    for i in range(len(todo)):
        for s in range(N_SAMPLES_PER_PROMPT):
            tile_tasks.append((i, s))

    total_tiles = len(tile_tasks)
    print(f"[GPU {rank}] Generating {total_tiles} tiles for {len(todo)} prompts "
          f"(shard {shard_start}-{shard_end}, {skipped} prompts already done)")

    # Buffer tiles per prompt; flush to disk as a 2x2 grid once all 4 arrive.
    tile_buffer = {i: [None] * N_SAMPLES_PER_PROMPT for i in range(len(todo))}

    batch_size = args.batch_size
    num_batches = math.ceil(total_tiles / batch_size)
    start_time = time.time()
    generated_count = 0

    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", position=rank,
                leave=True, disable=(rank != 0))

    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_tiles)
        batch_items = tile_tasks[batch_start:batch_end]
        bsz = len(batch_items)
        prompts_batch = [todo[i][1] for i, _ in batch_items]

        text_guidance = compute_text_guidance(
            prompts_batch, clip_tokenizer, clip_encoder, device)

        # Seed per-sample: varies across sample_idx AND batch position so
        # (a) the 4 samples for a prompt differ, and (b) the run is deterministic
        # for a given (args.seed, shard, batch_idx).
        set_all_seeds(args.seed + shard_start * 1_000_003 + batch_idx)

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

        for (prompt_idx, sample_idx), arr in zip(batch_items, images):
            tile_buffer[prompt_idx][sample_idx] = arr

            # Flush when all 4 tiles are ready
            if all(t is not None for t in tile_buffer[prompt_idx]):
                _save_grid(out_dir, todo[prompt_idx][0], tile_buffer[prompt_idx])
                tile_buffer[prompt_idx] = None  # free memory

        generated_count += bsz
        if rank == 0 and (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rate = generated_count / max(elapsed, 1e-6)
            remaining = (total_tiles - generated_count) / max(rate, 1e-6)
            pbar.set_postfix(
                rate=f"{rate:.1f} tiles/s", eta=f"{remaining / 60:.1f}min")

    elapsed = time.time() - start_time
    print(f"[GPU {rank}] Done: {generated_count} tiles "
          f"in {elapsed / 60:.1f}min ({generated_count / max(elapsed, 1):.1f} tiles/s)")


def _save_grid(out_dir, key, tiles):
    """tiles: list of 4 HxWx3 uint8 arrays; stitch into 2x2 and save."""
    R = TILE_RESOLUTION
    canvas = Image.new("RGB", (R * GRID_SIZE, R * GRID_SIZE))
    for idx, arr in enumerate(tiles):
        img = Image.fromarray(arr)
        if img.size != (R, R):
            img = img.resize((R, R), Image.BICUBIC)
        row, col = divmod(idx, GRID_SIZE)
        canvas.paste(img, (col * R, row * R))
    canvas.save(os.path.join(out_dir, f"{key}.png"))


def run_generation(args):
    torch, mp, *_ = _import_generation_stack()

    prompts = load_dpg_prompts()
    if args.num_samples:
        prompts = prompts[:args.num_samples]

    out_dir = images_dir_for(args.model, args.order, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model: MaskGen-KL-{args.model.upper()}")
    print(f"Order: {args.order}")
    print(f"Seed:  {args.seed}")
    print(f"Total prompts: {len(prompts)}  (samples/prompt: {N_SAMPLES_PER_PROMPT})")
    print(f"Output: {out_dir}")

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}")

    if num_gpus == 1:
        worker_generate(0, 1, prompts, out_dir, args)
    else:
        mp.spawn(
            worker_generate,
            args=(num_gpus, prompts, out_dir, args),
            nprocs=num_gpus,
            join=True,
        )

    generated = sum(1 for p in os.listdir(out_dir) if p.endswith(".png"))
    print(f"\nGeneration complete: {generated}/{len(prompts)} grids")


# -----------------------------------------------------------------
# Evaluation (drives dist_eval.sh via subprocess + parses the output txt)
# -----------------------------------------------------------------
def run_evaluation(args):
    out_dir = run_dir_name(args.model, args.order, args.seed)
    images_dir = images_dir_for(args.model, args.order, args.seed)

    if not os.path.isdir(images_dir):
        print(f"Images dir not found: {images_dir}")
        print(f"Run generation first: python eval_dpg.py --model {args.model} "
              f"--order {args.order}")
        sys.exit(1)

    n_grids = sum(1 for p in os.listdir(images_dir) if p.endswith(".png"))
    expected = args.num_samples or 1065
    if n_grids < expected:
        print(f"Warning: only {n_grids} grids in {images_dir} (expected {expected}).")
        if not args.force:
            print("  Pass --force to evaluate anyway.")
            sys.exit(1)

    # Choose number of processes for accelerate — default to visible GPU count
    import torch  # lightweight, for device count only
    num_procs = args.num_gpus or torch.cuda.device_count() or 1

    env = os.environ.copy()
    env["PROCESSES"] = str(num_procs)
    env["PIC_NUM"] = str(N_SAMPLES_PER_PROMPT)
    # unique port per run so parallel evaluations don't collide
    env["PORT"] = str(29500 + (hash((args.model, args.order)) % 1000))
    # Prepend the current Python's bin dir so dist_eval.sh finds `accelerate`
    # even when this script was invoked by absolute path (no `conda activate`).
    py_bin = os.path.dirname(sys.executable)
    env["PATH"] = py_bin + os.pathsep + env.get("PATH", "")

    print(f"\n{'='*60}")
    print(f"Evaluating: MaskGen-KL-{args.model.upper()} / {args.order}")
    print(f"Images dir: {images_dir}")
    print(f"PROCESSES={num_procs}  PIC_NUM={N_SAMPLES_PER_PROMPT}  "
          f"RESOLUTION={TILE_RESOLUTION}")
    print(f"{'='*60}\n")

    cmd = ["bash", DPG_EVAL_SCRIPT, images_dir, str(TILE_RESOLUTION)]
    print(f"Running (cwd={DPG_ROOT}): {' '.join(cmd)}")
    subprocess.run(cmd, cwd=DPG_ROOT, env=env, check=True)

    _parse_and_save_summary(images_dir, out_dir, args.model, args.order, args.seed)


def _latest_results_txt(images_dir):
    cands = glob.glob(os.path.join(images_dir, "dpg-bench_*_results.txt"))
    cands = [c for c in cands if not c.endswith("_detail.txt")]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _parse_and_save_summary(images_dir, out_dir, model, order, seed):
    """Parse the evaluator's results.txt into dpg_summary.json.

    Expected tail of the file:
        L1 category scores:
            entity: 84.12
            relation: 75.3
            ...
        L2 category scores:
            entity - whole (space): 91.2
            ...
        Image path: ...
        Save results to: ...
        DPG-Bench score: 82.5
    """
    results_txt = _latest_results_txt(images_dir)
    if results_txt is None:
        print(f"Could not find dpg-bench_*_results.txt in {images_dir}")
        return

    with open(results_txt) as f:
        text = f.read()

    overall_m = re.search(r"DPG-Bench score:\s*([\d.]+)", text)
    overall = float(overall_m.group(1)) if overall_m else None

    def _parse_section(header):
        m = re.search(
            rf"{re.escape(header)}:\s*\n((?:\s+[^\n]+\n)+)", text)
        if not m:
            return {}
        out = {}
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, val = line.rsplit(":", 1)
            try:
                out[name.strip()] = float(val.strip())
            except ValueError:
                continue
        return out

    l1 = _parse_section("L1 category scores")
    l2 = _parse_section("L2 category scores")

    summary = {
        "model": f"maskgen_kl_{model}",
        "order": order,
        "seed": seed,
        "overall": overall,
        "l1_categories": l1,
        "l2_categories": l2,
        "results_txt": os.path.relpath(results_txt, out_dir),
    }
    summary_path = os.path.join(out_dir, "dpg_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDPG-Bench overall: {overall}")
    if l1:
        print("L1 categories:")
        for k, v in l1.items():
            print(f"  {k:<20} {v:.2f}")
    print(f"Summary saved to {summary_path}")


# -----------------------------------------------------------------
# Comparison tables (L1 + L2, with question counts and random/prompt_sim deltas)
# -----------------------------------------------------------------
def _load_category_counts():
    """Count DPG-Bench questions and prompts per L1 / L2 category.

    Mirrors the category derivation in compute_dpg_bench.py:
        L2 = tuple.split('(')[0].strip()
        L1 = L2.split('-')[0].strip()

    Returns (l1_q, l2_q, l1_p, l2_p, total_q, total_p) where _q are question
    counts (Counter) and _p are prompt-coverage counts (dict[str, int]).
    """
    import csv
    from collections import Counter, defaultdict

    l1_q, l2_q = Counter(), Counter()
    l1_p = defaultdict(set)
    l2_p = defaultdict(set)
    all_prompts = set()
    total_q = 0
    with open(DPG_CSV) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # compute_dpg_bench.py skips the first data row (i == 0)
            if i == 0:
                continue
            tpl = (row.get("tuple") or "").strip()
            if not tpl:
                continue
            l2 = tpl.split("(")[0].strip()
            l1 = l2.split("-")[0].strip()
            item_id = row.get("item_id", "")
            l2_q[l2] += 1
            l1_q[l1] += 1
            l2_p[l2].add(item_id)
            l1_p[l1].add(item_id)
            all_prompts.add(item_id)
            total_q += 1
    l1_p = {k: len(v) for k, v in l1_p.items()}
    l2_p = {k: len(v) for k, v in l2_p.items()}
    return l1_q, l2_q, l1_p, l2_p, total_q, len(all_prompts)


def _mean_std(values):
    """Return (mean, std) over a list of numbers, ignoring None entries.
    std is population std (ddof=0). Returns (None, None) if no valid values."""
    vs = [v for v in values if v is not None]
    if not vs:
        return None, None
    if len(vs) == 1:
        return vs[0], 0.0
    m = sum(vs) / len(vs)
    var = sum((v - m) ** 2 for v in vs) / len(vs)
    return m, var ** 0.5


def _format_category_table(level, results, q_counts, p_counts,
                           total_q, total_p, key):
    """Render one table (L1 or L2). `key` is 'l1_categories' or 'l2_categories'.

    `results[model][order]` is a list of per-seed summary dicts. Cells are
    rendered as `mean±std` over seeds; the Δ column is mean(prompt_sim) -
    mean(random)."""
    models = [m for m in ["l", "xl"] if m in results]
    if not models:
        return ""

    # Categories sorted by question count (descending) for easy readability.
    all_cats = set()
    for m in models:
        for seeds_list in results[m].values():
            for s in seeds_list:
                all_cats.update((s.get(key) or {}).keys())
    cats = sorted(all_cats, key=lambda c: (-q_counts.get(c, 0), c))

    def _collect(seeds_list, cat=None):
        """Pull one scalar per seed (cat=None means 'overall')."""
        if cat is None:
            return [s.get("overall") for s in seeds_list]
        return [(s.get(key) or {}).get(cat) for s in seeds_list]

    def _fmt_ms(mean, std):
        if mean is None:
            return f"{'--':>13}"
        return f"{mean:>6.2f}±{std:<5.2f}"

    def _fmt_cell(rand_seeds, sim_seeds, cat=None):
        rm, rs = _mean_std(_collect(rand_seeds, cat))
        sm, ss = _mean_std(_collect(sim_seeds, cat))
        delta = (sm - rm) if (rm is not None and sm is not None) else None
        d_str = f"{delta:>+7.2f}" if delta is not None else f"{'--':>7}"
        return f"{_fmt_ms(rm, rs)} {_fmt_ms(sm, ss)} {d_str}"

    # Header — wider cells now to fit "mean±std" (13 chars each)
    col_w = 28
    header = f"{'Category':<{col_w}} | {'N_q':>5} | {'N_p':>5}"
    for m in models:
        header += f" | {m.upper()+'-rand':>13} {m.upper()+'-sim':>13} {'Δ'+m.upper():>7}"
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        f"DPG-Bench {level} Category Comparison  "
        f"(total: {total_q} questions, {total_p} prompts)",
        "=" * len(header),
        header,
        sep,
    ]
    for cat in cats:
        nq = q_counts.get(cat, 0)
        np_ = p_counts.get(cat, 0)
        row = f"{cat:<{col_w}} | {nq:>5} | {np_:>5}"
        for m in models:
            row += " | " + _fmt_cell(
                results[m].get("random", []),
                results[m].get("prompt_sim", []),
                cat=cat)
        lines.append(row)

    # Overall row
    lines.append(sep)
    overall = f"{'Overall':<{col_w}} | {total_q:>5} | {total_p:>5}"
    for m in models:
        overall += " | " + _fmt_cell(
            results[m].get("random", []),
            results[m].get("prompt_sim", []),
            cat=None)
    lines.append(overall)
    lines.append("=" * len(header))

    return "\n".join(lines)


def _seeds_present():
    """Inspect on-disk dirs to find {(model, order): [seeds...]} present."""
    pat = re.compile(r"^maskgen_kl_(l|xl)_(random|prompt_sim|prompt_sim_rev)_seed(\d+)$")
    found = {}
    if not os.path.isdir(OUTPUT_BASE):
        return found
    for name in os.listdir(OUTPUT_BASE):
        m = pat.match(name)
        if not m:
            continue
        model, order, seed = m.group(1), m.group(2), int(m.group(3))
        found.setdefault((model, order), []).append(seed)
    for k in found:
        found[k].sort()
    return found


def run_summary():
    # results[model][order] = list of per-seed summary dicts (sorted by seed)
    results = {}
    seeds_map = _seeds_present()
    for model in ["l", "xl"]:
        model_res = {}
        for order in ["random", "prompt_sim", "prompt_sim_rev"]:
            seeds_list = []
            for seed in seeds_map.get((model, order), []):
                summary_file = os.path.join(
                    run_dir_name(model, order, seed), "dpg_summary.json")
                if os.path.exists(summary_file):
                    with open(summary_file) as f:
                        seeds_list.append(json.load(f))
            if seeds_list:
                model_res[order] = seeds_list
        if model_res:
            results[model] = model_res

    if not results:
        print("No evaluation results found yet.")
        print(f"Expected: {OUTPUT_BASE}/maskgen_kl_<model>_<order>_seed<seed>/dpg_summary.json")
        return

    # Print which seeds we found
    print("Seeds discovered per (model, order):")
    for model in ["l", "xl"]:
        for order in ["random", "prompt_sim", "prompt_sim_rev"]:
            seeds = sorted(s.get("seed", "?") for s in results.get(model, {}).get(order, []))
            if seeds:
                print(f"  {model:>2} / {order:<14}: {seeds}")
    print()

    l1_q, l2_q, l1_p, l2_p, total_q, total_p = _load_category_counts()

    l1_table = _format_category_table(
        "L1", results, l1_q, l1_p, total_q, total_p, "l1_categories")
    l2_table = _format_category_table(
        "L2", results, l2_q, l2_p, total_q, total_p, "l2_categories")

    body = l1_table + "\n\n" + l2_table
    print(body)

    os.makedirs(OUTPUT_BASE, exist_ok=True)
    out_file = os.path.join(OUTPUT_BASE, "dpg_summary_tables.txt")
    with open(out_file, "w") as f:
        f.write(body + "\n")
    print(f"\nTables saved to {out_file}")


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type=str, default="xl",
                        choices=["l", "xl"])
    parser.add_argument("--order", type=str, default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--aes-score", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-GPU batch size of tiles (4 tiles / prompt)")
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit to first N prompts (smoke test)")

    parser.add_argument("--evaluate", action="store_true",
                        help="Run DPG evaluation after generation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, only run DPG evaluation")
    parser.add_argument("--summary", action="store_true",
                        help="Print comparison table of all completed runs")
    parser.add_argument("--force", action="store_true",
                        help="Run evaluation even if grids are incomplete")

    args = parser.parse_args()

    # Validate --order against the generator's supported list, but only when
    # actually generating (eval-only / summary do not need the generator stack).
    if not args.summary and not args.eval_only:
        from modeling.maskgen import MaskGen_KL
        if args.order not in MaskGen_KL.SUPPORTED_ORDER_TYPES:
            parser.error(
                f"--order must be one of {MaskGen_KL.SUPPORTED_ORDER_TYPES}")

    if args.summary:
        run_summary()
        return

    if not args.eval_only:
        run_generation(args)

    if args.evaluate or args.eval_only:
        run_evaluation(args)


if __name__ == "__main__":
    main()
