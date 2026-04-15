"""Evaluate MaskGen-KL on GenEval benchmark.

Generates images for all 553 GenEval prompts using MaskGen-KL (L/XL),
saves them in the GenEval-expected directory format, then runs the
GenEval object-detection-based evaluation pipeline.

Usage:
    # Step 1: Generate images (run for each model/order combo)
    python eval_geneval.py --model xl --order random
    python eval_geneval.py --model xl --order prompt_sim
    python eval_geneval.py --model l  --order random
    python eval_geneval.py --model l  --order prompt_sim

    # Step 2: Evaluate (requires mmdet + Mask2Former weights)
    python eval_geneval.py --model xl --order random --evaluate
    python eval_geneval.py --model xl --order prompt_sim --evaluate

    # Step 3: Summarize all results
    python eval_geneval.py --summary

    # Generate + evaluate in one shot
    python eval_geneval.py --model xl --order random --evaluate

    # Multi-GPU generation
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_geneval.py --model xl --order random
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"

import argparse
import json
import math
import time
import subprocess
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_KL


# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
GENEVAL_ROOT = "/home/hliu256/geneval"
GENEVAL_METADATA = os.path.join(GENEVAL_ROOT, "prompts/evaluation_metadata.jsonl")
OUTPUT_BASE = "/data3/haoyuliu/geneval_eval"
MASK2FORMER_DIR = os.path.join(OUTPUT_BASE, "mask2former_weights")


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_text_guidance(prompts, clip_tokenizer, clip_encoder, device):
    """Compute text guidance embeddings for de-tokenization (batched)."""
    text_guidance_ids = clip_tokenizer(prompts).to(device)
    cast_dtype = clip_encoder.transformer.get_cast_dtype()
    text_guidance = clip_encoder.token_embedding(text_guidance_ids).to(cast_dtype)
    text_guidance = text_guidance + clip_encoder.positional_embedding.to(cast_dtype)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.transformer(text_guidance, attn_mask=clip_encoder.attn_mask)
    text_guidance = text_guidance.permute(1, 0, 2)
    text_guidance = clip_encoder.ln_final(text_guidance)
    return text_guidance


def load_geneval_metadata():
    """Load all GenEval prompts and metadata."""
    metadatas = []
    with open(GENEVAL_METADATA) as f:
        for line in f:
            metadatas.append(json.loads(line.strip()))
    return metadatas


def run_dir_name(model, order):
    return os.path.join(OUTPUT_BASE, f"maskgen_kl_{model}_{order}")


def worker_generate(rank, num_gpus, metadatas, out_dir, args):
    """Each GPU generates its shard of prompts in GenEval format.

    Strategy: run n_samples rounds. In each round, batch all prompts together
    (like FID eval) so we can use large batch sizes for GPU utilization.
    """
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Shard across GPUs
    shard_size = math.ceil(len(metadatas) / num_gpus)
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, len(metadatas))
    shard_metas = list(enumerate(metadatas[shard_start:shard_end], start=shard_start))

    if not shard_metas:
        return

    # Pre-create output dirs and save metadata
    for global_idx, meta in shard_metas:
        prompt_dir = os.path.join(out_dir, f"{global_idx:05d}")
        sample_dir = os.path.join(prompt_dir, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "metadata.jsonl"), "w") as f:
            json.dump(meta, f)

    # Determine which (prompt_idx, sample_idx) pairs still need generation
    todo = []  # list of (global_idx, meta, sample_idx)
    for global_idx, meta in shard_metas:
        sample_dir = os.path.join(out_dir, f"{global_idx:05d}", "samples")
        for s in range(args.n_samples):
            img_path = os.path.join(sample_dir, f"{s:05d}.png")
            if not os.path.exists(img_path):
                todo.append((global_idx, meta, s))

    if not todo:
        print(f"[GPU {rank}] All images already exist, skipping")
        return

    print(f"[GPU {rank}] Loading models...")

    # Load models
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

    total_imgs = len(todo)
    print(f"[GPU {rank}] Generating {total_imgs} images "
          f"(shard {shard_start}-{shard_end}, "
          f"{len(shard_metas) * args.n_samples - total_imgs} skipped)")

    batch_size = args.batch_size
    num_batches = math.ceil(total_imgs / batch_size)
    start_time = time.time()
    generated_count = 0

    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", position=rank,
                leave=True, disable=(rank != 0))

    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_imgs)
        batch_items = todo[batch_start:batch_end]
        bsz = len(batch_items)
        prompts = [meta["prompt"] for _, meta, _ in batch_items]

        # Compute text guidance for decoding
        text_guidance = compute_text_guidance(
            prompts, clip_tokenizer, clip_encoder, device)

        # Seed: deterministic per batch position within the shard
        set_all_seeds(args.seed + shard_start * 1000 + batch_idx)

        with torch.no_grad():
            tokens = maskgen.sample_tokens(
                bsz=bsz,
                clip_tokenizer=clip_tokenizer,
                clip_encoder=clip_encoder,
                num_iter=args.num_iter,
                cfg=args.cfg,
                captions=prompts,
                aes_scores=args.aes_score,
                order_type=args.order,
            )

            images = tatitok.decode_tokens(tokens, text_guidance)
            images = torch.clamp(images, 0.0, 1.0)
            images = (images * 255.0).permute(0, 2, 3, 1).to(
                "cpu", dtype=torch.uint8).numpy()

        # Save each image to its prompt's sample dir
        for i, (global_idx, _, sample_idx) in enumerate(batch_items):
            sample_dir = os.path.join(out_dir, f"{global_idx:05d}", "samples")
            img = Image.fromarray(images[i])
            img.save(os.path.join(sample_dir, f"{sample_idx:05d}.png"))

        generated_count += bsz

        if rank == 0 and (batch_idx + 1) % 5 == 0:
            elapsed = time.time() - start_time
            rate = generated_count / elapsed
            remaining = (total_imgs - generated_count) / max(rate, 1e-6)
            pbar.set_postfix(
                rate=f"{rate:.1f} img/s", eta=f"{remaining / 60:.1f}min")

    elapsed = time.time() - start_time
    print(f"[GPU {rank}] Done: {generated_count} images "
          f"in {elapsed / 60:.1f}min ({generated_count / max(elapsed, 1):.1f} img/s)")


def run_generation(args):
    """Generate images for all GenEval prompts."""
    metadatas = load_geneval_metadata()
    out_dir = run_dir_name(args.model, args.order)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Model: MaskGen-KL-{args.model.upper()}")
    print(f"Order: {args.order}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Total prompts: {len(metadatas)}")
    print(f"Output: {out_dir}")

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}")

    if num_gpus == 1:
        worker_generate(0, 1, metadatas, out_dir, args)
    else:
        mp.spawn(
            worker_generate,
            args=(num_gpus, metadatas, out_dir, args),
            nprocs=num_gpus,
            join=True,
        )

    # Verify
    total_expected = len(metadatas) * args.n_samples
    total_generated = sum(
        len([f for f in os.listdir(os.path.join(out_dir, d, "samples"))
             if f.endswith(".png")])
        for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d, "samples"))
    )
    print(f"\nGeneration complete: {total_generated}/{total_expected} images")


def run_evaluation(args):
    """Run GenEval evaluation on generated images."""
    out_dir = run_dir_name(args.model, args.order)
    results_file = os.path.join(out_dir, "results.jsonl")

    # Check Mask2Former weights
    m2f_weight = os.path.join(
        MASK2FORMER_DIR,
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"
    )
    if not os.path.exists(m2f_weight):
        print(f"Mask2Former weights not found at {MASK2FORMER_DIR}")
        print(f"Downloading...")
        os.makedirs(MASK2FORMER_DIR, exist_ok=True)
        subprocess.run([
            "bash", os.path.join(GENEVAL_ROOT, "evaluation/download_models.sh"),
            MASK2FORMER_DIR
        ], check=True)

    evaluate_script = os.path.join(GENEVAL_ROOT, "evaluation/evaluate_images.py")
    summary_script = os.path.join(GENEVAL_ROOT, "evaluation/summary_scores.py")

    # Step 1: Run object detection evaluation
    print(f"\n{'='*60}")
    print(f"Evaluating: MaskGen-KL-{args.model.upper()} / {args.order}")
    print(f"Images dir: {out_dir}")
    print(f"Results: {results_file}")
    print(f"{'='*60}\n")

    cmd_eval = [
        sys.executable, evaluate_script,
        out_dir,
        "--outfile", results_file,
        "--model-path", MASK2FORMER_DIR,
    ]
    print(f"Running: {' '.join(cmd_eval)}")
    subprocess.run(cmd_eval, check=True)

    # Step 2: Summarize scores
    print(f"\n{'='*60}")
    print(f"Summary: MaskGen-KL-{args.model.upper()} / {args.order}")
    print(f"{'='*60}")
    cmd_summary = [sys.executable, summary_script, results_file]
    subprocess.run(cmd_summary, check=True)

    # Step 3: Parse and save structured results
    _save_parsed_results(results_file, args.model, args.order)


def _save_parsed_results(results_file, model, order):
    """Parse results.jsonl and save a structured summary (prompt-level).

    Prompt-level: a prompt is correct if ANY of its images is correct.
    This matches the paper's reporting convention.
    """
    import pandas as pd

    df = pd.read_json(results_file, orient="records", lines=True)

    # Prompt-level accuracy: correct if any image for that prompt is correct
    prompt_df = df.groupby('metadata').agg(
        correct=('correct', 'any'),
        tag=('tag', 'first'),
    ).reset_index()

    task_scores = {}
    for tag, task_df in prompt_df.groupby('tag', sort=False):
        task_scores[tag] = task_df['correct'].mean()

    overall = np.mean(list(task_scores.values()))

    summary = {
        "model": f"maskgen_kl_{model}",
        "order": order,
        "overall": round(overall, 4),
        "tasks": {k: round(v, 4) for k, v in task_scores.items()},
        "total_images": len(df),
        "total_prompts": len(prompt_df),
        "metric": "prompt-level (correct if any sample is correct)",
    }

    summary_file = os.path.join(
        run_dir_name(model, order), "geneval_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStructured results saved to {summary_file}")


def run_summary():
    """Print a comparison table of all completed evaluations."""
    print(f"\n{'='*80}")
    print(f"GenEval Results Comparison")
    print(f"{'='*80}")

    # Paper reference values (README MaskGen Model Zoo, prompt-level)
    paper_results = {
        ("l", "random"): {
            "single_object": 0.99, "two_object": 0.57, "counting": 0.36,
            "colors": 0.80, "position": 0.11, "color_attr": 0.29,
            "overall": 0.52,
        },
        ("xl", "random"): {"overall": 0.55},
    }

    # Collect all results
    all_results = []
    for model in ["l", "xl"]:
        for order in ["random", "prompt_sim", "prompt_sim_rev"]:
            summary_file = os.path.join(
                run_dir_name(model, order), "geneval_summary.json")
            if os.path.exists(summary_file):
                with open(summary_file) as f:
                    result = json.load(f)
                all_results.append((model, order, result))

    if not all_results:
        print("No evaluation results found yet.")
        print(f"Expected location: {OUTPUT_BASE}/maskgen_kl_<model>_<order>/geneval_summary.json")
        return

    # Print table
    task_order = ["single_object", "two_object", "counting",
                  "colors", "position", "color_attr"]
    task_short = ["S.Obj", "T.Obj", "Count", "Colors", "Pos.", "C.Attr"]
    header = f"{'Model':<28} | " + " | ".join(f"{t:>6}" for t in task_short) + " | Overall"
    print(header)
    print("-" * len(header))

    # Print paper baselines first
    for (model, order), scores in paper_results.items():
        label = f"Paper MaskGen-{model.upper()} (KL)"
        row = f"{label:<28} | "
        row += " | ".join(
            f"{scores[t]:>6.2f}" if t in scores else f"{'--':>6}"
            for t in task_order)
        row += f" | {scores['overall']:>6.2f}"
        print(row)

    print("-" * len(header))

    # Print our results
    for model, order, result in all_results:
        label = f"MaskGen-{model.upper()} ({order})"
        row = f"{label:<28} | "
        tasks = result.get("tasks", {})
        row += " | ".join(f"{tasks.get(t, 0):>6.2f}" for t in task_order)
        row += f" | {result['overall']:>6.2f}"
        print(row)

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type=str, default="xl",
                        choices=["l", "xl"],
                        help="MaskGen-KL model size")
    parser.add_argument("--order", type=str, default="random",
                        choices=MaskGen_KL.SUPPORTED_ORDER_TYPES,
                        help="Token generation order strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--aes-score", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per GPU (multiple prompts batched together)")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of images per prompt (GenEval default: 4)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (default: all visible)")

    parser.add_argument("--evaluate", action="store_true",
                        help="Run GenEval evaluation after generation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--summary", action="store_true",
                        help="Print comparison table of all results")

    args = parser.parse_args()

    if args.summary:
        run_summary()
        return

    if not args.eval_only:
        run_generation(args)

    if args.evaluate or args.eval_only:
        run_evaluation(args)


if __name__ == "__main__":
    main()
