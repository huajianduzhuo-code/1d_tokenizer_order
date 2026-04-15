"""Evaluate MaskGen-KL generation quality on MJHQ-30K via FID.

Generates images for all 30K prompts in MJHQ-30K using all available GPUs,
then computes FID between generated and real images using clean-fid.

Usage:
    # Generate with default random order, all GPUs (reproduces paper baseline)
    python eval_mjhq_fid.py --order random

    # Generate with prompt_sim order
    python eval_mjhq_fid.py --order prompt_sim

    # Use a subset for quick testing
    python eval_mjhq_fid.py --order random --num-samples 1000

    # Specify GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_mjhq_fid.py --order random

    # Only compute FID from already-generated images
    python eval_mjhq_fid.py --fid-only --gen-dir /path/to/generated
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"

import argparse
import json
import math
import time
import numpy as np
import torch
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip
from cleanfid import fid

from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_KL


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


def load_mjhq_prompts(meta_path, num_samples=None):
    """Load prompts and image keys from MJHQ-30K metadata.

    Returns:
        list of (image_key, category, prompt) tuples, sorted by key for
        reproducibility.
    """
    with open(meta_path) as f:
        meta = json.load(f)

    entries = sorted(
        [(k, v["category"], v["prompt"]) for k, v in meta.items()],
        key=lambda x: x[0],
    )
    if num_samples is not None and num_samples < len(entries):
        entries = entries[:num_samples]
    return entries


def worker_generate(rank, num_gpus, entries, gen_dir, args):
    """Worker function: each GPU generates its shard of entries."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Split entries across GPUs
    shard_size = math.ceil(len(entries) / num_gpus)
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, len(entries))
    shard_entries = entries[shard_start:shard_end]

    if not shard_entries:
        return

    gen_dir = Path(gen_dir)

    # Check which images already exist (for resuming)
    existing = set()
    for f_path in gen_dir.iterdir():
        if f_path.suffix == ".png":
            existing.add(f_path.stem)

    # Filter out already-generated entries
    todo_entries = [e for e in shard_entries if e[0] not in existing]
    if not todo_entries:
        print(f"[GPU {rank}] All {len(shard_entries)} images already exist, skipping")
        return

    print(f"[GPU {rank}] Loading models...")

    # Load models on this GPU
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

    print(f"[GPU {rank}] Generating {len(todo_entries)} images "
          f"(shard {shard_start}-{shard_end}, {len(shard_entries) - len(todo_entries)} skipped)")

    batch_size = args.batch_size
    num_batches = math.ceil(len(todo_entries) / batch_size)
    start_time = time.time()
    generated_count = 0

    pbar = tqdm(range(num_batches), desc=f"GPU {rank}", position=rank,
                leave=True, disable=(rank != 0 and num_batches < 10))

    for batch_idx in pbar:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(todo_entries))
        batch_entries = todo_entries[batch_start:batch_end]
        prompts = [e[2] for e in batch_entries]
        bsz = len(prompts)

        # Compute text guidance for decoding
        text_guidance = compute_text_guidance(prompts, clip_tokenizer, clip_encoder, device)

        # Set seed: use global batch index for reproducibility
        global_batch_idx = (shard_start + batch_start) // batch_size
        set_all_seeds(args.seed + global_batch_idx)

        # Generate
        with torch.no_grad():
            tokens = maskgen.sample_tokens(
                bsz=bsz, clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder,
                num_iter=args.num_iter, cfg=args.cfg, captions=prompts,
                aes_scores=args.aes_score, order_type=args.order,
            )

            # Decode to images
            images = tatitok.decode_tokens(tokens, text_guidance)
            images = torch.clamp(images, 0.0, 1.0)
            images = (images * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save each image
        for i, (key, _, _) in enumerate(batch_entries):
            img = Image.fromarray(images[i])
            img.save(gen_dir / f"{key}.png")

        generated_count += bsz

        if rank == 0 and (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            rate = generated_count / elapsed
            remaining_this_gpu = (len(todo_entries) - generated_count) / max(rate, 1e-6)
            pbar.set_postfix(rate=f"{rate:.1f} img/s", eta=f"{remaining_this_gpu/60:.0f}min")

    elapsed = time.time() - start_time
    print(f"[GPU {rank}] Done: {generated_count} images in {elapsed/60:.1f}min "
          f"({generated_count/max(elapsed,1):.1f} img/s)")


def prepare_real_images_flat(mjhq_img_dir, entries, real_flat_dir):
    """Create a flat directory of real images (resized) matching the generated set."""
    real_flat_dir = Path(real_flat_dir)
    if real_flat_dir.exists() and len(list(real_flat_dir.iterdir())) >= len(entries):
        print(f"Real image flat dir already prepared ({len(entries)} images)")
        return

    real_flat_dir.mkdir(parents=True, exist_ok=True)
    mjhq_img_dir = Path(mjhq_img_dir)

    print(f"Preparing flat real image directory ({len(entries)} images)...")
    for key, category, _ in tqdm(entries, desc="Linking real images"):
        dst = real_flat_dir / f"{key}.jpg"
        if dst.exists():
            continue
        src = mjhq_img_dir / category / f"{key}.jpg"
        if not src.exists():
            src = mjhq_img_dir / category / f"{key}.png"
        if src.exists():
            # Symlink to original resolution — let clean-fid handle resize to 299x299
            os.symlink(src.resolve(), dst)
        else:
            print(f"  WARNING: real image not found: {src}")


def compute_fid_score(real_dir, gen_dir):
    """Compute FID between two directories of images using clean-fid."""
    print(f"Computing FID...")
    print(f"  Real images: {real_dir}")
    print(f"  Generated images: {gen_dir}")
    score = fid.compute_fid(str(real_dir), str(gen_dir))
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default="xl",
                        choices=["l", "xl"],
                        help="MaskGen-KL model size (l or xl)")
    parser.add_argument("--order", type=str, default="random",
                        choices=MaskGen_KL.SUPPORTED_ORDER_TYPES,
                        help="Token generation order strategy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--aes-score", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size per GPU for generation")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to generate (default: all 30K)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all visible)")
    parser.add_argument("--mjhq-dir", type=str,
                        default="/data3/haoyuliu/mjhq30k",
                        help="Path to MJHQ-30K dataset root")
    parser.add_argument("--output-dir", type=str,
                        default="/data3/haoyuliu/mjhq30k_eval",
                        help="Base output directory for generated images and results")
    parser.add_argument("--fid-only", action="store_true",
                        help="Skip generation, only compute FID from existing images")
    parser.add_argument("--gen-dir", type=str, default=None,
                        help="Path to pre-generated images (for --fid-only)")
    args = parser.parse_args()

    meta_path = os.path.join(args.mjhq_dir, "meta_data.json")
    mjhq_img_dir = args.mjhq_dir  # category dirs (animals/, art/, ...) are directly under this

    # Load prompts
    entries = load_mjhq_prompts(meta_path, args.num_samples)
    n = len(entries)
    print(f"Loaded {n} prompts from MJHQ-30K")

    # Create a per-run directory: output_dir/eval_<order>_<suffix>_seed<seed>/
    suffix = f"n{n}" if args.num_samples else "30k"
    run_dir = os.path.join(args.output_dir, f"eval_{args.model}_{args.order}_{suffix}_seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    gen_dir = args.gen_dir or os.path.join(run_dir, "generated")
    real_flat_dir = os.path.join(run_dir, "real_resized")

    if not args.fid_only:
        # Determine number of GPUs
        num_gpus = args.num_gpus or torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs, batch_size={args.batch_size} per GPU")
        print(f"Effective batch: {num_gpus * args.batch_size} images/round")

        # Create output dir before spawning workers
        Path(gen_dir).mkdir(parents=True, exist_ok=True)

        if num_gpus == 1:
            worker_generate(0, 1, entries, gen_dir, args)
        else:
            mp.spawn(
                worker_generate,
                args=(num_gpus, entries, gen_dir, args),
                nprocs=num_gpus,
                join=True,
            )

        # Verify all images were generated
        generated_count = len(list(Path(gen_dir).glob("*.png")))
        print(f"\nTotal generated images: {generated_count}/{n}")

    # Prepare real images (resize to 256x256)
    prepare_real_images_flat(mjhq_img_dir, entries, real_flat_dir)

    # Compute FID
    fid_score = compute_fid_score(real_flat_dir, gen_dir)
    paper_fid = {"xl": 6.53, "l": 7.24}[args.model]
    print(f"\n{'='*50}")
    print(f"  Model: MaskGen-{args.model.upper()} KL")
    print(f"  FID ({args.order}, {n} images): {fid_score:.2f}")
    print(f"  (Paper reports: {paper_fid})")
    print(f"{'='*50}")

    # Save result
    result_path = os.path.join(run_dir, "fid_result.txt")
    with open(result_path, "w") as f:
        f.write(f"model: maskgen_kl_{args.model}\n")
        f.write(f"order: {args.order}\n")
        f.write(f"num_samples: {n}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"num_iter: {args.num_iter}\n")
        f.write(f"cfg: {args.cfg}\n")
        f.write(f"aes_score: {args.aes_score}\n")
        f.write(f"num_gpus: {args.num_gpus or torch.cuda.device_count()}\n")
        f.write(f"batch_size_per_gpu: {args.batch_size}\n")
        f.write(f"fid: {fid_score:.4f}\n")
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
