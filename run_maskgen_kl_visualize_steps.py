"""Generate images with MaskGen-KL and visualize progressive generation.

Supports both single-prompt mode and multi-prompt comparison mode.

Single prompt:
    python run_maskgen_kl_visualize_steps.py --prompt "A cat on a table" --order random

Compare random vs prompt_sim for all built-in prompts:
    python run_maskgen_kl_visualize_steps.py --compare
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"

import argparse
import math
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import open_clip
from modeling.tatitok import TATiTok
from modeling.maskgen import MaskGen_KL


# ── Prompts designed to showcase generation reasoning ────────────────────────
# Each has clear compositional structure / spatial hierarchy / semantic focus
# so that random vs similarity-based ordering should produce visibly different
# generation trajectories.
COMPARISON_PROMPTS = [
    # Main subject vs background — does sim-order prioritize the subject?
    "A golden retriever sitting in a field of sunflowers.",
    # Spatial relationship — which object gets generated first?
    "A cat sitting on top of a stack of books.",
    # Counting + distinct objects — ordering of semantically similar items
    "Three red apples and two green pears on a wooden table.",
    # Unusual composition — tests semantic understanding
    "An astronaut riding a horse on the surface of the moon.",
    # Fine-grained detail + hierarchy — cake vs candles vs background
    "A birthday cake with colorful candles on a kitchen counter.",
]


def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda"


def compute_text_guidance(prompts, clip_tokenizer, clip_encoder):
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


def decode_snapshots(snapshots, tokenizer, text_guidance):
    """Decode a list of snapshot tensors into numpy images.

    Args:
        snapshots: list of (bsz, C, 1, seq_len) tensors
        text_guidance: (bsz, 77, dim) tensor

    Returns:
        list of lists: snapshot_images[step][batch_idx] = (H, W, 3) uint8 array
    """
    all_step_images = []
    for snap_tokens in snapshots:
        img = tokenizer.decode_tokens(snap_tokens, text_guidance)
        img = torch.clamp(img, 0.0, 1.0)
        img = (img * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        all_step_images.append(img)  # (bsz, H, W, 3)
    return all_step_images


def visualize_single(snapshot_images, prompt, order_type, out_path):
    """Grid visualization for a single prompt."""
    num_steps = len(snapshot_images)
    ncols = 8
    nrows = math.ceil(num_steps / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.4 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        if idx < num_steps:
            ax.imshow(snapshot_images[idx])
            ax.set_title(f"step {idx + 1}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= num_steps:
            ax.axis("off")

    fig.suptitle(f'"{prompt}"  (order={order_type})', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_comparison(random_images, sim_images, prompt, num_iter, out_path):
    """Side-by-side comparison: random (top row) vs prompt_sim (bottom row).

    Shows a subset of steps to fit in one figure.
    """
    # Pick evenly-spaced steps to display
    max_cols = 10
    if num_iter <= max_cols:
        step_indices = list(range(num_iter))
    else:
        step_indices = [int(round(i * (num_iter - 1) / (max_cols - 1)))
                        for i in range(max_cols)]
        step_indices = sorted(set(step_indices))
    ncols = len(step_indices)

    fig, axes = plt.subplots(2, ncols, figsize=(2.8 * ncols, 6.5))

    for col, step_idx in enumerate(step_indices):
        # Top row: random
        axes[0, col].imshow(random_images[step_idx])
        axes[0, col].set_title(f"step {step_idx + 1}", fontsize=9)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Bottom row: prompt_sim
        axes[1, col].imshow(sim_images[step_idx])
        axes[1, col].set_title(f"step {step_idx + 1}", fontsize=9)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("random", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("prompt_sim", fontsize=12, fontweight="bold")

    fig.suptitle(f'"{prompt}"', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt for single-prompt mode")
    parser.add_argument("--order", type=str, default="random",
                        choices=MaskGen_KL.SUPPORTED_ORDER_TYPES,
                        help="Order type for single-prompt mode")
    parser.add_argument("--compare", action="store_true",
                        help="Compare random vs prompt_sim for all built-in prompts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num-iter", type=int, default=32, help="Number of sampling steps")
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--aes-score", type=float, default=6.5, help="Aesthetic score")
    args = parser.parse_args()

    if args.prompt is None and not args.compare:
        args.prompt = "Three cups of coffee on a table."

    set_all_seeds(args.seed)

    # ── Load models ──────────────────────────────────────────────────────
    print("Loading KL tokenizer...")
    tatitok_kl_tokenizer = TATiTok.from_pretrained("turkeyju/tokenizer_tatitok_bl32_vae")
    tatitok_kl_tokenizer.eval().requires_grad_(False).to(device)

    print("Loading KL generator...")
    maskgen_kl_generator = MaskGen_KL.from_pretrained("turkeyju/generator_maskgen_kl_xl")
    maskgen_kl_generator.eval().requires_grad_(False).to(device)

    print("Loading CLIP encoder...")
    clip_encoder, _, _ = open_clip.create_model_and_transforms('ViT-L-14-336', pretrained='openai')
    del clip_encoder.visual
    clip_tokenizer = open_clip.get_tokenizer('ViT-L-14-336')
    clip_encoder.transformer.batch_first = False
    clip_encoder.eval().requires_grad_(False).to(device)

    os.makedirs("image_outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.compare:
        # ── Comparison mode: batch all prompts, run both orders ──────────
        prompts = COMPARISON_PROMPTS
        bsz = len(prompts)
        print(f"Comparison mode: {bsz} prompts x 2 orders")

        text_guidance = compute_text_guidance(prompts, clip_tokenizer, clip_encoder)

        results = {}
        for order_type in ["random", "prompt_sim"]:
            set_all_seeds(args.seed)
            print(f"\nSampling with order={order_type} (batch={bsz})...")
            _, snapshots = maskgen_kl_generator.sample_tokens(
                bsz=bsz, clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder,
                num_iter=args.num_iter, cfg=args.cfg, captions=prompts,
                aes_scores=args.aes_score, order_type=order_type,
                record_snapshots=True,
            )
            print(f"Decoding {len(snapshots)} snapshots...")
            all_step_images = decode_snapshots(snapshots, tatitok_kl_tokenizer, text_guidance)
            results[order_type] = all_step_images

        # Save per-prompt comparison figures
        for p_idx, prompt in enumerate(prompts):
            random_imgs = [step_batch[p_idx] for step_batch in results["random"]]
            sim_imgs = [step_batch[p_idx] for step_batch in results["prompt_sim"]]
            prompt_slug = prompt.lower().replace(" ", "_").replace(".", "").replace(",", "")[:50]

            # Comparison figure
            cmp_path = f"image_outputs/compare_{prompt_slug}_{args.seed}_{timestamp}.png"
            visualize_comparison(random_imgs, sim_imgs, prompt, args.num_iter, cmp_path)
            print(f"Saved {cmp_path}")

            # Also save full step grids for each order
            for order_type in ["random", "prompt_sim"]:
                imgs = [step_batch[p_idx] for step_batch in results[order_type]]
                grid_path = f"image_outputs/maskgen_kl_{order_type}_{prompt_slug}_{args.seed}_{timestamp}.png"
                visualize_single(imgs, prompt, order_type, grid_path)
                print(f"Saved {grid_path}")

    else:
        # ── Single prompt mode ───────────────────────────────────────────
        text = [args.prompt]
        print(f"Generating image for: {text[0]}")
        print(f"Order type: {args.order}, seed: {args.seed}, num_iter: {args.num_iter}")

        text_guidance = compute_text_guidance(text, clip_tokenizer, clip_encoder)
        set_all_seeds(args.seed)

        print("Sampling tokens...")
        _, snapshots = maskgen_kl_generator.sample_tokens(
            bsz=1, clip_tokenizer=clip_tokenizer, clip_encoder=clip_encoder,
            num_iter=args.num_iter, cfg=args.cfg, captions=text,
            aes_scores=args.aes_score, order_type=args.order,
            record_snapshots=True,
        )

        print("Decoding snapshots...")
        all_step_images = decode_snapshots(snapshots, tatitok_kl_tokenizer, text_guidance)
        snapshot_images = [step_batch[0] for step_batch in all_step_images]
        for i in range(len(snapshot_images)):
            print(f"  decoded step {i+1}/{len(snapshot_images)}")

        prompt_slug = text[0].lower().replace(" ", "_").replace(".", "")[:50]
        out_path = f"image_outputs/maskgen_kl_{args.order}_{prompt_slug}_{args.seed}_{timestamp}.png"
        visualize_single(snapshot_images, text[0], args.order, out_path)
        print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    main()
