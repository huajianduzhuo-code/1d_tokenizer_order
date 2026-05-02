"""Evaluate MaskGen-KL on the PromptBind-Hard (PBH) benchmark.

Generates images for each PBH prompt, asks a VLM (GPT-4o or Gemini) the
prompt's questions, and aggregates per-question / per-prompt accuracies.
Mirrors the structure of eval_geneval.py.

Usage:
    # Step 1: Generate images (run for each model/order combo)
    python eval_pbh.py --model xl --order random
    python eval_pbh.py --model xl --order prompt_sim
    python eval_pbh.py --model l  --order random
    python eval_pbh.py --model l  --order prompt_sim

    # Step 2: Judge (requires OPENAI_API_KEY or GOOGLE_API_KEY in env)
    python eval_pbh.py --model xl --order random --eval-only --judge-provider openai

    # Generate + judge in one shot
    python eval_pbh.py --model xl --order random --evaluate --judge-provider openai

    # Re-aggregate from cache without calling the API
    python eval_pbh.py --model xl --order random --re-aggregate

    # Cross-config summary table
    python eval_pbh.py --summary

    # Multi-GPU generation
    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_pbh.py --model xl --order random --num-gpus 4
"""

import os
os.environ["HF_HOME"] = "/data3/haoyuliu/huggingface_cache"

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
import open_clip

_SUPPORTED_ORDER_TYPES = ["random", "prompt_sim", "prompt_sim_rev"]


# -----------------------------------------------------------------
# Paths
# -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
PBH_DATASET = REPO_ROOT / "benchmarks" / "pbh" / "prompts.jsonl"
OUTPUT_BASE = Path("/data3/haoyuliu/pbh_eval")


def _output_base_for_dataset(dataset_path: Path) -> Path:
    """Derive an output base from the dataset filename so different datasets
    (e.g. prompts.jsonl vs prompts_lite.jsonl) write to separate dirs.

    - prompts.jsonl       -> /data3/haoyuliu/pbh_eval
    - prompts_lite.jsonl  -> /data3/haoyuliu/pbh_eval_lite
    - prompts_v2.jsonl    -> /data3/haoyuliu/pbh_eval_v2
    """
    stem = dataset_path.stem
    if stem == "prompts":
        return Path("/data3/haoyuliu/pbh_eval")
    suffix = stem[len("prompts_"):] if stem.startswith("prompts_") else stem
    return Path(f"/data3/haoyuliu/pbh_eval_{suffix}")


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


def load_pbh_records():
    records = []
    with open(PBH_DATASET) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_dir_name(model, order, seed) -> Path:
    return OUTPUT_BASE / f"maskgen_kl_{model}_{order}_seed{seed}"


# -----------------------------------------------------------------
# Phase 1: Generation
# -----------------------------------------------------------------

def worker_generate(rank, num_gpus, records, out_dir, args):
    """Each GPU generates its shard of prompts in PBH format."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    out_dir = Path(out_dir)

    # Shard across GPUs
    shard_size = math.ceil(len(records) / num_gpus)
    shard_start = rank * shard_size
    shard_end = min(shard_start + shard_size, len(records))
    shard_recs = list(enumerate(records[shard_start:shard_end], start=shard_start))

    if not shard_recs:
        return

    # Pre-create output dirs and save metadata
    for global_idx, rec in shard_recs:
        prompt_dir = out_dir / f"{global_idx:05d}"
        sample_dir = prompt_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        with open(prompt_dir / "metadata.json", "w") as f:
            json.dump(rec, f)

    # Determine which (prompt_idx, sample_idx) pairs still need generation
    todo = []
    for global_idx, rec in shard_recs:
        sample_dir = out_dir / f"{global_idx:05d}" / "samples"
        for s in range(args.n_samples):
            img_path = sample_dir / f"{s:05d}.png"
            if not img_path.exists():
                todo.append((global_idx, rec, s))

    if not todo:
        print(f"[GPU {rank}] All images already exist, skipping")
        return

    print(f"[GPU {rank}] Loading models...")
    from modeling.tatitok import TATiTok
    from modeling.maskgen import MaskGen_KL

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
          f"{len(shard_recs) * args.n_samples - total_imgs} skipped)")

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
        prompts = [rec["prompt"] for _, rec, _ in batch_items]

        text_guidance = compute_text_guidance(
            prompts, clip_tokenizer, clip_encoder, device)

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

        for i, (global_idx, _, sample_idx) in enumerate(batch_items):
            sample_dir = out_dir / f"{global_idx:05d}" / "samples"
            img = Image.fromarray(images[i])
            img.save(sample_dir / f"{sample_idx:05d}.png")

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
    records = load_pbh_records()
    if args.num_samples is not None:
        records = records[:args.num_samples]

    out_dir = run_dir_name(args.model, args.order, args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: MaskGen-KL-{args.model.upper()}")
    print(f"Order: {args.order}")
    print(f"Seed:  {args.seed}")
    print(f"Samples per prompt: {args.n_samples}")
    print(f"Total prompts: {len(records)}")
    print(f"Output: {out_dir}")

    num_gpus = args.num_gpus or torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s), batch_size={args.batch_size}")

    if num_gpus == 1:
        worker_generate(0, 1, records, str(out_dir), args)
    else:
        mp.spawn(
            worker_generate,
            args=(num_gpus, records, str(out_dir), args),
            nprocs=num_gpus,
            join=True,
        )

    total_expected = len(records) * args.n_samples
    total_generated = 0
    for d in out_dir.iterdir():
        if not d.is_dir() or not d.name.isdigit():
            continue
        sd = d / "samples"
        if sd.exists():
            total_generated += sum(1 for f in sd.iterdir() if f.suffix == ".png")
    print(f"\nGeneration complete: {total_generated}/{total_expected} images")


# -----------------------------------------------------------------
# Phase 2: VLM judging
# -----------------------------------------------------------------

def run_judging(args):
    """Call the VLM judge on all generated images for this run."""
    import pbh_judge

    out_dir = run_dir_name(args.model, args.order, args.seed)
    if not out_dir.exists():
        print(f"ERROR: run dir does not exist: {out_dir}")
        sys.exit(1)

    records = load_pbh_records()
    if args.num_samples is not None:
        records = records[:args.num_samples]

    providers = (["openai", "gemini"] if args.judge_provider == "both"
                 else [args.judge_provider])

    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Judging with provider={provider}: {out_dir}")
        print(f"{'='*60}")
        pbh_judge.judge_run(out_dir, records, provider, force=args.force)


# -----------------------------------------------------------------
# Phase 3: Aggregation
# -----------------------------------------------------------------

def _load_judge_results(out_dir: Path, provider: str,
                        model: str | None = None) -> list[dict]:
    """Load cached judge results for a provider in this run dir.

    If `model` is given, only return results produced by that model — this
    avoids mixing in stale results from a previously-used judge model.
    """
    cache_root = out_dir / "judge_cache" / provider
    if not cache_root.exists():
        return []
    results = []
    for prompt_dir in cache_root.iterdir():
        if not prompt_dir.is_dir():
            continue
        for f in prompt_dir.iterdir():
            if not f.suffix == ".json":
                continue
            try:
                with open(f) as fh:
                    rec = json.load(fh)
                if model is not None and rec.get("model") != model:
                    continue
                results.append(rec)
            except Exception:
                pass
    return results


def aggregate_run(args, write: bool = True) -> dict | None:
    """Build pbh_summary.json from cached judge results."""
    out_dir = run_dir_name(args.model, args.order, args.seed)
    if not out_dir.exists():
        print(f"ERROR: run dir does not exist: {out_dir}")
        return None

    # Aggregate ALL providers that have cache on disk, not just the one
    # the user passed via --judge-provider. This avoids overwriting a
    # previously-aggregated provider's data when re-judging just one.
    providers = []
    cache_root = out_dir / "judge_cache"
    if cache_root.exists():
        for p in sorted(cache_root.iterdir()):
            if p.is_dir() and p.name in ("openai", "gemini"):
                providers.append(p.name)

    # records keyed by id, used for prompt-level "all-correct" check
    records = load_pbh_records()
    if args.num_samples is not None:
        records = records[:args.num_samples]
    rec_by_id = {r["id"]: r for r in records}

    summary = {
        "model": f"maskgen_kl_{args.model}",
        "order": args.order,
        "seed": args.seed,
        "n_samples": args.n_samples,
        "n_prompts": len(records),
        "providers": {},
    }

    import pbh_judge
    provider_models = {
        "openai": pbh_judge.DEFAULT_OPENAI_MODEL,
        "gemini": pbh_judge.DEFAULT_GEMINI_MODEL,
    }
    for provider in providers:
        results = _load_judge_results(out_dir, provider, provider_models[provider])
        if not results:
            print(f"[provider={provider}] no judge results found, skipping")
            continue

        # Per-question accuracy by category
        cat_q_total = {}
        cat_q_correct = {}
        for r in results:
            cat = r["category"]
            cat_q_total[cat] = cat_q_total.get(cat, 0) + 1
            cat_q_correct[cat] = cat_q_correct.get(cat, 0) + (1 if r["correct"] else 0)
        per_question_by_cat = {
            cat: round(cat_q_correct.get(cat, 0) / max(cat_q_total[cat], 1), 4)
            for cat in cat_q_total
        }
        per_question_overall = round(
            sum(cat_q_correct.values()) / max(sum(cat_q_total.values()), 1), 4)

        # Index results by (prompt_id, sample_idx, qid)
        by_psq = {}
        for r in results:
            by_psq[(r["prompt_id"], r["sample_idx"], r["qid"])] = r

        # Per-prompt strict accuracy (best-of-n_samples): a prompt counts as
        # correct if at least one sample answered ALL its questions correctly.
        # Per-prompt mean: average over (prompt, sample) of the all-correct
        # indicator -> equivalent to "fraction of (prompt, sample) pairs
        # where all questions are correct".
        cat_strict_total = {}
        cat_strict_hit = {}
        cat_mean_sum = {}
        cat_mean_n = {}
        for rid, rec in rec_by_id.items():
            cat = rec["category"]
            n_q = len(rec["questions"])
            n_correct_per_sample = {}
            for q_idx in range(n_q):
                for s_idx in range(args.n_samples):
                    r = by_psq.get((rid, s_idx, q_idx))
                    if r is None:
                        continue
                    n_correct_per_sample.setdefault(s_idx, 0)
                    if r["correct"]:
                        n_correct_per_sample[s_idx] += 1
            if not n_correct_per_sample:
                continue
            best_all = any(c == n_q for c in n_correct_per_sample.values())
            cat_strict_total[cat] = cat_strict_total.get(cat, 0) + 1
            cat_strict_hit[cat] = cat_strict_hit.get(cat, 0) + (1 if best_all else 0)
            for c in n_correct_per_sample.values():
                cat_mean_sum[cat] = cat_mean_sum.get(cat, 0.0) + (1.0 if c == n_q else 0.0)
                cat_mean_n[cat] = cat_mean_n.get(cat, 0) + 1

        per_prompt_strict_by_cat = {
            cat: round(cat_strict_hit.get(cat, 0) / max(cat_strict_total[cat], 1), 4)
            for cat in cat_strict_total
        }
        per_prompt_strict_overall = round(
            sum(cat_strict_hit.values()) / max(sum(cat_strict_total.values()), 1), 4)
        per_prompt_mean_by_cat = {
            cat: round(cat_mean_sum.get(cat, 0.0) / max(cat_mean_n[cat], 1), 4)
            for cat in cat_mean_n
        }
        per_prompt_mean_overall = round(
            sum(cat_mean_sum.values()) / max(sum(cat_mean_n.values()), 1), 4)

        n_errors = sum(1 for r in results if r.get("error"))

        summary["providers"][provider] = {
            "per_question_overall": per_question_overall,
            "per_question_by_cat": per_question_by_cat,
            "per_prompt_strict_overall": per_prompt_strict_overall,
            "per_prompt_strict_by_cat": per_prompt_strict_by_cat,
            "per_prompt_mean_overall": per_prompt_mean_overall,
            "per_prompt_mean_by_cat": per_prompt_mean_by_cat,
            "n_results": len(results),
            "n_errors": n_errors,
        }

        print(f"\n[provider={provider}] per-question overall: "
              f"{per_question_overall:.4f}")
        for cat, v in per_question_by_cat.items():
            print(f"    {cat:>26}: {v:.4f}")
        print(f"[provider={provider}] per-prompt strict (best-of-{args.n_samples}) "
              f"overall: {per_prompt_strict_overall:.4f}")
        if n_errors:
            print(f"[provider={provider}] {n_errors} errored responses "
                  f"(see judge_responses_{provider}.jsonl)")

    if write:
        summary_file = out_dir / "pbh_summary.json"
        tmp = summary_file.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, summary_file)
        print(f"\nSummary saved to {summary_file}")
    return summary


# -----------------------------------------------------------------
# Cross-run summary
# -----------------------------------------------------------------

def _seeds_present():
    pat = re.compile(r"^maskgen_kl_(l|xl)_(random|prompt_sim|prompt_sim_rev)_seed(\d+)$")
    found = {}
    if not OUTPUT_BASE.exists():
        return found
    for p in OUTPUT_BASE.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
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


def run_summary(args):
    """Print mean±std comparison across all completed configs."""
    seeds_map = _seeds_present()
    if not seeds_map:
        print("No PBH evaluation results found yet.")
        print(f"Expected location: {OUTPUT_BASE}/maskgen_kl_<model>_<order>_seed<seed>/pbh_summary.json")
        return

    all_summaries = {}  # (model, order) -> [summary dicts]
    for (model, order), seeds in seeds_map.items():
        for seed in seeds:
            sf = run_dir_name(model, order, seed) / "pbh_summary.json"
            if sf.exists():
                with open(sf) as f:
                    all_summaries.setdefault((model, order), []).append(json.load(f))

    if not all_summaries:
        print("No pbh_summary.json files found. Run --re-aggregate first.")
        return

    # Discover providers from the first summary
    providers = set()
    for sums in all_summaries.values():
        for s in sums:
            providers.update(s.get("providers", {}).keys())
    providers = sorted(providers)
    if not providers:
        print("No provider results in summaries.")
        return

    cats_order = [
        "counting_hard", "color_binding_multi", "count_x_color",
        "composition_layout", "negation_substitution", "relational_reasoning",
    ]
    cats_short = ["Count", "ColorBind", "CountxColor", "Layout", "Negation", "Reason"]
    cell_w = 13

    def _fmt_ms(mean, std):
        if mean is None:
            return f"{'--':>{cell_w}}"
        return f"{mean:>5.3f}±{std:<5.3f}".rjust(cell_w)

    lines_all = []
    for provider in providers:
        for metric, label in [
            ("per_question_by_cat", "per-question"),
            ("per_prompt_strict_by_cat", "per-prompt strict (best-of-N)"),
        ]:
            overall_key = ("per_question_overall" if metric.startswith("per_question")
                           else "per_prompt_strict_overall")

            lines = []
            lines.append("=" * 110)
            lines.append(f"PBH ({label}) — provider={provider} — mean±std across seeds")
            lines.append("=" * 110)
            lines.append("Seeds discovered per (model, order):")
            for (model, order), sums in sorted(all_summaries.items()):
                seeds = sorted(s.get("seed", "?") for s in sums)
                lines.append(f"  {model:>2} / {order:<14}: {seeds}")
            lines.append("")

            header = f"{'Model':<28} | " + " | ".join(
                f"{t:>{cell_w}}" for t in cats_short) + f" | {'Overall':>{cell_w}}"
            lines.append(header)
            lines.append("-" * len(header))

            order_rank = {"random": 0, "prompt_sim": 1, "prompt_sim_rev": 2}
            for (model, order) in sorted(all_summaries,
                                          key=lambda k: (k[0], order_rank.get(k[1], 99))):
                sums = all_summaries[(model, order)]
                n = len(sums)
                row = f"MaskGen-{model.upper()} ({order}, n={n})"
                row = f"{row:<28} | "
                for c in cats_order:
                    vals = []
                    for s in sums:
                        prov = s.get("providers", {}).get(provider, {})
                        v = prov.get(metric, {}).get(c)
                        if v is not None:
                            vals.append(v)
                    m, sd = _mean_std(vals)
                    row += _fmt_ms(m, sd) + " | "
                ovs = []
                for s in sums:
                    prov = s.get("providers", {}).get(provider, {})
                    v = prov.get(overall_key)
                    if v is not None:
                        ovs.append(v)
                m, sd = _mean_std(ovs)
                row += _fmt_ms(m, sd)
                lines.append(row)

            lines.append("=" * 110)
            body = "\n".join(lines)
            print(body)
            print()
            lines_all.append(body)

    out_file = OUTPUT_BASE / "pbh_summary_table.txt"
    with open(out_file, "w") as f:
        f.write("\n\n".join(lines_all) + "\n")
    print(f"Table saved to {out_file}")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    global PBH_DATASET, OUTPUT_BASE
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type=str, default="xl",
                        choices=["l", "xl"], help="MaskGen-KL model size")
    parser.add_argument("--order", type=str, default="random",
                        choices=_SUPPORTED_ORDER_TYPES,
                        help="Token generation order strategy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iter", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--aes-score", type=float, default=6.5)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size per GPU")
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Number of images per prompt (default: 4)")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs (default: all visible)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Smoke-test: limit to first N prompts")

    parser.add_argument("--evaluate", action="store_true",
                        help="Run VLM judging + aggregation after generation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip generation, only run judging + aggregation")
    parser.add_argument("--re-aggregate", action="store_true",
                        help="Skip API; re-read judge cache + emit summary")
    parser.add_argument("--summary", action="store_true",
                        help="Print cross-config comparison table from existing pbh_summary.json files")
    parser.add_argument("--judge-provider", type=str, default="openai",
                        choices=["openai", "gemini", "both"],
                        help="Which VLM to use for judging")
    parser.add_argument("--force", action="store_true",
                        help="Re-judge ignoring on-disk cache")
    parser.add_argument("--dataset", type=str, default=str(PBH_DATASET),
                        help="Path to a PBH-format JSONL dataset. "
                             "Different datasets write to different output dirs "
                             "(e.g. prompts_lite.jsonl -> /data3/haoyuliu/pbh_eval_lite/).")

    args = parser.parse_args()

    # Apply --dataset: redirect both the input and the output base
    PBH_DATASET = Path(args.dataset).resolve()
    OUTPUT_BASE = _output_base_for_dataset(PBH_DATASET)
    if not PBH_DATASET.exists():
        print(f"ERROR: dataset not found: {PBH_DATASET}")
        sys.exit(1)
    print(f"Dataset: {PBH_DATASET}")
    print(f"Output base: {OUTPUT_BASE}")

    if args.summary:
        run_summary(args)
        return

    if args.re_aggregate:
        aggregate_run(args, write=True)
        return

    if not args.eval_only:
        run_generation(args)

    if args.evaluate or args.eval_only:
        run_judging(args)
        aggregate_run(args, write=True)


if __name__ == "__main__":
    main()
