"""Run GenEval evaluation in parallel across multiple GPUs.

Shards the prompt directories across GPUs, runs evaluate_images.py
on each shard independently, then merges the results.

Usage:
    conda activate geneval
    python eval_geneval_parallel.py /data3/haoyuliu/geneval_eval/maskgen_kl_l_prompt_sim --num-gpus 8
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import math


GENEVAL_ROOT = "/home/hliu256/geneval"
EVAL_SCRIPT = os.path.join(GENEVAL_ROOT, "evaluation/evaluate_images.py")
MASK2FORMER_DIR = "/data3/haoyuliu/geneval_eval/mask2former_weights"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagedir", type=str, help="GenEval image directory")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--outfile", type=str, default=None,
                        help="Output results file (default: <imagedir>/results.jsonl)")
    args = parser.parse_args()

    imagedir = Path(args.imagedir)
    outfile = args.outfile or str(imagedir / "results.jsonl")
    num_gpus = args.num_gpus

    # Find all prompt subdirectories
    prompt_dirs = sorted([
        d.name for d in imagedir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ])
    print(f"Found {len(prompt_dirs)} prompt directories")
    print(f"Using {num_gpus} GPUs")

    # Create sharded symlink directories
    tmpdir = Path(f"/data3/haoyuliu/geneval_eval/_eval_shards_{imagedir.name}")
    if tmpdir.exists():
        shutil.rmtree(tmpdir)
    tmpdir.mkdir(parents=True)

    shard_size = math.ceil(len(prompt_dirs) / num_gpus)
    shard_dirs = []
    shard_outfiles = []

    for gpu_id in range(num_gpus):
        start = gpu_id * shard_size
        end = min(start + shard_size, len(prompt_dirs))
        shard_prompts = prompt_dirs[start:end]
        if not shard_prompts:
            continue

        shard_dir = tmpdir / f"shard_{gpu_id}"
        shard_dir.mkdir()

        # Symlink prompt directories into shard
        for pdir in shard_prompts:
            os.symlink(str(imagedir / pdir), str(shard_dir / pdir))

        shard_dirs.append(shard_dir)
        shard_outfiles.append(str(tmpdir / f"results_{gpu_id}.jsonl"))

    print(f"Created {len(shard_dirs)} shards, launching evaluations...")

    # Launch parallel evaluations
    procs = []
    for gpu_id, (shard_dir, shard_out) in enumerate(zip(shard_dirs, shard_outfiles)):
        n_prompts = len(list(shard_dir.iterdir()))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            sys.executable, EVAL_SCRIPT,
            str(shard_dir),
            "--outfile", shard_out,
            "--model-path", MASK2FORMER_DIR,
        ]
        print(f"  GPU {gpu_id}: {n_prompts} prompts -> {shard_out}")
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((gpu_id, proc))

    # Wait for all to finish
    for gpu_id, proc in procs:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(f"  GPU {gpu_id}: FAILED (exit {proc.returncode})")
            print(stderr.decode()[-500:])
        else:
            print(f"  GPU {gpu_id}: done")

    # Merge results
    print(f"\nMerging results -> {outfile}")
    all_lines = []
    for shard_out in shard_outfiles:
        if os.path.exists(shard_out):
            with open(shard_out) as f:
                all_lines.extend(f.readlines())

    with open(outfile, "w") as f:
        f.writelines(all_lines)

    print(f"Total results: {len(all_lines)} images")

    # Cleanup shard dirs
    shutil.rmtree(tmpdir)

    # Print summary
    print(f"\n{'='*60}")
    summary_script = os.path.join(GENEVAL_ROOT, "evaluation/summary_scores.py")
    subprocess.run([sys.executable, summary_script, outfile])


if __name__ == "__main__":
    main()
