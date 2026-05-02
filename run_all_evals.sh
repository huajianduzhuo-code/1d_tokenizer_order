#!/bin/bash
# One-shot driver for all 4 evaluations × multiple seeds.
#
# Pipeline:
#   1. Generation phase (maskgen env, sequential, each step uses 8 GPUs):
#      DPG-Bench  →  GenEval  →  T2I-CompBench  →  MJHQ-FID (FID computed inline)
#   2. Evaluation phase (each step switches to its own conda env):
#      DPG-Bench (dpg env)  →  GenEval (geneval env)  →  T2I-CompBench (t2i_compbench env)
#   3. Summaries (maskgen env): print mean±std tables for all 4.
#
# Idempotent: each underlying wrapper already skips combos that already have
# a finished summary on disk. Re-running this script is safe.
#
# ──────────────── Configuration via env vars ────────────────
#   SEEDS                — uniform override for all 4 pipelines
#   MJHQ_SEEDS           — default "0 1 2"     (MJHQ has its own seed convention)
#   DPG_SEEDS            — default "42 43 44"
#   GENEVAL_SEEDS        — default "42 43 44"
#   T2ICB_SEEDS          — default "42 43 44"
#   SKIP_MJHQ=1          — skip the MJHQ-FID block
#   SKIP_DPG=1           — skip the DPG block
#   SKIP_GENEVAL=1       — skip the GenEval block
#   SKIP_T2ICB=1         — skip the T2I-CompBench block
#   GEN_ONLY=1           — run only the generation phase, stop before eval
#   EVAL_ONLY=1          — skip generation, run only eval (assumes images on disk)
#
# ──────────────── Examples ────────────────
#   bash run_all_evals.sh                              # all defaults
#   SEEDS="42 43 44" bash run_all_evals.sh             # uniform seeds (incl. MJHQ)
#   T2ICB_SEEDS="42 43" bash run_all_evals.sh          # only 2 seeds for T2I-CB
#   SKIP_MJHQ=1 SKIP_GENEVAL=1 bash run_all_evals.sh   # only DPG + T2I-CB
#   GEN_ONLY=1 bash run_all_evals.sh                   # generate everything overnight, eval later
#
# Run inside tmux — total wall time ≈ 18-25 hours for 3 seeds × 4 evals.

set -e

cd /home/hliu256/1d-tokenizer

# Per-eval seed configs (per-eval override > SEEDS > pipeline default)
MJHQ_SEEDS=${MJHQ_SEEDS:-${SEEDS:-"0 1 2"}}
DPG_SEEDS=${DPG_SEEDS:-${SEEDS:-"42 43 44"}}
GENEVAL_SEEDS=${GENEVAL_SEEDS:-${SEEDS:-"42 43 44"}}
T2ICB_SEEDS=${T2ICB_SEEDS:-${SEEDS:-"42 43 44"}}

LOG_DIR=/data3/haoyuliu/all_evals_logs
mkdir -p "$LOG_DIR"
START_TS=$(date '+%Y%m%d_%H%M%S')

echo "============================================================"
echo " Combined evaluation driver — start $(date)"
echo "============================================================"
echo "  MJHQ_SEEDS    = $MJHQ_SEEDS"
echo "  DPG_SEEDS     = $DPG_SEEDS"
echo "  GENEVAL_SEEDS = $GENEVAL_SEEDS"
echo "  T2ICB_SEEDS   = $T2ICB_SEEDS"
echo "  Skips: MJHQ=${SKIP_MJHQ:-0} DPG=${SKIP_DPG:-0} GENEVAL=${SKIP_GENEVAL:-0} T2ICB=${SKIP_T2ICB:-0}"
echo "  Phase:  GEN_ONLY=${GEN_ONLY:-0}  EVAL_ONLY=${EVAL_ONLY:-0}"
echo "  Logs:   $LOG_DIR/run_${START_TS}_*.log"
echo "============================================================"

# Helper: time + log a step. Aborts on failure (set -e).
step() {
    local name="$1"; shift
    local logf="$LOG_DIR/run_${START_TS}_${name}.log"
    local t0=$(date +%s)
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> $name  (start $(date '+%H:%M:%S'),  log → $logf)"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    "$@" 2>&1 | tee "$logf"
    local t1=$(date +%s)
    echo "<<< $name  done in $(( (t1-t0)/60 )) min"
}

# ──────────────── 1. Generation phase ────────────────
if [ -z "${EVAL_ONLY:-}" ]; then
    [ -z "${SKIP_DPG:-}" ]     && step "gen_dpg"     bash -c "SEEDS='$DPG_SEEDS' bash run_dpg_all.sh"
    [ -z "${SKIP_GENEVAL:-}" ] && step "gen_geneval" bash -c "SEEDS='$GENEVAL_SEEDS' bash run_geneval_all.sh"
    [ -z "${SKIP_T2ICB:-}" ]   && step "gen_t2icb"   bash -c "SEEDS='$T2ICB_SEEDS' bash run_t2i_compbench_all.sh"
    # MJHQ does generation + FID in one script — counts as both gen and eval.
    [ -z "${SKIP_MJHQ:-}" ]    && step "gen_eval_mjhq" bash -c "SEEDS='$MJHQ_SEEDS' bash run_mjhq_fid_all.sh"
fi

if [ -n "${GEN_ONLY:-}" ]; then
    echo ""
    echo "GEN_ONLY=1 set — stopping before evaluation phase."
    exit 0
fi

# ──────────────── 2. Evaluation phase ────────────────
[ -z "${SKIP_DPG:-}" ]     && step "eval_dpg"     bash -c "SEEDS='$DPG_SEEDS' bash run_dpg_evaluate.sh"
[ -z "${SKIP_GENEVAL:-}" ] && step "eval_geneval" bash -c "SEEDS='$GENEVAL_SEEDS' bash run_geneval_evaluate.sh"
[ -z "${SKIP_T2ICB:-}" ]   && step "eval_t2icb"   bash -c "SEEDS='$T2ICB_SEEDS' bash run_t2i_compbench_evaluate.sh"
# MJHQ FID is already done in the gen step above.

# ──────────────── 3. Summaries ────────────────
echo ""
echo "============================================================"
echo " Final summaries (mean±std across seeds)"
echo " $(date)"
echo "============================================================"

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate maskgen

[ -z "${SKIP_MJHQ:-}" ]    && step "summary_mjhq"    python eval_mjhq_fid.py --summary --summary-suffix 30k
[ -z "${SKIP_DPG:-}" ]     && step "summary_dpg"     python eval_dpg.py --summary
[ -z "${SKIP_GENEVAL:-}" ] && step "summary_geneval" python eval_geneval.py --summary
[ -z "${SKIP_T2ICB:-}" ]   && step "summary_t2icb"   python eval_t2i_compbench.py --summary

echo ""
echo "============================================================"
echo " ALL DONE — $(date)"
echo " Per-step logs in $LOG_DIR/run_${START_TS}_*.log"
echo "============================================================"
