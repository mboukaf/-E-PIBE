#!/bin/bash
# ---------------------------------------------------------------------------
# PIBE vs EPIBE under a sensor bias -- Inria Margaret, cluster scale.
# Docs/EPIBE.md §10.  Separate from train_pibe.sbatch: its own job name, log
# names and run directories, so it cannot resume or overwrite the big_* runs.
#
#   scripts/slurm/submit_biasid.sh            <-- use this (all 12 tasks)
#   scripts/slurm/submit_biasid.sh 0-3        # seed 0 only
#
# Array index -> (seed, estimator, measurement):
#     seed      = idx / 4                      0, 1, 2
#     estimator = (idx / 2) % 2                0 PIBE      1 EPIBE
#     noise     = idx % 2                      0 Gaussian  1 bias 0.5
#
#     idx:  0 pibe gauss   1 pibe b0.5   2 epibe gauss   3 epibe b0.5   (s0)
#           4 .. 7  the same for seed 1,   8 .. 11 for seed 2
#
# The noise-free condition is not trained: it is evaluated on the Gaussian runs
# by feeding y = x1 (scripts/plot_pibe_epibe_boxplots.py).
#
# Runs:  outputs/biasid_big_{pibe,epibe}_{gauss,b0.5}_s{0,1,2}
#
# Budget (both arms take the same number of full-chain optimizer steps):
#   PIBE   Algorithm 1, final_global_iters 60000                       ~ 19 h
#   EPIBE  Algorithm 2, final_global_iters 30000, + 20000 amortization
#          + 10000 re-centring + offset search on 128 trajectories      ~ 24 h
# Estimated from 0.77 s per global step at 16 threads (train_pibe.sbatch) and a
# measured 0.5 s per shooting step at 128 trajectories; --time leaves margin.
#
# RESUMABLE.  Algorithm 1/2 resume from <run>/state.pt; EPIBE's stages after
# it resume from <run>/stages.json, stage_bank.pt and stage_state.pt.  All are
# removed on success (the offset search record is kept as offset_search.json).
# ---------------------------------------------------------------------------
#SBATCH --job-name=biasid
#SBATCH --output=logs/biasid_%A_%a.out
#SBATCH --error=logs/biasid_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --requeue

set -euo pipefail

IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( IDX / 4 ))
if (( (IDX / 2) % 2 == 0 )); then EST="pibe"; else EST="epibe"; fi
if (( IDX % 2 == 0 )); then NOISE="gauss"; BIAS="0.0"; else NOISE="b0.5"; BIAS="0.5"; fi

CONFIG="configs/automatica_n3v2_biasid_big_${EST}.yaml"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }
RUN="outputs/biasid_big_${EST}_${NOISE}_s${SEED}"

mkdir -p logs "${RUN}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "host      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION:-?}"
echo "job       : ${SLURM_JOB_ID:-none}  array task ${IDX}"
echo "estimator : ${EST}"
echo "noise     : sigma 0.05, bias ${BIAS}"
echo "seed      : ${SEED}"
echo "config    : ${CONFIG}"
echo "run dir   : ${RUN}"
echo "threads   : ${OMP_NUM_THREADS}"
echo "resuming  : $({ [ -f "${RUN}/state.pt" ] || [ -f "${RUN}/stages.json" ]; } && echo yes || echo 'no (fresh start)')"
echo

# --- environment (same as train_pibe.sbatch) --------------------------------
# module load python/3.12
source .venv/bin/activate
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if ! "${PYTHON}" -V >/dev/null 2>&1; then
    echo "FATAL: ${PYTHON} does not run on $(hostname)." >&2
    exit 1
fi
echo "python    : ${PYTHON} ($(${PYTHON} -V 2>&1))"
"${PYTHON}" -c "import torch; print(f'torch     : {torch.__version__}  cuda={torch.cuda.is_available()}')"
echo

srun "${PYTHON}" scripts/train_pibe.py \
    --config "${CONFIG}" \
    --set seed="${SEED}" training.seed="${SEED}" data.noise_bias="${BIAS}" \
    --output-dir "${RUN}" \
    --threads "${OMP_NUM_THREADS}"

echo
echo "done. artifacts in ${RUN}:"
ls -1 "${RUN}" "${RUN}/figures" 2>/dev/null
