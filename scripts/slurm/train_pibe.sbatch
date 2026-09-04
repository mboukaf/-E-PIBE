#!/bin/bash
# ---------------------------------------------------------------------------
# PIBE large run -- Inria Margaret (margaret02.saclay.inria.fr)
#
#   sbatch --array=0-5 scripts/slurm/train_pibe.sbatch
#
# Array index -> (config, seed):
#     idx 0,1 -> seed 0    idx 2,3 -> seed 1    idx 4,5 -> seed 2
#     even idx -> Omega = 2pi/5   (baseline, matches the spec)
#     odd  idx -> Omega = 2pi/10  (low-frequency variant)
#
#   --array=0-1     one seed of each frequency
#   --array=0-5     three seeds of each  (recommended)
#   --array=0-4:2   baseline only, three seeds
#
# RESUMABLE.  <run>/state.pt is written every `checkpoint_every` steps and a
# restarted job continues from it automatically.  This is why --requeue is
# safe and why the preemptible "-best" partitions are usable: preemption costs
# at most 2000 steps, not the run.  state.pt is deleted on success, so
# resubmitting a finished job retrains rather than exiting immediately.
# ---------------------------------------------------------------------------
#SBATCH --job-name=pibe
#SBATCH --output=logs/pibe_%A_%a.out
#SBATCH --error=logs/pibe_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --requeue
#
# ---------------------------------------------------------------------------
# CPU vs GPU on this cluster
#
# The default above is the `normal` CPU partition: 30 nodes idle at last look,
# a 7-day limit, and no queue for scarce GPUs.  Measured cost is 209 ms per
# global step on 8 threads -> ~15.6 h for the full 260k-step schedule, so it
# fits comfortably.  16 cores are requested here for headroom.
#
# For GPU, the dtype matters more than the card's headline speed.  Every number
# in Docs/IMPLEMENTATION.md was measured in float64, and FP64 throughput splits
# these cards sharply:
#
#     good  (FP64 at 1/2 rate) : A100_40G, V100_32G, H100_96G, P100_16G
#     poor  (FP64 at 1/32)     : A40_48G, RTX6000_24G, 2080TI, 1080TI, 2080
#
# A float64 job on an A40 or 2080Ti will be slower than the CPU partition.
# At last look A100 had 2 free and RTX6000 2 free, while V100/H100/A40 were
# fully allocated -- so a GPU job may queue longer than the run itself takes.
#
# To use a GPU, uncomment the two lines below and set the constraint to a
# FP64-capable card.  Confirm the exact gres/constraint spelling first:
#
#     sinfo -o "%P %G %f" | sort -u
#
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
##SBATCH --constraint=NV_A100_40G
#
# If you can only get an A40/RTX6000/2080Ti, add dtype=float32 to the --set
# list below rather than running float64 on it -- but treat that as a separate
# experiment, since no reported number was measured in float32.
# ---------------------------------------------------------------------------

set -euo pipefail

IDX="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$(( IDX / 2 ))
if (( IDX % 2 == 0 )); then
    CONFIG="configs/automatica_n3v2_big.yaml";          TAG="w5"
else
    CONFIG="configs/automatica_n3v2_big_lowomega.yaml"; TAG="w10"
fi
RUN="outputs/big_${TAG}_s${SEED}"

mkdir -p logs "${RUN}"

# One BLAS thread per allocated core.  Oversubscription is a common reason a
# cluster run ends up slower than a laptop.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "host      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION:-?}"
echo "job       : ${SLURM_JOB_ID:-none}  array task ${IDX}"
echo "config    : ${CONFIG}"
echo "seed      : ${SEED}"
echo "run dir   : ${RUN}"
echo "threads   : ${OMP_NUM_THREADS}"
echo "resuming  : $([ -f "${RUN}/state.pt" ] && echo yes || echo 'no (fresh start)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "gpu       : none (CPU run)"
echo

# --- environment ------------------------------------------------------------
# Adapt to the module system on Margaret; check availability with `module avail`.
# module load python/3.12
# module load cuda/12.4          # only for a GPU partition
source .venv/bin/activate
python -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}')"
echo

srun python scripts/train_pibe.py \
    --config "${CONFIG}" \
    --set seed="${SEED}" training.seed="${SEED}" \
    --output-dir "${RUN}" \
    --threads "${OMP_NUM_THREADS}"

echo
echo "done. artifacts in ${RUN}:"
ls -1 "${RUN}" "${RUN}/figures" 2>/dev/null
