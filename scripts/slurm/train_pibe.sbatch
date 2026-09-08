#!/bin/bash
# ---------------------------------------------------------------------------
# PIBE large run -- Inria Margaret (margaret02.saclay.inria.fr)
#
#   scripts/slurm/submit.sh 0-5          <-- use this
#   sbatch --array=0-5 scripts/slurm/train_pibe.sbatch   (needs logs/ to exist)
#
# NOTE: SLURM opens the --output path below BEFORE this script runs, so the
# `mkdir -p logs` further down is too late.  If logs/ is absent the job fails
# instantly, with no output and no trace in squeue.  submit.sh creates it
# first; logs/.gitkeep keeps it present after a clone.
#
# v2: the first attempt collapsed after ~8 h (see Docs/CLUSTER_POSTMORTEM.md).
# Every parameter head and one state decoder saturated against its admissible
# box; the tanh gradient fell to 4e-16 and the remaining 16 h made no progress.
# The configs now set architecture.saturation_limit, and the run directories
# carry a _v2 suffix so the old, wrecked state.pt files are not resumed.
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
# a 7-day limit, and no queue for scarce GPUs.  The first attempt measured
# 0.77 s per global step at 16 threads -- far above the 209 ms this estimate
# originally assumed -- which is why its 260k-step schedule needed 55.6 h and
# died at the 24 h wall.  v2's schedule is 90k steps, about 19 h at the
# measured rate.  That fits, but not generously: raise --time if the partition
# allows, or move to a GPU and restore batch_size 128.
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
# Which generation of the configs to run.  Override at submission:
#   PIBE_VERSION=v2 sbatch --array=0-5 scripts/slurm/train_pibe.sbatch
#   v2: theta fixed across trajectories; x0 and the disturbance vary.
#   v3: theta, x0 and the disturbance all vary per trajectory (default).
VERSION="${PIBE_VERSION:-v3}"

if (( IDX % 2 == 0 )); then
    CONFIG="configs/automatica_n3v2_big_${VERSION}.yaml";          TAG="w5"
else
    CONFIG="configs/automatica_n3v2_big_lowomega_${VERSION}.yaml"; TAG="w10"
fi
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }
# The _v2 suffix is not cosmetic.  The first attempt left a state.pt in
# outputs/big_${TAG}_s${SEED}, and train_pibe.py resumes from state.pt
# automatically -- pointing the corrected configs at the old directories would
# silently continue the collapsed run instead of starting the fixed one.
RUN="outputs/big_${TAG}_${VERSION}_s${SEED}"

mkdir -p logs "${RUN}"

# One BLAS thread per allocated core.  Oversubscription is a common reason a
# cluster run ends up slower than a laptop.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

echo "host      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION:-?}"
echo "job       : ${SLURM_JOB_ID:-none}  array task ${IDX}"
echo "config    : ${CONFIG}  (version ${VERSION})"
echo "seed      : ${SEED}"
echo "run dir   : ${RUN}"
echo "threads   : ${OMP_NUM_THREADS}"
echo "resuming  : $([ -f "${RUN}/state.pt" ] && echo yes || echo 'no (fresh start)')"
if [ -f "${RUN}/state.pt" ]; then
    echo "            (found ${RUN}/state.pt -- this job CONTINUES that run."
    echo "             To start over instead, delete it or pass --no-resume.)"
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "gpu       : none (CPU run)"
echo

# --- environment ------------------------------------------------------------
# A venv's `activate` only edits PATH; it does not guarantee the interpreter it
# points at exists on THIS node.  If the venv was built against a module-provided
# python, a compute node without that module loaded gives
# "python: command not found" (exit 127) one second into the job.  So load the
# module first, then verify the interpreter before anything expensive starts.
#
# Uncomment and adjust; `module avail python` lists what Margaret offers.
# module load python/3.12
# module load cuda/12.4          # only for a GPU partition

source .venv/bin/activate

# Call the interpreter by path rather than relying on PATH, and check it runs.
PYTHON="${PYTHON:-$PWD/.venv/bin/python}"
if ! "${PYTHON}" -V >/dev/null 2>&1; then
    echo "FATAL: ${PYTHON} does not run on $(hostname)." >&2
    echo "       .venv/bin/python -> $(readlink -f "${PWD}/.venv/bin/python" 2>/dev/null || echo '?')" >&2
    echo "       Load the matching python module above, or rebuild the venv" >&2
    echo "       with an interpreter available on the compute nodes." >&2
    exit 1
fi
echo "python    : ${PYTHON} ($(${PYTHON} -V 2>&1))"
"${PYTHON}" -c "import torch; print(f'torch     : {torch.__version__}  cuda={torch.cuda.is_available()}')"
echo

srun "${PYTHON}" scripts/train_pibe.py \
    --config "${CONFIG}" \
    --set seed="${SEED}" training.seed="${SEED}" \
    --output-dir "${RUN}" \
    --threads "${OMP_NUM_THREADS}"

echo
echo "done. artifacts in ${RUN}:"
ls -1 "${RUN}" "${RUN}/figures" 2>/dev/null
