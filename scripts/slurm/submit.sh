#!/bin/bash
# Submit the PIBE array, having first created what SLURM needs to exist.
#
#   scripts/slurm/submit.sh              # three seeds of each frequency
#   scripts/slurm/submit.sh 0-1          # one seed of each
#   scripts/slurm/submit.sh 0-4:2        # baseline frequency only
#
# The reason this wrapper exists: #SBATCH --output=logs/... is opened by SLURM
# *before* the job script runs, so a `mkdir -p logs` inside the script is too
# late.  Without the directory the job fails at startup with an empty elapsed
# time and never appears in squeue -- which looks exactly like the submission
# having been ignored.
set -euo pipefail

cd "$(dirname "$0")/../.."
ARRAY="${1:-0-5}"

mkdir -p logs outputs

for cfg in configs/automatica_n3v2_big_v2.yaml \
           configs/automatica_n3v2_big_lowomega_v2.yaml; do
    [ -f "$cfg" ] || { echo "missing config: $cfg" >&2; exit 1; }
done

# Resuming an old run by accident is the other silent failure mode: the v1
# attempt left state.pt files behind, and train_pibe.py continues from them.
for d in outputs/big_w5_v2_s* outputs/big_w10_v2_s*; do
    [ -e "$d/state.pt" ] || continue
    echo "NOTE: $d/state.pt exists -- that task will RESUME, not restart."
done

echo "submitting array ${ARRAY}"
sbatch --array="${ARRAY}" scripts/slurm/train_pibe.sbatch
