#!/bin/bash
# Submit the PIBE-vs-EPIBE sensor-bias array (train_biasid.sbatch).
#
#   scripts/slurm/submit_biasid.sh          # all 12 tasks: 3 seeds x {PIBE, EPIBE} x {Gaussian, bias 0.5}
#   scripts/slurm/submit_biasid.sh 0-3      # seed 0 only
#
# Independent of submit.sh / train_pibe.sbatch: different job name, logs
# (logs/biasid_*) and run directories (outputs/biasid_big_*).
set -euo pipefail

cd "$(dirname "$0")/../.."
ARRAY="${1:-0-11}"

mkdir -p logs outputs

for cfg in configs/automatica_n3v2_biasid_big_pibe.yaml configs/automatica_n3v2_biasid_big_epibe.yaml; do
    [ -f "$cfg" ] || { echo "missing config: $cfg" >&2; exit 1; }
done

# A leftover resume file makes a task continue instead of starting over.
for d in outputs/biasid_big_*; do
    [ -d "$d" ] || continue
    for f in state.pt stages.json; do
        [ -e "$d/$f" ] && echo "NOTE: $d/$f exists -- that task will RESUME, not restart."
    done
    [ -e "$d/bank.pt" ] && [ ! -e "$d/state.pt" ] && [ ! -e "$d/stages.json" ] && \
        echo "NOTE: $d already holds a finished run -- resubmitting it RETRAINS and overwrites it."
done

echo "submitting biasid array ${ARRAY}"
sbatch --array="${ARRAY}" scripts/slurm/train_biasid.sbatch
