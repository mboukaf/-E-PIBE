#!/bin/bash
# Push the project to the cluster without breaking its Python environment.
#
#   scripts/slurm/sync.sh mboukaf@margaret02.saclay.inria.fr:Automatica
#
# .venv is excluded deliberately.  A virtualenv contains a platform-specific
# interpreter binary; copying a macOS one onto a Linux node replaces a working
# environment with something that fails as "cannot execute binary file"
# (exit 126), or as "python: command not found" (exit 127) once PATH points at
# it.  Build the venv once on the cluster and leave it alone.
#
# outputs/ is excluded too: it is large, it is generated, and pushing a stale
# copy can overwrite results computed on the cluster.
set -euo pipefail
cd "$(dirname "$0")/../.."

DEST="${1:?usage: sync.sh user@host:path}"

rsync -avz --progress \
    --exclude '.venv/' \
    --exclude 'outputs/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude '*.pt' \
    ./ "${DEST}/"

echo
echo "pushed. On the cluster, verify the environment before submitting:"
echo "  ./.venv/bin/python scripts/train_pibe.py \\"
echo "      --config configs/automatica_n3v2_big_v2.yaml --dry-run"
