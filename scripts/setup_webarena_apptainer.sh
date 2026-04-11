#!/bin/bash
# ══════════════════════════════════════════════════════════════════════
# Setup WebArena via Apptainer (Singularity) on HPC cluster
#
# Converts WebArena-Verified Docker images to Apptainer SIF files.
# No Docker daemon needed — Apptainer pulls directly from Docker Hub.
#
# Usage:
#   # On login node (one-time setup, ~30 min for all images):
#   bash scripts/setup_webarena_apptainer.sh
#
#   # Then launch services on a compute node via SLURM:
#   sbatch scripts/eval/run_online_webarena_apptainer.sh
# ══════════════════════════════════════════════════════════════════════

set -e

# Load Apptainer
module load apptainer/1.1.9 2>/dev/null || true

if ! command -v apptainer &>/dev/null; then
    echo "ERROR: apptainer not found. Load with: module load apptainer/1.1.9"
    exit 1
fi

echo "=== WebArena Apptainer Setup ==="
echo "Apptainer: $(apptainer --version)"
echo ""

# Directory for SIF images
SIF_DIR="/orcd/home/002/yuexing/CUA2026/containers"
mkdir -p "$SIF_DIR"

# Use a writable tmp dir (cluster /tmp may be too small)
export APPTAINER_TMPDIR="${SIF_DIR}/tmp"
export APPTAINER_CACHEDIR="${SIF_DIR}/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# ── WebArena-Verified images (optimized, 92% smaller) ──────────────
# Source: https://hub.docker.com/u/am1n3e

IMAGES=(
    "shopping:am1n3e/webarena-verified-shopping"
    "shopping_admin:am1n3e/webarena-verified-shopping_admin"
    "reddit:am1n3e/webarena-verified-reddit"
    "gitlab:am1n3e/webarena-verified-gitlab"
    "wikipedia:am1n3e/webarena-verified-wikipedia"
    "map:am1n3e/webarena-verified-map"
)

echo "Will pull ${#IMAGES[@]} images to ${SIF_DIR}/"
echo ""

for entry in "${IMAGES[@]}"; do
    name="${entry%%:*}"
    image="${entry#*:}"
    sif_path="${SIF_DIR}/webarena_${name}.sif"

    if [ -f "$sif_path" ]; then
        echo "[SKIP] ${name} — already exists: ${sif_path}"
        continue
    fi

    echo "[PULL] ${name} from docker://${image} ..."
    apptainer pull "$sif_path" "docker://${image}"
    echo "  -> Saved: ${sif_path} ($(du -h "$sif_path" | cut -f1))"
    echo ""
done

# Cleanup tmp
rm -rf "$APPTAINER_TMPDIR"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "SIF images in: ${SIF_DIR}/"
ls -lh "${SIF_DIR}"/webarena_*.sif 2>/dev/null
echo ""
echo "Next step: Submit the WebArena evaluation job:"
echo "  sbatch scripts/eval/run_online_webarena_apptainer.sh"
