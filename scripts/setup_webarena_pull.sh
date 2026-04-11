#!/bin/bash
#SBATCH --job-name=wa-pull
#SBATCH --partition=pi_mghassem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=results/slurm_logs/slurm-wa-pull-%j.out

# ══════════════════════════════════════════════════════════════════════
# Pull WebArena-Verified Docker images as Apptainer SIF files
# Must run on a compute node (login node has insufficient resources)
# ══════════════════════════════════════════════════════════════════════

set -e
module load apptainer/1.1.9

echo "=== Pulling WebArena-Verified Images ==="
echo "Node: $(hostname) | Date: $(date)"
echo "Apptainer: $(apptainer --version)"
echo ""

SIF_DIR="/orcd/home/002/yuexing/CUA2026/containers"
mkdir -p "$SIF_DIR"

export APPTAINER_TMPDIR="${SIF_DIR}/tmp"
export APPTAINER_CACHEDIR="${SIF_DIR}/cache"
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

pull_image() {
    local name=$1
    local docker_image=$2
    local sif_path="${SIF_DIR}/webarena_${name}.sif"

    if [ -f "$sif_path" ]; then
        echo "[SKIP] ${name} — already exists ($(du -h "$sif_path" | cut -f1))"
        return 0
    fi

    echo "[PULL] ${name} from docker://${docker_image} ..."
    if apptainer pull "$sif_path" "docker://${docker_image}"; then
        echo "  -> OK: ${sif_path} ($(du -h "$sif_path" | cut -f1))"
    else
        echo "  -> FAILED: ${name}"
        rm -f "$sif_path"
        return 1
    fi
    echo ""
}

# Pull images (start with smallest first)
pull_image wikipedia   am1n3e/webarena-verified-wikipedia
pull_image map         am1n3e/webarena-verified-map
pull_image shopping    am1n3e/webarena-verified-shopping
pull_image shopping_admin am1n3e/webarena-verified-shopping_admin
pull_image reddit      am1n3e/webarena-verified-reddit
pull_image gitlab      am1n3e/webarena-verified-gitlab

# Cleanup
rm -rf "$APPTAINER_TMPDIR"

echo ""
echo "=== Pull Complete ==="
ls -lh "${SIF_DIR}"/webarena_*.sif 2>/dev/null || echo "No SIF files found!"
echo ""
echo "Date: $(date)"
