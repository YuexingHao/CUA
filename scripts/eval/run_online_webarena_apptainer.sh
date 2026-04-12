#!/bin/bash
#SBATCH --job-name=online-wa
#SBATCH --partition=pi_mghassem
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=150gb
#SBATCH --time=24:00:00
#SBATCH --output=results/slurm_logs/slurm-online-wa-%j.out

# ══════════════════════════════════════════════════════════════════════
# InteraSkill Online WebArena Evaluation via Apptainer
#
# This script:
#   1. Launches WebArena services as Apptainer instances on the compute node
#   2. Waits for all services to be ready
#   3. Runs the InteraSkill agent against live WebArena
#   4. Cleans up services on exit
#
# Prerequisites:
#   bash scripts/setup_webarena_apptainer.sh  (one-time, on login node)
# ══════════════════════════════════════════════════════════════════════

set -e

echo "=== InteraSkill Online WebArena (Apptainer) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Date: $(date)"
echo ""

cd /orcd/home/002/yuexing/CUA2026
source ~/miniconda/etc/profile.d/conda.sh
conda activate base
module load apptainer/1.1.9
export PYTHONUNBUFFERED=1

SIF_DIR="/orcd/compute/mghassem/001/gobi1/huggingface/datasets/webarena_containers"
LOG_DIR="results/slurm_logs/webarena_services"
mkdir -p "$LOG_DIR"

# ── Cleanup on exit ─────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "=== Stopping WebArena services ==="
    for name in shopping shopping_admin reddit gitlab wikipedia map; do
        apptainer instance stop "wa_${name}" 2>/dev/null && \
            echo "  Stopped wa_${name}" || true
    done
    echo "Cleanup done."
}
trap cleanup EXIT

# ── Port assignments ────────────────────────────────────────────────
# Apptainer shares host network, so containers bind directly.
# Using high ports to avoid conflicts on shared compute nodes.
SHOPPING_PORT=17770
SHOPPING_ADMIN_PORT=17780
REDDIT_PORT=19999
GITLAB_PORT=18023
WIKIPEDIA_PORT=18888
MAP_PORT=13000

# ── Launch services ─────────────────────────────────────────────────
# Apptainer on HPC typically shares the host network namespace,
# so services bind directly to the host ports. No --net needed.
echo "=== Starting WebArena services ==="

launch_service() {
    local name=$1
    local sif=$2
    local port=$3
    local internal_port=${4:-80}

    if [ ! -f "${SIF_DIR}/${sif}" ]; then
        echo "  [SKIP] ${name}: ${SIF_DIR}/${sif} not found"
        return 1
    fi

    echo "  Starting ${name} on port ${port}..."
    # Run as background instance; host network by default on HPC
    # The container's web server listens on internal_port which maps
    # to the same port on the host (shared network namespace).
    # We set env vars to override default ports where possible.
    apptainer instance start \
        --writable-tmpfs \
        --env "PORT=${port}" \
        --env "LISTEN_PORT=${port}" \
        "${SIF_DIR}/${sif}" "wa_${name}" \
        > "${LOG_DIR}/${name}.log" 2>&1

    echo "  [OK] wa_${name} -> localhost:${port}"
}

launch_service shopping webarena_shopping.sif $SHOPPING_PORT 80
launch_service shopping_admin webarena_shopping_admin.sif $SHOPPING_ADMIN_PORT 80
launch_service reddit webarena_reddit.sif $REDDIT_PORT 80
launch_service gitlab webarena_gitlab.sif $GITLAB_PORT 8023
launch_service wikipedia webarena_wikipedia.sif $WIKIPEDIA_PORT 80
launch_service map webarena_map.sif $MAP_PORT 3000

# ── Wait for services ───────────────────────────────────────────────
echo ""
echo "Waiting for services to be ready..."

wait_for_service() {
    local name=$1
    local port=$2
    local max_wait=${3:-120}
    local waited=0

    while [ $waited -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/" 2>/dev/null | grep -qE "^[23]"; then
            echo "  [READY] ${name} (${waited}s)"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "  [TIMEOUT] ${name} after ${max_wait}s"
    return 1
}

wait_for_service shopping $SHOPPING_PORT 120
wait_for_service shopping_admin $SHOPPING_ADMIN_PORT 120
wait_for_service reddit $REDDIT_PORT 120
wait_for_service gitlab $GITLAB_PORT 300  # GitLab is slow to start
wait_for_service wikipedia $WIKIPEDIA_PORT 120
wait_for_service map $MAP_PORT 120

# ── Set environment variables for BrowserGym ────────────────────────
export WA_SHOPPING="http://localhost:${SHOPPING_PORT}"
export WA_SHOPPING_ADMIN="http://localhost:${SHOPPING_ADMIN_PORT}"
export WA_REDDIT="http://localhost:${REDDIT_PORT}"
export WA_GITLAB="http://localhost:${GITLAB_PORT}"
export WA_WIKIPEDIA="http://localhost:${WIKIPEDIA_PORT}"
export WA_MAP="http://localhost:${MAP_PORT}"
export WA_HOMEPAGE="http://localhost:${SHOPPING_PORT}"

echo ""
echo "=== Environment variables set ==="
echo "  WA_SHOPPING=$WA_SHOPPING"
echo "  WA_REDDIT=$WA_REDDIT"
echo "  WA_GITLAB=$WA_GITLAB"
echo "  WA_WIKIPEDIA=$WA_WIKIPEDIA"
echo "  WA_MAP=$WA_MAP"
echo ""

# ── Install BrowserGym if needed ────────────────────────────────────
python -c "import browsergym.core" 2>/dev/null || {
    echo "Installing BrowserGym..."
    pip install browsergym-core browsergym-webarena 2>/dev/null || pip install browsergym
    playwright install chromium
}

MODEL="Qwen/Qwen3-8B"
LORA_ADAPTER="results/qwen3_lora/final_adapter"

# ══════════════════════════════════════════════════════════════════════
# RUN AGENT
# ══════════════════════════════════════════════════════════════════════

# ── 1. LoRA agent (main result) ─────────────────────────────────────
echo "=== [1/3] Qwen3-8B LoRA — online WebArena (50 tasks) ==="
python -m interaskill.agent_online \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --benchmark webarena --max-tasks 50 --max-steps 15

# ── 2. Zero-shot baseline ──────────────────────────────────────────
echo ""
echo "=== [2/3] Qwen3-8B zero-shot — online WebArena (50 tasks) ==="
python -m interaskill.agent_online \
    --model "$MODEL" \
    --benchmark webarena --max-tasks 50 --max-steps 15

# ── 3. Ablation: no skill stage ─────────────────────────────────────
echo ""
echo "=== [3/3] Ablation: direct action, no skill (50 tasks) ==="
python -m interaskill.agent_online \
    --model "$MODEL" --adapter "$LORA_ADAPTER" \
    --benchmark webarena --max-tasks 50 --max-steps 15 \
    --grounding-mode direct

echo ""
echo "=== All experiments complete ==="
echo "Date: $(date)"
