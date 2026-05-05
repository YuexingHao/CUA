#!/bin/bash
# Setup BrowserGym with WebArena integration for InteraSkill online evaluation
#
# Prerequisites:
#   - conda environment with Python >= 3.10
#   - GPU node with internet access (for model downloads)
#
# Usage:
#   bash scripts/setup_browsergym.sh

set -e
echo "=== Setting up BrowserGym for InteraSkill ==="

# ── 1. Install BrowserGym packages ──────────────────────────────────
echo ""
echo "[1/4] Installing BrowserGym..."
pip install browsergym-core browsergym-webarena 2>/dev/null || \
    pip install browsergym  # fallback to full install

# ── 2. Install Playwright browser ───────────────────────────────────
echo ""
echo "[2/4] Installing Playwright Chromium..."
playwright install chromium

# ── 3. NLTK data (required by WebArena eval) ────────────────────────
echo ""
echo "[3/4] Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>/dev/null || true

# ── 4. Verify installation ──────────────────────────────────────────
echo ""
echo "[4/4] Verifying installation..."
python -c "
import gymnasium as gym
import browsergym.core
print('  browsergym.core: OK')
try:
    import browsergym.webarena
    print('  browsergym.webarena: OK')
except ImportError:
    print('  browsergym.webarena: NOT INSTALLED (optional)')
print()
print('Installation complete!')
print()
print('=== WebArena Setup (if using WebArena benchmark) ===')
print('WebArena requires running Docker services. Options:')
print()
print('Option A: AWS AMI (easiest)')
print('  Deploy WebArena AMI on t3a.xlarge with 1TB EBS')
print('  All services pre-configured')
print()
print('Option B: Docker locally')
print('  docker run -d -p 8082:80 --name shopping shopping_final_0712')
print('  docker run -d -p 8080:80 --name reddit reddit_final_0712')
print('  docker run -d -p 9001:80 --name gitlab gitlab_final_0712')
print('  ... (see WebArena docs)')
print()
print('Then set environment variables:')
print('  export WA_SHOPPING=\"http://HOST:8082/\"')
print('  export WA_REDDIT=\"http://HOST:8080\"')
print('  export WA_GITLAB=\"http://HOST:9001\"')
print('  export WA_WIKIPEDIA=\"http://HOST:8081/...\"')
print('  export WA_MAP=\"http://HOST:443\"')
print()
print('=== Quick Test (no WebArena needed) ===')
print('  python -m interaskill.agent_online \\\\')
print('    --model Qwen/Qwen3-8B --adapter results/qwen3_lora/final_adapter \\\\')
print('    --url \"https://www.google.com\" --goal \"Search for BrowserGym\" \\\\')
print('    --no-headless')
"
