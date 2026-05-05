# Data Preparation Scripts

Run these commands from the repository root. The raw IW benchmark file is not
distributed; the IW parse/summarize steps require a local copy at
`data/interaskill/raw/iw-benchmark-examples.json`.

```bash
# IW / InteraSkill
python scripts/data/parse_iw_benchmark.py
python scripts/data/summarize_iw_benchmark.py
python scripts/data/fabricate_trajectories.py --num 500 --seed 42
python scripts/data/generate_conversations.py --num 1500 --seed 42

# WebArena
python scripts/data/download_webarena.py --max-trajs 1000 --success-only
python scripts/data/generate_wa_conversations.py --max-trajs 200 --seed 42

# Mind2Web cache
python scripts/data/download_mind2web.py --max-tasks 500

# BrowseComp+
python scripts/data/generate_bc_conversations.py --max-queries 200 --seed 42

# WorkArena
python scripts/data/generate_workarena_conversations.py \
  --input data/workarena/trajectories/workarena_trajectories.json \
  --output data/workarena/conversations/workarena_conversations.jsonl \
  --max-trajs 200
```
