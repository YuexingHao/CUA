# Data Layout

Datasets live under one top-level `data/` directory, with one subfolder per
benchmark or source.

```text
data/
  interaskill/
    raw/                         # Optional IW source file location; raw IW is not distributed
    processed/                   # Public synthetic IW trajectories; private parsed IW files stay local
    conversations/               # IW train/validation JSONL conversations
  webarena/
    trajectories/                # WebArena trajectory JSON
    conversations/               # WebArena conversation JSONL
  browsecomp_plus/
    conversations/               # BrowseComp+ conversation JSONL
    repo/                        # Optional external BrowseComp+ checkout
  mind2web/
    cache/                       # Cached Mind2Web split files
  workarena/
    trajectories/                # Optional WorkArena rollout JSON
    conversations/               # Generated WorkArena conversation JSONL
    workarena_nlp.json           # Generated text-only WorkArena-NLP benchmark; ignored by git
```

The public repository includes synthetic IW trajectories and conversations, not
the raw IW benchmark file or intermediate parsed IW files. Keep any local raw
copy at `data/interaskill/raw/iw-benchmark-examples.json`; that file is ignored
by git.

Data preparation scripts are in `scripts/data/`. Paths are centralized in
`interaskill.paths`, so Python modules should import path constants instead of
hard-coding `data/*.json` locations.

Typical synthetic-data rebuild if you have the raw IW file locally:

```bash
python scripts/data/parse_iw_benchmark.py
python scripts/data/summarize_iw_benchmark.py
python scripts/data/fabricate_trajectories.py --num 500 --seed 42
python scripts/data/generate_conversations.py --num 1500 --seed 42
```
