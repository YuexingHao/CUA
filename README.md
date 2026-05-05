# InteraSkill: From Hard-Coded SKILL.md to Learned Behaviors

**Automatic Skill Discovery for Computer-Using Agents**

Under submission to NeurIPS 2026

[Project Website](website/website.html)

---

## Overview

Computer-using agents today rely on manually written `SKILL.md` files that map actions to fixed UI coordinates. These are brittle, inflexible, and don't scale. **InteraSkill** discovers reusable skills automatically from interaction trajectories — no manual engineering required.

## Key Results

| Model | Type | IW Acc. | WA Acc. | IW Edit Dist. |
|-------|------|---------|---------|---------------|
| SKILL.md | Fixed table | 0.140 | 0.087 | 0.633 |
| Frequency | Most common | 0.349 | 0.285 | 0.480 |
| AWM | Learned transitions | 0.334 | 0.788 | 0.479 |
| Transformer (ours) | Sequence model | 0.349 | 0.410 | 0.480 |
| **Qwen3-8B LoRA (ours)** | **Fine-tuned LLM** | **0.850** | **0.437** | **0.148** |

Zero-shot LLM baselines (no fine-tuning):
| Model | IW Acc. (partial) |
|-------|------------------|
| Llama-3.1-70B | ~30% |
| Gemma-4-31B | ~15% |
| OLMo-3-7B | ~6% |

## Repository Structure

```
CUA2026/
├── interaskill/               # Core pipeline code
│   ├── data.py                # Data loading and featurization
│   ├── segment.py             # Trajectory segmentation
│   ├── discover.py            # Skill discovery (clustering + InfoNCE)
│   ├── compose.py             # Skill composition (MLP, Transformer)
│   ├── evaluate.py            # Metrics and visualization
│   ├── baselines.py           # Baseline implementations
│   ├── finetune_qwen.py       # Qwen3-8B LoRA fine-tuning
│   ├── eval_qwen.py           # Qwen3 evaluation (IW + WA)
│   └── eval_model.py          # Generic model evaluation
│
├── data/                      # Dataset folders, one subfolder per benchmark
│   ├── README.md
│   ├── interaskill/
│   │   ├── raw/                # IW benchmark source files
│   │   ├── processed/          # Parsed and synthetic IW trajectories
│   │   └── conversations/      # IW train/validation conversation JSONL
│   ├── webarena/
│   │   ├── trajectories/
│   │   └── conversations/
│   ├── browsecomp_plus/
│   │   ├── conversations/
│   │   └── repo/               # Optional external checkout; ignored by git
│   ├── mind2web/cache/
│   └── workarena/
│       ├── trajectories/
│       ├── conversations/
│       └── workarena_nlp.json
│
├── skills/                    # Structured skill definitions (SKILL.md format)
│   ├── README.md
│   ├── document_edit/SKILL.md     # → anthropic docx
│   ├── presentation_edit/SKILL.md # → anthropic pptx
│   ├── data_transfer/SKILL.md     # → anthropic xlsx
│   ├── export_publish/SKILL.md    # → anthropic pdf
│   ├── search_navigate/SKILL.md
│   ├── review_content/SKILL.md
│   ├── send_message/SKILL.md
│   ├── collaborate/SKILL.md
│   ├── schedule_meeting/SKILL.md
│   ├── organize_files/SKILL.md
│   ├── monitor_status/SKILL.md
│   └── generic_action/SKILL.md
│
├── scripts/
│   ├── data/                  # Data download/conversion/generation scripts
│   │   ├── parse_iw_benchmark.py
│   │   ├── summarize_iw_benchmark.py
│   │   ├── fabricate_trajectories.py
│   │   ├── generate_conversations.py
│   │   ├── download_webarena.py
│   │   └── generate_wa_conversations.py
│   ├── eval/                  # SLURM evaluation scripts
│   │   ├── run_eval_wa.sh         # Qwen3 on WebArena
│   │   ├── run_eval_llama70b.sh   # Llama-3.1-70B zero-shot
│   │   ├── run_eval_gemma31b.sh   # Gemma-4-31B zero-shot
│   │   ├── run_eval_olmo7b.sh     # OLMo-3-7B zero-shot
│   │   ├── run_eval_phi4mini.sh   # Phi-4-mini zero-shot
│   │   ├── run_eval_deepseek_r1.sh# DeepSeek-R1 zero-shot
│   │   ├── run_baselines.sh       # All baselines
│   │   ├── run_webarena.sh        # Full WebArena pipeline
│   │   └── run_eval_only.sh       # Eval-only (skip training)
│   └── train/
│       ├── run_finetune.sh        # Qwen3-8B LoRA training
│       └── run_pipeline.sh        # Full InteraSkill pipeline
│
├── paper/                     # NeurIPS 2026 paper (LaTeX)
│   ├── main.tex
│   └── citation.bib
│
├── results/                   # Experiment outputs
│   ├── metrics.json               # IW pipeline metrics
│   ├── qwen3_eval_metrics.json    # Qwen3 IW results
│   ├── qwen3_eval_metrics_wa.json # Qwen3 WebArena results
│   ├── baseline_metrics_iw.json
│   ├── baseline_metrics_wa.json
│   └── webarena/                  # WebArena-specific results
│
├── website/                   # Project website
│   └── website.html
│
└── docs/
    ├── literature_survey.md
    └── research_proposal.md
```

## Quick Start

```bash
# 1. Run the pipeline on the included synthetic IW data
python -m interaskill.main

# 2. Optional: regenerate synthetic IW data if you have a local raw IW copy
python scripts/data/parse_iw_benchmark.py
python scripts/data/summarize_iw_benchmark.py
python scripts/data/fabricate_trajectories.py --num 500 --seed 42
python scripts/data/generate_conversations.py --num 1500 --seed 42

# 3. Evaluate on WebArena after downloading trajectories
python scripts/data/download_webarena.py --max-trajs 1000 --success-only
python scripts/data/generate_wa_conversations.py --max-trajs 200 --seed 42
sbatch scripts/eval/run_webarena.sh

# 4. Fine-tune Qwen3-8B LoRA
sbatch scripts/train/run_finetune.sh

# 5. Evaluate any model zero-shot
python -m interaskill.eval_model \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset wa --max-convs 200
```

The raw IW benchmark file and intermediate parsed IW files are not distributed
in this repository. The repo includes synthetic IW trajectories and
conversations derived from the benchmark for reproducible clone-and-run
experiments.

## Skill Library

InteraSkill outputs structured `SKILL.md` files following the [anthropics/skills](https://github.com/anthropics/skills) format. Each discovered skill includes:
- Trigger conditions and workflow steps
- Transition probabilities to/from other skills
- Error handling patterns learned from failure trajectories

See `skills/` directory for the 12 discovered skill definitions.

## Citation

```bibtex
@article{interaskill2026,
  title={From Hard-Coded {SKILL.md} to Learned Behaviors: Automatic Skill Discovery for Computer-Using Agents},
  author={Anonymous},
  journal={NeurIPS 2026 Submission},
  year={2026}
}
```
