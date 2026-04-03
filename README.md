# InteraSkill: From Hard-Coded SKILL.md to Learned Behaviors

**Automatic Skill Discovery for Computer-Using Agents**

Under Submission

Interactive demo available: https://yuexinghao.github.io/CUA/website/website.html

---

## Overview

Computer-using agents (CUAs) today rely on manually written `SKILL.md` files that map actions to fixed UI coordinates. These are brittle, inflexible, and don't scale. **InteraSkill** proposes an alternative: agents that discover reusable skills automatically from interaction trajectories — no manual engineering required.

### Key Idea

```
Raw Interaction Trajectories
    → Trajectory Segmentation (action discontinuities)
    → Wasserstein Clustering (group similar segments)
    → InfoNCE Skill Embedding (contrastive learning)
    → Hierarchical Composition (options framework)
    → Reusable Skill Library (replaces SKILL.md)
```

### What Makes This Different

| Approach | Skill Source | Online | Interaction-Derived | Transferable |
|----------|-------------|--------|---------------------|-------------|
| SKILL.md | Manual engineering | No | No | No |
| AWM (Wang et al.) | Trajectory mining | Partial | No | Yes |
| SkillWeaver (Zheng et al.) | Self-exploration | No | No | Yes |
| ICAL (Sarch et al.) | Demos + feedback | Yes | Partial | No |
| **InteraSkill (Ours)** | **Interaction trajectories** | **Yes** | **Yes** | **Yes** |

## Repository Structure

```
CUA2026/
├── README.md
├── .gitignore
├── data/
│   ├── iw-benchmark-examples.json  # IW Benchmark: 45 real M365 workflows
│   ├── parse_iw_benchmark.py       # Parse raw benchmark → trajectory format
│   ├── summarize_iw_benchmark.py   # Condense + extract 12 skill templates
│   └── fabricate_trajectories.py   # Generate synthetic trajectories from templates
├── paper/
│   ├── main.tex                    # NeurIPS 2026 paper (LaTeX)
│   ├── citation.bib                # Bibliography (37 verified citations)
│   ├── neurips_2026.sty            # NeurIPS style file
│   ├── Fig/                        # Figures
│   └── diagram.jpg
├── website/
│   └── website.html                # Interactive project website
├── literature_survey.md            # 37-paper literature survey
└── research_proposal.md            # Experiment design for WebArena evaluation
```

## Data Pipeline

### 1. Parse IW Benchmark

Converts 45 real-world information worker workflows (Word, PowerPoint, Teams, Outlook) into structured trajectories.

```bash
cd data
python parse_iw_benchmark.py
```

**Input:** `iw-benchmark-examples.json` (45 categories x 2 workflows = 90 trajectories)  
**Output:** `parsed_trajectories.json`

### 2. Summarize & Extract Skill Templates

Condenses 20-60 sub-tasks per trajectory into 3-7 high-level steps and discovers 12 reusable skill templates.

```bash
python summarize_iw_benchmark.py
```

**Output:**
- `iw_summary.json` — condensed trajectory summaries
- `iw_skill_templates.json` — 12 skill templates with action sequences

**Discovered Skills:**

| Skill | Frequency | Actions |
|-------|-----------|---------|
| `document_edit` | 33.4% | click → select → type → format → save |
| `send_message` | 12.0% | compose → recipient → write → send |
| `review_content` | 11.8% | open → read → comment → write → submit |
| `schedule_meeting` | 11.4% | calendar → new → title → attendees → time → send |
| `collaborate` | 5.9% | open chat → message → share → react |
| + 7 more | | |

### 3. Fabricate Synthetic Trajectories

Generates realistic synthetic trajectories using the discovered skill templates. Each trajectory has normalized coordinates, 5% action failure rate, and realistic duration noise.

```bash
python fabricate_trajectories.py --num 500 --seed 42
```

**Output:** `fabricated_trajectories.json`

| Stat | Value |
|------|-------|
| Trajectories | 500 (balanced low/medium/high) |
| Actions per trajectory | 7-40 (mean 20.8) |
| Segments per trajectory | 2-7 (mean 4.2) |
| Duration | 3.5-83.7 min (mean 35.9) |

### Trajectory Format

```json
{
  "trajectory_id": "fab_medium_0016",
  "objective": "Run customer feedback analysis and team action planning",
  "complexity": "medium",
  "apps_involved": ["outlook", "teams"],
  "total_duration_minutes": 39.9,
  "segments": [
    {
      "skill_type": "send_message",
      "actions": [
        {"action_type": "click", "target": "compose_button", "coordinates": {"x": 0.45, "y": 0.08}},
        {"action_type": "type", "target": "recipient_field", "text_length": 42},
        {"action_type": "type", "target": "message_body", "text_length": 156},
        {"action_type": "click", "target": "send_button", "coordinates": {"x": 0.82, "y": 0.12}}
      ]
    }
  ],
  "skill_sequence": ["send_message", "document_edit", "review_content", "collaborate"]
}
```

## Paper

The LaTeX source is in `paper/main.tex` with 37 verified citations in `citation.bib`. Key sections:

- **Related Work** — Comparison table against AWM, SkillWeaver, ICAL, AutoManual, etc.
- **Problem Formulation** — Hierarchical MDP with continuous action space
- **Skill Discovery** — InfoNCE + Wasserstein clustering + Jacobian regularization
- **Skill Composition** — Options framework with learned termination conditions
- **Cross-Domain Transfer** — CLIP-based semantic state matching

## Website

`website/website.html` is a self-contained interactive website with:

- **Overview tab** — Problem statement, gap analysis, pipeline, results
- **Interaction Demos tab** — WebArena scenarios (e-commerce, forum, GitLab transfer), IW benchmark real data, 6 failure modes with recovery, interactive chat demo
- **Theory tab** — Plain-language math explanations with analogies

Open in any browser: `open website/website.html`

## Evaluation Plan

**Testbed:** [WebArena](https://github.com/web-arena-x/webarena) (812 tasks, 6 domains)

| Training | Test (Unseen) |
|----------|--------------|
| E-commerce | GitLab |
| Reddit | OpenStreetMap |
| Shopping Admin | Wikipedia |

**Baselines:** SKILL.md agent, AWM, SkillWeaver, GPT-4o zero-shot  
**Metrics:** Task success rate, generalization gap, skill reuse rate, trajectory efficiency

See `research_proposal.md` for the full experiment design.

## Citation

```bibtex
@article{interaskill2026,
  title={From Hard-Coded {SKILL.md} to Learned Behaviors: Automatic Skill Discovery for Computer-Using Agents},
  author={Anonymous},
  journal={NeurIPS 2026 Submission},
  year={2026}
}
```

## License

This project is for research purposes. Please contact the authors for usage inquiries.
