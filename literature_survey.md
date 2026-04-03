# Literature Survey: From Hard-Coded SKILL.md to Learned Behaviors

**For:** NeurIPS 2026 submission  
**Total papers:** 37  
**Year range:** 1999–2026

---

## Theme 1: Skill Discovery & Workflow Learning for CUA Agents

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 1 | Wang et al., "Agent Workflow Memory" | ICML 2025 | Induces reusable workflows from agent trajectories (offline + online). +51.1% on WebArena, reduces step count. | Most directly related. AWM learns workflows from trajectories but via LLM induction, not user corrections. We extend this with interactive, user-grounded skill discovery. |
| 2 | Zheng et al., "SkillWeaver: Web Agents can Self-Improve by Discovering and Honing Skills" | arXiv 2504.07079, Apr 2025 | Agents autonomously discover website functionalities, practice them, distill into reusable Python API skills. +31.8% on WebArena, cross-agent transfer +54.3%. | Key comparison. SkillWeaver discovers via self-play; we discover from user interaction patterns, yielding more user-aligned skills. |
| 3 | Chen et al., "AutoManual: Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning" | NeurIPS 2024 | Planner + Builder agents construct Markdown instruction manuals through exploration. 97.4% ALFWorld, 65.1% WebArena (Reddit). | Closest to "automated SKILL.md" baseline. AutoManual generates text manuals from exploration, not user corrections. |
| 4 | Wang et al., "Voyager: An Open-Ended Embodied Agent with Large Language Models" | arXiv 2305.16291, 2023 | First LLM lifelong learning agent (Minecraft) with growing executable skill library and self-verification. | Foundational skill library concept, but in Minecraft, not GUI/web. We apply to real computer-use settings. |
| 5 | Sarch et al., "ICAL: Continual Learning of Multimodal Agents by Transforming Trajectories into Actionable Insights" | NeurIPS 2024 | VLM agents distill sub-optimal demos + human feedback into cognitive abstractions. VisualWebArena 14.3% → 22.7%. | **Closest to learning from user feedback.** But produces per-instance memory, not reusable transferable skill definitions like ours. |
| 6 | Sodhi et al., "SteP: Stacked LLM Policies for Web Actions" | COLM 2024 | Decomposes web tasks into dynamic stack of specialized LLM policies. WebArena 14.9% → 33.5%. | Demonstrates sub-policy decomposition is effective. Our learned skills are automatically discovered policies for such stacks. |

## Theme 2: Training & RL for Web/GUI Agents

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 7 | Qi et al., "WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL" | ICLR 2025 | Self-evolving curriculum RL. Llama-3.1-8B: 4.8% → 42.4% on WebArena-Lite, surpassing GPT-4-Turbo. | RL from online interaction improves agents; our skills provide complementary structured learning signal. |
| 8 | Bai et al., "DigiRL: Training In-The-Wild Device-Control Agents with Autonomous RL" | NeurIPS 2024 | Offline-to-online RL for device control with VLM evaluator. 17.7% → 67.2% on Android-in-the-Wild. | Agents can learn from real interaction; we focus on extracting declarative skill knowledge from interactions. |
| 9 | Xu et al., "AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials" | ICLR 2025 Spotlight | Three-stage pipeline: harvest tutorials → task specs → VLM replay → verification. $0.55/trajectory. | Mines static tutorials for training data. We go further: learning from dynamic user-agent interactions. |

## Theme 3: Foundation Models for Computer Use

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 10 | Guo et al., "OpAgent: Operator Agent for Web Navigation" | arXiv 2602.13559, Feb 2026 | Multi-agent system (Planner, Grounder, Reflector, Summarizer). 71.6% on WebArena — current SOTA. | Performance ceiling. Can learned skills achieve similar benefits with simpler architectures? |
| 11 | Anthropic, "Introducing Computer Use" | Blog, Oct 2024 | First general-purpose CUA in public beta. Claude interacts via screenshots + cursor + clicks + typing. | Primary substrate model. Our skill learning adds a layer on top of base CUA capability. |
| 12 | OpenAI, "Computer-Using Agent / Operator" | Blog + System Card, Jan 2025 | CUA model: 58.1% WebArena, 87% WebVoyager. Released as Operator product. | Major competing CUA. Motivates need for skill learning to improve reliability. |
| 13 | Yang et al., "UltraCUA: A Foundation Model with Hybrid Action" | arXiv 2510.17790, Oct 2025 | Unifies GUI primitives with programmatic tool calls. +22% on OSWorld, 11% faster. | Hybrid action approach is complementary; learned skills could serve as high-level tool calls. |

## Theme 4: Large-Scale CUA Datasets & Pipelines

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 14 | Wang et al., "OpenCUA: Open Foundations for Computer-Use Agents" | NeurIPS 2025 Spotlight | AgentNet: 22.6K trajectories across 3 OSes, 200+ apps. OpenCUA-72B: 45.0% on OSWorld-Verified. | Large-scale trajectory data our skill discovery could mine. Annotation infra could capture user corrections. |
| 15 | Zhang et al., "AgentOhana: Unified Data and Training Pipeline for Agent Learning" | arXiv 2402.15506, Feb 2024 | Aggregates agent trajectories into standardized format. xLAM-v0.1 with strong cross-benchmark performance. | Unified trajectory format enables cross-environment skill transfer. |
| 16 | Liu et al., "LearnAct: Few-Shot Mobile GUI Agent with Unified Demonstration Benchmark" | arXiv 2504.13805, Apr 2025 | LearnGUI dataset. Single demonstration improves Gemini-1.5-Pro from 19.3% → 51.7% (+198.9%). | **Validates our thesis:** even one demo dramatically helps. We automate reusable skill extraction from such demos. |

## Theme 5: Benchmarks

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 17 | Zhou et al., "WebArena: A Realistic Web Environment for Building Autonomous Agents" | ICLR 2024 | 812 tasks across 6 self-hosted websites. GPT-4: 14.41% vs human 78.24%. | **Primary evaluation benchmark.** |
| 18 | Koh et al., "VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks" | ACL 2024 | 910 visually grounded web tasks. Best VLM: 16.4% vs human 88.7%. | Extended multimodal evaluation. |
| 19 | Deng et al., "Mind2Web: Towards a Generalist Agent for the Web" | NeurIPS 2023 | 2,000+ tasks across 137 websites, 31 domains. MindAct two-stage approach. | Large offline dataset for evaluating skill induction from logged trajectories. |
| 20 | Yao et al., "WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents" | NeurIPS 2022 | Simulated e-commerce, 1.18M products, sim-to-real transfer to Amazon/eBay. | Early benchmark establishing web agent paradigm. Sim-to-real motivates transferable skills. |
| 21 | Xie et al., "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments" | NeurIPS 2024 | First real OS-level benchmark (Ubuntu/Win/Mac), 369 tasks. Best: 12.24% vs human 72.36%. | OS-level evaluation. Tests skill generalization beyond web to full desktop. |
| 22 | Xie et al., "Scaling Computer-Use Grounding via UI Decomposition and Synthesis" (OSWorld-G / Jedi) | arXiv 2505.13227, May 2025 | 564 grounding samples + 4M Jedi dataset. OSWorld 5% → 27%. | Addresses grounding bottleneck; better grounding enables more reliable skill execution. |

## Theme 6: Unsupervised Skill Discovery in RL (Foundational)

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 23 | Eysenbach et al., "Diversity Is All You Need: Learning Skills without a Reward Function" (DIAYN) | ICLR 2019 | Discovers diverse skills by maximizing MI(z; s) using a discriminator, no extrinsic reward. | **Foundational.** Our InfoNCE objective generalizes DIAYN's discriminability. We use trajectory-level MI instead of state-level. |
| 24 | Hansen et al., "Fast Task Inference with Variational Intrinsic Successor Features" (VISR) | ICLR 2020 | Combines successor features with variational skill discovery for zero-shot transfer. | Relevant to transferring GUI skills to new tasks without retraining. |
| 25 | Sharma et al., "Dynamics-Aware Discovery of Skills" (DADS) | ICLR 2020 | Maximizes MI between skills and next-state dynamics, yielding predictable skills in continuous spaces. | Analogous to discovering GUI skills with predictable state transitions (e.g., "open dialog" → specific UI state). |
| 26 | Laskin et al., "CIC: Contrastive Intrinsic Control for Unsupervised Skill Discovery" | ICLR 2022 | Uses InfoNCE to maximize MI between skills and state transitions without explicit discriminator. Scales to high-dim. | **Most directly relevant foundational method.** Our method follows CIC's contrastive formulation, extended to multimodal GUI observations. |
| 27 | Park et al., "LSD: Lipschitz Skill Discovery" | ICML 2023 | Lipschitz-constrained skill discovery for smooth, geometrically structured skill spaces. | Connects to our Wasserstein clustering — both seek metrically meaningful skill organization. |

## Theme 7: Options Framework & Hierarchical RL

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 28 | Sutton et al., "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL" | Artificial Intelligence, 1999 | Foundational options framework: (initiation, policy, termination) triples for temporal abstraction. | **Core formalism.** Each discovered GUI skill is an option with learned initiation and termination. |
| 29 | Bacon et al., "The Option-Critic Architecture" | AAAI 2017 | Policy-gradient theorems for learning options end-to-end, including termination functions. | Enables learning when GUI skills should terminate jointly with skill policies. |

## Theme 8: Contrastive & Mutual Information Methods

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 30 | van den Oord et al., "Representation Learning with Contrastive Predictive Coding" (CPC/InfoNCE) | arXiv 1807.03748, 2018 | Introduces InfoNCE contrastive loss as tractable MI lower bound. | **Core building block** of our skill embedding objective. |
| 31 | Laskin et al., "CURL: Contrastive Unsupervised Representations for RL" | ICML 2020 | Contrastive learning on augmented observations for sample-efficient pixel-based RL. | Contrastive objectives yield useful visual features from pixels — directly relevant to learning from GUI screenshots. |

## Theme 9: Learning from Demonstrations & Human Interaction

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 32 | Ross et al., "A Reduction of Imitation Learning to No-Regret Online Learning" (DAgger) | AISTATS 2011 | Iterative dataset aggregation querying expert under learner's distribution, addressing covariate shift. | User corrections in our online loop are DAgger-style: human feedback under agent-induced states. |
| 33 | Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" (RLHF) | NeurIPS 2022 | Reward model from human preferences + PPO fine-tuning at scale for LLMs. | Paradigm for converting implicit user preferences into trainable reward signal for GUI skills. |
| 34 | Humphreys et al., "A Data-Driven Approach for Learning to Control Computers" | ICML 2022 | Behavioral cloning with transformer over screenshots for click/type/scroll. Large-scale human demo dataset. | Baseline data-collection paradigm. Our skill discovery applies on top of such demonstration datasets. |

## Theme 10: Multimodal Grounding & Meta-Learning

| # | Citation | Venue | Summary | Relation to Our Work |
|---|----------|-------|---------|---------------------|
| 35 | Radford et al., "Learning Transferable Visual Models from Natural Language Supervision" (CLIP) | ICML 2021 | Joint vision-language embeddings via contrastive pre-training. Zero-shot visual recognition + grounding. | CLIP embeddings serve as our observation encoder, grounding skills in semantics aligned with language. |
| 36 | Shaw et al., "From Pixels to UI Actions: A GUI Agent with Backbone Actionable Transformer" (Pix2Act) | NeurIPS 2023 Workshop | End-to-end pixel-to-action mapping for GUIs using ViT backbone. | Pixel-to-action feasibility for GUIs. Our skill discovery layers temporal abstraction on top. |
| 37 | Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (MAML) | ICML 2017 | Learns initialization for few-gradient-step adaptation to new tasks. | Enables rapid adaptation of discovered skills to new users/GUIs with minimal interaction data. |

---

## Key Gap Analysis

| Approach | Skill Source | Online Learning | User-Adaptive | Reusable/Transferable |
|----------|-------------|----------------|---------------|----------------------|
| SKILL.md (manual) | Human engineering | No | No | No (domain-specific) |
| AWM (Wang et al., 2025) | Offline trajectory mining | Partial | No | Yes (workflows) |
| SkillWeaver (Zheng et al., 2025) | Agent self-exploration | No | No | Yes (Python APIs) |
| AutoManual (Chen et al., 2024) | Agent exploration | No | No | Yes (text manuals) |
| ICAL (Sarch et al., 2024) | Demos + human feedback | Yes | Partially | No (per-instance) |
| LearnAct (Liu et al., 2025) | Single demonstration | No | No | No (one-shot) |
| Voyager (Wang et al., 2023) | Self-play (Minecraft) | Yes | No | Yes (code library) |
| **InteraSkill (Ours)** | **User interaction traces** | **Yes** | **Yes** | **Yes (skill embeddings)** |

**The gap:** No existing work combines (1) online skill discovery, (2) from user interactions, (3) producing reusable transferable skill definitions, (4) with continuous improvement. We are the first to unify the reusability of SkillWeaver/AWM with the human-grounded learning signal of ICAL/LearnAct.

---

## Recommended BibTeX Additions

The following papers should be added to `citation.bib` (beyond the 6 already present):

- AWM (Wang et al., 2025) — arXiv:2409.07429
- SkillWeaver (Zheng et al., 2025) — arXiv:2504.07079
- AutoManual (Chen et al., 2024) — arXiv:2405.16247
- Voyager (Wang et al., 2023) — arXiv:2305.16291
- ICAL (Sarch et al., 2024) — arXiv:2406.14596
- WebRL (Qi et al., 2025) — arXiv:2411.02337
- AgentTrek (Xu et al., 2025) — arXiv:2412.09605
- OpAgent (Guo et al., 2026) — arXiv:2602.13559
- OpenCUA (Wang et al., 2025) — arXiv:2508.09123
- LearnAct (Liu et al., 2025) — arXiv:2504.13805
- OSWorld (Xie et al., 2024) — arXiv:2404.07972
- WebArena (Zhou et al., 2024) — arXiv:2307.13854
- VisualWebArena (Koh et al., 2024) — arXiv:2401.13649
- CIC (Laskin et al., 2022)
- DADS (Sharma et al., 2020)
- Option-Critic (Bacon et al., 2017)
- CLIP (Radford et al., 2021)
- DAgger (Ross et al., 2011)
- RLHF (Ouyang et al., 2022)
- CPC/InfoNCE (van den Oord et al., 2018)
- UltraCUA (Yang et al., 2025) — arXiv:2510.17790
