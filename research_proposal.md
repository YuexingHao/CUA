# Research Proposal: From Hard-Coded SKILL.md to Learned Behaviors via User Interaction

**Venue:** NeurIPS 2026  
**Testbed:** WebArena (web-arena-x/webarena)  
**Timeline:** ~4 weeks (April–May 2026)

---

## 1. Problem Statement

Current computer-using agents (CUAs) rely on hard-coded skill specifications (SKILL.md) that map semantic intents to fixed UI actions. This paradigm is:
- **Brittle** — UI changes break predefined coordinate mappings
- **Inflexible** — new interaction patterns require manual engineering  
- **Non-personalized** — agents cannot adapt to individual user workflows

Recent work (AWM, SkillWeaver, AgentTrek) has begun addressing skill reuse, but none learns skills **from user interactions in a closed loop**:

| Approach | Skill Source | Online Learning | Self-Improving |
|----------|-------------|----------------|----------------|
| SKILL.md (baseline) | Manual engineering | No | No |
| AWM (Wang et al.) | Offline trajectory mining | Partial (online variant) | No |
| SkillWeaver (Apr 2025) | Agent self-exploration | No | No |
| AgentTrek (ICLR 2025) | Web tutorials | No | No |
| **Ours** | **Simulated interaction trajectories** | **Yes** | **Yes** |

## 2. Core Hypothesis

> **H:** An agent that discovers and encodes skills from user interaction trajectories can match or exceed the performance of hard-coded SKILL.md agents on WebArena, while exhibiting superior generalization to unseen tasks and continuous improvement over time.

### Sub-hypotheses:
- **H1 (Emergence):** Natural clusters in user trajectory space correspond to semantically meaningful, reusable skills
- **H2 (Efficiency):** Learned skills reduce the number of primitive actions needed per task (fewer steps = higher success)
- **H3 (Generalization):** Skills learned on one WebArena domain (e.g., e-commerce) transfer to others (e.g., GitLab, CMS)
- **H4 (Self-Improvement):** Agents that iteratively refine skills from their own simulated interactions outperform static-skill agents over successive episodes

## 3. Proposed Framework: InteraSkill

A three-phase pipeline that maps directly to the theory in our paper:

### Phase 1: Simulated Interaction & Trajectory Segmentation
- Seed from sample JSON trajectory files; run baseline agent on WebArena to generate additional trajectories
- Record all interactions as structured JSON logs (screenshots, DOM states, actions, timing)
- Segment trajectories at action discontinuities: `Δa_t = ||a_t - a_{t-1}||₂ > θ`
- Each segment = candidate skill instance

### Phase 2: Skill Discovery & Embedding
- Cluster segments via Wasserstein distance on action distributions (as in paper Sec. 4.3)
- Learn skill embeddings via InfoNCE contrastive loss: `max_z I(z; τ | s₀)`
- Ground skills multimodally (visual + text + action) for cross-domain transfer
- Produce a **learned skill library** — the replacement for SKILL.md

### Phase 3: Hierarchical Skill Composition & Self-Improving Loop
- Train hierarchical policy: high-level selects skills `π_H(z | b_t)`, low-level executes `π_L(a | s_t, z)`
- **Self-improving simulation loop:** after each batch of simulated episodes, update skill embeddings and discover new skills from the agent's own trajectories
- Skills accumulate in a persistent **Skill Memory** that grows across episodes (contrast with AWM's static workflow memory)

### Architecture Diagram:
```
Sample JSON Seeds → Baseline Agent Rollouts → Trajectory Recording → Segmentation → Clustering
                                                                                        ↓
                                                                        Skill Embedding (InfoNCE)
                                                                                        ↓
                                    Skill Memory ← Self-Improving Loop ← Hierarchical Policy
                                         ↓                                       ↑
                                Skill Composition ──────────────────→ Task Execution (WebArena)
                                                                                  ↓
                                                                     New Trajectories (feedback)
```

## 4. Experimental Design on WebArena

### 4.1 Baselines
1. **SKILL.md agent** — Standard WebArena agent with hard-coded action primitives (ReAct + accessibility tree)
2. **AWM agent** — Agent Workflow Memory with offline-mined workflows
3. **SkillWeaver agent** — Self-explored skill library
4. **GPT-4o / Claude baseline** — Zero-shot prompting (no skill library)

### 4.2 Our Conditions
1. **InteraSkill-Offline** — Skills learned from a fixed dataset of user trajectories (ablation)
2. **InteraSkill-Online** — Full online learning loop with continuous skill discovery
3. **InteraSkill-Transfer** — Train on 3 WebArena domains, test on remaining 3

### 4.3 Evaluation Metrics
- **Task success rate** (WebArena's primary metric — functional correctness)
- **Generalization gap** — Success on held-out domains vs. training domains
- **Skill reuse rate** — % of tasks solved using previously discovered skills
- **Trajectory efficiency** — Average number of primitive actions per successful task
- **Online improvement curve** — Success rate as a function of # simulated episodes
- **Skill interpretability** — Human agreement on semantic labels for discovered skills

### 4.4 WebArena Domain Split
| Training Domains | Test Domains (held-out) |
|-----------------|----------------------|
| OneStopShop (e-commerce) | GitLab |
| Reddit (social forum) | OpenStreetMap |
| Shopping Admin (CMS) | Wikipedia |

### 4.5 Data Collection Protocol (Simulation-Only)
All data is collected via simulated agent runs — no human demonstrations required.

1. **Phase A (Seed):** Bootstrap from sample JSON trajectory files (existing interaction logs). Parse into unified trajectory format $(x, y, a, d)$ with DOM state snapshots.
2. **Phase B (Exploration):** Run a baseline agent (ReAct + GPT-4o) on WebArena training domains. Record all interaction trajectories as JSON logs (screenshots, DOM trees, actions, rewards).
3. **Phase C (Online Skill Discovery):** Deploy InteraSkill agent on training domains. The agent discovers skills from its own interaction trajectories and iteratively refines them across episodes — a self-improving simulation loop with no human in the loop.
4. **Phase D (Eval):** Freeze skills, evaluate on all 6 domains (812 WebArena tasks).

## 5. Connecting Theory to Experiments

The paper's theoretical contributions map directly to measurable outcomes:

| Theory (paper) | Experiment |
|----------------|-----------|
| Trajectory segmentation (Sec 4.3, Step 1) | Measure segmentation quality via cluster coherence and downstream task success |
| InfoNCE skill embedding (Sec 4.1) | Measure cluster purity and silhouette scores of learned skills |
| Cross-domain transfer bound (Sec 4.5) | Measure generalization gap across WebArena domains |
| Hierarchical composition (Sec 3.3) | Measure task success with composed vs. single skills |
| Compositionality guarantee (Sec 4.4) | Verify I(z_i; τ) high and I(z_i; z_j | τ) low for learned skills |

## 6. Timeline (4 Weeks)

| Week | Tasks |
|------|-------|
| **Week 1** (Apr 3-10) | Literature survey finalized. WebArena cloned & running. Parse sample JSON seeds. |
| **Week 2** (Apr 10-17) | Implement trajectory recording + segmentation + clustering. Run baseline agent rollouts for seed trajectories. Run SKILL.md baseline. |
| **Week 3** (Apr 17-24) | Implement InfoNCE skill embedding. Train InteraSkill-Offline. Run AWM + SkillWeaver baselines. Begin self-improving simulation loop. |
| **Week 4** (Apr 24-May 1) | Full evaluation on all 812 tasks. Cross-domain transfer experiments. Write results into paper. |

## 7. Expected Contributions

1. **InteraSkill framework** — First system for online skill discovery from simulated agent interactions in CUA
2. **Self-improving skill memory** — Persistent, growing skill library that improves across simulated episodes
3. **WebArena experiments** — Comprehensive comparison against AWM, SkillWeaver, and SKILL.md baselines
4. **Theory validation** — Empirical verification of the information-theoretic skill discovery framework

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Insufficient seed data | Use sample JSON files as warm-start; run additional baseline agent rollouts; use AWM's offline trajectories if needed |
| WebArena setup complexity on HPC | Use Docker/Singularity containers; fallback to VisualWebArena if needed |
| Online learning too slow to converge in 4 weeks | Pre-train on offline data, only fine-tune online; report learning curves even if not converged |
| Cross-domain transfer underwhelms | Report per-domain results; analyze which skill types transfer and which don't |

## 9. Connection to QBR Demo

The QBR Agent demo (qbr_agent_combined.html) serves as a **motivating case study** in the paper introduction:
- It demonstrates the real-world need (M365 workflow automation)
- Line 1506: "Corrections and preferences the user gives naturally — no SKILL.md, no prompt engineering required" — this is exactly InteraSkill's value proposition
- The QBR demo's 5-layer architecture (Planner → Executor → Reflector → Memory → Learner) maps to our hierarchical framework
- Can be referenced as an industry deployment scenario that motivates the WebArena experiments
