#!/usr/bin/env python3
"""
Generate multi-turn user-agent conversation data from BrowseComp-Plus
trajectories for evaluating models on deep-research skill prediction.

BrowseComp-Plus trajectories consist of interleaved reasoning and search
tool calls. We map these to our skill taxonomy:
  - reasoning steps -> review_content (analyzing/synthesizing information)
  - search tool calls -> search_navigate (querying for information)
  - get_document calls -> review_content (reading specific documents)
  - final output -> export_publish (producing final answer)

Usage:
    python data/generate_bc_conversations.py [--max-queries 200] [--seed 42]
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent
BC_QUERIES = DATA_DIR / "browsecomp-plus/repo/data/browsecomp_queries.jsonl"
BC_RUNS = DATA_DIR / "browsecomp-plus/repo/data/decrypted_run_files/bm25/gpt5.jsonl"
OUTPUT_FILE = DATA_DIR / "bc_conversations.jsonl"

SYSTEM_PROMPT = """\
You are InteraSkill, an AI deep-research agent that helps users answer \
complex questions by searching through documents, analyzing content, \
and synthesizing findings.

You can execute the following skills:
- search_navigate: Search for documents or navigate to information sources
- review_content: Read, analyze, and verify document content
- document_edit: Take notes, summarize, or compile findings
- data_transfer: Extract and organize data from sources
- export_publish: Produce final answers, reports, or summaries
- collaborate: Discuss findings or coordinate with other agents
- generic_action: Other interactions (UI clicks, form fills)

When executing a research task:
1. Plan your search strategy
2. Search for relevant documents
3. Review and analyze findings
4. Synthesize information across sources
5. Produce a final answer with evidence"""

# Map BrowseComp-Plus step types to our skill taxonomy
def map_step_to_skill(step):
    """Map a BrowseComp-Plus trajectory step to an InteraSkill skill."""
    if step["type"] == "tool_call":
        tool = step.get("tool_name", "")
        if tool == "search":
            return "search_navigate"
        elif tool == "get_document":
            return "review_content"
        else:
            return "generic_action"
    elif step["type"] == "reasoning":
        return "review_content"
    elif step["type"] == "output_text":
        return "export_publish"
    return "generic_action"


def extract_skill_sequence(result_steps):
    """Extract a simplified skill sequence from trajectory steps.

    Consecutive steps of the same skill are collapsed into one.
    """
    skills = []
    for step in result_steps:
        skill = map_step_to_skill(step)
        if not skills or skills[-1] != skill:
            skills.append(skill)
    return skills


REASONING_TEMPLATES = [
    "Let me think about what to search for next.",
    "I need to analyze these results and refine my query.",
    "Based on what I found, I should search for more specific information.",
    "Let me cross-reference this with other sources.",
    "I'm synthesizing the information I've gathered so far.",
    "I need to verify this finding against other documents.",
]

SEARCH_TEMPLATES = [
    "Searching for: {query}",
    "Looking up: {query}",
    "Querying: {query}",
]

REVIEW_TEMPLATES = [
    "Reviewing the search results. Found relevant information.",
    "Analyzing the document content for key details.",
    "Cross-referencing findings with previous results.",
    "Reading through the evidence documents.",
]

USER_FOLLOWUPS = [
    "What did you find?",
    "Keep searching.",
    "Try a different angle.",
    "Can you find more specific information?",
    "What else can you find?",
    "Continue with the research.",
    "Look deeper into this.",
]


def generate_conversation(query_data, run_data, conv_id):
    """Generate a conversation from a BrowseComp-Plus query + trajectory."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    question = query_data["query"]
    skill_sequence = extract_skill_sequence(run_data["result"])

    # Limit skill sequence length for tractable eval
    if len(skill_sequence) > 10:
        skill_sequence = skill_sequence[:10]

    # First user message: the research question
    messages.append({"role": "user", "content": question})

    # Build conversation from skill sequence
    for i, skill in enumerate(skill_sequence):
        # Agent turn
        if i == 0:
            agent_msg = (
                f"**[Thinking]** Let me plan my research approach for this question.\n\n"
                f"**[Action: {skill}]** Starting the research process.\n\n"
                f"**[Observation]** Beginning search..."
            )
        elif skill == "search_navigate":
            # Extract actual search query if available
            search_queries = [s for s in run_data["result"]
                            if s["type"] == "tool_call" and s.get("tool_name") == "search"]
            if search_queries:
                sq = search_queries[min(i, len(search_queries)-1)]
                try:
                    q = json.loads(sq["arguments"])["query"][:80]
                except:
                    q = "relevant information"
                tmpl = random.choice(SEARCH_TEMPLATES).replace("{query}", q)
            else:
                tmpl = "Searching for more information..."
            agent_msg = (
                f"**[Thinking]** {random.choice(REASONING_TEMPLATES)}\n\n"
                f"**[Action: {skill}]** {tmpl}\n\n"
                f"**[Observation]** Found several relevant results."
            )
        elif skill == "review_content":
            agent_msg = (
                f"**[Thinking]** {random.choice(REASONING_TEMPLATES)}\n\n"
                f"**[Action: {skill}]** {random.choice(REVIEW_TEMPLATES)}\n\n"
                f"**[Observation]** Gathered useful information."
            )
        elif skill == "export_publish":
            agent_msg = (
                f"**[Thinking]** I have enough information to formulate an answer.\n\n"
                f"**[Action: {skill}]** Compiling final answer.\n\n"
                f"**[Observation]** Answer ready."
            )
        else:
            agent_msg = (
                f"**[Thinking]** Processing...\n\n"
                f"**[Action: {skill}]** Executing step.\n\n"
                f"**[Observation]** Step completed."
            )
        messages.append({"role": "assistant", "content": agent_msg})

        # User follow-up (except for last step)
        if i < len(skill_sequence) - 1:
            messages.append({"role": "user", "content": random.choice(USER_FOLLOWUPS)})

    # Closing
    messages.append({"role": "user", "content": "Thanks, that's all I needed."})
    messages.append({"role": "assistant", "content":
        f"You're welcome! I completed the research using {len(skill_sequence)} steps.\n"
        f"Skills used: {' -> '.join(skill_sequence)}"
    })

    return {
        "conversation_id": conv_id,
        "task": question[:100],
        "complexity": "high" if run_data["search_counts"] > 15 else "medium",
        "apps": ["search_engine", "document_viewer"],
        "skill_flow": skill_sequence,
        "num_turns": len(messages),
        "domain": "deep_research",
        "source": "browsecomp-plus",
        "search_counts": run_data["search_counts"],
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate BrowseComp-Plus conversations for eval")
    parser.add_argument("--max-queries", type=int, default=200,
                        help="Max queries to convert (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-file", type=str, default=str(BC_RUNS),
                        help="Path to decrypted run file")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load queries
    print(f"Loading queries from {BC_QUERIES}...")
    queries = {}
    with open(BC_QUERIES) as f:
        for line in f:
            d = json.loads(line)
            queries[d["query_id"]] = d
    print(f"  Loaded {len(queries)} queries")

    # Load run trajectories
    print(f"Loading trajectories from {args.run_file}...")
    runs = {}
    with open(args.run_file) as f:
        for line in f:
            d = json.loads(line)
            runs[d["query_id"]] = d
    print(f"  Loaded {len(runs)} trajectories")

    # Match queries to runs
    matched = [(queries[qid], runs[qid]) for qid in queries if qid in runs]
    random.shuffle(matched)
    matched = matched[:args.max_queries]
    print(f"  Using {len(matched)} matched query-trajectory pairs")

    # Generate conversations
    conversations = []
    for i, (query, run) in enumerate(matched):
        conv_id = f"bc_conv_{i:05d}"
        conv = generate_conversation(query, run, conv_id)
        conversations.append(conv)

    # Save
    with open(OUTPUT_FILE, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"\nSaved {len(conversations)} conversations to {OUTPUT_FILE}")

    # Statistics
    print(f"\n{'='*60}")
    print("BROWSECOMP-PLUS CONVERSATION STATISTICS")
    print(f"{'='*60}")

    skill_counts = Counter()
    seq_lens = []
    search_counts = []
    for c in conversations:
        seq_lens.append(len(c["skill_flow"]))
        search_counts.append(c["search_counts"])
        for s in c["skill_flow"]:
            skill_counts[s] += 1

    print(f"Skill sequence length: min={min(seq_lens)}, max={max(seq_lens)}, "
          f"mean={sum(seq_lens)/len(seq_lens):.1f}")
    print(f"Search counts: min={min(search_counts)}, max={max(search_counts)}, "
          f"mean={sum(search_counts)/len(search_counts):.1f}")
    print(f"\nSkill distribution:")
    for skill, count in skill_counts.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/sum(skill_counts.values()):.1f}%)")


if __name__ == "__main__":
    main()
