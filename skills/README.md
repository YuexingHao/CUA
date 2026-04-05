# InteraSkill: Learned Skill Library

Structured skill definitions discovered from user interaction trajectories,
formatted following the [anthropics/skills](https://github.com/anthropics/skills) standard.

## Skill Taxonomy (12 skills)

| Skill | Anthropic Equivalent | Domain |
|-------|---------------------|--------|
| `document_edit` | `docx` | Document authoring |
| `presentation_edit` | `pptx` | Slide creation/editing |
| `data_transfer` | `xlsx` (partial) | Cross-app data movement |
| `export_publish` | `pdf` | Export/convert/publish |
| `search_navigate` | — | Finding files, info, pages |
| `review_content` | — | Reading, commenting, verifying |
| `send_message` | `internal-comms` (partial) | Email/chat composition |
| `collaborate` | `internal-comms` (partial) | Real-time teamwork |
| `schedule_meeting` | — | Calendar/invite management |
| `organize_files` | — | File/folder management |
| `monitor_status` | — | Dashboard/notification checks |
| `generic_action` | — | Catch-all UI interactions |

## Format

Each skill follows the anthropic SKILL.md structure:
- **Quick Reference** — one-line action guide
- **When to Use** — trigger conditions
- **Workflow** — step-by-step execution
- **Transitions** — likely preceding/following skills
- **Error Handling** — common failure modes and recovery
