# Presentation Edit

## Quick Reference

| Task | Action |
|------|--------|
| Edit slides | Open PowerPoint/Slides, modify content |
| Create deck | Build slides from template or scratch |
| Add visuals | Insert charts, images, diagrams |

## When to Use

Trigger when the task requires **modifying slide presentations** (PowerPoint, Google Slides). Includes:
- Updating slide content (titles, body text, data)
- Adding/removing/reordering slides
- Inserting charts, images, or diagrams
- Applying design themes and layouts

## Workflow

1. Open presentation file
2. Navigate to target slide
3. Edit content: text, images, charts, layout
4. Apply consistent styling across slides
5. Save and optionally export

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `data_transfer` | 0.30 | Charts/data pulled in, now placing on slides |
| `document_edit` | 0.20 | Report written, now making deck |
| `search_navigate` | 0.20 | Found template, now editing |

| To | Probability | Context |
|----|------------|---------|
| `review_content` | 0.35 | Review deck before sharing |
| `export_publish` | 0.30 | Export as PDF for distribution |
| `send_message` | 0.15 | Email the finished deck |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Embedded media missing | Re-link or replace with static image |
| Template corruption | Fall back to blank slide with manual styling |
| Large file (>100MB) | Compress images, split deck |

## Anthropic Skill Mapping

Corresponds to `pptx` in [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/pptx). Key techniques:
- Use `pptxgenjs` for creation from scratch
- Use unpack/edit XML/repack for template editing
- Always do visual QA — convert to images and inspect
