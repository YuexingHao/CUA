# Document Edit

## Quick Reference

| Task | Action |
|------|--------|
| Edit text document | Open in Word/editor, modify content, save |
| Format content | Apply styles, headings, tables, lists |
| Update data | Replace values, add sections, restructure |

## When to Use

Trigger when the task requires **modifying text content** in a document (Word, Google Docs, web CMS). Includes:
- Updating reports with new data
- Adding/removing sections
- Formatting text (headings, tables, lists)
- Inserting charts or images into documents

## Workflow

1. Open target document (may follow `search_navigate`)
2. Navigate to the section requiring changes
3. Make edits: type, delete, format, insert
4. Verify changes visually
5. Save document

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `search_navigate` | 0.35 | Found the doc, now editing |
| `data_transfer` | 0.20 | Pasted data, now formatting |
| `review_content` | 0.15 | Reviewed, now fixing issues |

| To | Probability | Context |
|----|------------|---------|
| `review_content` | 0.30 | Verify edits look correct |
| `export_publish` | 0.25 | Done editing, export as PDF |
| `data_transfer` | 0.15 | Pull edited content elsewhere |

## Error Handling

| Failure | Recovery |
|---------|----------|
| File locked by another user | Wait, notify user, or request unlock |
| Permission denied | Escalate to user for access |
| Autosave conflict | Merge changes or reload latest |

## Anthropic Skill Mapping

Corresponds to `docx` in [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/docx). Key techniques from that skill:
- `.docx` files are ZIP archives of XML — can unpack/edit/repack
- Use `docx-js` for programmatic creation
- Always validate output after generation
