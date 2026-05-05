# Export & Publish

## Quick Reference

| Task | Action |
|------|--------|
| Export to PDF | File → Export/Save As → PDF |
| Print document | File → Print, select printer/settings |
| Publish content | Upload to web, share publicly |

## When to Use

Trigger when the task requires **converting, exporting, or publishing** finished content. Includes:
- Exporting documents/slides as PDF
- Printing to physical or virtual printers
- Publishing to web platforms or shared drives
- Converting between file formats

## Workflow

1. Open the completed document/presentation
2. Select export format and options
3. Execute export/conversion
4. Verify output file (check formatting, page count)
5. Save to target location

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `review_content` | 0.35 | Reviewed and approved, now exporting |
| `document_edit` | 0.30 | Finished editing, creating final output |
| `presentation_edit` | 0.20 | Deck done, exporting as PDF |

| To | Probability | Context |
|----|------------|---------|
| `send_message` | 0.45 | Exported, now emailing the file |
| `organize_files` | 0.25 | Filing the exported document |

## Error Handling

| Failure | Recovery |
|---------|----------|
| PDF rendering failure (embedded objects) | Convert linked objects to static images first |
| Memory overflow on large files | Split and convert in parts, then merge |
| Format loss during conversion | Use higher-fidelity export settings |

## Anthropic Skill Mapping

Corresponds to `pdf` in [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/pdf). Relies on LibreOffice and Poppler for conversion pipelines.
