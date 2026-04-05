# Data Transfer

## Quick Reference

| Task | Action |
|------|--------|
| Copy data between apps | Select source → copy → switch app → paste |
| Import/export data | Use import/export dialogs or APIs |
| Sync information | Pull data from one system to another |

## When to Use

Trigger when the task requires **moving data between applications or systems**. Includes:
- Copying tables/charts from spreadsheets to documents
- Importing data from databases or APIs
- Syncing information between platforms (e.g., CRM to email)
- Clipboard operations across applications

## Workflow

1. Open source application/data
2. Select and copy relevant data
3. Switch to target application
4. Paste or import data
5. Verify formatting and completeness

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `search_navigate` | 0.35 | Found the data source, now transferring |
| `monitor_status` | 0.15 | Checked metrics, now pulling them |

| To | Probability | Context |
|----|------------|---------|
| `document_edit` | 0.40 | Data pasted, now formatting in doc |
| `presentation_edit` | 0.25 | Data ready, inserting into slides |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Format mismatch on paste | Convert format, use paste-special |
| API rate limit | Wait and retry with backoff |
| Schema incompatibility | Create mapping file, transform data |

## Anthropic Skill Mapping

Partially corresponds to `xlsx` in [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/xlsx) when spreadsheet data is involved. Key techniques:
- Use `openpyxl` for Excel formula preservation
- Always use Excel formulas over hardcoded values
- Recalculate after modifications
