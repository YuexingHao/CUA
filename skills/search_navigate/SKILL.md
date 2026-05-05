# Search & Navigate

## Quick Reference

| Task | Action |
|------|--------|
| Find files | Search shared drive, recent docs, or file browser |
| Search content | Use in-app search, Ctrl+F, or web search |
| Navigate to page | Click links, use address bar, browse menus |

## When to Use

Trigger when the task requires **locating information or navigating to a target**. Includes:
- Searching for files, emails, or messages
- Navigating to specific pages or locations in an app
- Using search bars, filters, or directory listings
- Browsing folder structures or menu hierarchies

## Workflow

1. Identify what needs to be found
2. Choose search method (search bar, file browser, URL)
3. Enter search query or navigate
4. Review results / verify correct destination
5. Open or select the target

## Transitions

| From | Probability | Context |
|------|------------|---------|
| (start) | 0.40 | Often the first skill in a workflow |
| `monitor_status` | 0.15 | Saw notification, navigating to it |

| To | Probability | Context |
|----|------------|---------|
| `document_edit` | 0.30 | Found the doc, now editing |
| `review_content` | 0.30 | Found it, now reading |
| `data_transfer` | 0.15 | Found source data, now copying |

## Error Handling

| Failure | Recovery |
|---------|----------|
| No results found | Broaden search, try alternative terms |
| Stale search index | Supplement with real-time activity log |
| Permission denied on result | Request access or find alternative copy |
