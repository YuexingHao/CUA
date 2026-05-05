# Review Content

## Quick Reference

| Task | Action |
|------|--------|
| Read document | Open and scroll through content |
| Add comments | Use review/comment feature |
| Verify accuracy | Cross-check data, check formatting |

## When to Use

Trigger when the task requires **reading, inspecting, or verifying** content without modifying the primary document. Includes:
- Reading through a document or email
- Adding review comments or annotations
- Checking data accuracy against sources
- Verifying formatting and layout

## Workflow

1. Open document/content for review
2. Read through systematically
3. Note issues or add comments
4. Summarize findings if needed
5. Decide: approve, request changes, or edit directly

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `search_navigate` | 0.30 | Found the content, now reviewing |
| `document_edit` | 0.25 | Made edits, now verifying |
| `presentation_edit` | 0.15 | Slides done, reviewing deck |

| To | Probability | Context |
|----|------------|---------|
| `document_edit` | 0.30 | Found issues, now fixing |
| `export_publish` | 0.25 | Approved, ready to export |
| `collaborate` | 0.20 | Sharing feedback with team |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Track changes not visible | Enable track changes view |
| Comments not saving | Check permissions, try different tool |
