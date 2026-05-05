# Organize Files

## Quick Reference

| Task | Action |
|------|--------|
| Create folders | New folder in file browser |
| Move/rename files | Drag-drop, right-click rename |
| Archive old files | Move to archive folder or compress |

## When to Use

Trigger when the task requires **managing file organization** in a file system or drive. Includes:
- Creating folder structures
- Moving files between folders
- Renaming files for consistency
- Archiving or cleaning up old files

## Workflow

1. Open file browser or drive
2. Identify files to organize
3. Create folder structure if needed
4. Move/rename/archive files
5. Verify organization is correct

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `search_navigate` | 0.30 | Found files, now organizing them |
| `export_publish` | 0.25 | Exported file, now filing it |

| To | Probability | Context |
|----|------------|---------|
| `document_edit` | 0.25 | Files organized, now editing |
| `send_message` | 0.20 | Organized, notifying team of new location |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Files locked during move | Notify users, retry after release |
| Permission denied on destination | Request access or use alternative location |
| Name collision | Append version number or timestamp |
