# Generic Action

## Quick Reference

| Task | Action |
|------|--------|
| UI interaction | Click, scroll, fill form, toggle setting |
| Miscellaneous | Any action not covered by specific skills |

## When to Use

Trigger when the task requires a **general UI interaction** that does not fit into any of the specialized skill categories. This is a catch-all for:
- Form fills and submissions
- Settings adjustments
- UI navigation clicks
- Dialog confirmations

## Workflow

1. Identify the target UI element
2. Perform the interaction (click, type, select)
3. Verify the action completed
4. Proceed to next step

## Transitions

| From / To | Context |
|-----------|---------|
| Any skill | Generic actions can appear anywhere in a workflow |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Element not found | Wait for page load, try alternative selector |
| Action had no effect | Retry, check if element is disabled |
