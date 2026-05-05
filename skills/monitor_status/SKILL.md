# Monitor Status

## Quick Reference

| Task | Action |
|------|--------|
| Check dashboard | Open project board or monitoring tool |
| Review notifications | Check notification panel |
| Verify system health | Look at status indicators |

## When to Use

Trigger when the task requires **checking the current state** of a system, project, or process. Includes:
- Viewing project dashboards or task boards
- Checking notification panels
- Monitoring build/deploy status
- Reviewing task completion progress

## Workflow

1. Open dashboard or monitoring tool
2. Review key metrics and indicators
3. Check for alerts or blockers
4. Note any items requiring action
5. Report findings or take action

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `send_message` | 0.25 | Sent message, now checking for updates |
| (start) | 0.20 | Starting workflow by checking status |

| To | Probability | Context |
|----|------------|---------|
| `document_edit` | 0.30 | Found issue, updating records |
| `search_navigate` | 0.25 | Need to investigate an alert |
| `collaborate` | 0.20 | Flagging issue to team |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Dashboard not loading | Try alternative monitoring tool |
| Stale data | Refresh, check data pipeline |
