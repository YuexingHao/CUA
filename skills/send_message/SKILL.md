# Send Message

## Quick Reference

| Task | Action |
|------|--------|
| Send email | Open compose, add recipients, write, send |
| Post chat message | Open channel/DM, type, send |
| Share notification | Compose and distribute alert |

## When to Use

Trigger when the task requires **composing and sending a message** to one or more recipients. Includes:
- Writing and sending emails (Outlook, Gmail)
- Posting messages in chat channels (Teams, Slack)
- Sending direct messages
- Distributing notifications or alerts

## Workflow

1. Open compose window (email or chat)
2. Add recipients (To, CC, channel)
3. Write subject and body
4. Attach files if needed
5. Review and send

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `export_publish` | 0.30 | Exported PDF, now emailing it |
| `collaborate` | 0.20 | Discussion done, sending summary |
| `document_edit` | 0.15 | Doc ready, notifying stakeholders |

| To | Probability | Context |
|----|------------|---------|
| `monitor_status` | 0.25 | Sent, now watching for replies |
| `collaborate` | 0.20 | Message sent, continuing discussion |
| (end) | 0.30 | Often the last skill in a workflow |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Recipient not found | Verify address, check directory |
| Attachment too large | Compress or share via link |
| Send permission denied | Use different sender or escalate |

## Anthropic Skill Mapping

Partially corresponds to `internal-comms` in anthropics/skills.
