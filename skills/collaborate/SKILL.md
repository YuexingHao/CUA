# Collaborate

## Quick Reference

| Task | Action |
|------|--------|
| Share in channel | Post update in Teams/Slack channel |
| Co-edit document | Open shared doc, work simultaneously |
| Tag team members | @mention for attention or assignment |

## When to Use

Trigger when the task requires **real-time coordination with other people**. Includes:
- Posting in team channels for discussion
- Co-editing shared documents
- Tagging/mentioning team members
- Coordinating handoffs between people

## Workflow

1. Open collaboration platform (Teams, Slack, shared doc)
2. Navigate to relevant channel/thread/document
3. Post message, share content, or tag people
4. Monitor for responses
5. Continue discussion or hand off

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `document_edit` | 0.25 | Doc ready for team input |
| `review_content` | 0.20 | Review done, sharing feedback |

| To | Probability | Context |
|----|------------|---------|
| `send_message` | 0.25 | Follow up with formal email |
| `document_edit` | 0.25 | Received feedback, making changes |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Channel permission denied | Post in alternative channel, request access |
| Concurrent edit conflict | Merge changes, reload latest version |
| User not found / offline | Leave message, try alternative contact |
