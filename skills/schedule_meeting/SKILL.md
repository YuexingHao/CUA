# Schedule Meeting

## Quick Reference

| Task | Action |
|------|--------|
| Create event | Open calendar, new event, set details |
| Send invites | Add attendees, send invitation |
| Check availability | View free/busy for attendees |

## When to Use

Trigger when the task requires **creating calendar events or coordinating schedules**. Includes:
- Creating meetings with time, location, agenda
- Sending calendar invitations
- Checking attendee availability
- Booking rooms or resources

## Workflow

1. Open calendar application
2. Check availability of attendees
3. Create new event (title, time, duration, location)
4. Add attendees and set reminders
5. Send invitation

## Transitions

| From | Probability | Context |
|------|------------|---------|
| `collaborate` | 0.30 | Team discussed, now scheduling follow-up |
| `document_edit` | 0.20 | Agenda prepared, now creating meeting |

| To | Probability | Context |
|----|------------|---------|
| `send_message` | 0.40 | Meeting created, sending heads-up email |
| `collaborate` | 0.25 | Posted meeting link in channel |

## Error Handling

| Failure | Recovery |
|---------|----------|
| Calendar sync conflict | Verify with user, treat soft blocks as overridable |
| Room unavailable | Suggest alternative rooms or virtual meeting |
| Attendee conflict | Propose alternative times |
