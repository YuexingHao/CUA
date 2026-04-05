#!/usr/bin/env python3
"""
Generate realistic multi-turn user-agent conversation data
for fine-tuning Qwen3-8B on CUA skill prediction.

Produces conversations that combine:
  1. User-agent task dialogues (user requests, agent executes)
  2. Agent reasoning traces (chain-of-thought before acting)
  3. User corrections & feedback (agent adapts mid-workflow)

Each conversation simulates a full workflow session where a user
asks a CUA agent to complete a complex task across M365 apps.

Usage:
    python generate_conversations.py [--num 1000] [--seed 42]
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent
TEMPLATES_FILE = DATA_DIR / "iw_skill_templates.json"
OUTPUT_TRAIN = DATA_DIR / "train_conversations.jsonl"
OUTPUT_VAL = DATA_DIR / "val_conversations.jsonl"

# ── System prompt for the CUA agent ─────────────────────────────────

SYSTEM_PROMPT = """\
You are InteraSkill, an AI computer-using agent that helps users complete \
tasks across Microsoft 365 applications (Word, PowerPoint, Outlook, Teams).

You can execute the following skills:
- document_edit: Edit text documents (Word)
- presentation_edit: Edit slides (PowerPoint)
- send_message: Compose and send emails/messages
- schedule_meeting: Create calendar events and invites
- search_navigate: Find files, emails, or information
- review_content: Read and add comments/feedback
- collaborate: Work with team members in real-time
- data_transfer: Copy data between applications
- export_publish: Export or print documents
- organize_files: Move, rename, or archive files
- monitor_status: Check dashboards, tasks, notifications
- generic_action: Other UI interactions

When executing a task:
1. Think step-by-step about what skills are needed
2. Explain what you're doing and why
3. Report what you observe on screen
4. Ask for clarification when the user's intent is ambiguous
5. Adapt when the user gives corrections or changes direction"""

# ── Rich scenario templates ──────────────────────────────────────────

SCENARIOS = [
    {
        "task": "Prepare the quarterly business review materials",
        "apps": ["word", "powerpoint", "outlook"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "document_edit", "data_transfer",
                       "presentation_edit", "review_content", "export_publish",
                       "send_message"],
        "user_messages": [
            "I need to prepare the QBR materials for next Monday's leadership meeting. Can you help?",
            "Start by finding last quarter's report — it should be in the shared drive.",
            "Good. Now update the revenue numbers in the Word doc. Q3 revenue was $4.2M, up 12% from Q2.",
            "Can you pull the key charts from the Word doc into the PowerPoint deck?",
            "The slide deck needs updating too — change the title to 'Q3 2026 Business Review'.",
            "Let me review everything before we send it out.",
            "Looks good overall. Export the deck as PDF.",
            "Now email the PDF to the leadership team — sarah@company.com and mike@company.com.",
        ],
        "corrections": [
            (3, "Actually, use the updated chart from the appendix, not the one in the main body."),
            (5, "Wait, the market share numbers on slide 4 look wrong. Can you double-check against the Word doc?"),
        ],
    },
    {
        "task": "Organize the team sprint planning session",
        "apps": ["teams", "outlook", "word"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "document_edit", "schedule_meeting",
                       "collaborate", "send_message", "monitor_status"],
        "user_messages": [
            "We need to set up sprint planning for next week. Help me get everything ready.",
            "First, find the sprint backlog document from the project channel in Teams.",
            "Update the backlog — move items 'API refactor' and 'Auth fix' to the sprint candidates section.",
            "Schedule a 1-hour planning meeting for Tuesday at 10am. Invite the engineering team.",
            "Share the updated backlog in the #engineering channel so people can review before the meeting.",
            "Send a heads-up email to the team leads about the meeting.",
            "Check if there are any blockers in the project board that we should discuss.",
        ],
        "corrections": [
            (3, "Make it 90 minutes instead of 60 — we had issues with time last sprint."),
            (4, "Also tag @alex and @priya in the Teams message, they're the new team members."),
        ],
    },
    {
        "task": "Draft and distribute the product launch announcement",
        "apps": ["word", "outlook", "teams"],
        "complexity": "medium",
        "skill_flow": ["document_edit", "review_content", "collaborate",
                       "export_publish", "send_message"],
        "user_messages": [
            "I need to write a product launch announcement for our new feature. Help me draft it.",
            "Create a new Word doc. The feature is called 'SmartSync' — it's an AI-powered file synchronization tool.",
            "Let me review what you've written so far.",
            "Share the draft with the marketing team in the #product-launches channel for feedback.",
            "Export it as a PDF for the press kit.",
            "Now compose an email announcement to all-staff@company.com with the key highlights.",
        ],
        "corrections": [
            (1, "Actually, let's use the company announcement template, not a blank document."),
            (5, "Add the CEO as CC on that email — ceo@company.com."),
        ],
    },
    {
        "task": "Process and respond to client feedback",
        "apps": ["outlook", "word", "teams"],
        "complexity": "medium",
        "skill_flow": ["search_navigate", "review_content", "document_edit",
                       "collaborate", "send_message"],
        "user_messages": [
            "I got a batch of client feedback emails this morning. Help me process them.",
            "Search my inbox for emails with subject containing 'feedback' from the last week.",
            "Open the first one from Acme Corp and let me read through their comments.",
            "Create a summary document with the key feedback points organized by theme.",
            "Share the summary with the product team in our Teams channel.",
            "Draft a response to Acme Corp acknowledging their feedback and outlining next steps.",
        ],
        "corrections": [
            (2, "Actually, filter for emails from the last 3 days, not a full week."),
            (3, "Add a priority column to the summary — mark 'login issues' as critical."),
        ],
    },
    {
        "task": "Set up onboarding materials for new hires",
        "apps": ["word", "powerpoint", "outlook", "teams"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "document_edit", "presentation_edit",
                       "organize_files", "schedule_meeting", "send_message",
                       "collaborate"],
        "user_messages": [
            "We have three new engineers starting next Monday. I need to prepare their onboarding materials.",
            "Find the onboarding checklist template in the HR shared drive.",
            "Update the checklist with our team-specific items: setup dev environment, get access to GitHub, join Slack channels.",
            "Now update the welcome presentation — change the team name to 'Platform Engineering' and update the org chart on slide 3.",
            "Create a folder called 'Onboarding-2026-Q2' and move all the materials there.",
            "Schedule a welcome meeting for Monday at 9am with the new hires and the team.",
            "Send the new hires a welcome email with the meeting invite and a link to the materials folder.",
            "Post in the #team channel that new members are joining and to give them a warm welcome.",
        ],
        "corrections": [
            (2, "Oh, also add 'complete security training' to the checklist — HR just made it mandatory."),
            (5, "Change it to 9:30am, I just realized I have a standup at 9."),
        ],
    },
    {
        "task": "Prepare meeting minutes and follow-up actions",
        "apps": ["word", "outlook", "teams"],
        "complexity": "low",
        "skill_flow": ["document_edit", "review_content", "send_message"],
        "user_messages": [
            "I just finished a project review meeting. Help me write up the minutes.",
            "Create a meeting minutes document. Date: April 3, 2026. Attendees: Sarah, Mike, Alex, and me.",
            "Let me review and add a few more action items.",
            "Email the minutes to all attendees.",
        ],
        "corrections": [
            (2, "Add Lisa to the attendees list — she joined halfway through."),
        ],
    },
    {
        "task": "Compile and share the weekly status report",
        "apps": ["word", "powerpoint", "teams"],
        "complexity": "medium",
        "skill_flow": ["search_navigate", "data_transfer", "document_edit",
                       "collaborate", "export_publish"],
        "user_messages": [
            "It's Friday and I need to put together the weekly status report.",
            "Find this week's task completion data from the project board.",
            "Pull the completion metrics into the status report template.",
            "Update the highlights section — we completed the API migration and started the UI redesign.",
            "Share the draft in the #leadership channel for comments.",
            "Export as PDF once everyone's signed off.",
        ],
        "corrections": [
            (3, "Actually, mention the API migration had a 2-day delay but we recovered. Be transparent about it."),
        ],
    },
    {
        "task": "Coordinate vendor evaluation process",
        "apps": ["outlook", "word", "teams", "powerpoint"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "organize_files", "document_edit",
                       "review_content", "schedule_meeting", "presentation_edit",
                       "send_message"],
        "user_messages": [
            "We're evaluating three cloud vendors and need to organize the comparison.",
            "Find all the vendor proposals we received — they should be in my email attachments from last week.",
            "Organize them into a folder called 'Vendor-Eval-2026'.",
            "Create a comparison matrix document with columns: vendor, pricing, features, support, security.",
            "I want to review the comparison before sharing.",
            "Schedule a decision meeting with the infrastructure team for next Thursday at 2pm.",
            "Create a summary slide deck with the top 3 findings for each vendor.",
            "Send the comparison doc and deck to the team before the meeting.",
        ],
        "corrections": [
            (3, "Add a 'compliance' column too — legal wants us to check GDPR compliance."),
            (6, "Make it a 5-slide max deck — the VP doesn't have patience for long presentations."),
        ],
    },
    {
        "task": "Handle urgent client escalation",
        "apps": ["outlook", "teams", "word"],
        "complexity": "medium",
        "skill_flow": ["search_navigate", "review_content", "collaborate",
                       "document_edit", "send_message"],
        "user_messages": [
            "We just got an urgent escalation from BigCorp. Their integration is down. Help me coordinate the response.",
            "Find the last support ticket and any related emails from BigCorp.",
            "Check the incident thread in #support-escalations in Teams.",
            "Ping the on-call engineer in the channel — we need a status update ASAP.",
            "Draft an incident response document with what we know so far.",
            "Send a status update email to the BigCorp account manager — be professional but transparent.",
        ],
        "corrections": [
            (4, "Add the timeline of events — the issue started at 2:15 PM EST today."),
        ],
    },
    {
        "task": "Plan team offsite logistics",
        "apps": ["outlook", "word", "teams"],
        "complexity": "low",
        "skill_flow": ["document_edit", "schedule_meeting", "send_message",
                       "collaborate"],
        "user_messages": [
            "We're planning a team offsite next month. Help me get the logistics started.",
            "Create an agenda document for a 2-day offsite. Day 1: strategy sessions. Day 2: team building.",
            "Schedule a planning call with the admin team for this Friday at 3pm.",
            "Send the draft agenda to the team and ask for topic suggestions.",
            "Post in #general asking people to fill out the dietary preferences form.",
        ],
        "corrections": [
            (1, "Make it a 3-day offsite actually — leadership approved an extra day for hackathon."),
        ],
    },
]

# ── Failure / multi-API scenarios ────────────────────────────────────
#
# These scenarios involve calling multiple tools/APIs where things break:
#   - API timeouts, auth failures, rate limits
#   - Cross-app data format mismatches
#   - Permission errors, stale references
#   - Cascading failures (tool A fails → tool B has no input)
#   - Conflicting concurrent edits
#
# The agent must: detect the failure, diagnose, attempt recovery, and
# communicate clearly to the user.

FAILURE_SCENARIOS = [
    {
        "task": "Cross-app data pipeline: pull metrics from dashboard, update report, email stakeholders",
        "apps": ["teams", "word", "outlook"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "data_transfer", "document_edit",
                       "data_transfer", "export_publish", "send_message"],
        "user_messages": [
            "Pull the latest project metrics from the Teams dashboard and put them into the weekly report, then email it out.",
            "Start by grabbing the KPI data from the #analytics channel.",
            "Now paste that into the weekly report document.",
            "Update the summary paragraph with the new numbers.",
            "Also pull the burn-down chart and add it to the report.",
            "Export as PDF and email to stakeholders.",
            "Send to the distribution list: reports@company.com.",
        ],
        "failures": [
            {
                "position": 1,
                "skill": "data_transfer",
                "failure_type": "api_timeout",
                "error": "The Teams Graph API timed out while fetching the channel data. The request exceeded the 30-second limit.",
                "observation": "**[Error]** API call to `GET /teams/{team-id}/channels/{channel-id}/messages` returned HTTP 504 Gateway Timeout after 30s.",
                "recovery": "retry_with_backoff",
                "recovery_detail": "I'll retry with exponential backoff. Waiting 5 seconds before the next attempt...",
                "recovery_success": True,
            },
            {
                "position": 3,
                "skill": "data_transfer",
                "failure_type": "format_mismatch",
                "error": "The burn-down chart is an embedded Power BI visual in Teams. It can't be directly copied — the clipboard contains only a placeholder image, not the live data.",
                "observation": "**[Error]** Paste operation succeeded but the chart appears as a static low-resolution image (72 DPI) instead of the interactive chart. Data labels are unreadable.",
                "recovery": "alternative_approach",
                "recovery_detail": "I'll take a different approach: export the chart directly from Power BI as a high-resolution PNG, then insert it into Word as an image.",
                "recovery_success": True,
            },
        ],
        "corrections": [],
    },
    {
        "task": "Sync meeting notes across Outlook, Teams, and shared drive with version control",
        "apps": ["outlook", "teams", "word"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "document_edit", "collaborate",
                       "data_transfer", "organize_files", "send_message"],
        "user_messages": [
            "I need to sync the meeting notes from today's call across all our systems.",
            "Find today's meeting recording and transcript in Outlook calendar.",
            "Create a clean summary document from the transcript.",
            "Share it in the Teams project channel.",
            "Copy the action items into the team's shared task tracker.",
            "Archive the original transcript in the project folder.",
            "Send a follow-up email to attendees with the summary link.",
        ],
        "failures": [
            {
                "position": 0,
                "skill": "search_navigate",
                "failure_type": "auth_expired",
                "error": "My OAuth token for the Microsoft Graph API has expired. I need to re-authenticate to access calendar data.",
                "observation": "**[Error]** API call to `GET /me/calendar/events` returned HTTP 401 Unauthorized. Token expired at 2026-04-04T13:00:00Z.",
                "recovery": "reauth",
                "recovery_detail": "Refreshing the authentication token using the refresh grant... Token refreshed successfully. Retrying the request.",
                "recovery_success": True,
            },
            {
                "position": 3,
                "skill": "data_transfer",
                "failure_type": "concurrent_edit_conflict",
                "error": "The shared task tracker has been modified by another user (Alex) since I loaded it. My changes would overwrite their updates.",
                "observation": "**[Error]** Write conflict detected on `shared_task_tracker.xlsx`. Last modified by Alex at 14:23, my cached version is from 14:15. Conflicting cells: B12-B18.",
                "recovery": "merge_changes",
                "recovery_detail": "I'll reload the latest version, merge Alex's changes with the new action items, and then save. Let me show you both versions so you can confirm.",
                "recovery_success": True,
            },
        ],
        "corrections": [
            (2, "Make sure to tag the action items with owner names — don't just list them generically."),
        ],
    },
    {
        "task": "Automate report generation: query database, format in Word, convert to PDF, email",
        "apps": ["word", "outlook"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "data_transfer", "document_edit",
                       "document_edit", "export_publish", "send_message"],
        "user_messages": [
            "Generate the monthly sales report automatically — pull data, format it, and send it out.",
            "First, query the sales database for March 2026 numbers.",
            "Import the query results into the report template.",
            "Format the data table — add borders, header row, and alternating colors.",
            "Add the executive summary based on the numbers.",
            "Export as PDF.",
            "Email the PDF to the sales leadership DL.",
        ],
        "failures": [
            {
                "position": 1,
                "skill": "data_transfer",
                "failure_type": "rate_limit",
                "error": "The database API returned a rate limit error. We've exceeded the allowed 100 requests per minute for this service principal.",
                "observation": "**[Error]** API call to `POST /api/v2/query` returned HTTP 429 Too Many Requests. Rate limit: 100/min, current: 103/min. Retry-After: 45 seconds.",
                "recovery": "wait_and_retry",
                "recovery_detail": "I'll wait 45 seconds as indicated by the Retry-After header, then retry the query. In the meantime, I'll prepare the report template.",
                "recovery_success": True,
            },
            {
                "position": 4,
                "skill": "export_publish",
                "failure_type": "rendering_failure",
                "error": "The PDF export failed — the Word document contains an embedded OLE object (Excel chart) that the PDF renderer can't process.",
                "observation": "**[Error]** Export to PDF failed with error: 'Unable to render embedded object at page 3, position (2.5in, 4.1in). Object type: Excel.Chart.12'. The chart was inserted as a linked object, not embedded.",
                "recovery": "convert_and_retry",
                "recovery_detail": "I'll convert the linked Excel chart to a static image first, then retry the PDF export. This will lose the live data link but produce a valid PDF.",
                "recovery_success": True,
            },
        ],
        "corrections": [
            (3, "Use the company brand colors for the table — navy headers, light gray alternating rows."),
        ],
    },
    {
        "task": "Multi-system notification: update CRM, notify Slack, create Jira ticket, email client",
        "apps": ["outlook", "teams"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "document_edit", "send_message",
                       "collaborate", "send_message", "monitor_status"],
        "user_messages": [
            "A critical bug was reported by Acme Corp. I need to create tickets, notify the team, and respond to the client — all at once.",
            "First, find the original bug report email from Acme Corp.",
            "Create an incident document with the details — severity P1, affects payment processing.",
            "Post a P1 alert in the #incidents channel in Teams with the incident doc link.",
            "Notify the on-call engineer via Teams DM — tag them as urgent.",
            "Send an acknowledgment email to the Acme Corp contact confirming we're on it.",
            "Check the monitoring dashboard to see if the issue is still active.",
        ],
        "failures": [
            {
                "position": 2,
                "skill": "collaborate",
                "failure_type": "permission_denied",
                "error": "I don't have permission to post in the #incidents channel. It's restricted to the SRE team and incident commanders.",
                "observation": "**[Error]** API call to `POST /teams/{team-id}/channels/{channel-id}/messages` returned HTTP 403 Forbidden. Error: 'User does not have permission to post messages in this channel. Required role: Incident Commander or SRE Team Member.'",
                "recovery": "escalate_to_user",
                "recovery_detail": "I can't post there directly. I have two options: (1) I can post in #engineering instead and ask someone with access to cross-post, or (2) you can grant me temporary access. Which do you prefer?",
                "recovery_success": False,
            },
            {
                "position": 3,
                "skill": "send_message",
                "failure_type": "recipient_not_found",
                "error": "The on-call schedule API returned an empty result — no engineer is currently assigned to the on-call rotation for this service.",
                "observation": "**[Error]** API call to `GET /oncall/schedule?service=payments` returned 200 OK but with empty `current_oncall` field. The rotation may not have been updated for this week.",
                "recovery": "fallback_escalation",
                "recovery_detail": "The on-call roster is empty. I'll escalate to the engineering manager (James) instead, and also flag this gap in the on-call schedule to the ops team.",
                "recovery_success": True,
            },
        ],
        "corrections": [
            (4, "CC the account manager on that email too — they should be in the loop."),
        ],
    },
    {
        "task": "Bulk document processing: convert, watermark, and distribute 15 files",
        "apps": ["word", "powerpoint", "outlook"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "organize_files", "document_edit",
                       "export_publish", "export_publish", "send_message"],
        "user_messages": [
            "I need to process all the proposal documents for the board meeting — convert to PDF, add confidential watermarks, and send to board members.",
            "Find all the proposal docs in the 'Board-Q3' folder.",
            "Organize them by department — finance, engineering, marketing.",
            "Add 'CONFIDENTIAL - Board Only' watermark to each document.",
            "Convert all Word docs to PDF.",
            "Also convert the PowerPoint decks to PDF.",
            "Email the full package to the board distribution list.",
        ],
        "failures": [
            {
                "position": 2,
                "skill": "document_edit",
                "failure_type": "file_locked",
                "error": "3 of the 15 documents are currently locked for editing by other users. I can't add watermarks to locked files.",
                "observation": "**[Error]** Files locked:\n- `finance_proposal_v3.docx` — locked by Sarah (since 13:45)\n- `eng_roadmap.docx` — locked by Mike (since 14:02)\n- `marketing_budget.docx` — locked by Priya (since 13:58)\nRemaining 12 files are available.",
                "recovery": "partial_proceed",
                "recovery_detail": "I'll process the 12 unlocked files now and queue the 3 locked ones. I'll notify Sarah, Mike, and Priya via Teams to close their files, and retry in 10 minutes.",
                "recovery_success": True,
            },
            {
                "position": 4,
                "skill": "export_publish",
                "failure_type": "memory_overflow",
                "error": "The batch PDF conversion ran out of memory processing the large engineering deck (180 slides with embedded videos).",
                "observation": "**[Error]** PDF export crashed with MemoryError on `eng_roadmap.pptx` (180 slides, 450MB). System memory: 95% used. Other conversions completed successfully.",
                "recovery": "split_and_retry",
                "recovery_detail": "The engineering deck is too large for single-pass conversion. I'll split it into 3 parts (60 slides each), convert separately, then merge the PDFs.",
                "recovery_success": True,
            },
        ],
        "corrections": [],
    },
    {
        "task": "Cross-platform calendar sync with conflicting data sources",
        "apps": ["outlook", "teams"],
        "complexity": "medium",
        "skill_flow": ["search_navigate", "schedule_meeting", "collaborate",
                       "send_message", "monitor_status"],
        "user_messages": [
            "I need to schedule the quarterly review but there are conflicting calendars. Help me sort it out.",
            "Check availability for all 8 VPs across both Outlook and Teams calendars.",
            "Find a 2-hour slot next week that works for everyone.",
            "Post the proposed time in the #leadership channel for confirmation.",
            "Send calendar invites once we get 3 confirmations.",
            "Set up a reminder to check for RSVPs by end of day.",
        ],
        "failures": [
            {
                "position": 1,
                "skill": "schedule_meeting",
                "failure_type": "calendar_sync_conflict",
                "error": "The Outlook and Teams calendars show different availability for VP Chen. Outlook shows free 2-3 PM Tuesday, but Teams shows a conflicting 'Focus Time' block.",
                "observation": "**[Error]** Calendar mismatch detected for chen@company.com:\n- Outlook: Available Tue 2-3 PM\n- Teams: 'Focus Time' block Tue 1-4 PM (auto-generated by Viva Insights)\nThe calendars may be out of sync, or the Focus Time block may be overridable.",
                "recovery": "verify_with_user",
                "recovery_detail": "I found a sync conflict. The Teams 'Focus Time' blocks are auto-generated and usually overridable. Should I treat them as soft blocks (ignorable) or hard blocks? I'll also ping Chen to confirm their actual availability.",
                "recovery_success": False,
            },
        ],
        "corrections": [
            (3, "Actually, send the invite first and let people decline if they can't make it — we don't have time to wait for confirmations."),
        ],
    },
    {
        "task": "Migrate project data between old and new systems",
        "apps": ["teams", "word", "outlook"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "data_transfer", "data_transfer",
                       "organize_files", "review_content", "collaborate",
                       "send_message"],
        "user_messages": [
            "We're migrating Project Alpha from the old SharePoint to the new Teams workspace. Move everything over.",
            "Start by listing all files in the old SharePoint site.",
            "Copy the project documents to the new Teams file storage.",
            "Now migrate the conversation history from the old Slack channel to Teams.",
            "Create the new folder structure: /docs, /design, /meeting-notes, /archive.",
            "Verify that all files transferred correctly — check file counts and sizes.",
            "Announce the migration in both old and new channels.",
            "Email the project team with the new links and updated access instructions.",
        ],
        "failures": [
            {
                "position": 1,
                "skill": "data_transfer",
                "failure_type": "schema_mismatch",
                "error": "The old SharePoint uses a legacy metadata schema that's incompatible with the new Teams file structure. Custom columns (Priority, Owner, Due Date) will be lost during transfer.",
                "observation": "**[Error]** Schema validation failed during migration:\n- Column 'Priority' (Choice type) → not supported in Teams files\n- Column 'Owner' (Person type) → requires user mapping\n- Column 'Due Date' (DateTime) → format mismatch (UTC vs local)\n85 files affected.",
                "recovery": "transform_and_migrate",
                "recovery_detail": "I'll create a metadata mapping file to preserve the custom columns. Priority and Due Date will be stored in a companion JSON file. Owner fields will be mapped to the new Teams user IDs. This adds an extra step but no data will be lost.",
                "recovery_success": True,
            },
            {
                "position": 2,
                "skill": "data_transfer",
                "failure_type": "partial_failure",
                "error": "The Slack-to-Teams message migration partially failed. Text messages transferred but file attachments and threaded replies were not included.",
                "observation": "**[Error]** Migration results:\n- Text messages: 1,247/1,247 ✓\n- File attachments: 0/89 ✗ (Slack API requires paid plan for attachment export)\n- Threaded replies: 0/342 ✗ (nested structure not supported by migration tool)\n- Reactions/emoji: 856/856 ✓",
                "recovery": "partial_with_workaround",
                "recovery_detail": "Text and reactions migrated successfully. For the 89 file attachments, I'll download them directly from Slack's CDN (the URLs are still active for 30 days) and upload to Teams. For threaded replies, I'll flatten them into timestamped messages with a 'Reply to:' prefix.",
                "recovery_success": True,
            },
        ],
        "corrections": [
            (5, "Also check that file permissions transferred — we don't want sensitive docs to be publicly accessible in the new workspace."),
        ],
    },
    {
        "task": "Automated compliance check across multiple document repositories",
        "apps": ["word", "outlook", "teams"],
        "complexity": "high",
        "skill_flow": ["search_navigate", "search_navigate", "review_content",
                       "document_edit", "collaborate", "send_message"],
        "user_messages": [
            "Legal needs us to verify that all customer-facing documents have the updated privacy disclaimer. Audit everything.",
            "Search all Word documents in the shared drive modified in the last 90 days.",
            "Cross-reference with the list of customer-facing docs in the compliance tracker.",
            "Check each document for the 2026 privacy disclaimer text.",
            "For any missing or outdated disclaimers, add the current version.",
            "Report findings in the #compliance channel.",
            "Email the compliance summary to legal@company.com.",
        ],
        "failures": [
            {
                "position": 0,
                "skill": "search_navigate",
                "failure_type": "search_index_stale",
                "error": "The SharePoint search index is 3 days behind. Documents modified between April 1-3 may not appear in results.",
                "observation": "**[Error]** Search API warning: `index_freshness: 2026-04-01T00:00:00Z`. Current time: 2026-04-04T14:30:00Z. Index lag: ~72 hours. Results may be incomplete for recently modified documents.",
                "recovery": "supplement_search",
                "recovery_detail": "The search index is stale. I'll supplement the search results by also checking the 'Recent Files' activity log via the Graph API, which is real-time. This will catch any documents modified in the last 3 days that the search index missed.",
                "recovery_success": True,
            },
            {
                "position": 2,
                "skill": "review_content",
                "failure_type": "cascading_failure",
                "error": "The compliance tracker spreadsheet references a lookup table that was moved. All VLOOKUP formulas return #REF! errors, making the doc classification unreliable.",
                "observation": "**[Error]** `compliance_tracker.xlsx` has 47 #REF! errors in column E ('Doc Classification'). The lookup table `classification_rules.xlsx` was moved from `/legal/` to `/compliance/policies/` on March 28. All classifications after that date are broken.",
                "recovery": "fix_upstream_then_proceed",
                "recovery_detail": "This is a cascading failure — the moved file broke the tracker. I'll first fix the broken reference in the compliance tracker (update the VLOOKUP path to the new location), recalculate all classifications, then proceed with the audit. I'll also notify the person who moved the file.",
                "recovery_success": True,
            },
        ],
        "corrections": [],
    },
]

# Merge failure scenarios into main SCENARIOS list
SCENARIOS.extend(FAILURE_SCENARIOS)

# ── Screen observation templates ─────────────────────────────────────

SCREEN_OBSERVATIONS = {
    "search_navigate": [
        "I can see the search results. There are {n} matches for your query.",
        "I found the file '{file}' in the {location}. Opening it now.",
        "The search returned several results. The most recent one is from {date}.",
        "I'm navigating to the {location}. The folder structure shows several relevant files.",
    ],
    "document_edit": [
        "I have the document open in Word. Currently viewing page {page} of {total}.",
        "I've placed the cursor at the {section} section. Ready to make edits.",
        "The document currently has {n} pages. I'll add the new content at the end of section {section}.",
        "I can see the existing text. I'll update the relevant paragraph now.",
    ],
    "presentation_edit": [
        "The PowerPoint deck is open with {n} slides. Currently on slide {slide}.",
        "I can see the slide layout. The title placeholder and content area are both editable.",
        "I'm on slide {slide}. The current title reads '{title}'. I'll update it now.",
        "The design tab is open. I can see several layout options available.",
    ],
    "send_message": [
        "I've opened the compose window. The To field is empty and ready.",
        "The email draft is ready. Subject line: '{subject}'. Shall I send?",
        "I've added the recipients and composed the body. Preview looks good.",
        "The message is composed and ready to send.",
    ],
    "schedule_meeting": [
        "The calendar is open. I can see {day} has {n} existing meetings.",
        "I've created a new event. The suggested time slot at {time} is available for all attendees.",
        "The meeting invite is set up. Title: '{title}', Duration: {duration} minutes.",
        "All attendees have been added. The room is automatically booked.",
    ],
    "review_content": [
        "I'm reviewing the document. Scrolling through the content now.",
        "I can see {n} existing comments in the review panel.",
        "I've read through the content. Here are the key points I noticed...",
        "The track changes view shows {n} recent modifications.",
    ],
    "collaborate": [
        "I've opened the Teams channel. The most recent message is from {person}.",
        "I can see {n} team members are currently online in the channel.",
        "The shared file is open for collaborative editing. {n} others are viewing.",
        "I've posted the message in the channel. The notification has been sent.",
    ],
    "data_transfer": [
        "I've selected the data from {source}. Copying to clipboard.",
        "Switching to {target} now. I can see the target location.",
        "The data has been pasted successfully. Verifying formatting...",
        "The transfer is complete. The data appears correctly in {target}.",
    ],
    "export_publish": [
        "The export dialog is open. Available formats: PDF, DOCX, PPTX.",
        "Exporting as {format}... The progress bar shows {pct}%.",
        "Export complete. The file has been saved to {location}.",
        "The PDF is ready. File size: {size}.",
    ],
    "organize_files": [
        "I can see the file browser. The current folder contains {n} items.",
        "I've selected the files. The move dialog is open.",
        "The folder '{folder}' has been created. Moving files now.",
        "Files have been organized. The destination folder now contains {n} items.",
    ],
    "monitor_status": [
        "The dashboard shows {n} active tasks and {m} completed this week.",
        "I can see the notification panel. There are {n} unread notifications.",
        "The project board shows {n} items in progress and {m} blocked.",
        "The status overview is loaded. All systems show green except {item}.",
    ],
    "generic_action": [
        "I've clicked on the {element}. The UI is responding.",
        "The action was completed successfully.",
        "I can see the confirmation dialog. Proceeding with the action.",
        "The interface has updated with the new changes.",
    ],
}

# ── Reasoning trace templates ────────────────────────────────────────

REASONING_TEMPLATES = {
    "planning": [
        "Let me think about the best approach here. To {task}, I'll need to: {steps}.",
        "I'll break this down into steps. First, I need to {first_step}, then {next_step}.",
        "For this task, the most efficient workflow would be: {steps}.",
    ],
    "skill_selection": [
        "I'll use the **{skill}** skill here because {reason}.",
        "The next step requires **{skill}** — {reason}.",
        "This calls for **{skill}**. {reason}.",
    ],
    "adaptation": [
        "Got it, I'll adjust my approach. Let me {adjustment} instead.",
        "Understood — changing course. I'll {adjustment}.",
        "No problem, I'll {adjustment}. That makes more sense.",
    ],
    "completion": [
        "Done! I've {action}. {observation}",
        "That's complete. {observation} What would you like me to do next?",
        "Finished {action}. {observation}",
    ],
    "failure_detected": [
        "I've encountered an issue. Let me diagnose what went wrong.",
        "Something didn't work as expected. Let me investigate.",
        "I hit a problem here. Let me figure out what happened and how to fix it.",
    ],
    "recovery_attempt": [
        "I have a recovery plan. {detail}",
        "Here's what I'll do to work around this: {detail}",
        "I can recover from this. {detail}",
    ],
    "recovery_success": [
        "Recovery successful! The issue is resolved and I've completed the step.",
        "That worked. The problem is fixed and we can continue.",
        "Recovered successfully. Everything is back on track.",
    ],
    "recovery_failed": [
        "I wasn't able to resolve this automatically. I need your input to proceed.",
        "This requires manual intervention. Here are the options I see:",
        "I can't fix this on my own. Let me explain the situation so we can decide together.",
    ],
    "user_input_on_failure": [
        "Go with option 1.",
        "Try the workaround approach you suggested.",
        "Let's go with the alternative — that sounds safer.",
        "Use the fallback. We can fix it properly later.",
        "OK, do what you can and we'll deal with the rest manually.",
        "Skip that step for now and continue with the rest.",
    ],
}

# ── Skill-specific reasoning ────────────────────────────────────────

SKILL_REASONS = {
    "search_navigate": "I need to locate the relevant file/information first",
    "document_edit": "the document content needs to be updated",
    "presentation_edit": "the slides need modifications",
    "send_message": "we need to communicate this to the relevant people",
    "schedule_meeting": "we need to coordinate a time for everyone",
    "review_content": "it's important to verify the content before proceeding",
    "collaborate": "this requires real-time coordination with the team",
    "data_transfer": "we need to move data from one application to another",
    "export_publish": "the document needs to be exported in the final format",
    "organize_files": "the files need to be organized for easy access",
    "monitor_status": "I should check the current state before making changes",
    "generic_action": "this step requires a general interface interaction",
}

# ── Action description templates ─────────────────────────────────────

ACTION_DESCRIPTIONS = {
    "search_navigate": [
        "Searching for '{query}' in {app}...",
        "Navigating to the {location} folder...",
        "Looking up the file in the recent documents...",
    ],
    "document_edit": [
        "Editing the {section} section of the document...",
        "Adding the new content to paragraph {n}...",
        "Updating the {field} with the latest information...",
        "Formatting the text and adjusting the layout...",
    ],
    "presentation_edit": [
        "Editing slide {n} — updating the {element}...",
        "Adjusting the layout on slide {n}...",
        "Adding new content to the {placeholder}...",
    ],
    "send_message": [
        "Composing email to {recipient}...",
        "Writing the subject line and body...",
        "Adding attachments and finalizing...",
        "Sending the message now...",
    ],
    "schedule_meeting": [
        "Creating a new calendar event for {date} at {time}...",
        "Adding {n} attendees to the invite...",
        "Setting the meeting duration to {duration} minutes...",
        "Sending the invitation...",
    ],
    "review_content": [
        "Reading through the document content...",
        "Checking the {section} section for accuracy...",
        "Adding a review comment about {topic}...",
    ],
    "collaborate": [
        "Opening the Teams channel #{channel}...",
        "Posting the update in the group chat...",
        "Sharing the file with the team...",
        "Tagging {person} for their input...",
    ],
    "data_transfer": [
        "Selecting the data in {source}...",
        "Copying the {element} to clipboard...",
        "Switching to {target} and pasting...",
    ],
    "export_publish": [
        "Opening the export dialog...",
        "Selecting {format} as the output format...",
        "Exporting the file...",
    ],
    "organize_files": [
        "Creating the folder '{folder}'...",
        "Moving the selected files to {destination}...",
        "Renaming the file to '{name}'...",
    ],
    "monitor_status": [
        "Opening the project dashboard...",
        "Checking the notification panel...",
        "Reviewing the task board status...",
    ],
    "generic_action": [
        "Clicking on the {element}...",
        "Entering the information in the form...",
        "Confirming the action...",
    ],
}

# ── Helper functions ─────────────────────────────────────────────────

NAMES = ["Sarah", "Mike", "Alex", "Priya", "James", "Lisa", "Chen", "Maria"]
CHANNELS = ["engineering", "product-launches", "general", "leadership",
            "support-escalations", "project-alpha", "design-review"]
FOLDERS = ["Q3-Reports", "Project-Files", "Shared-Documents", "Archive",
           "Onboarding-2026", "Vendor-Eval", "Meeting-Notes"]
APPS = ["Word", "PowerPoint", "Outlook", "Teams"]
DATES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIMES = ["9:00 AM", "9:30 AM", "10:00 AM", "11:00 AM", "1:00 PM",
         "2:00 PM", "2:30 PM", "3:00 PM", "4:00 PM"]


def fill_template(template: str) -> str:
    """Fill placeholders in a template string with random values."""
    replacements = {
        "{n}": str(random.randint(2, 15)),
        "{m}": str(random.randint(1, 8)),
        "{page}": str(random.randint(1, 5)),
        "{total}": str(random.randint(5, 20)),
        "{slide}": str(random.randint(1, 12)),
        "{section}": random.choice(["introduction", "summary", "methodology",
                                     "results", "conclusion", "appendix"]),
        "{title}": random.choice(["Q3 Review", "Project Update", "Sprint Plan",
                                   "Weekly Status", "Product Launch"]),
        "{subject}": random.choice(["Project Update", "Action Items",
                                     "Meeting Follow-up", "Review Required"]),
        "{person}": random.choice(NAMES),
        "{file}": random.choice(["Q3_Report.docx", "status_update.pptx",
                                  "meeting_notes.docx", "backlog.xlsx"]),
        "{location}": random.choice(["shared drive", "project folder",
                                      "team channel", "recent documents"]),
        "{date}": random.choice(DATES),
        "{time}": random.choice(TIMES),
        "{duration}": str(random.choice([30, 45, 60, 90])),
        "{day}": random.choice(DATES),
        "{source}": random.choice(APPS),
        "{target}": random.choice(APPS),
        "{format}": random.choice(["PDF", "DOCX", "PPTX"]),
        "{pct}": str(random.choice([25, 50, 75, 100])),
        "{size}": f"{random.uniform(0.5, 15.0):.1f} MB",
        "{folder}": random.choice(FOLDERS),
        "{channel}": random.choice(CHANNELS),
        "{app}": random.choice(APPS),
        "{element}": random.choice(["title", "chart", "table", "text box"]),
        "{placeholder}": random.choice(["title", "subtitle", "content area"]),
        "{recipient}": f"{random.choice(NAMES).lower()}@company.com",
        "{query}": random.choice(["quarterly report", "meeting notes",
                                   "project plan", "status update"]),
        "{field}": random.choice(["revenue figures", "completion dates",
                                   "team assignments", "budget numbers"]),
        "{topic}": random.choice(["formatting", "data accuracy",
                                   "missing section", "unclear wording"]),
        "{name}": random.choice(["Final_v2.docx", "Approved_Draft.pptx"]),
        "{destination}": random.choice(FOLDERS),
        "{item}": random.choice(["CI/CD pipeline", "staging server",
                                  "email delivery", "SSO service"]),
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def generate_agent_turn(skill: str, is_first: bool = False,
                        task_description: str = "") -> str:
    """Generate a full agent response with reasoning + action + observation."""
    parts = []

    # 1. Reasoning trace (thinking out loud)
    if is_first:
        steps = ", ".join(random.sample(list(SKILL_REASONS.keys()), min(3, len(SKILL_REASONS))))
        reasoning = fill_template(random.choice(REASONING_TEMPLATES["planning"]))
        reasoning = reasoning.replace("{task}", task_description[:60])
        reasoning = reasoning.replace("{steps}", steps)
        reasoning = reasoning.replace("{first_step}", f"use {skill}")
        reasoning = reasoning.replace("{next_step}", "proceed to the next step")
        parts.append(f"**[Thinking]** {reasoning}")
    else:
        reason = SKILL_REASONS.get(skill, "this is the next logical step")
        skill_reasoning = fill_template(random.choice(REASONING_TEMPLATES["skill_selection"]))
        skill_reasoning = skill_reasoning.replace("{skill}", skill)
        skill_reasoning = skill_reasoning.replace("{reason}", reason)
        parts.append(f"**[Thinking]** {skill_reasoning}")

    # 2. Action description
    actions = ACTION_DESCRIPTIONS.get(skill, ACTION_DESCRIPTIONS["generic_action"])
    action_desc = fill_template(random.choice(actions))
    parts.append(f"**[Action: {skill}]** {action_desc}")

    # 3. Screen observation
    observations = SCREEN_OBSERVATIONS.get(skill, SCREEN_OBSERVATIONS["generic_action"])
    observation = fill_template(random.choice(observations))
    parts.append(f"**[Observation]** {observation}")

    # 4. Completion note
    completion = fill_template(random.choice(REASONING_TEMPLATES["completion"]))
    completion = completion.replace("{action}", f"completed the {skill} step")
    completion = completion.replace("{observation}", "Everything looks good.")
    parts.append(completion)

    return "\n\n".join(parts)


def generate_correction_response(correction_msg: str, skill: str) -> str:
    """Generate agent response to a user correction."""
    adaptation = fill_template(random.choice(REASONING_TEMPLATES["adaptation"]))
    adaptation = adaptation.replace("{adjustment}", f"redo the {skill} step with your updated instructions")

    action_desc = fill_template(random.choice(
        ACTION_DESCRIPTIONS.get(skill, ACTION_DESCRIPTIONS["generic_action"])))
    observation = fill_template(random.choice(
        SCREEN_OBSERVATIONS.get(skill, SCREEN_OBSERVATIONS["generic_action"])))

    return (
        f"**[Adapting]** {adaptation}\n\n"
        f"**[Action: {skill}]** {action_desc}\n\n"
        f"**[Observation]** {observation}\n\n"
        f"I've incorporated your feedback. The changes are reflected now."
    )


def generate_failure_turn(failure: dict, skill: str) -> str:
    """Generate an agent turn that encounters and handles a failure."""
    parts = []

    # 1. Reasoning — skill selection (normal start)
    reason = SKILL_REASONS.get(skill, "this is the next logical step")
    skill_reasoning = fill_template(random.choice(REASONING_TEMPLATES["skill_selection"]))
    skill_reasoning = skill_reasoning.replace("{skill}", skill)
    skill_reasoning = skill_reasoning.replace("{reason}", reason)
    parts.append(f"**[Thinking]** {skill_reasoning}")

    # 2. Action attempt
    actions = ACTION_DESCRIPTIONS.get(skill, ACTION_DESCRIPTIONS["generic_action"])
    action_desc = fill_template(random.choice(actions))
    parts.append(f"**[Action: {skill}]** {action_desc}")

    # 3. Failure observation
    parts.append(failure["observation"])

    # 4. Failure diagnosis
    diagnosis = random.choice(REASONING_TEMPLATES["failure_detected"])
    parts.append(f"**[Thinking]** {diagnosis}")
    parts.append(f"**[Diagnosis]** {failure['error']}")

    # 5. Recovery plan
    recovery = random.choice(REASONING_TEMPLATES["recovery_attempt"])
    recovery = recovery.replace("{detail}", failure["recovery_detail"])
    parts.append(f"**[Recovery: {failure['recovery']}]** {recovery}")

    # 6. Recovery outcome
    if failure.get("recovery_success", True):
        outcome = random.choice(REASONING_TEMPLATES["recovery_success"])
        parts.append(outcome)
    else:
        outcome = random.choice(REASONING_TEMPLATES["recovery_failed"])
        parts.append(outcome)

    return "\n\n".join(parts)


def generate_conversation(scenario: dict, conv_id: str) -> dict:
    """Generate a full multi-turn conversation from a scenario template."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    skill_flow = scenario["skill_flow"]
    user_messages = list(scenario["user_messages"])  # copy to avoid mutation
    corrections = dict(scenario.get("corrections", []))

    # Build failure lookup: position → failure dict
    failures_by_pos = {}
    for f in scenario.get("failures", []):
        failures_by_pos[f["position"]] = f

    # Ensure we have enough user messages for the skill flow
    while len(user_messages) < len(skill_flow) + 1:
        user_messages.append("Continue with the next step.")

    # First user message: the task request
    messages.append({"role": "user", "content": user_messages[0]})

    # Agent acknowledges and plans
    agent_intro = generate_agent_turn(
        skill_flow[0], is_first=True, task_description=scenario["task"])
    messages.append({"role": "assistant", "content": agent_intro})

    # Multi-turn execution
    for i, skill in enumerate(skill_flow[1:], start=1):
        # User provides next instruction
        if i < len(user_messages):
            messages.append({"role": "user", "content": user_messages[i]})
        else:
            messages.append({"role": "user", "content": "Go ahead with the next step."})

        # Check if this step has a failure
        if i in failures_by_pos:
            failure = failures_by_pos[i]
            # Agent attempts skill but encounters failure
            agent_response = generate_failure_turn(failure, skill)
            messages.append({"role": "assistant", "content": agent_response})

            # If recovery failed, user must provide input
            if not failure.get("recovery_success", True):
                user_input = random.choice(REASONING_TEMPLATES["user_input_on_failure"])
                messages.append({"role": "user", "content": user_input})
                # Agent responds to user input and retries
                retry_response = (
                    f"**[Thinking]** Understood. I'll proceed with your guidance.\n\n"
                    f"**[Action: {skill}]** "
                    + fill_template(random.choice(
                        ACTION_DESCRIPTIONS.get(skill, ACTION_DESCRIPTIONS["generic_action"])))
                    + f"\n\n**[Observation]** "
                    + fill_template(random.choice(
                        SCREEN_OBSERVATIONS.get(skill, SCREEN_OBSERVATIONS["generic_action"])))
                    + "\n\nDone — completed with your guidance. Moving on."
                )
                messages.append({"role": "assistant", "content": retry_response})
        else:
            # Normal execution (no failure)
            agent_response = generate_agent_turn(skill)
            messages.append({"role": "assistant", "content": agent_response})

        # Check if there's a correction at this position
        if i in corrections:
            correction_text = corrections[i]
            messages.append({"role": "user", "content": correction_text})
            correction_response = generate_correction_response(correction_text, skill)
            messages.append({"role": "assistant", "content": correction_response})

    # Final user message
    closing_messages = [
        "Great, that's everything. Thanks for the help!",
        "Perfect, we're all set. Thanks!",
        "Looks good, appreciate the help!",
        "That's all I needed. Thank you!",
        "Excellent work, thanks for handling all of that!",
    ]
    messages.append({"role": "user", "content": random.choice(closing_messages)})
    messages.append({"role": "assistant", "content":
        "You're welcome! Here's a summary of what we accomplished:\n\n"
        + "\n".join(f"- **{skill}**: completed" for skill in skill_flow)
        + "\n\nLet me know if you need anything else!"
    })

    return {
        "conversation_id": conv_id,
        "task": scenario["task"],
        "complexity": scenario["complexity"],
        "apps": scenario["apps"],
        "skill_flow": skill_flow,
        "num_turns": len(messages),
        "has_failures": bool(failures_by_pos),
        "messages": messages,
    }


def augment_scenario(base_scenario: dict) -> dict:
    """Create a variation of a scenario by shuffling/modifying details."""
    scenario = {k: v for k, v in base_scenario.items()}

    # Randomly modify skill flow (swap adjacent, add/remove one)
    flow = list(scenario["skill_flow"])
    if len(flow) > 3 and random.random() < 0.3:
        # Swap two adjacent skills
        idx = random.randint(0, len(flow) - 2)
        flow[idx], flow[idx + 1] = flow[idx + 1], flow[idx]
    if random.random() < 0.2:
        # Add an extra skill
        extra = random.choice(list(SKILL_REASONS.keys()))
        pos = random.randint(1, len(flow))
        flow.insert(pos, extra)
    if len(flow) > 3 and random.random() < 0.15:
        # Remove a skill
        flow.pop(random.randint(1, len(flow) - 1))

    scenario["skill_flow"] = flow

    # Vary corrections
    if random.random() < 0.4:
        pos = random.randint(1, max(1, len(flow) - 2))
        correction_templates = [
            "Wait, let me change that — use a different approach.",
            "Actually, can you redo that? I want it formatted differently.",
            "Hold on, I forgot to mention — also include the latest numbers.",
            "Let me correct that — the date should be next week, not this week.",
            "One thing — add a note about the deadline at the bottom.",
        ]
        new_corrections = list(scenario.get("corrections", []))
        new_corrections.append((pos, random.choice(correction_templates)))
        scenario["corrections"] = new_corrections

    # Randomly inject a failure into non-failure scenarios
    if "failures" not in scenario and random.random() < 0.25:
        pos = random.randint(1, max(1, len(flow) - 2))
        skill_at_pos = flow[pos] if pos < len(flow) else flow[-1]
        random_failures = [
            {
                "position": pos,
                "skill": skill_at_pos,
                "failure_type": "api_timeout",
                "error": f"The API call for {skill_at_pos} timed out after 30 seconds. The service may be experiencing high load.",
                "observation": f"**[Error]** Request to the {skill_at_pos} service timed out (HTTP 504). Attempt 1 of 3 failed.",
                "recovery": "retry_with_backoff",
                "recovery_detail": "I'll retry with exponential backoff — waiting 5 seconds, then 15, then 30.",
                "recovery_success": True,
            },
            {
                "position": pos,
                "skill": skill_at_pos,
                "failure_type": "stale_cache",
                "error": f"The cached data for this {skill_at_pos} operation is stale. The underlying resource has been modified since the cache was populated.",
                "observation": f"**[Error]** Cache validation failed — ETag mismatch. Cached version: `v3`, current version: `v5`. The data has been updated by another process.",
                "recovery": "invalidate_and_refresh",
                "recovery_detail": "I'll invalidate the cache, fetch the latest version, and redo the operation with fresh data.",
                "recovery_success": True,
            },
            {
                "position": pos,
                "skill": skill_at_pos,
                "failure_type": "network_error",
                "error": "The network connection to the service was reset unexpectedly. This could be a transient issue.",
                "observation": "**[Error]** `ConnectionResetError: [Errno 104] Connection reset by peer` during API call. The server may have been restarted.",
                "recovery": "retry",
                "recovery_detail": "This looks like a transient network issue. I'll wait a moment and retry the connection.",
                "recovery_success": True,
            },
        ]
        scenario["failures"] = [random.choice(random_failures)]
    elif "failures" in scenario:
        # Keep existing failures but adjust positions if skill flow changed
        adjusted = []
        for f in scenario["failures"]:
            if f["position"] < len(flow):
                adjusted.append(f)
        scenario["failures"] = adjusted

    return scenario


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn CUA conversation data")
    parser.add_argument("--num", type=int, default=1000,
                        help="Number of conversations to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Generating {args.num} multi-turn conversations...")
    print(f"Base scenarios: {len(SCENARIOS)}")

    conversations = []
    for i in range(args.num):
        # Pick a base scenario and augment it
        base = random.choice(SCENARIOS)
        scenario = augment_scenario(base)
        conv_id = f"conv_{i:05d}"
        conv = generate_conversation(scenario, conv_id)
        conversations.append(conv)

    # Shuffle and split
    random.shuffle(conversations)
    split = int(0.85 * len(conversations))
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    # Save as JSONL
    for path, convs in [(OUTPUT_TRAIN, train_convs), (OUTPUT_VAL, val_convs)]:
        with open(path, "w") as f:
            for conv in convs:
                f.write(json.dumps(conv) + "\n")
        print(f"Saved {len(convs)} conversations to {path}")

    # Statistics
    print(f"\n{'='*60}")
    print("CONVERSATION STATISTICS")
    print(f"{'='*60}")
    print(f"Total conversations: {len(conversations)}")
    print(f"Train: {len(train_convs)}, Val: {len(val_convs)}")

    turns = [c["num_turns"] for c in conversations]
    print(f"\nTurns per conversation: min={min(turns)}, max={max(turns)}, "
          f"mean={sum(turns)/len(turns):.1f}")

    skill_counts = Counter()
    for c in conversations:
        for s in c["skill_flow"]:
            skill_counts[s] += 1
    print(f"\nSkill distribution across all conversations:")
    for skill, count in skill_counts.most_common():
        print(f"  {skill:20s}: {count:5d} ({100*count/sum(skill_counts.values()):.1f}%)")

    complexity_counts = Counter(c["complexity"] for c in conversations)
    print(f"\nComplexity distribution:")
    for comp, count in complexity_counts.most_common():
        print(f"  {comp:10s}: {count}")

    # Failure statistics
    failure_convs = sum(1 for c in conversations if c.get("has_failures", False))
    failure_types = Counter()
    for c in conversations:
        for msg in c["messages"]:
            if msg["role"] == "assistant" and "**[Error]**" in msg["content"]:
                if "Timeout" in msg["content"] or "timed out" in msg["content"]:
                    failure_types["api_timeout"] += 1
                elif "403" in msg["content"] or "Permission" in msg["content"]:
                    failure_types["permission_denied"] += 1
                elif "401" in msg["content"] or "Unauthorized" in msg["content"]:
                    failure_types["auth_expired"] += 1
                elif "429" in msg["content"] or "Rate" in msg["content"]:
                    failure_types["rate_limit"] += 1
                elif "conflict" in msg["content"].lower() or "locked" in msg["content"].lower():
                    failure_types["conflict/lock"] += 1
                elif "mismatch" in msg["content"].lower() or "schema" in msg["content"].lower():
                    failure_types["format/schema"] += 1
                elif "cache" in msg["content"].lower() or "stale" in msg["content"].lower():
                    failure_types["stale_cache"] += 1
                elif "Connection" in msg["content"] or "network" in msg["content"].lower():
                    failure_types["network_error"] += 1
                elif "Memory" in msg["content"] or "crashed" in msg["content"]:
                    failure_types["memory/crash"] += 1
                else:
                    failure_types["other"] += 1

    print(f"\nFailure statistics:")
    total_failures = sum(failure_types.values())
    print(f"  Conversations with failures: ~{total_failures} failure events")
    for ftype, count in failure_types.most_common():
        print(f"  {ftype:20s}: {count}")

    # Print example conversation
    print(f"\n{'='*60}")
    print("EXAMPLE CONVERSATION (first 6 turns)")
    print(f"{'='*60}")
    ex = train_convs[0]
    print(f"Task: {ex['task']}")
    print(f"Skills: {' → '.join(ex['skill_flow'])}")
    print(f"Turns: {ex['num_turns']}")
    for msg in ex["messages"][:6]:
        role = msg["role"].upper()
        content = msg["content"][:300]
        print(f"\n[{role}]")
        print(content + ("..." if len(msg["content"]) > 300 else ""))


if __name__ == "__main__":
    main()
