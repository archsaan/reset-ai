"""
Central configuration for the whole app.

Everything here is a constant read at import time. Nothing that depends on
a request, a user, or the database belongs in this file — that's what
app/db and app/agent are for. Keeping all the "knobs" in one place means
that when this moves to a real settings system (pydantic-settings + env
vars) later, this is the only file that changes shape.
"""

import os

from dotenv import load_dotenv

# Loads variables from a local .env file (if present) into os.environ.
# In deployment (Render/Railway/etc.) you set these as real environment
# variables in the platform's dashboard instead of shipping a .env file.
load_dotenv()

# ---------------------------------------------------------
# External schedule API (ProfitConnect)
# ---------------------------------------------------------

SCHEDULE_API_URL = "https://crmapi.profitconnect.co/calendar/get/dayschedule"
SCHEDULE_API_KEY = "PASTE_KEY_HERE_IF_NEEDED"

# ---------------------------------------------------------
# Booking API (real, once details are available)
# ---------------------------------------------------------
# TODO: fill these in once the real booking endpoints are confirmed.
# Until then, app/agent/tools/booking_tools.py falls back to an
# in-memory mock so the rest of the graph keeps working.

BOOKING_CREATE_API_URL = ""
BOOKING_CANCEL_API_URL = ""
BOOKING_API_KEY = ""

# ---------------------------------------------------------
# Databases
# ---------------------------------------------------------
# DATABASE_URL comes from your .env file locally (see .env.example) or
# from the hosting platform's environment variables in production.
# Example (Neon): postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require
#
# Both the admin DB (config/KB/usage) and the LangGraph checkpointer
# (conversation memory) now live in this SAME Postgres database, in
# separate tables — one connection string, one thing to back up.

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Add it to a .env file in the project root "
        "(see .env.example) or set it as a real environment variable in "
        "your hosting platform's dashboard."
    )

# ---------------------------------------------------------
# RAG / embeddings (knowledge base retrieval)
# ---------------------------------------------------------
# Voyage AI is Anthropic's recommended embedding provider for RAG with
# Claude. Get a free-tier key at https://dash.voyageai.com
#
# voyage-3-lite = 512 dimensions, cheap and fast — plenty for a KB of
# pricing/FAQ/policy docs. If retrieval quality ever becomes the
# bottleneck, switch to voyage-3 (1024 dims) — just update both
# EMBEDDING_MODEL and EMBEDDING_DIM together and re-index the KB.

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_DIM = 512
KB_RETRIEVAL_TOP_K = 4  # how many chunks to pull per question

# ---------------------------------------------------------
# Auth (placeholder — swap for the real member/admin auth system later)
# ---------------------------------------------------------

MOCK_ADMIN_TOKEN = "dev-admin-token-change-me"

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

AVAILABLE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-5"]

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
# RULES is shared across every agent since all agents currently use the
# same tool set (schedule/booking). If an agent ever gets its own tools
# (e.g. a PFC-specific CRM tool), split this into per-agent rules too.

RULES = (
    "Only answer factual questions (pricing, location, policies, services) using the "
    "Knowledge Base above. Never guess or invent details not contained in it. "
    "If the answer isn't in the Knowledge Base, say: 'Our team will have all "
    "the details and will be in touch with you very shortly! 😊' "
    "\n\nFor anything related to class times, schedules, or availability on a specific "
    "date — NEVER use static text, and never guess. Always use the get_schedule or "
    "check_availability tool instead, since those pull real, live data. "
    "\n\nUse get_schedule when a lead asks generally what's on or wants to browse classes. "
    "Use check_availability when a lead names a specific class and wants to know if "
    "there's space, or wants to book — this tool also checks upcoming days automatically "
    "if the requested date is full. Each result includes a 'day_label' field — always "
    "trust and use that label rather than calculating the day yourself. "
    "Use book_class ONLY after check_availability has confirmed there is space, and only "
    "once you have the lead's name."
)

# ---------------------------------------------------------
# Default agents — seeded into the `agents` table the first time the app
# starts against a fresh database. Editable afterward through the admin
# dashboard; this is just the starting point.
# ---------------------------------------------------------

DEFAULT_TRIBE_APP_SYSTEM_PROMPT = (
    "You are Riley, the Reset Fitness AI Assistant. Your goal is to make every lead "
    "feel heard and excited about Reset Fitness — answering questions accurately and "
    "helping them book classes. You are warm, friendly, confident, and never robotic. "
    "If asked your name, say 'I am Riley, the Reset Fitness AI Assistant! 😊' "
    "If asked if you are human, say 'I am Riley, Reset Fitness' AI Assistant — here to "
    "help just like a real team member would! 😊' Never claim to be human. "
    "Keep responses concise (2-3 sentences where possible) and use 1-2 emojis, but "
    "never on complaints or sensitive topics."
)

DEFAULT_PFC_SYSTEM_PROMPT = (
    "You are the Reset Fitness PFC Assistant, a support tool for Reset Fitness studio "
    "staff using the ProfitConnect (PFC) system. Your job is to help staff quickly find "
    "class schedule information, check class capacity, and support day-to-day front-desk "
    "and CRM tasks. You speak to trained staff, not members — be direct, concise, and "
    "skip the member-facing warmth and emojis. If asked to do something outside what "
    "your current tools support (e.g. editing a member's profile, processing a refund), "
    "say plainly that this isn't wired up yet rather than guessing at an answer."
)

DEFAULT_AGENTS = [
    {
        "slug": "tribe-app",
        "name": "Tribe App Agent",
        "description": "Helps Reset Fitness app members book classes and check membership status.",
        "system_prompt": DEFAULT_TRIBE_APP_SYSTEM_PROMPT,
        "active_model": AVAILABLE_MODELS[0],
    },
    {
        "slug": "pfc",
        "name": "PFC Agent",
        "description": "Helps Reset Fitness studio staff with CRM/ProfitConnect activities.",
        "system_prompt": DEFAULT_PFC_SYSTEM_PROMPT,
        "active_model": AVAILABLE_MODELS[0],
    },
]
