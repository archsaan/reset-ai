"""
System prompt assembly.

The system prompt is built fresh on every request from the specific
agent's own row in the `agents` table, plus that agent's own knowledge
base — so an admin edit to one agent's prompt or KB takes effect
immediately, and never affects any other agent.

Knowledge base content comes from RAG retrieval (only the chunks
relevant to the user's actual question, scoped to this agent) once that
agent's KB has been indexed into pgvector. Until then — or if retrieval
fails for any reason — it falls back to the old behavior of dumping
every active KB doc's full text into the prompt, so the agent never goes
from "has an answer" to "has nothing" because of a RAG wiring issue.
"""

from datetime import date

from app.core.config import RULES
from app.db.admin_db import get_active_kb_text
from app.rag.retrieval import retrieve_relevant_kb_text


def build_system_prompt(agent: dict, lead_email: str | None = None, user_question: str | None = None) -> str:
    """`agent` is a row from the agents table (see admin_db.get_agent_by_id
    / get_agent_by_slug) — needs at least id and system_prompt."""
    kb_text = _get_kb_text(agent["id"], user_question)
    today_str = date.today().isoformat()
    identity_note = f"\n\nThis lead's email is {lead_email}." if lead_email else ""
    return (
        agent["system_prompt"]
        + "\n\n--- KNOWLEDGE BASE ---\n" + (kb_text or "(no active knowledge base documents)")
        + "\n\n--- RULES ---\n" + RULES
        + f"\n\nToday's actual date is {today_str}. Always trust this over any assumption."
        + identity_note
    )


def _get_kb_text(agent_id: int, user_question: str | None) -> str:
    """RAG retrieval when possible, full-dump fallback otherwise."""
    if user_question:
        try:
            retrieved = retrieve_relevant_kb_text(agent_id, user_question)
            if retrieved is not None:
                return retrieved
        except Exception:
            # RAG is an enhancement, not a dependency — if Voyage/pgvector
            # has a hiccup, degrade to the full KB dump rather than
            # breaking the whole chat response.
            pass
    return get_active_kb_text(agent_id)
