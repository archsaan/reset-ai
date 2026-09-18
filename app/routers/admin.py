"""
Admin dashboard routes — agent manager (create/edit/delete agents, each
with its own system prompt + knowledge base + model), usage/cost view,
and per-agent test chat.

Every route below (other than /admin/login) depends on require_admin, so
swapping mock auth for real auth later is a one-line change in
app/core/security.py, not a change to any route here.
"""

import uuid

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from langchain_core.messages import HumanMessage

from app.agent.graph import get_compiled_graph
from app.core.config import AVAILABLE_MODELS, MOCK_ADMIN_TOKEN
from app.core.security import require_admin
from app.db.admin_db import (
    create_agent,
    delete_agent,
    delete_kb_doc as db_delete_kb_doc,
    get_agent_by_id,
    get_usage_summary,
    insert_kb_doc,
    list_agents,
    list_kb_docs as db_list_kb_docs,
    toggle_kb_doc as db_toggle_kb_doc,
    update_agent,
)
from app.rag.retrieval import index_kb_doc
from app.schemas.agent import AgentCreate, AgentUpdate
from app.schemas.chat import TestChatRequest

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/login")
def admin_login(payload: dict):
    """Mock login — replace with real auth later. Accepts any non-empty
    username/password for now just to unblock frontend development."""
    username = payload.get("username", "")
    password = payload.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    # TODO: replace with real credential check against your auth system
    return {"token": MOCK_ADMIN_TOKEN}


# ---------------------------------------------------------
# Agents
# ---------------------------------------------------------

@router.get("/agents")
def get_agents(authorization: str = Header(None)):
    require_admin(authorization)
    return list_agents()


@router.post("/agents")
def create_agent_route(payload: AgentCreate, authorization: str = Header(None)):
    require_admin(authorization)
    if payload.active_model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"active_model must be one of {AVAILABLE_MODELS}")
    try:
        new_id = create_agent(
            payload.slug, payload.name, payload.description,
            payload.system_prompt, payload.active_model
        )
    except Exception as exc:
        # Most likely a duplicate slug (agents.slug has a UNIQUE constraint)
        raise HTTPException(status_code=400, detail=f"Could not create agent: {exc}")
    return {"status": "created", "id": new_id}


@router.get("/agents/{agent_id}")
def get_agent_route(agent_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    agent = get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/agents/{agent_id}")
def update_agent_route(agent_id: int, payload: AgentUpdate, authorization: str = Header(None)):
    require_admin(authorization)
    if not get_agent_by_id(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    if payload.active_model is not None and payload.active_model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"active_model must be one of {AVAILABLE_MODELS}")
    update_agent(
        agent_id,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        active_model=payload.active_model,
    )
    return {"status": "updated"}


@router.delete("/agents/{agent_id}")
def delete_agent_route(agent_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    if not get_agent_by_id(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    delete_agent(agent_id)
    return {"status": "deleted"}


# ---------------------------------------------------------
# Knowledge base (scoped to an agent)
# ---------------------------------------------------------

@router.get("/agents/{agent_id}/knowledge-base")
def list_kb_docs_route(agent_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    if not get_agent_by_id(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    rows = db_list_kb_docs(agent_id)
    return [{"id": r["id"], "filename": r["filename"], "active": bool(r["active"]), "uploaded_at": r["uploaded_at"]} for r in rows]


@router.post("/agents/{agent_id}/knowledge-base")
async def upload_kb_doc_route(agent_id: int, file: UploadFile = File(...), authorization: str = Header(None)):
    require_admin(authorization)
    if not get_agent_by_id(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")

    content_bytes = await file.read()
    content_text = content_bytes.decode("utf-8", errors="ignore")
    doc_id = insert_kb_doc(agent_id, file.filename, content_text)

    try:
        chunk_count = index_kb_doc(doc_id, content_text)
    except Exception as exc:
        # The doc is still saved and usable via the old full-KB-dump
        # fallback in prompts.py even if embedding fails (e.g. Voyage
        # API key missing/invalid) — surface the error but don't fail
        # the upload itself.
        return {"status": "uploaded", "filename": file.filename, "indexed": False, "index_error": str(exc)}

    return {"status": "uploaded", "filename": file.filename, "indexed": True, "chunks": chunk_count}


@router.put("/agents/{agent_id}/knowledge-base/{doc_id}/toggle")
def toggle_kb_doc_route(agent_id: int, doc_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    new_state = db_toggle_kb_doc(doc_id)
    if new_state is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "updated", "active": bool(new_state)}


@router.delete("/agents/{agent_id}/knowledge-base/{doc_id}")
def delete_kb_doc_route(agent_id: int, doc_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    db_delete_kb_doc(doc_id)
    return {"status": "deleted"}


# ---------------------------------------------------------
# Usage
# ---------------------------------------------------------

def _summarize_usage(rows) -> list[dict]:
    rate_per_million = {"claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
                         "claude-sonnet-4-5": {"input": 3.0, "output": 15.0}}
    summary = []
    for model, requests, total_input, total_output in rows:
        total_input = total_input or 0
        total_output = total_output or 0
        rates = rate_per_million.get(model, {"input": 0, "output": 0})
        est_cost = (total_input / 1_000_000 * rates["input"]) + (total_output / 1_000_000 * rates["output"])
        summary.append({
            "model": model, "requests": requests,
            "total_input_tokens": total_input, "total_output_tokens": total_output,
            "estimated_cost_usd": round(est_cost, 4),
        })
    return summary


@router.get("/usage")
def get_global_usage(authorization: str = Header(None)):
    """Usage summed across every agent."""
    require_admin(authorization)
    return _summarize_usage(get_usage_summary())


@router.get("/agents/{agent_id}/usage")
def get_agent_usage(agent_id: int, authorization: str = Header(None)):
    require_admin(authorization)
    if not get_agent_by_id(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return _summarize_usage(get_usage_summary(agent_id))


# ---------------------------------------------------------
# Test chat (scoped to an agent, throwaway thread)
# ---------------------------------------------------------

@router.post("/agents/{agent_id}/test-chat")
def test_chat_route(agent_id: int, request: TestChatRequest, authorization: str = Header(None)):
    """Runs the given agent with a throwaway thread_id, so testing in the
    dashboard never pollutes real member/staff conversation history."""
    require_admin(authorization)
    agent = get_agent_by_id(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    graph, _ = get_compiled_graph(agent["slug"])
    test_thread_id = f"admin-test:{agent['slug']}:{uuid.uuid4()}"
    config = {"configurable": {"thread_id": test_thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)], "_thread_id": test_thread_id},
        config=config
    )
    return {"reply": result["messages"][-1].content}
