"""
Public chat endpoint — used by whichever app is calling it (the Tribe
member app, the PFC staff tool, WhatsApp, etc.), disambiguated by the
`agent` slug in the request body.
"""

import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.agent.graph import get_compiled_graph
from app.schemas.chat import ChatRequest

router = APIRouter(tags=["chat"])


def make_thread_id(agent_slug: str, email: str | None, session_id: str | None) -> str:
    """Email is the stable cross-session identity key: same lead, same
    thread, regardless of which device or channel they message from.
    The agent slug is included so the same person's Tribe App
    conversation and PFC conversation (if they ever overlap) never
    share memory — each agent gets its own thread per person."""
    identity = email.strip().lower() if email else (session_id or str(uuid.uuid4()))
    return f"{agent_slug}:{identity}"


@router.post("/chat")
def chat(request: ChatRequest):
    graph, agent = get_compiled_graph(request.agent)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"No agent found with slug '{request.agent}'")

    thread_id = make_thread_id(request.agent, request.email, request.session_id)
    config = {"configurable": {"thread_id": thread_id}}

    input_state = {"messages": [HumanMessage(content=request.message)], "_thread_id": thread_id}
    if request.email:
        input_state["lead_email"] = request.email.strip().lower()

    result = graph.invoke(input_state, config=config)

    reply_text = result["messages"][-1].content
    return {"session_id": thread_id, "agent": agent["slug"], "reply": reply_text}
