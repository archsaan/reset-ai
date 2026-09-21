"""
Public chat endpoint — used by whichever app is calling it (the Tribe
member app, the PFC staff tool, WhatsApp, etc.), disambiguated by the
`agent` slug in the request body.
"""

import uuid

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage
from langgraph.types import Command

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

    # If the previous turn paused on an approval gate (e.g. awaiting a
    # booking confirmation), this incoming message is the member's answer
    # to that question, not a fresh request — resume the paused graph with
    # it rather than starting a new turn. state.next is non-empty only
    # when a node is paused mid-execution waiting on interrupt().
    pending_state = graph.get_state(config)
    if pending_state.next:
        result = graph.invoke(Command(resume=request.message), config=config)
    else:
        input_state = {"messages": [HumanMessage(content=request.message)], "_thread_id": thread_id}
        if request.email:
            input_state["lead_email"] = request.email.strip().lower()
        result = graph.invoke(input_state, config=config)

    if "__interrupt__" in result:
        # Graph paused again (e.g. asking the member to confirm a
        # booking). Surface the question as the reply text so the calling
        # app (Tribe app, etc.) needs no changes — it just looks like the
        # assistant asked a question, and the member's next message will
        # be treated as the answer via the branch above.
        interrupt_obj = result["__interrupt__"][0]
        reply_text = interrupt_obj.value.get("question", "Can you confirm this booking? (yes/no)")
    else:
        reply_text = result["messages"][-1].content

    return {"session_id": thread_id, "agent": agent["slug"], "reply": reply_text}
