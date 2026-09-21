"""
LangGraph wiring: the agent node, the tool node, the conditional edge
that lets the model loop tools -> agent as many times as it needs in one
turn, and the compiled graph the routers actually call.

The graph is rebuilt per request using whichever agent (system prompt +
KB + model) the caller asked for, by slug. That's a deliberate, simple
choice for now — cache or optimize this once request volume makes it
worth it. All agents currently share the same tool set (all_tools);
give an agent its own tools later by branching on agent["slug"] in
build_graph() if that's ever needed.
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.agent.tools import all_tools
from app.agent.tools.booking_tools import execute_booking
from app.core.config import XAI_API_KEY, XAI_BASE_URL
from app.db.admin_db import get_agent_by_slug, log_usage
from app.db.checkpointer import checkpointer

# Deny-by-default: a reply only counts as confirmation if it's clearly a
# "yes". Anything else (a question, "wait", silence-equivalent, a typo,
# or anything starting with a negation like "please cancel") is treated
# as NOT confirmed — this gates a real booking, so ambiguous must never
# resolve to "go ahead". This is a deterministic keyword check, not an
# LLM judgment call, for the same reason. Deliberately narrow: words like
# "please" or "book" are excluded from the first-word check because they
# also start plenty of declines ("please don't", "book a different one").
_AFFIRMATIVE_PHRASES = {
    "yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "sure",
    "ok", "okay", "correct", "that's right", "sounds good", "go ahead",
    "book it", "please book it", "confirm it", "yes please", "yes book it",
}
_AFFIRMATIVE_FIRST_WORDS = {
    "yes", "y", "yeah", "yep", "yup", "confirm", "confirmed", "sure",
    "ok", "okay", "correct",
}


def _is_affirmative(reply: str) -> bool:
    normalized = (reply or "").strip().lower().rstrip(".!")
    if not normalized:
        return False
    if normalized in _AFFIRMATIVE_PHRASES:
        return True
    first_word = normalized.split()[0]
    return first_word in _AFFIRMATIVE_FIRST_WORDS


def _latest_human_message(messages: list) -> str | None:
    """Finds the most recent user message in the conversation, used as
    the retrieval query for KB search. Walking backwards handles the
    tools -> agent loop, where the last message might be a ToolMessage
    rather than the user's actual question."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return None


def get_model_with_tools(model_name: str):
    """Picks the right client by model name. Grok models go through
    ChatOpenAI pointed at xAI's OpenAI-compatible endpoint (xAI ships no
    separate SDK/langchain package needed for this); everything else is
    assumed to be a Claude model."""
    if model_name.startswith("grok"):
        if not XAI_API_KEY:
            raise RuntimeError(
                "XAI_API_KEY is not set. Add it to your .env file (see .env.example) "
                "to use Grok models — get one at https://console.x.ai"
            )
        model = ChatOpenAI(model=model_name, api_key=XAI_API_KEY, base_url=XAI_BASE_URL, max_tokens=500)
    else:
        model = ChatAnthropic(model=model_name, max_tokens=500)
    return model.bind_tools(all_tools)


def make_agent_node(agent: dict):
    model_with_tools = get_model_with_tools(agent["active_model"])

    def agent_node(state: AgentState):
        lead_email = state.get("lead_email")
        user_question = _latest_human_message(state["messages"])
        system_prompt = build_system_prompt(agent, lead_email, user_question)
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = model_with_tools.invoke(messages)

        usage = getattr(response, "usage_metadata", None) or {}
        thread_id = state.get("_thread_id", "unknown")
        log_usage(
            agent["id"], thread_id, agent["active_model"],
            usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        )
        return {"messages": [response]}
    return agent_node


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return END
    # book_class is the only tool with real-world consequences, so any
    # call to it is routed to the approval gate instead of the generic
    # tool node. Current prompting only ever has the model call book_class
    # on its own (after a prior check_availability turn), so this doesn't
    # need to split a mixed batch of tool calls — confirm_booking_node
    # still handles that case defensively, see its own comments.
    if any(call["name"] == "book_class" for call in tool_calls):
        return "confirm_booking"
    return "tools"


def confirm_booking_node(state: AgentState):
    """The approval gate. Runs instead of the generic ToolNode whenever
    the model's last message contains a book_class call. Physically
    pauses the graph with interrupt() — no real HTTP call happens until
    a resumed invocation supplies the member's actual reply, and even
    then only if that reply reads as a clear yes (see _is_affirmative).
    This is enforced here in code, not left to the model's judgment or
    to prompt instructions, because a prompt can be misread or bypassed
    but a paused graph can't proceed without this code choosing to let it.
    """
    last_message = state["messages"][-1]
    tool_messages = []

    for call in last_message.tool_calls:
        if call["name"] != "book_class":
            # Defensive: current prompting never has the model batch other
            # tool calls together with book_class, but if it ever does,
            # don't silently execute or silently drop it — surface a
            # clear tool_result so the model retries it on its own turn.
            tool_messages.append(ToolMessage(
                content=json.dumps({
                    "success": False,
                    "reason": "This tool must be called on its own turn, separately from book_class.",
                }),
                tool_call_id=call["id"],
            ))
            continue

        args = call["args"]
        question = (
            f"Please confirm: book {args.get('class_name')} on {args.get('target_date')} "
            f"at {args.get('start_time')} for {args.get('lead_name')}? (yes/no)"
        )
        # Pauses here. The checkpointer persists this state; the graph
        # only continues past this line on a resumed invocation, and
        # member_reply below is exactly the value passed to that resume.
        member_reply = interrupt({
            "type": "booking_confirmation",
            "calendar_schedule_id": args.get("calendar_schedule_id"),
            "class_name": args.get("class_name"),
            "target_date": args.get("target_date"),
            "start_time": args.get("start_time"),
            "lead_name": args.get("lead_name"),
            "question": question,
        })

        if _is_affirmative(str(member_reply)):
            result = execute_booking(
                calendar_schedule_id=args.get("calendar_schedule_id"),
                class_name=args.get("class_name"),
                target_date=args.get("target_date"),
                start_time=args.get("start_time"),
                lead_name=args.get("lead_name"),
            )
        else:
            result = {
                "success": False,
                "reason": "The member did not confirm, so this booking was not made.",
            }

        tool_messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))

    return {"messages": tool_messages}


tool_node = ToolNode(all_tools)


def build_graph(agent: dict) -> StateGraph:
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", make_agent_node(agent))
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("confirm_booking", confirm_booking_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "confirm_booking": "confirm_booking", END: END}
    )
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge("confirm_booking", "agent")
    return graph_builder


def get_compiled_graph(agent_slug: str):
    """Looks up the agent by slug and rebuilds the graph for it.
    Simple approach: rebuild per request. Fine at this scale;
    cache/optimize later. Returns None if the slug doesn't match any
    agent, so the caller can return a clean 404 instead of a crash."""
    agent = get_agent_by_slug(agent_slug)
    if agent is None:
        return None, None
    return build_graph(agent).compile(checkpointer=checkpointer), agent
