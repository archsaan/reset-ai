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

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.prompts import build_system_prompt
from app.agent.state import AgentState
from app.agent.tools import all_tools
from app.db.admin_db import get_agent_by_slug, log_usage
from app.db.checkpointer import checkpointer


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
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


tool_node = ToolNode(all_tools)


def build_graph(agent: dict) -> StateGraph:
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", make_agent_node(agent))
    graph_builder.add_node("tools", tool_node)
    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph_builder.add_edge("tools", "agent")
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
