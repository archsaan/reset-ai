"""
The state shape LangGraph carries between nodes for one turn of the
conversation graph. Extend this as the agent needs to track more
per-conversation context (e.g. a pending booking awaiting approval).
"""

from typing import Optional

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    lead_email: Optional[str]
