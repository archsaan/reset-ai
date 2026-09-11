from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import uuid
import httpx
from datetime import date, datetime, timedelta

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

SCHEDULE_API_URL = "https://crmapi.profitconnect.co/calendar/get/dayschedule"
SCHEDULE_API_KEY = "PASTE_KEY_HERE_IF_NEEDED"  # or None if not required

DB_FILE = "conversations.db"

# ---------------------------------------------------------
# Agent identity + knowledge base
# ---------------------------------------------------------

AGENT_IDENTITY = (
    "You are Riley, the Reset Fitness AI Assistant. Your goal is to make every lead "
    "feel heard and excited about Reset Fitness — answering questions accurately and "
    "helping them book classes. You are warm, friendly, confident, and never robotic. "
    "If asked your name, say 'I am Riley, the Reset Fitness AI Assistant! 😊' "
    "If asked if you are human, say 'I am Riley, Reset Fitness' AI Assistant — here to "
    "help just like a real team member would! 😊' Never claim to be human. "
    "Keep responses concise (2-3 sentences where possible) and use 1-2 emojis, but "
    "never on complaints or sensitive topics."
)

with open("knowledge_base.md", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

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
    "once you have the lead's name. If the requested class/date is full, use "
    "check_availability's results to offer the nearest real alternative before booking "
    "anything. You may call multiple tools in sequence within the same turn if needed — "
    "for example, checking availability and then immediately booking once confirmed."
)

def build_system_prompt() -> str:
    today_str = date.today().isoformat()
    return (
        AGENT_IDENTITY
        + "\n\n--- KNOWLEDGE BASE ---\n" + KNOWLEDGE_BASE
        + "\n\n--- RULES ---\n" + RULES
        + f"\n\nToday's actual date is {today_str}. Always trust this over any assumption."
    )

# ---------------------------------------------------------
# Database helpers (unchanged from before)
# ---------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_history(session_id: str):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
        (session_id,)
    ).fetchall()
    conn.close()
    return rows

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# Tools — now defined with LangChain's @tool decorator,
# which auto-generates the schema Claude needs from the
# function signature and docstring (no more manual JSON schema)
# ---------------------------------------------------------

def _fetch_schedule(target_date: str = None):
    """Internal helper — raw schedule fetch, used by all three tools below."""
    if target_date is None:
        target_date = date.today().isoformat()

    payload = {"facility_id": 1, "room_id": "", "date": target_date}
    headers = {}
    if SCHEDULE_API_KEY and SCHEDULE_API_KEY != "PASTE_KEY_HERE_IF_NEEDED":
        headers["Authorization"] = f"Bearer {SCHEDULE_API_KEY}"

    response = httpx.post(SCHEDULE_API_URL, json=payload, headers=headers, timeout=10)
    data = response.json()

    cleaned = []
    for cls in data.get("schedule", []):
        capacity = int(cls["capacity"])
        booked = int(cls["booked"])
        cancelled = int(cls.get("cancelled", 0))
        spots_left = max(capacity - (booked - cancelled), 0)
        cleaned.append({
            "class_name": cls["class_name"],
            "discipline": cls["discipline_name"],
            "start_time": cls["start_time"],
            "end_time": cls["end_time"],
            "coach": f'{cls["coach"][0]["firstname"]} {cls["coach"][0]["lastname"]}' if cls.get("coach") else None,
            "spots_left": spots_left,
            "class_type": cls["class_type"],
        })
    return cleaned


@tool
def get_schedule(target_date: str = None) -> list:
    """Get the full class schedule for Reset Fitness on a given date.
    Use this when a lead asks generally what's on, or what classes are
    available on a specific day. target_date should be in YYYY-MM-DD format;
    defaults to today if not given."""
    return _fetch_schedule(target_date)


@tool
def check_availability(class_name: str, target_date: str = None, max_days_ahead: int = 3) -> list:
    """Check if a specific class has open spots on a given date, and
    automatically checks the next few days too. Use this when a lead names
    a specific class and wants to know if there's space, or wants to book.
    Each result includes a 'day_label' field (e.g. 'today', 'tomorrow', or
    a weekday name) — always trust and use that label rather than
    calculating the day yourself. target_date defaults to today if not given."""
    if target_date is None:
        target_date = date.today().isoformat()

    start = datetime.strptime(target_date, "%Y-%m-%d")
    today = date.today()

    results = []
    for i in range(max_days_ahead + 1):
        check_date_obj = (start + timedelta(days=i)).date()
        check_date = check_date_obj.isoformat()

        if check_date_obj == today:
            day_label = "today"
        elif check_date_obj == today + timedelta(days=1):
            day_label = "tomorrow"
        else:
            day_label = check_date_obj.strftime("%A, %B %d")

        schedule = _fetch_schedule(check_date)
        matches = [c for c in schedule if class_name.lower() in c["class_name"].lower()]

        for cls in matches:
            results.append({
                "date": check_date,
                "day_label": day_label,
                "class_name": cls["class_name"],
                "start_time": cls["start_time"],
                "spots_left": cls["spots_left"],
                "coach": cls["coach"],
            })
    return results


# Fake in-memory bookings — resets on restart. Mock only, no real write.
mock_bookings = []

@tool
def book_class(class_name: str, target_date: str, start_time: str, lead_name: str) -> dict:
    """Book a lead into a specific class. Only call this after
    check_availability has confirmed there is space at the exact date/time.
    Requires the lead's name. target_date in YYYY-MM-DD format, start_time
    matching the schedule format exactly (e.g. '06:30 PM')."""
    matches = [
        cls for cls in _fetch_schedule(target_date)
        if class_name.lower() in cls["class_name"].lower()
        and cls["start_time"] == start_time
    ]

    if not matches:
        return {"success": False, "reason": "Class not found on that date/time."}

    cls = matches[0]
    if cls["spots_left"] <= 0:
        return {"success": False, "reason": "Class is fully booked."}

    booking = {
        "class_name": cls["class_name"],
        "date": target_date,
        "start_time": start_time,
        "lead_name": lead_name,
        "booking_id": str(uuid.uuid4())[:8],
    }
    mock_bookings.append(booking)
    return {"success": True, "booking": booking}


all_tools = [get_schedule, check_availability, book_class]

# ---------------------------------------------------------
# LangGraph setup
# ---------------------------------------------------------

model = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=500)
model_with_tools = model.bind_tools(all_tools)

def agent_node(state: MessagesState):
    """Calls Claude with the current conversation. Claude decides whether
    it needs to call a tool or can answer directly."""
    system_prompt = build_system_prompt()
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: MessagesState):
    """The conditional edge: if the last message has tool calls, go run
    them. Otherwise, the agent is done — end the graph."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

tool_node = ToolNode(all_tools)

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after tools run, loop back to the agent

graph = graph_builder.compile()

# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.get("/test-bookings")
def test_bookings():
    return mock_bookings

# ---------------------------------------------------------
# Chat endpoint — now runs the LangGraph graph instead of
# manual tool-call handling
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    save_message(session_id, "user", request.message)
    history_rows = get_history(session_id)[-20:]

    # Rebuild LangChain message objects from stored history
    lc_messages = []
    for role, content in history_rows:
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))

    result = graph.invoke({"messages": lc_messages})

    final_message = result["messages"][-1]
    reply_text = final_message.content

    save_message(session_id, "assistant", reply_text)

    return {"session_id": session_id, "reply": reply_text}