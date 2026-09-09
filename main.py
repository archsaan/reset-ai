from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import anthropic
import sqlite3
import uuid
import httpx
from datetime import date, datetime, timedelta

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

SCHEDULE_API_URL = "https://crmapi.profitconnect.co/calendar/get/dayschedule"
SCHEDULE_API_KEY = "PASTE_KEY_HERE_IF_NEEDED"  # or None if not required

DB_FILE = "conversations.db"

# ---------------------------------------------------------
# Agent identity
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

# ---------------------------------------------------------
# Knowledge base — loaded once at startup from file
# ---------------------------------------------------------

with open("knowledge_base.md", "r", encoding="utf-8") as f:
    KNOWLEDGE_BASE = f.read()

BASE_SYSTEM_PROMPT = (
    AGENT_IDENTITY
    + "\n\n--- KNOWLEDGE BASE ---\n"
    + KNOWLEDGE_BASE
    + "\n\n--- RULES ---\n"
    + "Only answer factual questions (pricing, location, policies, services) using the "
      "Knowledge Base above. Never guess or invent details not contained in it. "
      "If the answer isn't in the Knowledge Base, say: 'Our team will have all "
      "the details and will be in touch with you very shortly! 😊' "
      "\n\nFor anything related to class times, schedules, or availability on a specific "
      "date — NEVER use static text, and never guess. Always use the get_schedule or "
      "check_availability tool instead, since those pull real, live data. The Knowledge "
      "Base intentionally does not contain a fixed schedule for this reason. "
      "\n\nUse get_schedule when a lead asks generally what's on or wants to browse classes. "
      "Use check_availability when a lead names a specific class and wants to know if "
      "there's space, or wants to book — this tool also checks upcoming days automatically "
      "if the requested date is full. Each result includes a 'day_label' field — always "
      "trust and use that label rather than calculating the day yourself. "
      "Use book_class ONLY after check_availability has confirmed there is space, and only "
      "once you have the lead's name."
)

tools = [
    {
        "name": "get_schedule",
        "description": "Get the full class schedule for Reset Fitness on a given date. Use this when a lead asks generally what's on, or what classes are available on a specific day.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format. Defaults to today if not specified."
                }
            }
        }
    },
    {
        "name": "check_availability",
        "description": "Check if a specific class has open spots on a given date, and automatically checks the next few days too if needed. Use this when a lead asks about a specific class by name and wants to know if there's space, especially if they want to book. Each result includes a 'day_label' field — always trust and use this label rather than calculating the day yourself.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {
                    "type": "string",
                    "description": "The name of the class to check, e.g. 'Firestarter', 'Red Light Treatment'"
                },
                "target_date": {
                    "type": "string",
                    "description": "Date in YYYY-MM-DD format to start checking from. Defaults to today if not specified."
                }
            },
            "required": ["class_name"]
        }
    },
    {
        "name": "book_class",
        "description": "Book a lead into a specific class. Only call this after confirming with check_availability that there is space at the exact date/time. Requires the lead's name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "class_name": {"type": "string"},
                "target_date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "e.g. '06:30 PM', matching the schedule format exactly"},
                "lead_name": {"type": "string"}
            },
            "required": ["class_name", "target_date", "start_time", "lead_name"]
        }
    }
]

# ---------------------------------------------------------
# App setup
# ---------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev; restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic()

# ---------------------------------------------------------
# Database helpers
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
    return [{"role": role, "content": content} for role, content in rows]

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# Tool 1: get_schedule — full schedule for one day
# ---------------------------------------------------------

def get_schedule(target_date: str = None):
    if target_date is None:
        target_date = date.today().isoformat()

    payload = {
        "facility_id": 1,
        "room_id": "",
        "date": target_date
    }

    headers = {}
    if SCHEDULE_API_KEY and SCHEDULE_API_KEY != "PASTE_KEY_HERE_IF_NEEDED":
        headers["Authorization"] = f"Bearer {SCHEDULE_API_KEY}"
        # or headers["x-api-key"] = SCHEDULE_API_KEY — depends on what the API expects

    response = httpx.post(SCHEDULE_API_URL, json=payload, headers=headers, timeout=10)
    data = response.json()

    cleaned = []
    for cls in data.get("schedule", []):
        capacity = int(cls["capacity"])
        booked = int(cls["booked"])
        cancelled = int(cls.get("cancelled", 0))
        spots_left = capacity - (booked - cancelled)

        cleaned.append({
            "class_name": cls["class_name"],
            "discipline": cls["discipline_name"],
            "start_time": cls["start_time"],
            "end_time": cls["end_time"],
            "coach": f'{cls["coach"][0]["firstname"]} {cls["coach"][0]["lastname"]}' if cls.get("coach") else None,
            "spots_left": max(spots_left, 0),
            "class_type": cls["class_type"],
        })

    return cleaned

# Temporary debug endpoint — test the schedule API directly, no Claude call, no cost.
# Remove this before treating the project as "done".
@app.get("/test-schedule")
def test_schedule():
    return get_schedule()

# ---------------------------------------------------------
# Tool 2: check_availability — one class, across multiple days
# ---------------------------------------------------------

def check_availability(class_name: str, target_date: str = None, max_days_ahead: int = 3):
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

        schedule = get_schedule(check_date)

        matches = [
            cls for cls in schedule
            if class_name.lower() in cls["class_name"].lower()
        ]

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

# Temporary debug endpoint — test this tool directly, no Claude call, no cost.
# Remove this before treating the project as "done".
@app.get("/test-availability")
def test_availability(class_name: str, target_date: str = None):
    return check_availability(class_name, target_date)

# ---------------------------------------------------------
# Tool 3: book_class — MOCK booking, no real API call
# ---------------------------------------------------------

mock_bookings = []

def book_class(class_name: str, target_date: str, start_time: str, lead_name: str):
    matches = [
        cls for cls in get_schedule(target_date)
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

@app.get("/test-bookings")
def test_bookings():
    return mock_bookings

# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/chat")
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    save_message(session_id, "user", request.message)
    history = get_history(session_id)
    recent_history = history[-20:]

    today_str = date.today().isoformat()
    system_prompt = (
        BASE_SYSTEM_PROMPT
        + f"\n\nToday's actual date is {today_str}. Always trust this over any "
          f"assumption about what day it is."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system_prompt,
        messages=recent_history,
        tools=tools
    )

    print("STOP REASON:", response.stop_reason)  # temporary debug line — remove later

    if response.stop_reason == "tool_use":
        tool_use_block = next(b for b in response.content if b.type == "tool_use")
        print("TOOL CALLED:", tool_use_block.name, tool_use_block.input)  # temporary debug line

        if tool_use_block.name == "get_schedule":
            result = get_schedule(tool_use_block.input.get("target_date"))
        elif tool_use_block.name == "check_availability":
            result = check_availability(
                tool_use_block.input["class_name"],
                tool_use_block.input.get("target_date")
            )
        elif tool_use_block.name == "book_class":
            result = book_class(**tool_use_block.input)
        else:
            result = {"error": "unknown tool"}

        follow_up = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=system_prompt,
            messages=recent_history + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": str(result)
                    }
                ]}
            ]
        )
        reply_text = next(b.text for b in follow_up.content if b.type == "text")
    else:
        reply_text = next(b.text for b in response.content if b.type == "text")

    save_message(session_id, "assistant", reply_text)

    return {"session_id": session_id, "reply": reply_text}