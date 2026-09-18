"""
Tools backed by the real ProfitConnect schedule API. Read-only — these
never write anything, so they carry no approval-gate concerns.
"""

from datetime import date, datetime, timedelta

import httpx
from langchain_core.tools import tool

from app.core.config import SCHEDULE_API_KEY, SCHEDULE_API_URL


def _fetch_schedule(target_date: str | None = None) -> list:
    """Raw helper (not a tool itself) — calls the live schedule API and
    strips it down to only what the model needs, so no unrelated member
    PII ever reaches the LLM."""
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
    """Get the full class schedule for Reset Fitness on a given date."""
    return _fetch_schedule(target_date)


@tool
def check_availability(class_name: str, target_date: str = None, max_days_ahead: int = 3) -> list:
    """Check if a specific class has open spots on a given date, checking
    the next few days too. Each result includes a 'day_label' field."""
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
