"""
Booking tool — real ProfitConnect integration.

Two-step flow against PFC, both required before a booking is real:
  1. POST BOOKING_VALIDATE_API_URL — confirms the member's membership/plan
     actually allows this booking (active plan, credits available, etc.)
  2. POST BOOKING_CREATE_API_URL — reserves the spot, only once validate
     has confirmed it's allowed.

member_id is currently hardcoded to TEST_MEMBER_ID (see app/core/config.py)
because /chat has no real member auth yet — the Tribe app doesn't tell us
who's logged in. THIS IS TEMPORARY. Before this is used with real members,
member_id must come from the actual logged-in member (e.g. a member_id
field on the /chat request, once the Tribe app can send it), not this
constant. Search for TEST_MEMBER_ID to find every place that needs to
change.

Approval gate: the model can PROPOSE a booking by calling book_class, but
the actual HTTP calls only happen after an explicit member confirmation.
That gate lives in app/agent/graph.py (see confirm_booking_node) using
LangGraph's interrupt()/Command(resume=...) — the graph physically pauses
before this code runs, and only resumes into execute_booking() if the
member's reply is a clear yes. book_class itself is never invoked as a
tool by the graph; it exists only so the model has a typed tool signature
to call. All the real work is in execute_booking(), which confirm_booking_node
calls directly once approved.
"""

import httpx
from langchain_core.tools import tool

from app.core.config import (
    BOOKING_API_KEY,
    BOOKING_CREATE_API_URL,
    BOOKING_VALIDATE_API_URL,
    FACILITY_ID,
    TEST_MEMBER_ID,
)


def _auth_headers() -> dict:
    headers = {}
    if BOOKING_API_KEY and BOOKING_API_KEY != "PASTE_KEY_HERE_IF_NEEDED":
        headers["Authorization"] = f"Bearer {BOOKING_API_KEY}"
    return headers


def _extract_failure_reason(data: dict) -> str:
    """PFC's exact error shape isn't confirmed yet, so this checks the
    common field names an API like this tends to use, falling back to a
    generic message rather than guessing wrong and hiding a real reason."""
    for key in ("message", "error", "reason", "detail"):
        if data.get(key):
            return str(data[key])
    return "The booking system declined this booking."


def execute_booking(calendar_schedule_id: int, class_name: str, target_date: str, start_time: str, lead_name: str) -> dict:
    """The real validate -> book HTTP flow. Plain function, not a tool —
    call this directly once a member has explicitly confirmed. Never call
    this straight from a tool call the model made; that's the approval
    gate's whole point (see module docstring)."""
    member_id = TEST_MEMBER_ID  # TEMPORARY — see module docstring.

    # Step 1: validate the member is actually allowed to book this class.
    try:
        validate_resp = httpx.post(
            BOOKING_VALIDATE_API_URL,
            json={"calendar_schedule_id": calendar_schedule_id, "facility_id": FACILITY_ID, "member_id": member_id},
            headers=_auth_headers(),
            timeout=10,
        )
        validate_resp.raise_for_status()
        validate_data = validate_resp.json()
    except httpx.HTTPError as exc:
        return {"success": False, "reason": f"Couldn't reach the booking system to validate: {exc}"}

    if validate_data.get("valid") is False or validate_data.get("success") is False:
        return {"success": False, "reason": _extract_failure_reason(validate_data)}

    # Step 2: actually book it.
    try:
        book_resp = httpx.post(
            BOOKING_CREATE_API_URL,
            json={
                "calendar_schedule_id": calendar_schedule_id,
                "facility_id": FACILITY_ID,
                "member_id": [member_id],
                "planStatus": "Active",
                "status": "Booked",
                "validateBy": "Membership",
            },
            headers=_auth_headers(),
            timeout=10,
        )
        book_resp.raise_for_status()
        book_data = book_resp.json()
    except httpx.HTTPError as exc:
        return {"success": False, "reason": f"Validated but couldn't complete the booking: {exc}"}

    if book_data.get("success") is False:
        return {"success": False, "reason": _extract_failure_reason(book_data)}

    return {
        "success": True,
        "booking": {
            "class_name": class_name,
            "date": target_date,
            "start_time": start_time,
            "lead_name": lead_name,
            "calendar_schedule_id": calendar_schedule_id,
        },
        "raw_response": book_data,  # keep visible until the real response shape is confirmed
    }


@tool
def book_class(calendar_schedule_id: int, class_name: str, target_date: str, start_time: str, lead_name: str) -> dict:
    """Propose booking a member into a class. Only call after
    check_availability has confirmed there is space — pass the exact
    calendar_schedule_id that check_availability returned for the
    class/time the member chose. Calling this does NOT book the class yet
    — the member will be asked to explicitly confirm first."""
    # This body is never actually executed by the graph (confirm_booking_node
    # intercepts every book_class tool call before it would reach here — see
    # should_continue in app/agent/graph.py). It exists only so the shape is
    # right if this tool is ever invoked directly, e.g. in a test.
    return execute_booking(calendar_schedule_id, class_name, target_date, start_time, lead_name)
