"""
Booking tool.

Currently MOCK — appends to an in-memory list after re-validating against
the real schedule API. This is the exact spot to wire in the real
create-booking endpoint once its URL/method/payload shape is confirmed
(see app/core/config.py BOOKING_CREATE_API_URL).

When the real API is wired in, this is also where the approval-gate
pattern belongs: the model should only be allowed to call this tool
after (1) check_availability confirmed space, and (2) — once real member
auth exists — an explicit member confirmation step, not just "the model
decided to."
"""

import uuid

from langchain_core.tools import tool

from app.agent.tools.schedule_tools import _fetch_schedule

# TODO: replace with a real persistence layer (DB table or the booking
# API's own record) once the real booking endpoint is wired in. This
# in-memory list resets whenever the process restarts.
mock_bookings: list = []


@tool
def book_class(class_name: str, target_date: str, start_time: str, lead_name: str) -> dict:
    """Book a lead into a class. Only call after check_availability confirms space."""
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

    # TODO: swap this block for a real POST to BOOKING_CREATE_API_URL once
    # the endpoint details are confirmed. Keep the same return shape so
    # nothing else in the graph needs to change.
    booking = {
        "class_name": cls["class_name"], "date": target_date, "start_time": start_time,
        "lead_name": lead_name, "booking_id": str(uuid.uuid4())[:8],
    }
    mock_bookings.append(booking)
    return {"success": True, "booking": booking}
