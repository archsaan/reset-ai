from app.agent.tools.schedule_tools import get_schedule, check_availability
from app.agent.tools.booking_tools import book_class, mock_bookings

# Central registry of every tool the agent can call.
# Add new tools here (and to their own module in this package) as the
# project grows — this is the one place the graph looks for the full list.
all_tools = [get_schedule, check_availability, book_class]

__all__ = [
    "get_schedule",
    "check_availability",
    "book_class",
    "mock_bookings",
    "all_tools",
]
