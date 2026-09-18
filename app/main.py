"""
App entrypoint. Run with:

    uvicorn app.main:app --reload

Kept deliberately thin — this file's only job is to create the FastAPI
app, wire up CORS, initialize the admin DB, and mount the routers. All
real logic lives in core/, db/, agent/, and routers/.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.tools import mock_bookings
from app.db.admin_db import init_admin_db
from app.db.vector_store import init_vector_store
from app.routers import admin, chat

init_admin_db()
init_vector_store()

app = FastAPI(title="Reset Fitness Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}


@app.get("/test-bookings")
def test_bookings():
    """Debug route — inspect the in-memory mock bookings while the real
    booking API isn't wired in yet. Remove once real bookings persist
    somewhere queryable."""
    return mock_bookings
