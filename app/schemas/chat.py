from pydantic import BaseModel


class ChatRequest(BaseModel):
    agent: str  # agent slug, e.g. "tribe-app" or "pfc"
    session_id: str | None = None
    email: str | None = None
    message: str


class TestChatRequest(BaseModel):
    message: str
