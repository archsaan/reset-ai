from pydantic import BaseModel


class AgentCreate(BaseModel):
    slug: str  # URL/identifier-safe, e.g. "tribe-app" — used in /chat requests
    name: str
    description: str = ""
    system_prompt: str
    active_model: str


class AgentUpdate(BaseModel):
    """All fields optional — only the ones provided get changed.
    `slug` is deliberately not editable: it's the stable identifier other
    systems (the Tribe app, the PFC tool) reference in their /chat calls."""
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    active_model: str | None = None
