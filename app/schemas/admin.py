"""
Placeholder for typed admin request/response models.

The original app took raw `dict` bodies for several admin routes
(system-prompt update, model update, login). That still works fine — but
as this grows, replace those `dict` payloads with real Pydantic models
here (e.g. SystemPromptUpdate, ModelUpdate, AdminLoginRequest) so FastAPI
can validate and document them automatically.
"""
