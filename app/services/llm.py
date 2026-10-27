from app.core.config import settings
from app.services.history import trim_by_tokens
from pathlib import Path

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")

async def build_request(client, messages, system=SYSTEM_PROMPT) -> dict:
    # Feature 4: Prompt Caching
    """Everything both routes send to Claude, with caching switched on."""

    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": system,
        "messages": await trim_by_tokens(client, messages, system),
        # Automatic caching: the API puts the breakpoint on the last cacheable
        # block and moves it forward as the conversation grows.
        "cache_control": {"type": "ephemeral"},
        "thinking": {"type": "enabled", "budget_tokens": 200},
    }

async def reply(client, messages, system=SYSTEM_PROMPT):
    return await client.message.create(**await build_request(client, messages, system))
