from app.core.config import settings
from app.services.history import trim_by_tokens
from pathlib import Path

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system.md").read_text(encoding="utf-8")

async def reply(client, messages, system=SYSTEM_PROMPT):
    res = await client.messages.create(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=system,
        messages=await trim_by_tokens(client, messages, system),
    )
    return res
