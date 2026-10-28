# One shared async Claude client for the whole app. Creating a client per request is
# slow, and an async client keeps the server free while it waits for Claude.
# Routes get it through Depends(get_client), so tests can swap in a fake.

from anthropic import AsyncAnthropic
from app.core.config import settings

_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

def get_client() -> AsyncAnthropic:
    return _client