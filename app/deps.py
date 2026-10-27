from anthropic import AsyncAnthropic
from app.core.config import settings

_client = AsyncAnthropic(api_key=settings.anthropic_api_key)

def get_client() -> AsyncAnthropic:
    return _client