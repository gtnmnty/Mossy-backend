# Feature 5: Tool - current time
# Claude cannot know the real time, so it calls this with a timezone name
# and gets back the date and time there.
# Needs the tzdata package on Windows, which has no built-in timezone database.

from datetime import datetime
from zoneinfo import ZoneInfo

SCHEMA = {
    "name": "get_current_time",
    "description": "Get the current date and time in an IANA timezone, e.g. Asia/Manila or America/New_York.",
    "input_schema": {
        "type": "object",
        "properties": {"timezone": {"type": "string", "description": "IANA timezone name"}},
        "required": ["timezone"],
    },
}

async def run(timezone: str) -> str:
    return datetime.now(ZoneInfo(timezone)).strftime("%A, %Y-%m-%d %H:%M:%S %Z")