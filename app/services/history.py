# Feature 1: Multi-turn history
# Claude needs past messages on every call, but long chats cost more tokens.
# These helpers keep only the recent part of the chat.

from app.core.config import settings
from app.schemas.schema import Msg

# Keeps the last N messages. Claude requires the history to start with a user
# message, so any assistant messages left at the front are dropped.
def trim(messages: list[Msg]) -> list[dict]:
    recent = messages[-settings.max_history_turns:]

    while recent and recent[0].role != "user":
        recent = recent[1:]

    return [m.model_dump() for m in recent]

# Optional upgrade: trims by token count instead of message count.
# Flow: cut to N messages -> ask the API how many tokens that is -> while it is over
# the budget, drop the oldest user/assistant pair and count again.
async def trim_by_tokens(client, messages: list[Msg], system, budget: int = 10000) -> list[dict]:
    msgs = trim(messages)                               # turn cap first

    while len(msgs) > 1:
        count = await client.messages.count_tokens(
            model=settings.model,
            system=system,
            messages=msgs,
        )
        if count.input_tokens <= budget:
            break

        msgs = msgs[2:]                                 # drop the oldest user/assistant pair
        while msgs and msgs[0]["role"] != "user":
            msgs = msgs[1:]

    return msgs
