from app.core.config import settings
from app.schemas.schema import Msg

def trim(messages: list[Msg]) -> list[dict]:
    recent = messages[-settings.max_history_turns:]

    while recent and recent[0].role != "user":
        recent = recent[1:]

    return [m.model_dump() for m in recent]

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
