import anthropic
import json
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.core.config import settings
from app.deps import get_client
from app.schemas.schema import ChatRequest
from app.services.history import trim
from app.services.llm import reply, SYSTEM_PROMPT

router = APIRouter()

# Non-streaming fallback: waits for the full answer, then returns it as one JSON body.
@router.post("/chat")
async def chat(req: ChatRequest, client=Depends(get_client)):
    try:
        res = await reply(client, req.messages)
    except anthropic.APIError as e:
        # HTTPException keeps CORS headers, so the browser sees the real error
        raise HTTPException(status_code=502, detail=str(e))
    text = "".join(b.text for b in res.content if b.type == "text")
    return {"reply": text}


# Feature 3: Streaming: sends Claude's reply to the browser piece by piece over
# Server-Sent Events (SSE) so text appears as it is generated instead of after a long wait.
@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, client=Depends(get_client)):
    # Async generator: every `yield` becomes one SSE event on the open HTTP response.
    # Events: "delta" (a text chunk), "done" (finished + token usage), "error" (API failure).
    async def gen():
        try:
            # AsyncAnthropic only: a sync client here would block the whole event loop.
            async with client.messages.stream(
                model=settings.model,
                max_tokens=settings.max_tokens,
                system=SYSTEM_PROMPT,
                messages=trim(req.messages),  # validated history, capped to the last N turns
            ) as stream:
                # text_stream yields each new piece of text as Claude generates it.
                async for text in stream.text_stream:
                    yield {"event": "delta", "data": json.dumps({"text": text})}
                # After the text ends, the full message gives us the token usage.
                final = await stream.get_final_message()
                yield {"event": "done",
                       "data": json.dumps({"usage": final.usage.model_dump()})}
        except anthropic.APIError as e:
            # The HTTP status is already 200 once streaming starts, so failures
            # must be reported as an event the frontend knows how to read.
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    # Wrapping the generator is what actually sends it: sse-starlette frames each
    # yielded dict as "event: ...\r\ndata: ...\r\n\r\n". Without the return, nothing is sent.
    return EventSourceResponse(gen())
