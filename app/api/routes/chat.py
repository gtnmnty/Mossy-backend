import anthropic
import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.deps import get_client
from app.schemas.schema import ChatMeta, ChatMetaRequest, ChatRequest
from app.services.llm import make_meta, reply, run_with_tools

router = APIRouter()
log = logging.getLogger("mossy")


# Feature 1: Multi-turn conversation
# Both routes take the full chat history ({"messages": [...]}), because Claude does not
# remember past calls. ChatRequest checks the history before any paid API call is made.

# Non-streaming fallback: waits for the full answer, then returns it as one JSON body.
@router.post("/chat")
async def chat(req: ChatRequest, client=Depends(get_client)):
    try:
        res = await reply(client, req.messages, think=req.think)
    except anthropic.APIError as e:
        # HTTPException keeps CORS headers, so the browser sees the real error
        raise HTTPException(status_code=502, detail=str(e))
    text = "".join(b.text for b in res.content if b.type == "text")
    return {"reply": text}


# Feature 3 + Feature 7: Streaming
# Sends Claude's reply to the browser piece by piece over Server-Sent Events (SSE),
# so text appears as it is written instead of after a long wait.
# Events: "thinking" (a reasoning chunk, only when req.think is true), "delta" (an answer
# text chunk), "tool" (Mossy is using a tool), "done" (finished, with token usage),
# "error" (something failed).
@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, client=Depends(get_client)):
    # Every `yield` becomes one SSE event on the open HTTP response.
    async def gen():
        try:
            async for ev in run_with_tools(client, req.messages, think=req.think):
                yield ev
        except anthropic.APIError as e:
            log.error("Claude API error: %s", e)
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    # sse-starlette frames each yielded dict as "event: ...\r\ndata: ...\r\n\r\n".
    return EventSourceResponse(gen())


# Feature 8: Structured outputs
# A separate route, called after a reply finishes, so a slow or failed title call never
# holds up the chat itself. On any failure it falls back to the start of the first message
# as the title and no follow-ups, instead of breaking the UI.
@router.post("/chat/meta", response_model=ChatMeta)
async def chat_meta(req: ChatMetaRequest, client=Depends(get_client)):
    try:
        return await make_meta(client, req.messages)
    except Exception:
        log.exception("chat/meta failed, falling back to a plain title")
        first = req.messages[0].content[:40]
        return ChatMeta(title=first, followups=[])
