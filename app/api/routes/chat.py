import anthropic
from fastapi import APIRouter, Depends, HTTPException

from app.deps import get_client
from app.schemas.schema import ChatRequest
from app.services.llm import reply

router = APIRouter()

@router.post("/chat")
async def chat(req: ChatRequest, client=Depends(get_client)):
    try:
        res = await reply(client, req.messages)
    except anthropic.APIError as e:
        # HTTPException keeps CORS headers, so the browser sees the real error
        raise HTTPException(status_code=502, detail=str(e))
    text = "".join(b.text for b in res.content if b.type == "text")
    return {"reply": text}
