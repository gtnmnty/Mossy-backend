# App entry point: creates FastAPI, allows the frontend to call it, and plugs in the routes.
import logging

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, health

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mossy")

app = FastAPI(title="MossyChatbot")

# CORS: lets the Next.js frontend on port 3000 call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Feature 1: logs why a request failed validation (for example, history ending with an
# assistant message) and returns a 422. jsonable_encoder turns the ValueError inside the
# error details into text; without it the response itself crashes with a 500.
@app.exception_handler(RequestValidationError)
async def show_422(request, exc):
    log.warning("422: %s", exc.errors())
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


app.include_router(chat.router)
app.include_router(health.router)
