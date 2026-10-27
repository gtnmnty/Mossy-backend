from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import chat, health
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MossyChatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def show_422(request, exc):
    print("422:", exc.errors())
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

app.include_router(chat.router)
app.include_router(health.router)
