# Feature 1: Multi-turn history
# Request shapes. Pydantic checks them before any Claude call, so bad input
# returns a 422 instead of costing money.

from typing import Literal
from pydantic import BaseModel, Field, model_validator

# One chat message. Content must be 1 to 8000 characters.
class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


# The full chat history (1 to 200 messages). The last one must be from the
# user, because Claude has to have something to answer.
class ChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=200)
    # Feature 7: "Think harder" toggle. When true, Claude reasons before answering.
    think: bool = False

    @model_validator(mode="after")
    def is_user_last(self):
        if self.messages[-1].role != "user":
            raise ValueError("last message must be from the user")

        return self


# Feature 8: Structured outputs (titles and follow-ups)
# Request for /chat/meta. No "last message is from the user" check here: this is called
# after Mossy's reply is already in the history, so the list naturally ends with Mossy's turn.
class ChatMetaRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=200)


# What /chat/meta returns: a short title for the conversation and some follow-up questions
# the user might want to ask next. Limits (3 to 6 words, exactly 3 follow-ups) are written
# into the descriptions, not as Field(min_length=...) rules, because Claude's structured
# outputs support only a small set of JSON Schema features — no minLength/maxLength and no
# minItems other than 0 or 1.
class ChatMeta(BaseModel):
    title: str = Field(description="A 3 to 6 word title for the conversation, no quotes")
    followups: list[str] = Field(description="Exactly 3 short follow-up questions the user might ask next")

