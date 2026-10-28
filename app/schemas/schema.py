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


