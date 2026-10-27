from typing import Literal
from pydantic import BaseModel, Field, model_validator

class Msg(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def is_user_last(self):
        if self.messages[-1].role != "user":
            raise ValueError("last message must be from the user")

        return self


