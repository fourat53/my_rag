from pydantic import BaseModel, Field
from typing import Literal


class ChatRequest(BaseModel):
    query: str = Field(default="How does RAG work?", min_length=1)
    model: str = Field(default="gemini-2.5-flash-lite", min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)


class ModelsRequest(BaseModel):
    with_details: bool = Field(
        default=True,
        description="Whether to include detailed information about each model in the response.",
    )


class LoadFileRequest(BaseModel):
    file: str = Field(default="wp.png", min_length=1)
    loader_type: Literal["unstructured", "manual"] = Field(
        default="manual",
        description="unstructured or manual loader",
    )
