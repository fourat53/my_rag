from pydantic import BaseModel, Field
from typing import Literal


class ChatRequest(BaseModel):
    query: str = Field(default="How does RAG work?", min_length=1)
    model: str = Field(default="models/gemini-flash-latest")
    temperature: float = Field(default=0.2, ge=0, le=2)


class LoadDocumentRequest(BaseModel):
    file_path: str = Field(default="journal_p.pdf", min_length=1)
    loader_type: Literal["unstructured", "manual"] = Field(
        default="unstructured",
        description="Select 'unstructured' for automated parsing or 'manual' for the custom logic for each file type.",
    )


class LoadDocumentsRequest(BaseModel):
    folder_path: str = Field(examples=["data"], min_length=1)
    loader_type: Literal["unstructured", "manual"] = Field(
        default="unstructured",
        description="Select 'unstructured' for automated parsing or 'manual' for the custom logic for each file type.",
    )
