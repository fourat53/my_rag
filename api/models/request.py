from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., example="How does RAG work?", min_length=1)
    model: str = Field(default="gemini-2.0-flash")
    temperature: float = Field(default=0.7, ge=0, le=2)

class DocumentProcessing(BaseModel):
    file_path: str=Field(..., example="repport_f.pdf", min_length=1)