from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(examples=["How does RAG work?"], min_length=1)
    model: str = Field(default="models/gemini-flash-latest")
    temperature: float = Field(default=0.2, ge=0, le=2)


class DocumentProcessing(BaseModel):
    file_path: str = Field(examples=["repport_f.pdf"], min_length=1)
