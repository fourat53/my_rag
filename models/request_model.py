from pydantic import BaseModel, Field


class AddTextRequest(BaseModel):
    index_id: str
    text: str


class DeleteTextRequest(BaseModel):
    index_id: str


class QueryRequest(BaseModel):
    question: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100, description="Number of top chunks to return")
