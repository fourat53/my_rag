from pydantic import BaseModel, Field


class AddTextRequest(BaseModel):
    index_id: str
    text: str


class DeleteTextRequest(BaseModel):
    index_id: str


class QueryRequest(BaseModel):
    question: str


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMQueryRequest(BaseModel):
    provider: str
    model: str
    question: str
    base_url: str | None = Field(
        default=None,
        description="Per-provider API base URL (from the app's MongoDB), overrides the env fallback",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns (oldest first), excluding the current question",
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of top chunks to return"
    )
