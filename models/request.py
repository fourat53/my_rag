from pydantic import BaseModel, Field


class AddTextRequest(BaseModel):
    index_id: str
    text: str


class UploadUrlRequest(BaseModel):
    url: str = Field(
        description="Public http(s) URL of the stored file (UploadThing)"
    )
    index_id: str = Field(
        description="Unique id used as Pinecone metadata for every chunk of this file"
    )
    filename: str | None = Field(
        default=None,
        description="Original file name; used to detect the file extension for extraction",
    )


class DeleteTextRequest(BaseModel):
    index_id: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    index_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional index_id metadata values. When provided, retrieval is scoped to the chunks stored under these documents only."
        ),
    )


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
    api_key: str | None = Field(
        default=None,
        description="Optional per-user API key (from the app's database); overrides the env fallback for this request",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns (oldest first), excluding the current question",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of top chunks to return"
    )
    index_ids: list[str] | None = Field(
        default=None,
        description=(
            "Optional index_id metadata values. When provided, retrieval is scoped to the chunks stored under these documents only."
        ),
    )
