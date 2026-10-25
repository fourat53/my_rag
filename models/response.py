from pydantic import BaseModel, Field


class AddTextResponse(BaseModel):
    status: str
    added_chunks: list[str] = Field(
        default_factory=list, description="Content of each added chunk"
    )
    added_count: int = Field(description="Number of chunks added")


class UploadTextResponse(BaseModel):
    status: str
    filename: str = Field(description="Name of the uploaded file")
    index_id: str = Field(
        description="Unique id used as Pinecone metadata for every chunk of this file"
    )
    added_chunks: list[str] = Field(
        default_factory=list, description="Content of each added chunk"
    )
    added_count: int = Field(description="Number of chunks added")


class DeleteTextResponse(BaseModel):
    status: str
    deleted_chunks: list[str] = Field(
        default_factory=list, description="Content of each deleted chunk"
    )
    deleted_count: int = Field(description="Number of chunks deleted")


class QueryResponse(BaseModel):
    latency: float = Field(description="Time taken to answer the question")
    answer: str = Field(description="Answer to the question")


class ChunkSimilarityItem(BaseModel):
    content: str = Field(description="Chunk text content")
    score: float = Field(description="Similarity score (higher is more similar)")


class SearchResponse(BaseModel):
    latency: float = Field(description="Time taken to answer the question")
    total_count: int = Field(description="Number of chunks returned")
    chunks: list[ChunkSimilarityItem] = Field(
        default_factory=list, description="Top-k chunks by similarity"
    )
