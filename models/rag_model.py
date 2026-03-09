from pydantic import BaseModel


class BasicResponse(BaseModel):
    status: str


class AddTextRequest(BaseModel):
    index_id: str
    text: str


class DeleteTextRequest(BaseModel):
    index_id: str


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
