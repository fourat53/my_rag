from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing import List, Dict, Any


# Error Response Models
class ErrorResponse(BaseModel):
    status: str = Field(..., description="Error status")
    message: str = Field(..., description="Error message")


# Chat Response Models
class ChatSuccessResponse(BaseModel):
    status: str = Field(..., description="Response status")
    response: Dict[str, Any] = Field(..., description="Full AI response object")


ChatResponse = ChatSuccessResponse | ErrorResponse


# Models Response Models
class ModelsSuccessResponse(BaseModel):
    status: str = Field(..., description="Response status")
    models_count: int = Field(..., description="Number of models")
    models: List[Dict[str, Any]] | List[str] = Field(..., description="List of models")


ModelsResponse = ModelsSuccessResponse | ErrorResponse


# Document Loading Response Models
class LoadFileSuccessResponse(BaseModel):
    status: str = Field(..., description="Response status")
    documents_count: int = Field(..., description="Number of documents")
    documents: List[Document] = Field(..., description="Loaded document text")


LoadFileResponse = LoadFileSuccessResponse | ErrorResponse
