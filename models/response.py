from pydantic import BaseModel, Field
from typing import Union, List, Dict, Any


class ChatSuccessResponse(BaseModel):
    status: str = Field(..., description="Response status")
    response: Dict[str, Any] = Field(..., description="Full AI response object")


class ErrorResponse(BaseModel):
    status: str = Field(..., description="Error status")
    message: str = Field(..., description="Error message")


ChatResponse = ChatSuccessResponse | ErrorResponse


class ModelsResponse(BaseModel):
    status: str = Field(..., description="Response status")
    models_count: int = Field(..., description="Number of models")
    models: List[Dict[str, Any]] = Field(..., description="List of available models")


class LoadDocumentResponse(BaseModel):
    status: str = Field(..., description="Response status")
    response: Dict[str, Any] = Field(..., description="Loaded document text")
