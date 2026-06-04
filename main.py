from api.chat.router import chat_router
from api.documents.router import rag_router
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="API for a Retrieval-Augmented Generation system",
    version="1.0.0",
)

app.include_router(rag_router, prefix="/rag", tags=["RAG Operations"])
app.include_router(chat_router, prefix="/chat", tags=["Chat LLMs Operations"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
