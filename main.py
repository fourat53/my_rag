from api import chat_router, rag_router

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

app = FastAPI(
    title="RAG Pipeline API",
    description="API for RAG and LLM orchestration",
    version="1.0.0",
)
app.include_router(rag_router, prefix="/rag", tags=["RAG Operations"])
app.include_router(chat_router, prefix="/chat", tags=["Chat LLMs Operations"])


def main():
    print("Server started at http://localhost:8000")


if __name__ == "__main__":
    main()
