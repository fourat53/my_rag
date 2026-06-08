from routers import chat_router, documents_router
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="API for a Retrieval-Augmented Generation system",
    version="1.0.0",
)


app.include_router(documents_router)
app.include_router(chat_router)
