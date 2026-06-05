from models.router import main_router
from dotenv import load_dotenv
from fastapi import FastAPI
import uvicorn

load_dotenv()

app = FastAPI(
    title="RAG API",
    description="API for a Retrieval-Augmented Generation system",
    version="1.0.0",
)

app.include_router(main_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
