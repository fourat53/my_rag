from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from api.rag_api import router as rag_router

app = FastAPI(title="AI Microservice")

app.include_router(rag_router)
