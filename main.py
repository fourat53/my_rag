from dotenv import load_dotenv

load_dotenv()

import logging
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api import models_router, query_router, rag_router
from services.rag import EmbeddingRateLimitError

logger = logging.getLogger("my_rag")

app = FastAPI(title="Chat AI Microservice")

origins = [os.getenv("NEXT_PUBLIC_URL")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EmbeddingRateLimitError)
async def handle_embedding_rate_limit(
    request: Request, exc: EmbeddingRateLimitError
) -> JSONResponse:
    logger.warning("Embedding rate limit reached on %s", request.url.path)
    return JSONResponse(status_code=429, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred while processing the request."
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "message": "Chat AI Microservice is running"}


app.include_router(rag_router)
app.include_router(query_router)
app.include_router(models_router)
