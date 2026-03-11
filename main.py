from dotenv import load_dotenv

load_dotenv()

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.rag_api import router as rag_router
from api.local_lama_api import router as local_lama_router

app = FastAPI(title="AI Microservice")
next_prod_url = os.getenv("NEXT_PROD_URL")
next_dev_url = os.getenv("NEXT_DEV_URL")

origins = [next_prod_url, next_dev_url]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router)
app.include_router(local_lama_router)
