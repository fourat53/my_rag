import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

lama_url = os.getenv("LLAMA_DEV_URL")

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.3,
    base_url=lama_url,
    num_gpu=1,
)


def query_db_stream(question: str):
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
