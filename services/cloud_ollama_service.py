import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url=os.getenv("OLLAMA_BASE_URL"),
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
