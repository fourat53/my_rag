import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

openai_key = os.getenv("OPENAI_API_KEY", "")


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        api_key=openai_key,
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
