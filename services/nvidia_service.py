import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL"),
    )


def query_db_stream(question: str, model: str):
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    try:
        for chunk in chain.stream({"question": question}):
            if chunk.content:
                yield chunk.content
    except Exception as exc:
        yield f"\n[error] NVIDIA request failed: {exc!s}\n"
