import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

nvidia_key = os.getenv("NVIDIA_API_KEY", "")


def get_llm(model: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=nvidia_key,
    )


def query_db_stream(question: str, model: str):
    if not nvidia_key:
        yield "[error] NVIDIA_API_KEY is not set.\n"
        return
    llm = get_llm(model)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm
    try:
        for chunk in chain.stream({"question": question}):
            if chunk.content:
                yield chunk.content
    except Exception as exc:
        yield f"\n[error] NVIDIA request failed: {exc!s}\n"
