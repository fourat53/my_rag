import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage


def get_llm(model: str, base_url: str):
    return ChatOpenAI(
        model=model,
        temperature=0.3,
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=base_url,
    )


def query_stream(
    question: str,
    model: str,
    base_url: str,
    history: list[dict] | None = None,
):
    llm = get_llm(model, base_url)
    messages = [
        (
            HumanMessage(content=m["content"])
            if m.get("role") == "user"
            else AIMessage(content=m["content"])
        )
        for m in (history or [])
        if m.get("content")
    ]
    messages.append(HumanMessage(content=question))
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as exc:
        yield f"\n[error] NVIDIA request failed: {exc!s}\n"
