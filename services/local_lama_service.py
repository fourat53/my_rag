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
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Whenever you need to emphasize text, do not use **bold** syntax. "
                    "Instead, use HTML with a paragraph and the class 'font-semibold', "
                    "for example: <p class='font-semibold'>This is bold</p>.\n\n"
                    "Here is a one-shot example:\n"
                    "Q: What is the capital of France?\n"
                    "A: Paris"
                ),
            ),
            (
                "user",
                "{question}",
            ),
        ]
    )
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
