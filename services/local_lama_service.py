import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from core.pinecone import INDEX_NAME

MAX_CHUNKS_FOR_DELETE = 10_000

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)

vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.3,
    base_url=os.getenv("LLAMA_DEV_URL"),
    num_gpu=1,
)


def process_text(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped.split()) < 4:
            continue
        if stripped not in seen:
            seen.add(stripped)
            cleaned_lines.append(stripped)
    cleaned_text = " ".join(cleaned_lines)
    print(cleaned_text)
    return cleaned_text


def add_text_to_db(index_id: str, raw_text: str) -> Tuple[List[str], int]:
    cleaned_text = process_text(raw_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(cleaned_text)

    docs = [
        Document(page_content=chunk, metadata={"index_id": index_id})
        for chunk in chunks
    ]
    if docs:
        vector_store.add_documents(docs)
    return chunks, len(chunks)


def delete_text_from_db(index_id: str) -> Tuple[List[str], int]:
    docs = vector_store.similarity_search(
        " ", k=MAX_CHUNKS_FOR_DELETE, filter={"index_id": index_id}
    )
    deleted_chunks = [doc.page_content for doc in docs]
    vector_store.delete(filter={"index_id": index_id})
    return deleted_chunks, len(deleted_chunks)


def similarity_search_chunks(query: str, k: int = 5) -> List[Tuple[str, float]]:
    results = vector_store.similarity_search_with_score(query, k=k)
    return [(doc.page_content, float(score)) for doc, score in results]


def query_db(question: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                "Question:\n{question}",
            ),
        ]
    )

    chain = prompt | llm
    res = chain.invoke({"question": question})
    return res.content


def query_db_stream(question: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                "Question:\n{question}",
            ),
        ]
    )
    chain = prompt | llm
    for chunk in chain.stream({"question": question}):
        if chunk.content:
            yield chunk.content
