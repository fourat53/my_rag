import os

from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from core.pinecone import get_index
from .embeddings import embed_documents, embed_query
from .text_processing import process_text

gemini_api_key = os.getenv("GEMINI_API_KEY")

index = get_index()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2,
    google_api_key=gemini_api_key,
)


def add_documents(index_id: str, raw_text: str) -> tuple[list[str], int]:
    cleaned_text = process_text(raw_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(cleaned_text)

    if not chunks:
        return chunks, 0

    vectors = embed_documents(chunks)

    pinecone_vectors = [
        {
            "id": str(uuid4()),
            "values": vector,
            "metadata": {
                "index_id": index_id,
                "text": chunk,
            },
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    response = index.upsert(
        vectors=pinecone_vectors,
        batch_size=200,
        show_progress=False,
    )

    if response.has_errors:
        raise RuntimeError(
            f"Pinecone failed to upsert {response.failed_item_count} vectors.",
            f"Errors: {response.errors}",
        )

    return chunks, len(chunks)


def delete_documents(index_id: str) -> tuple[list[str], int]:
    filter_dict = {"index_id": {"$eq": index_id}}
    deleted_chunks: list[str] = []
    pagination_token = None

    while True:
        response = index.fetch_by_metadata(
            filter=filter_dict,
            limit=1000,
            pagination_token=pagination_token,
        )

        for vector in response.vectors.values():
            if vector.metadata is None:
                continue
            text = vector.metadata.get("text")
            if isinstance(text, str):
                deleted_chunks.append(text)

        pagination_token = (
            response.pagination.next if response.pagination is not None else None
        )

        if pagination_token is None:
            break

    index.delete(filter=filter_dict)

    return deleted_chunks, len(deleted_chunks)


def similarity_search(
    query: str, k: int = 5, index_ids: list[str] | None = None
) -> list[tuple[str, float]]:
    query_vector = embed_query(query)

    if index_ids:
        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            filter={"index_id": {"$in": index_ids}},
        )
    else:
        results = index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
        )

    matches: list[tuple[str, float]] = []

    for match in results.matches:
        if match.metadata is None or match.score is None:
            continue
        text = match.metadata.get("text")
        if isinstance(text, str):
            matches.append((text, float(match.score)))

    return matches


def query_db(question: str, index_ids: list[str] | None = None) -> str:
    query_vector = embed_query(question)

    if index_ids:
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
            filter={"index_id": {"$in": index_ids}},
        )
    else:
        results = index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True,
        )

    context_parts: list[str] = []

    for match in results.matches:
        if match.metadata is None:
            continue
        text = match.metadata.get("text")
        if isinstance(text, str):
            context_parts.append(text)

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a precise and reliable assistant. "
                    "Your task is to answer questions using only the information provided in the context, "
                    "which consists of multiple retrieved text chunks. "
                    "Do not introduce external knowledge, assumptions, or additional details. "
                    "Synthesize the relevant information across chunks into a clear, natural, human-like answer. "
                    "Avoid lists, bullet points, markdown formatting, symbols, or emphasis markers. "
                    "Write clean sentences, one after the other, as a short paragraph. "
                    "If the context does not contain the answer or any part of it, respond exactly with: "
                    '"I do not have enough information to answer the question."'
                    "\n\nContext:\n{context}"
                ),
            ),
            (
                "user",
                (
                    "This is an example of how to answer a question using the context.\n"
                    "Question:\n"
                    "What technologies power the RAG microservice?\n\n"
                    "Context:\n"
                    "Chunk 1: The RAG microservice runs on an Nginx server and is implemented using Python and FastAPI.\n"
                    "Chunk 2: Vector storage is handled by ChromaDB in production and FAISS during development.\n"
                    "Chunk 3: The system uses OpenAI's text-embedding-ada-002 model for embedding generation and "
                    "integrates retrieval and generation through LangChain."
                ),
            ),
            (
                "assistant",
                (
                    "The RAG microservice is built with Python and FastAPI, running on an Nginx server. "
                    "For vector storage, it uses ChromaDB in production and FAISS during development. "
                    "The system generates embeddings using OpenAI's text-embedding-ada-002 model and "
                    "integrates retrieval and generation through the LangChain library."
                ),
            ),
            (
                "user",
                "Question:\n{question}",
            ),
        ]
    )

    chain = prompt | llm
    res = chain.invoke({"context": context, "question": question})

    if isinstance(res.content, str):
        return res.content

    return "\n".join(
        block["text"]
        for block in res.content
        if isinstance(block, dict) and block.get("type") == "text"
    )
