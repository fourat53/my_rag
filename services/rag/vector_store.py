from uuid import uuid4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.pinecone import get_index
from .embeddings import embed_documents, embed_query
from .text_processing import process_text

index = get_index()


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


def build_rag_question(
    question: str, context_chunks: list[str], document_names: list[str] | None = None
) -> str:
    lines = [
        "You are answering a question. "
        "Below is a document context made of chunks retrieved from the documents the user linked to it. "
        "Use the context together with your own knowledge.",
    ]
    if document_names:
        lines.append(f"Linked documents: {', '.join(document_names)}")
    lines += [
        "- If the question clearly targets the linked documents and they contain nothing that helps answering it, "
        'respond exactly with: "I don\'t know."',
        "- If the linked documents only contain related information, say that you do not have the exact answer "
        "and share the information that still helps "
        '(example: a file says "the box has a dark color", the question asks whether the box is blue, '
        'so answer "I don\'t know, but the box has a dark color.").',
        "- If different linked documents give different or conflicting answers, provide all of them.",
        "- If the question is not about the linked documents, answer normally with your own knowledge, "
        "using any relevant document content when it helps.",
        "- Deliver the answer in the shape that serves the question best: plain prose, a table, code, links, "
        "a list, or anything else. For example, when asked which rows of a CSV are the best, "
        "answer with a table of those rows preceded with an explanation sentence.",
        "",
        "Document context:",
        (
            "\n\n".join(context_chunks)
            if context_chunks
            else "(no relevant chunks were found)"
        ),
        "",
        f"Question: {question}",
    ]
    return "\n".join(lines)
