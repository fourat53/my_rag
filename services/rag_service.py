import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from core.pinecone import INDEX_NAME

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)
vector_store = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY")
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
    return " ".join(cleaned_lines)


def add_text_to_db(index_id: str, raw_text: str):
    cleaned_text = process_text(raw_text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(cleaned_text)

    docs = [
        Document(page_content=chunk, metadata={"index_id": index_id})
        for chunk in chunks
    ]
    if docs:
        vector_store.add_documents(docs)


def delete_text_from_db(index_id: str):
    vector_store.delete(filter={"index_id": index_id})


def query_db(question: str) -> str:
    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert answering questions. Use the given context.\nContext:\n{context}",
            ),
            ("user", "Here is the user query: {question}"),
            (
                "assistant",
                "I will use only the provided context to answer the query accurately.",
            ),
        ]
    )

    chain = prompt | llm
    res = chain.invoke({"context": context, "question": question})
    return res.content
