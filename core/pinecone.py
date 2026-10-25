import os, time

from pinecone import Pinecone, ServerlessSpec

PINECONE_INDEX_NAME = "chat-example"
PINECONE_INDEX_DIMENSION = 1536

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def get_index():
    return pc.Index(PINECONE_INDEX_NAME)


def create_index():
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if PINECONE_INDEX_NAME in existing_indexes:
        print(f"Index {PINECONE_INDEX_NAME} already exists.")
        return

    print(f"Creating index {PINECONE_INDEX_NAME} with dimension {PINECONE_INDEX_DIMENSION}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=PINECONE_INDEX_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not _is_ready(pc.describe_index(PINECONE_INDEX_NAME).status):
        time.sleep(1)


def _is_ready(status) -> bool:
    if isinstance(status, str):
        return status == "Ready"
    return bool(status.get("ready"))


if __name__ == "__main__":
    create_index()
