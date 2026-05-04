import os
import time
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def create_index(
    name: str = "chat-example",
    dimension: int = 3072,
    metric: str = "cosine",
    chunk_size: int = 1000,
    overlap: int = 100,
):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if name not in existing_indexes:
        print(f"Creating index {name} with dimension {dimension}, chunk_size {chunk_size}, overlap {overlap}")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
    else:
        print(f"Index {name} already exists.")


if __name__ == "__main__":
    create_index()
