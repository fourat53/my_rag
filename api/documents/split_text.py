from langchain_text_splitters import RecursiveCharacterTextSplitter
from routers.documents import documents_router


@documents_router.post("/split_text", response_model=list[str])
def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size,
        chunk_overlap,
    )

    return splitter.split_text(text)
