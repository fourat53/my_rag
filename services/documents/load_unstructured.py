from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from typing import List
import os


class LoadUnstructured:
    @classmethod
    def load_file(cls, file_path: str) -> List[Document] | None:
        relative_path = f"data/{file_path}"
        if not os.path.exists(relative_path):
            return print(f"Error: File data/{file_path} not found.")
        try:
            loader = UnstructuredLoader(
                file_path=relative_path,
                mode="elements",
                partition_via_api=True,
            )
            docs = loader.load()
            print("------------- DOCS -------------\n", docs)
            return docs

        except Exception as e:
            print(f"An error occurred during document processing: {e}")
