from typing import List
import os

from langchain_unstructured import UnstructuredLoader as Loader
from langchain_core.documents import Document


class LoaderUnstructured:
    @staticmethod
    def get_loader(file_path: str) -> Loader:
        return Loader(
            file_path=file_path,
            mode="elements",
            partition_via_api=True,
        )

    @classmethod
    def process_file(cls, file_path: str):
        if not os.path.exists(file_path):
            return print(f"Error: File {file_path} not found.")

        try:
            loader = cls.get_loader(file_path)
            docs = loader.load()
            return docs

        except Exception as e:
            print(f"An error occurred during document processing: {e}")
