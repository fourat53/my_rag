import os, io

from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def extract_docx(data: bytes) -> str:
    document = DocxDocument(io.BytesIO(data))
    parts: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            parts.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            row_text = " | ".join(cells).strip()
            if row_text:
                parts.append(row_text)
    return "\n\n".join(parts)


def extract_text(filename: str, data: bytes) -> str:
    extension = os.path.splitext(filename)[1].lower()
    if extension == ".pdf":
        text = extract_pdf(data)
    elif extension == ".docx":
        text = extract_docx(data)
    elif extension in {".txt", ".md", ".csv"}:
        text = data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {extension or 'unknown'}")
    return text
