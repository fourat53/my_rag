from services.documents import load_manual, load_unstructured, write_text
from routers.documents import documents_router
from fastapi import HTTPException, status
from models import (
    LoadFileRequest,
    LoadFileResponse,
    FileNotFoundError,
    FileTypeError,
    DocumentLoadError,
)


@documents_router.post("/load_file", response_model=LoadFileResponse)
def load_file(req: LoadFileRequest):
    file_path = f"data/{req.file}"

    try:
        if req.loader_type == "unstructured":
            docs = load_unstructured(file_path)

        elif req.loader_type == "manual":
            docs = load_manual(file_path)

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported loader type {req.loader_type}",
            )

        write_text(
            req.loader_type,
            req.file,
            "\n\n".join(doc.page_content for doc in docs),
        )

        return {
            "status": "success",
            "documents_count": len(docs),
            "documents": docs,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    except FileTypeError as e:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(e)
        )

    except DocumentLoadError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"❌ Internal server error: {str(e)}",
        )
