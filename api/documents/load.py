from services.documents.load_unstructured import LoadUnstructured
from services.documents.load_manual import LoadManual
from models.request import LoadDocumentRequest


async def load(req: LoadDocumentRequest):
    try:
        if req.loader_type == "unstructured":
            data = LoadUnstructured.load_file(req.file_path)
        else:
            data = LoadManual.load_file(req.file_path)
        return {"status": "success", "response": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}
