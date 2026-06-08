from routers.documents import documents_router


@documents_router.post("/process_text", response_model="str")
def process_text(text):
    processed_text = text.strip()
    return processed_text
