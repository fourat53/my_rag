import os, requests

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/models", tags=["Models"])


PROVIDER_API_KEYS: dict[str, str] = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "cloud-ollama": os.getenv("OLLAMA_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "nvidia": os.getenv("NVIDIA_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
}


@router.get("/{provider}")
def get_models(provider: str, base_url: str) -> dict[str, list[dict[str, str | None]]]:
    if provider not in PROVIDER_API_KEYS:
        raise HTTPException(
            status_code=400, detail=f"No API key for provider: {provider}"
        )
    try:
        api_key = PROVIDER_API_KEYS[provider]
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        # print("DATA:", json.dumps(data.get("data", [])[:1], indent=2))
        models = []

        for model in data.get("data", []):
            model_id = model.get("id")
            model_name = None
            if provider == "openrouter":
                model_name = model.get("name").split(": ")[-1]
            elif provider == "gemini":
                model_name = model.get("display_name")
            elif provider == "nvidia":
                model_name = model_id.split("/")[-1]
            else:
                model_name = model_id

            models.append({"id": model_id, "name": model_name})

        return {"models": models}
    except Exception as e:
        print(f"Error fetching provider models from {base_url}: {e}")
        return {"models": []}
