import os
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import requests
import json

router = APIRouter(prefix="/models", tags=["Models"])

PROVIDER_CONFIG = {
    "nvidia": {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "base_url": os.getenv("NVIDIA_BASE_URL"),
    },
    "lm-studio": {
        "api_key": None,
        "base_url": os.getenv("LM_STUDIO_LOCAL_URL"),
    },
    "local-ollama": {
        "api_key": None,
        "base_url": os.getenv("OLLAMA_LOCAL_URL"),
    },
    "cloud-ollama": {
        "api_key": os.getenv("OLLAMA_API_KEY"),
        "base_url": os.getenv("OLLAMA_BASE_URL"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
    },
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": os.getenv("GEMINI_BASE_URL"),
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": os.getenv("OPENROUTER_BASE_URL"),
    },
}


def fetch_openai_compatible_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append(
                {
                    "id": model_id,
                    "name": model.get("name", model_id).replace("models/", "").title(),
                    "description": model.get("description", ""),
                }
            )
        return models
    except Exception as e:
        print(f"Error fetching OpenAI-compatible models from {base_url}: {e}")
        return []


def fetch_openrouter_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append(
                {
                    "id": model_id,
                    "name": model.get(
                        "name", model_id.split("/")[-1] if "/" in model_id else model_id
                    ),
                    "description": f"Context: {model.get('context_length', 'Unknown')} tokens",
                }
            )
        return models
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
        return []


@router.get("/{provider_id}")
def get_models(provider_id: str):
    if provider_id not in PROVIDER_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")

    config = PROVIDER_CONFIG[provider_id]
    base_url = config["base_url"]
    api_key = config["api_key"]

    if not base_url:
        return {"models": []}

    if provider_id == "openrouter":
        models = fetch_openrouter_models(base_url, api_key)
    else:
        models = fetch_openai_compatible_models(base_url, api_key)

    for model in models:
        model["providerId"] = provider_id

    return {"models": models}


@router.get("/")
def get_all_models():
    all_models = []

    for provider_id, config in PROVIDER_CONFIG.items():
        try:
            base_url = config["base_url"]
            api_key = config["api_key"]

            if not base_url:
                continue

            if provider_id == "openrouter":
                models = fetch_openrouter_models(base_url, api_key)
            else:
                models = fetch_openai_compatible_models(base_url, api_key)

            for model in models:
                model["providerId"] = provider_id

            all_models.extend(models)
        except Exception as e:
            print(f"Error fetching models for {provider_id}: {e}")
            continue

    return {"models": all_models}
