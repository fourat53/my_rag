import os
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import requests

router = APIRouter(prefix="/models", tags=["Models"])

PROVIDER_CONFIG = {
    "local-ollama": {
        "base_url": os.getenv("LOCAL_OLLAMA_URL", "http://localhost:11434"),
        "api_key": None,
        "models_endpoint": "/api/tags",
    },
    "cloud-ollama": {
        "base_url": os.getenv("CLOUD_OLLAMA_URL", ""),
        "api_key": os.getenv("CLOUD_OLLAMA_API_KEY", ""),
        "models_endpoint": "/v1/models",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "models_endpoint": "/models",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "models_endpoint": "/models",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "models_endpoint": "/models",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "models_endpoint": "/models",
    },
}


def fetch_ollama_models(base_url: str) -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("models", []):
            model_name = model.get("name", model.get("model", ""))
            models.append({
                "id": model_name,
                "name": model_name.replace(":latest", "").replace(":", " ").title(),
                "description": f"Size: {model.get('size', 'Unknown')}",
            })
        return models
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []


def fetch_openai_compatible_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append({
                "id": model_id,
                "name": model.get("name", model_id),
                "description": model.get("description", ""),
            })
        return models
    except Exception as e:
        print(f"Error fetching OpenAI-compatible models: {e}")
        return []


def fetch_gemini_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            f"{base_url}/models?key={api_key}",
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("models", []):
            model_id = model.get("name", "").replace("models/", "")
            if "generateContent" in model.get("supportedGenerationMethods", []):
                models.append({
                    "id": model_id,
                    "name": model_id.replace("gemini-", "Gemini ").replace("-", " ").title(),
                    "description": f"Version: {model.get('version', 'Unknown')}",
                })
        return models
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
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
            models.append({
                "id": model_id,
                "name": model.get("name", model_id.split("/")[-1] if "/" in model_id else model_id),
                "description": f"Context: {model.get('context_length', 'Unknown')} tokens",
            })
        return models
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
        return []


def fetch_nvidia_models(base_url: str, api_key: str) -> List[Dict[str, Any]]:
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        models = []
        for model in data.get("data", []):
            model_id = model.get("id", "")
            models.append({
                "id": model_id,
                "name": model.get("name", model_id),
                "description": model.get("description", ""),
            })
        return models
    except Exception as e:
        print(f"Error fetching Nvidia models: {e}")
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
    
    if provider_id == "local-ollama":
        models = fetch_ollama_models(base_url)
    elif provider_id == "gemini":
        models = fetch_gemini_models(base_url, api_key)
    elif provider_id == "openrouter":
        models = fetch_openrouter_models(base_url, api_key)
    elif provider_id in ["openai", "cloud-ollama", "nvidia"]:
        models = fetch_openai_compatible_models(base_url, api_key)
    else:
        models = []
    
    for model in models:
        model["providerId"] = provider_id
    
    return {"models": models}


@router.get("/")
def get_all_models():
    all_models = []
    
    for provider_id in PROVIDER_CONFIG:
        try:
            config = PROVIDER_CONFIG[provider_id]
            base_url = config["base_url"]
            api_key = config["api_key"]
            
            if not base_url:
                continue
            
            if provider_id == "local-ollama":
                models = fetch_ollama_models(base_url)
            elif provider_id == "gemini":
                models = fetch_gemini_models(base_url, api_key)
            elif provider_id == "openrouter":
                models = fetch_openrouter_models(base_url, api_key)
            elif provider_id in ["openai", "cloud-ollama", "nvidia"]:
                models = fetch_openai_compatible_models(base_url, api_key)
            else:
                models = []
            
            for model in models:
                model["providerId"] = provider_id
            
            all_models.extend(models)
        except Exception as e:
            print(f"Error fetching models for {provider_id}: {e}")
            continue
    
    return {"models": all_models}
