import os, re, time, requests

from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException

router = APIRouter(prefix="/models", tags=["Models"])


PROVIDER_API_KEYS: dict[str, str] = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "cloud-ollama": os.getenv("OLLAMA_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "nvidia": os.getenv("NVIDIA_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
}

GEMINI_CHAT_METHOD = "generateContent"
GEMINI_PROBE_TIMEOUT_SECONDS = 10
GEMINI_PROBE_CACHE_TTL_SECONDS = 3600
GEMINI_VERSION_PATTERN = re.compile(r"gemini-(\d+(?:\.\d+)?)")
GEMINI_EXCLUDED_MARKERS = (
    "tts",
    "transcribe",
    "live",
    "image",
    "native-audio",
    "robotics",
    "nano-banana",
    "lyria",
    "aqa",
    "deep-research",
    "antigravity",
)

probe_cache: dict[tuple[str, str], tuple[float, list[str]]] = {}


def is_gemini_chat_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return not any(marker in normalized for marker in GEMINI_EXCLUDED_MARKERS)


def fetch_gemini_native_models(api_key: str, base_url: str) -> list[dict]:
    parsed = urlparse(base_url)
    response = requests.get(
        f"{parsed.scheme}://{parsed.netloc}/v1beta/models",
        headers={"x-goog-api-key": api_key},
        params={"pageSize": 200},
        timeout=30,
    )
    response.raise_for_status()
    return [
        m
        for m in response.json().get("models", [])
        if GEMINI_CHAT_METHOD in m.get("supportedGenerationMethods", [])
        and is_gemini_chat_model(m.get("name", ""))
    ]


def fetch_gemini_compat_models(api_key: str, base_url: str) -> list[dict]:
    response = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return [
        m
        for m in response.json().get("data", [])
        if m.get("id") and is_gemini_chat_model(m["id"])
    ]


def probe_gemini_model(base_url: str, api_key: str, model_id: str) -> str:
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            },
            timeout=GEMINI_PROBE_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        return "error"
    return "ok" if response.status_code == 200 else "unavailable"


def validate_gemini_models(
    base_url: str, api_key: str, model_ids: list[str]
) -> list[str]:
    cache_key = (base_url, api_key)
    cached = probe_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < GEMINI_PROBE_CACHE_TTL_SECONDS:
        return cached[1]

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(
            pool.map(lambda mid: probe_gemini_model(base_url, api_key, mid), model_ids)
        )
    valid = [mid for mid, status in zip(model_ids, statuses) if status == "ok"]

    if any(status != "error" for status in statuses):
        probe_cache[cache_key] = (time.monotonic(), valid)
    return valid


def order_gemini_models_by_version(model_id: str) -> tuple[int, float]:
    match = GEMINI_VERSION_PATTERN.search(model_id)
    if not match:
        return (1, 0.0)
    return (0, -float(match.group(1)))


def gemini_models(api_key: str | None, base_url: str) -> list[dict[str, str | None]]:
    if api_key is None:
        return []
    try:
        fetched = [
            (m.get("name"), m.get("displayName"))
            for m in fetch_gemini_native_models(api_key, base_url)
        ]
    except requests.RequestException:
        fetched = [
            (m.get("id"), m.get("display_name"))
            for m in fetch_gemini_compat_models(api_key, base_url)
        ]

    candidates = [(mid, name) for mid, name in fetched if mid]
    titles = dict(candidates)
    valid_ids = validate_gemini_models(base_url, api_key, [mid for mid, _ in candidates])
    valid_ids.sort(key=order_gemini_models_by_version)
    return [
        {"id": model_id, "name": titles.get(model_id) or model_id.split("/")[-1]}
        for model_id in valid_ids
    ]


@router.get("/{provider}")
def get_models(
    provider: str,
    base_url: str,
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
) -> dict[str, list[dict[str, str | None]]]:
    if provider not in PROVIDER_API_KEYS and x_api_key is None:
        raise HTTPException(
            status_code=400, detail=f"No API key for provider: {provider}"
        )
    api_key = x_api_key or PROVIDER_API_KEYS.get(provider)
    if provider == "gemini":
        return {"models": gemini_models(api_key, base_url)}
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        models = []

        for model in data.get("data", []):
            model_id = model.get("id")
            model_name = None
            if provider == "openrouter":
                model_name = model.get("name").split(": ")[-1]
            elif provider == "nvidia":
                model_name = model_id.split("/")[-1]
            else:
                model_name = model_id

            models.append({"id": model_id, "name": model_name})

        return {"models": models}
    except Exception as e:
        print(f"Error fetching provider models from {base_url}: {e}")
        return {"models": []}
