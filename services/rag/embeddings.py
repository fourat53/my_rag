import logging
import os
import random
import re
import time

from collections import deque
from collections.abc import Callable, Iterator
from typing import TypeVar

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.pinecone import PINECONE_INDEX_DIMENSION

logger = logging.getLogger(__name__)

RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED", "RATE LIMIT")
RETRY_DELAY_PATTERN = re.compile(r"retryDelay[^0-9]*(\d+(?:\.\d+)?)s")
MAX_ATTEMPTS = 4
MIN_RETRY_DELAY_SECONDS = 5.0
MAX_RETRY_DELAY_SECONDS = 90.0
RATE_LIMIT_MESSAGE = (
    "The embedding provider rate limit was reached. Wait a minute and try again "
    "or upload a smaller file."
)

CHARS_PER_TOKEN = 3.5
TOKENS_PER_MINUTE_BUDGET = 25000
QUOTA_WINDOW_SECONDS = 60.0
MAX_BATCH_TEXTS = 64
MAX_BATCH_CHARS = 20000

Result = TypeVar("Result")


class EmbeddingRateLimitError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(RATE_LIMIT_MESSAGE)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    output_dimensionality=PINECONE_INDEX_DIMENSION,
)

quota_window: deque[tuple[float, float]] = deque()


def is_rate_limit_error(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def retry_delay_seconds(error: Exception, attempt: int) -> float:
    match = RETRY_DELAY_PATTERN.search(str(error))
    delay = match.group(1) if match else MIN_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
    return min(MAX_RETRY_DELAY_SECONDS, max(MIN_RETRY_DELAY_SECONDS, float(delay)))


def run_with_rate_limit_retry(operation: Callable[[], Result]) -> Result:
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as error:
            rate_limited = is_rate_limit_error(error)
            if not rate_limited:
                raise
            if attempt >= MAX_ATTEMPTS:
                raise EmbeddingRateLimitError() from error
            delay = retry_delay_seconds(error, attempt) + random.uniform(0, 3)
            logger.warning(
                "Embedding rate limit reached (attempt %s/%s); retrying in %.0f seconds.",
                attempt,
                MAX_ATTEMPTS,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def estimate_tokens(texts: list[str]) -> float:
    return sum(len(text) for text in texts) / CHARS_PER_TOKEN


def wait_for_quota(tokens: float) -> None:
    while True:
        now = time.monotonic()
        while quota_window and now - quota_window[0][0] >= QUOTA_WINDOW_SECONDS:
            quota_window.popleft()
        used = sum(window_tokens for _, window_tokens in quota_window)
        if used + tokens <= TOKENS_PER_MINUTE_BUDGET:
            quota_window.append((now, tokens))
            return
        delay = QUOTA_WINDOW_SECONDS - (now - quota_window[0][0]) + 1
        logger.info("Embedding quota window full; pausing %.0f seconds.", delay)
        time.sleep(delay)


def batch_texts(texts: list[str]) -> Iterator[list[str]]:
    batch: list[str] = []
    batch_chars = 0
    for text in texts:
        if batch and (
            len(batch) >= MAX_BATCH_TEXTS or batch_chars + len(text) > MAX_BATCH_CHARS
        ):
            yield batch
            batch = []
            batch_chars = 0
        batch.append(text)
        batch_chars += len(text)
    if batch:
        yield batch


def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batch_texts(texts):
        wait_for_quota(estimate_tokens(batch))
        vectors.extend(
            run_with_rate_limit_retry(
                lambda batch=batch: embeddings.embed_documents(batch)
            )
        )
    return vectors


def embed_query(text: str) -> list[float]:
    return run_with_rate_limit_retry(lambda: embeddings.embed_query(text))
