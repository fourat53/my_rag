import requests
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_URL = os.getenv("LLAMA_DEV_URL", "http://localhost:11434")
SERVER_URL = "http://localhost:8000/llama/query"

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def get_available_models(base_url: str) -> list[str]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def stream_response(url: str, payload: dict) -> str | None:
    try:
        response = requests.post(url, json=payload, timeout=180, stream=True)
        response.raise_for_status()
        full = ""
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                decoded = chunk.decode("utf-8", errors="replace")
                print(decoded, end="", flush=True)
                full += decoded
        print()
        return full
    except requests.exceptions.ChunkedEncodingError:
        print(f"\n{RED}[Connection lost — server may have timed out or crashed]{RESET}")
    except requests.exceptions.Timeout:
        print(f"{RED}[Request timed out]{RESET}")
    except requests.exceptions.ConnectionError:
        print(f"{RED}[Could not connect to server]{RESET}")
    except requests.exceptions.HTTPError as e:
        print(f"{RED}[HTTP Error: {e}]{RESET}")
    return None


def print_banner():
    print()
    print(f"  {BOLD}{CYAN}╔══════════════════════════════════════╗{RESET}")
    print(f"  {BOLD}{CYAN}║       Local LLaMA Chat Client        ║{RESET}")
    print(f"  {BOLD}{CYAN}╚══════════════════════════════════════╝{RESET}")
    print()


def print_help():
    print(f"\n  {BOLD}Commands:{RESET}")
    print(f"    {GREEN}/model{RESET}       — Switch model")
    print(f"    {GREEN}/models{RESET}      — List available models")
    print(f"    {GREEN}/mode{RESET}        — Toggle Ollama direct / FastAPI server")
    print(f"    {GREEN}/clear{RESET}       — Clear chat history")
    print(f"    {GREEN}/history{RESET}     — Show chat history")
    print(f"    {GREEN}/help{RESET}        — Show this help")
    print(f"    {GREEN}/exit{RESET}        — Exit chat\n")


def print_status(model: str, mode: str):
    print(f"  {DIM}Model: {YELLOW}{model}{RESET}  {DIM}│  Mode: {YELLOW}{mode}{RESET}")


def query_ollama_direct(base_url: str, model: str, messages: list[dict]) -> str | None:
    url = f"{base_url}/api/chat"
    payload = {"model": model, "messages": messages, "stream": True}
    try:
        response = requests.post(url, json=payload, timeout=180, stream=True)
        response.raise_for_status()
        full = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full += token
                if data.get("done"):
                    break
        print()
        return full if full else None
    except requests.exceptions.ConnectionError:
        print(f"{RED}[Cannot reach Ollama at {base_url}]{RESET}")
    except requests.exceptions.Timeout:
        print(f"{RED}[Request timed out]{RESET}")
    except requests.exceptions.HTTPError as e:
        print(f"{RED}[HTTP Error: {e}]{RESET}")
    except Exception as e:
        print(f"{RED}[Error: {e}]{RESET}")
    return None


def query_fastapi_server(question: str) -> str | None:
    return stream_response(SERVER_URL, {"question": question})


def pick_model(base_url: str) -> str | None:
    models = get_available_models(base_url)
    if not models:
        print(f"  {RED}No models found or Ollama not reachable at {base_url}{RESET}")
        return None
    print(f"\n  {BOLD}Available models:{RESET}")
    for i, m in enumerate(models, 1):
        print(f"    {CYAN}{i}.{RESET} {m}")
    print()
    choice = input(f"  {GREEN}Select model (number or name): {RESET}").strip()
    if not choice:
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(models):
        return models[int(choice) - 1]
    if choice in models:
        return choice
    print(f"  {RED}Invalid selection{RESET}")
    return None


import json


def main():
    base_url = DEFAULT_URL
    models = get_available_models(base_url)
    current_model = models[0] if models else "qwen2.5:3b"
    mode = "ollama"
    history: list[dict] = []

    print_banner()
    print_status(current_model, mode)
    print(f"  {DIM}Type /help for commands{RESET}\n")

    while True:
        try:
            user_input = input(f"  {BOLD}{MAGENTA}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM}Bye!{RESET}\n")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "/exit" or cmd == "/quit":
            print(f"\n  {DIM}Bye!{RESET}\n")
            break

        elif cmd == "/help":
            print_help()

        elif cmd == "/model":
            new_model = pick_model(base_url)
            if new_model:
                current_model = new_model
                history.clear()
                print(f"\n  {GREEN}Switched to {BOLD}{current_model}{RESET}")
                print_status(current_model, mode)
                print()

        elif cmd == "/models":
            models = get_available_models(base_url)
            if models:
                print(f"\n  {BOLD}Available models:{RESET}")
                for m in models:
                    marker = f" {GREEN}◄{RESET}" if m == current_model else ""
                    print(f"    • {m}{marker}")
                print()
            else:
                print(f"  {RED}No models found or Ollama not reachable{RESET}")

        elif cmd == "/mode":
            mode = "fastapi" if mode == "ollama" else "ollama"
            label = "Ollama direct" if mode == "ollama" else "FastAPI server"
            print(f"\n  {GREEN}Mode → {BOLD}{label}{RESET}")
            print_status(current_model, mode)
            print()

        elif cmd == "/clear":
            history.clear()
            print(f"  {GREEN}Chat history cleared.{RESET}\n")

        elif cmd == "/history":
            if not history:
                print(f"  {DIM}No history yet.{RESET}\n")
            else:
                print()
                for msg in history:
                    role = msg["role"].capitalize()
                    color = CYAN if role == "User" else YELLOW
                    print(
                        f"  {color}{role}:{RESET} {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}"
                    )
                print()

        else:
            if mode == "ollama":
                history.append({"role": "user", "content": user_input})
                print(f"  {YELLOW}Assistant:{RESET} ", end="", flush=True)
                start = time.time()
                reply = query_ollama_direct(base_url, current_model, history)
                elapsed = time.time() - start
                if reply:
                    history.append({"role": "assistant", "content": reply})
                    print(f"  {DIM}[{elapsed:.1f}s]{RESET}")
                else:
                    history.pop()
                print()
            else:
                print(f"  {YELLOW}Assistant:{RESET} ", end="", flush=True)
                start = time.time()
                reply = query_fastapi_server(user_input)
                elapsed = time.time() - start
                if reply:
                    print(f"  {DIM}[{elapsed:.1f}s]{RESET}")
                print()


if __name__ == "__main__":
    main()
