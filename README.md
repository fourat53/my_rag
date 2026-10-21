# AI Microservice Backend

## Overview

This project is a FastAPI-based AI microservice providing robust endpoints for Retrieval-Augmented Generation (RAG) and Local LLM processing. It leverages Langchain, Pinecone for vector storage, Google GenAI, and Ollama to power the intelligent backend for our chat application.

## Installation

1. Navigate to the backend directory:
   ```bash
   cd my_rag
   ```
2. Create and activate a Python virtual environment:
   ```bash
   uv venv venv --python 3.12
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install the dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
4. Set up environment variables. Create a `.env` file in the root directory and configure necessary keys:
   ```env
   NEXT_PROD_URL=https://your-production-frontend-url.com
   NEXT_DEV_URL=http://localhost:3000
   # Add your specific API keys here (e.g., PINECONE_API_KEY, GEMINI_API_KEY)
   ```

## Usage

You can run the backend service either locally using Uvicorn or via Docker.

**Running locally:**

1. Start the FastAPI server using Uvicorn:
   ```bash
   uv run uvicorn main:app --reload
   ```
2. The API will be available at `http://localhost:8000`. You can explore the interactive API documentation at `http://localhost:8000/docs`.

**Running with Docker:**

1. Ensure Docker is installed and running.
2. Build and run the container:
   ```bash
   docker compose up --build
   ```

## Licensing

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! To contribute:

1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
