from typing import Any, List

import chromadb
import requests

from config import Config
from documents import DocumentChunk


def get_chroma_client():
    """Create and return a persistent Chroma client."""
    return chromadb.PersistentClient(path=Config.CHROMA_PATH)


def get_or_create_collection():
    """Get or create the Chroma collection for the knowledge assistant."""
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=Config.COLLECTION_NAME
    )


def get_embedding(text: str) -> list[float]:
    """Create an embedding using the configured Ollama embedding model."""
    url = f"{Config.OLLAMA_BASE_URL.rstrip('/')}/api/embed"

    response = requests.post(
        url,
        json={
            "model": Config.EMBEDDING_MODEL,
            "input": text,
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    embeddings = data.get("embeddings")

    if not embeddings:
        raise ValueError("Embedding model returned no embeddings.")

    return embeddings[0]


def seed_vector_store(chunks: List[DocumentChunk]) -> int:
    """Store document chunks and their embeddings in Chroma."""
    collection = get_or_create_collection()

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for chunk in chunks:
        ids.append(chunk.id)
        documents.append(chunk.text)

        metadatas.append(
            {
                "source": chunk.source,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index,
            }
        )

        embeddings.append(get_embedding(chunk.text))

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def retrieve_relevant_chunks(
    question: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant knowledge-base chunks."""
    collection = get_or_create_collection()

    if collection.count() == 0:
        return []

    embedding = get_embedding(question)

    number_to_retrieve = top_k or Config.TOP_K

    results = collection.query(
        query_embeddings=[embedding],
        n_results=number_to_retrieve,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []

    for index, text in enumerate(documents):
        metadata = metadatas[index] or {}

        chunk = {
            "text": text,
            "source": metadata.get("source", "unknown"),
            "title": metadata.get("title", "Unknown Source"),
            "chunk_index": metadata.get("chunk_index"),
        }

        if index < len(distances):
            chunk["distance"] = distances[index]

        chunks.append(chunk)

    return chunks
