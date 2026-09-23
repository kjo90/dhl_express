"""Local ChromaDB retrieval for synthetic customs knowledge only."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "data" / "customs_documents.jsonl"
DEFAULT_CHROMA_PATH = ROOT / ".chroma"


class SyntheticCustomsVectorStore:
    """Persistent ChromaDB collection using local sentence-transformer embeddings."""

    def __init__(self, persist_path: Path | str = DEFAULT_CHROMA_PATH, model_name: str = "all-MiniLM-L6-v2"):
        self.persist_path = str(persist_path)
        self.model_name = model_name
        self._model = None
        self._collection = None

    def _embedding_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _collection_handle(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_path)
            self._collection = client.get_or_create_collection(
                name="synthetic_customs_documents", metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedding_model().encode(texts, normalize_embeddings=True).tolist()

    def index_documents(self, documents_path: Path | str = DEFAULT_DOCUMENTS) -> int:
        """Idempotently embed and upsert local synthetic documents into ChromaDB."""
        documents = [json.loads(line) for line in Path(documents_path).read_text().splitlines() if line.strip()]
        collection = self._collection_handle()
        collection.upsert(
            ids=[document["document_id"] for document in documents],
            documents=[document["text"] for document in documents],
            metadatas=[{
                "title": document["title"], "country_code": document["country_code"],
                "hs_code": document["hs_code"], "document_type": document["document_type"],
            } for document in documents],
            embeddings=self._embed([document["text"] for document in documents]),
        )
        return len(documents)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return semantically similar synthetic documents, ranked by Chroma distance."""
        result = self._collection_handle().query(
            query_embeddings=self._embed([query]), n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"document_id": doc_id, "text": text, "metadata": metadata, "distance": distance}
            for doc_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]


def graph_context_to_search_query(context: dict) -> str:
    """Use graph facts as a focused vector-search query; no generated facts are added."""
    product = context["product"]
    destination = context["destination"]
    rules = "; ".join(rule["name"] for rule in context["applicable_customs_rules"])
    issues = "; ".join(case["issue"] for case in context["historical_similar_cases"])
    return (
        f"{product['name']} HS {product['hs_code']} destination {destination['name']}. "
        f"Applicable requirements: {rules}. Historical issues: {issues}."
    )
