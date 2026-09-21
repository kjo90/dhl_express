"""Phase 2 orchestration: graph evidence + vector retrieval + synthetic ML risk."""
from __future__ import annotations

from .risk_model import SyntheticDelayRiskModel
from .hybrid_retrieval import build_hybrid_evidence
from .vector_store import SyntheticCustomsVectorStore, graph_context_to_search_query


class CustomsDelayAnalyzer:
    def __init__(self, graph_queries, vector_store=None, risk_model=None):
        self.graph_queries = graph_queries
        self.vector_store = vector_store or SyntheticCustomsVectorStore()
        self.risk_model = risk_model or SyntheticDelayRiskModel().fit_from_synthetic_data()

    def analyze(self, shipment_id: str, top_k: int = 3) -> dict | None:
        context = self.graph_queries.get_graph_context(shipment_id)
        if context is None:
            return None
        search_query = graph_context_to_search_query(context)
        vector_documents = self.vector_store.search(search_query, top_k=top_k)
        return {
            "shipment_id": shipment_id,
            "graph_context": context,
            "synthetic_risk_assessment": self.risk_model.predict(context),
            "vector_search_query": search_query,
            "retrieved_synthetic_documents": vector_documents,
            "hybrid_evidence": build_hybrid_evidence(context, vector_documents),
            "disclaimer": "All graph data, documents, retrieval evidence, and risk scoring are synthetic demonstrations.",
        }
