"""Evidence assembly across Neo4j graph context and Chroma vector matches."""
from __future__ import annotations


def build_hybrid_evidence(graph_context: dict, vector_documents: list[dict]) -> list[dict]:
    """Create display-ready evidence cards without inventing relationships or scores."""
    evidence = []
    for rule in graph_context["applicable_customs_rules"]:
        evidence.append({
            "source": "Graph", "evidence_type": "Applicable synthetic rule", "id": rule.get("rule_id", "unknown-rule"),
            "title": rule.get("name", "Unnamed synthetic rule"), "detail": rule.get("description", "No description recorded"),
            "required_document": rule.get("required_document"), "risk_level": rule.get("risk_level"),
        })
    for case in graph_context["historical_similar_cases"]:
        evidence.append({
            "source": "Graph", "evidence_type": "Similar synthetic case", "id": case.get("case_id", "unknown-case"),
            "title": case.get("issue", "Unspecified synthetic issue"), "detail": case.get("resolution", "No resolution recorded"),
            "clearance_days": case.get("clearance_days"), "outcome": case.get("outcome"),
        })
    for document in vector_documents:
        metadata = document.get("metadata", {})
        evidence.append({
            "source": "Vector", "evidence_type": "Semantic document match", "id": document.get("document_id", "unknown-document"),
            "title": metadata.get("title", "Untitled synthetic document"), "detail": document.get("text", ""),
            "document_type": metadata.get("document_type", "document"), "distance": round(float(document.get("distance", 0)), 4),
        })
    return evidence
