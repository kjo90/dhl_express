import pytest

from src.explanations import deterministic_explanation, optional_openai_explanation
from src.hybrid_retrieval import build_hybrid_evidence


def analysis_fixture():
    graph_context = {
        "shipment": {"shipment_id": "DHL-SG-10001", "status": "CUSTOMS_CLEARANCE", "declared_value_eur": 12500},
        "product": {"name": "Semiconductor Component", "hs_code": "8541"},
        "destination": {"name": "Germany", "code": "DE"},
        "applicable_customs_rules": [{"rule_id": "RULE-001", "name": "Technical specification required", "description": "Synthetic requirement", "required_document": "Technical Specification", "risk_level": "HIGH"}],
        "historical_similar_cases": [{"case_id": "CASE-001", "issue": "Missing product specification", "resolution": "Document submitted", "clearance_days": 3, "outcome": "CLEARED"}],
        "events": [],
    }
    documents = [{"document_id": "SYN-DOC-001", "text": "Synthetic document", "metadata": {"title": "Technical checklist", "document_type": "checklist"}, "distance": 0.2}]
    return {
        "graph_context": graph_context,
        "retrieved_synthetic_documents": documents,
        "synthetic_risk_assessment": {
            "risk_band": "HIGH", "synthetic_delay_risk_probability": 0.8,
            "feature_contributions": [
                {"feature": "high_risk_rule_count", "value": 1, "coefficient_contribution": 1.1},
                {"feature": "rule_count", "value": 1, "coefficient_contribution": 0.2},
            ],
        },
    }


def test_hybrid_evidence_keeps_graph_and_vector_provenance():
    analysis = analysis_fixture()
    evidence = build_hybrid_evidence(analysis["graph_context"], analysis["retrieved_synthetic_documents"])
    assert [item["source"] for item in evidence] == ["Graph", "Graph", "Vector"]
    assert evidence[-1]["id"] == "SYN-DOC-001"


def test_deterministic_explanation_is_grounded_in_supplied_evidence():
    explanation = deterministic_explanation(analysis_fixture())
    assert "DHL-SG-10001" in explanation["headline"]
    assert "Technical Specification" in explanation["graph_evidence"]["applicable_requirements"]
    assert explanation["vector_evidence"] == ["Technical checklist"]
    assert "synthetic" in explanation["disclaimer"].lower()


def test_optional_llm_explanation_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Enter an OpenAI API key"):
        optional_openai_explanation(analysis_fixture())
