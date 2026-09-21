from src.customs_delay_analysis import CustomsDelayAnalyzer
from src.vector_store import graph_context_to_search_query


def sample_context():
    return {
        "shipment": {"shipment_id": "DHL-SG-10001", "status": "CUSTOMS_CLEARANCE", "declared_value_eur": 12500},
        "product": {"name": "Semiconductor Component", "hs_code": "8541"},
        "destination": {"name": "Germany", "code": "DE"},
        "applicable_customs_rules": [{"name": "Technical specification required", "risk_level": "HIGH"}],
        "historical_similar_cases": [{"issue": "Missing product specification", "clearance_days": 3}],
        "events": [],
    }


class FakeGraph:
    def get_graph_context(self, shipment_id):
        return sample_context() if shipment_id == "DHL-SG-10001" else None


class FakeVectorStore:
    def search(self, query, top_k):
        return [{"document_id": "SYN-DOC-001", "text": "synthetic", "metadata": {}, "distance": 0.1}]


class FakeRiskModel:
    def predict(self, context):
        return {"risk_band": "HIGH", "synthetic_delay_risk_probability": 0.8}


def test_vector_query_is_grounded_in_graph_context():
    query = graph_context_to_search_query(sample_context())
    assert "Semiconductor Component" in query
    assert "8541" in query
    assert "Germany" in query
    assert "Missing product specification" in query


def test_analyzer_combines_graph_vector_and_risk_evidence():
    result = CustomsDelayAnalyzer(FakeGraph(), FakeVectorStore(), FakeRiskModel()).analyze("DHL-SG-10001")
    assert result["synthetic_risk_assessment"]["risk_band"] == "HIGH"
    assert result["retrieved_synthetic_documents"][0]["document_id"] == "SYN-DOC-001"
    assert CustomsDelayAnalyzer(FakeGraph(), FakeVectorStore(), FakeRiskModel()).analyze("UNKNOWN") is None
