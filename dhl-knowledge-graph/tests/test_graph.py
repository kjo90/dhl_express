from pathlib import Path

import pandas as pd

from src.graph_queries import DHLGraphQueries

ROOT = Path(__file__).resolve().parents[1]


class FakeRecord:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query, **params):
        self.queries.append((query, params))
        return [FakeRecord(item) for item in next(self.responses)]


class FakeDriver:
    def __init__(self, responses):
        self.session_instance = FakeSession(responses)

    def session(self):
        return self.session_instance

    def close(self):
        pass


def test_synthetic_data_meets_requested_scale():
    assert len(pd.read_csv(ROOT / "data" / "shipments.csv")) >= 20
    assert len(pd.read_csv(ROOT / "data" / "customers.csv")) >= 8
    assert len(pd.read_csv(ROOT / "data" / "products.csv")) >= 10
    assert len(pd.read_csv(ROOT / "data" / "countries.csv")) == 6
    assert len(pd.read_csv(ROOT / "data" / "customs_rules.csv")) >= 10
    cases = pd.read_csv(ROOT / "data" / "customs_cases.csv")
    assert len(cases) >= 20
    assert set(["CASE-001", "CASE-002", "CASE-003"]).issubset(cases.case_id)


def test_context_returns_expected_synthetic_semiconductor_evidence():
    responses = [
        [{
            "shipment": {"shipment_id": "DHL-SG-10001", "status": "CUSTOMS_CLEARANCE"},
            "product": {"name": "Semiconductor Component", "hs_code": "8541", "product_id": "PROD-001"},
            "destination": {"name": "Germany", "code": "DE"},
            "rules": [{"rule_id": "RULE-001"}, {"rule_id": "RULE-002"}],
            "cases": [{"case_id": "CASE-001"}, {"case_id": "CASE-002"}, {"case_id": "CASE-003"}],
            "events": [{"event_type": "PICKED_UP"}],
        }],
    ]
    driver = FakeDriver(responses)
    context = DHLGraphQueries(driver=driver).get_graph_context("DHL-SG-10001")
    assert context["product"]["hs_code"] == "8541"
    assert context["destination"]["code"] == "DE"
    assert [case["case_id"] for case in context["historical_similar_cases"]] == ["CASE-001", "CASE-002", "CASE-003"]
    context_query = driver.session_instance.queries[0][0]
    assert all(term in context_query for term in ["HAS_HS_CODE", "HAS_CUSTOMS_RULE", "INVOLVED_PRODUCT", "OCCURRED_IN"])
