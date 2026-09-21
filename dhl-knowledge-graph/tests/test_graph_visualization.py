from src.graph_visualization import graph_context_explanation, rows_to_graph


def test_rows_to_graph_preserves_live_node_and_relationship_identity():
    graph = rows_to_graph([{
        "anchor_id": "shipment-1", "anchor_type": "Shipment", "anchor_properties": {"shipment_id": "DHL-SG-10001"},
        "neighbor_id": "product-1", "neighbor_type": "Product", "neighbor_properties": {"product_id": "PROD-001", "name": "Semiconductor Component"},
        "relationship_id": "relationship-1", "relationship_type": "CONTAINS", "source_id": "shipment-1", "target_id": "product-1",
    }])
    assert graph["nodes"]["shipment-1"]["display"] == "DHL-SG-10001"
    assert graph["nodes"]["product-1"]["display"] == "Semiconductor Component"
    assert graph["edges"]["relationship-1"]["label"] == "CONTAINS"


def test_graph_context_explanation_uses_context_values():
    context = {
        "shipment": {"shipment_id": "DHL-SG-10001", "status": "CUSTOMS_CLEARANCE"},
        "product": {"name": "Semiconductor Component", "hs_code": "8541"},
        "destination": {"name": "Germany"},
        "events": [{"event_type": "PICKED_UP"}, {"event_type": "CUSTOMS_CLEARANCE"}],
        "applicable_customs_rules": [{"required_document": "Technical Specification"}],
        "historical_similar_cases": [{"issue": "Missing product specification"}],
    }
    explanation = graph_context_explanation(context)
    assert "DHL-SG-10001" in explanation
    assert "Technical Specification" in explanation
    assert "Missing product specification" in explanation
