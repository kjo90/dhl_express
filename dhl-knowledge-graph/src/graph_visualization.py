"""Neo4j-backed graph traversal service for the Streamlit knowledge-graph explorer."""
from __future__ import annotations


def node_display(node_type: str, properties: dict) -> str:
    """Return a compact, human-readable node label from live Neo4j properties."""
    for key in ("shipment_id", "name", "rule_id", "case_id", "document_id", "event_id", "resolution_id", "customer_id", "product_id", "code"):
        if properties.get(key):
            return str(properties[key])
    return node_type


def rows_to_graph(rows: list[dict]) -> dict:
    """Normalize Neo4j relationship rows into deduplicated display nodes and edges."""
    nodes, edges = {}, {}
    for row in rows:
        for prefix in ("anchor", "neighbor"):
            node_id = row[f"{prefix}_id"]
            node_type = row[f"{prefix}_type"]
            properties = row[f"{prefix}_properties"]
            nodes[node_id] = {
                "id": node_id, "type": node_type, "properties": properties,
                "display": node_display(node_type, properties),
            }
        edges[row["relationship_id"]] = {
            "id": row["relationship_id"], "source": row["source_id"], "target": row["target_id"],
            "label": row["relationship_type"],
        }
    return {"nodes": nodes, "edges": edges}


class Neo4jGraphExplorer:
    """Read-only, generic one-hop expansion over the live synthetic Neo4j graph."""

    def __init__(self, graph_queries):
        self.driver = graph_queries.driver

    def _run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as session:
            return [record.data() for record in session.run(cypher, **params)]

    def shipment_neighborhood(self, shipment_id: str) -> dict:
        """Return the selected shipment and only its immediate Neo4j neighbours."""
        rows = self._run("""
            MATCH (anchor:Shipment {shipment_id: $shipment_id})-[relationship]-(neighbor)
            RETURN elementId(anchor) AS anchor_id, labels(anchor)[0] AS anchor_type, properties(anchor) AS anchor_properties,
                   elementId(neighbor) AS neighbor_id, labels(neighbor)[0] AS neighbor_type, properties(neighbor) AS neighbor_properties,
                   elementId(relationship) AS relationship_id, type(relationship) AS relationship_type,
                   elementId(startNode(relationship)) AS source_id, elementId(endNode(relationship)) AS target_id
        """, shipment_id=shipment_id)
        return rows_to_graph(rows)

    def expand_node(self, node_id: str) -> dict:
        """Return every immediate relationship around an already-selected Neo4j node."""
        rows = self._run("""
            MATCH (anchor)-[relationship]-(neighbor)
            WHERE elementId(anchor) = $node_id
            RETURN elementId(anchor) AS anchor_id, labels(anchor)[0] AS anchor_type, properties(anchor) AS anchor_properties,
                   elementId(neighbor) AS neighbor_id, labels(neighbor)[0] AS neighbor_type, properties(neighbor) AS neighbor_properties,
                   elementId(relationship) AS relationship_id, type(relationship) AS relationship_type,
                   elementId(startNode(relationship)) AS source_id, elementId(endNode(relationship)) AS target_id
        """, node_id=node_id)
        return rows_to_graph(rows)

    def node_details(self, node_id: str) -> dict | None:
        rows = self._run("""
            MATCH (node)
            WHERE elementId(node) = $node_id
            RETURN elementId(node) AS node_id, labels(node)[0] AS node_type, properties(node) AS properties
        """, node_id=node_id)
        if not rows:
            return None
        row = rows[0]
        return {"id": row["node_id"], "type": row["node_type"], "properties": row["properties"],
                "display": node_display(row["node_type"], row["properties"])}


def graph_context_explanation(context: dict) -> str:
    """Create a deterministic shipment explanation entirely from graph context values."""
    shipment, product, destination = context["shipment"], context["product"], context["destination"]
    documents = [rule["required_document"] for rule in context["applicable_customs_rules"] if rule.get("required_document")]
    issues = [case["issue"] for case in context["historical_similar_cases"]]
    event_types = [event["event_type"] for event in context["events"]]
    sentences = [
        f"{shipment['shipment_id']} contains {product['name']} (HS {product['hs_code']}) and is travelling to {destination['name']}.",
        f"Its current shipment status is {shipment['status']}; connected events are {', '.join(event_types) or 'not recorded'}.",
    ]
    if documents:
        sentences.append(f"The connected synthetic customs rules require: {', '.join(documents)}.")
    if issues:
        sentences.append(f"Related synthetic historical cases recorded: {', '.join(issues)}.")
    sentences.append("Potential synthetic triage action: review the connected evidence before operational follow-up; this is not customs or DHL policy.")
    return " ".join(sentences)
