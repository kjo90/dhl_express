"""Load the synthetic DHL Express customs Knowledge Graph into Neo4j."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def records(filename: str) -> list[dict]:
    """Read a CSV as Python-native values, preserving text identifiers."""
    return pd.read_csv(DATA / filename, keep_default_na=False).to_dict("records")


def run(session, query: str, **params) -> None:
    session.run(query, **params).consume()


def load(uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    with driver.session() as session:
        for statement in (ROOT / "graph" / "constraints.cypher").read_text().split(";"):
            if statement.strip():
                run(session, statement)

        run(session, "UNWIND $rows AS row MERGE (c:Country {code: row.code}) SET c += row", rows=records("countries.csv"))
        run(session, "UNWIND $rows AS row MERGE (c:Customer {customer_id: row.customer_id}) SET c += row", rows=records("customers.csv"))
        run(session, """
            UNWIND $rows AS row
            MERGE (h:HSCode {code: toString(row.hs_code)})
            MERGE (p:Product {product_id: row.product_id})
            SET p.name = row.name, p.category = row.category
            MERGE (p)-[:HAS_HS_CODE]->(h)
        """, rows=records("products.csv"))
        run(session, """
            UNWIND $rows AS row
            MATCH (customer:Customer {customer_id: row.customer_id})
            MATCH (product:Product {product_id: row.product_id})
            MATCH (origin:Country {code: row.origin_code})
            MATCH (destination:Country {code: row.destination_code})
            MERGE (s:Shipment {shipment_id: row.shipment_id})
            SET s.declared_value_eur = toFloat(row.declared_value_eur), s.status = row.status,
                s.shipment_date = row.shipment_date
            MERGE (customer)-[:SHIPS]->(s)
            MERGE (s)-[:CONTAINS]->(product)
            MERGE (s)-[:SHIPPED_FROM]->(origin)
            MERGE (s)-[:SHIPPED_TO]->(destination)
        """, rows=records("shipments.csv"))
        run(session, """
            UNWIND $rows AS row
            MATCH (country:Country {code: row.country_code})
            MERGE (h:HSCode {code: toString(row.hs_code)})
            MERGE (d:Document {document_id: row.document_id})
            SET d.name = row.document_name
            MERGE (r:CustomsRule {rule_id: row.rule_id})
            SET r.name = row.rule_name, r.description = row.description, r.risk_level = row.risk_level,
                r.hs_code = toString(row.hs_code)
            MERGE (country)-[:HAS_CUSTOMS_RULE]->(r)
            MERGE (r)-[:APPLIES_TO]->(h)
            MERGE (r)-[:REQUIRES]->(d)
        """, rows=records("customs_rules.csv"))
        run(session, """
            UNWIND $rows AS row
            MATCH (product:Product {product_id: row.product_id})
            MATCH (country:Country {code: row.country_code})
            MERGE (resolution:Resolution {resolution_id: row.resolution_id})
            SET resolution.description = row.resolution
            MERGE (cc:CustomsCase {case_id: row.case_id})
            SET cc.issue = row.issue, cc.outcome = row.outcome, cc.clearance_days = toInteger(row.clearance_days)
            MERGE (cc)-[:INVOLVED_PRODUCT]->(product)
            MERGE (cc)-[:OCCURRED_IN]->(country)
            MERGE (cc)-[:RESOLVED_BY]->(resolution)
        """, rows=records("customs_cases.csv"))

        shipments = records("shipments.csv")
        events = []
        for row in shipments:
            events.extend([
                {"event_id": f"{row['shipment_id']}-E1", "shipment_id": row["shipment_id"], "event_type": "PICKED_UP", "event_time": row["shipment_date"], "location": row["origin_code"]},
                {"event_id": f"{row['shipment_id']}-E2", "shipment_id": row["shipment_id"], "event_type": "CUSTOMS_CLEARANCE" if row["status"] == "CUSTOMS_CLEARANCE" else "IN_TRANSIT", "event_time": row["shipment_date"], "location": row["destination_code"]},
            ])
        run(session, """
            UNWIND $rows AS row
            MATCH (s:Shipment {shipment_id: row.shipment_id})
            MERGE (e:ShipmentEvent {event_id: row.event_id})
            SET e.event_type = row.event_type, e.event_time = row.event_time, e.location = row.location
            MERGE (s)-[:HAS_EVENT]->(e)
        """, rows=events)
    driver.close()


if __name__ == "__main__":
    load(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "password"),
    )
    print("Synthetic DHL Knowledge Graph loaded successfully.")
