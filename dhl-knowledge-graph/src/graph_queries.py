"""Read-only, parameterized queries for the synthetic DHL graph."""
from __future__ import annotations

import os
from neo4j import GraphDatabase


class DHLGraphQueries:
    def __init__(self, uri=None, user=None, password=None, driver=None):
        self.driver = driver or GraphDatabase.driver(
            uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.getenv("NEO4J_USER", "neo4j"), password or os.getenv("NEO4J_PASSWORD", "password")),
        )

    def close(self):
        self.driver.close()

    def _query(self, cypher, shipment_id):
        with self.driver.session() as session:
            return [record.data() for record in session.run(cypher, shipment_id=shipment_id)]

    def get_shipment(self, shipment_id):
        return self._query("MATCH (s:Shipment {shipment_id: $shipment_id}) RETURN s {.*} AS shipment", shipment_id)

    def get_shipment_product(self, shipment_id):
        return self._query("""
            MATCH (:Shipment {shipment_id: $shipment_id})-[:CONTAINS]->(p:Product)-[:HAS_HS_CODE]->(h:HSCode)
            RETURN p {.*, hs_code: h.code} AS product
        """, shipment_id)

    def get_destination_country(self, shipment_id):
        return self._query("""
            MATCH (:Shipment {shipment_id: $shipment_id})-[:SHIPPED_TO]->(c:Country)
            RETURN c {.*} AS country
        """, shipment_id)

    def get_customs_rules(self, shipment_id):
        return self._query("""
            MATCH (:Shipment {shipment_id: $shipment_id})-[:CONTAINS]->(:Product)-[:HAS_HS_CODE]->(h:HSCode)
            MATCH (:Shipment {shipment_id: $shipment_id})-[:SHIPPED_TO]->(c:Country)-[:HAS_CUSTOMS_RULE]->(r:CustomsRule)-[:APPLIES_TO]->(h)
            OPTIONAL MATCH (r)-[:REQUIRES]->(d:Document)
            RETURN r {.*, required_document: d.name} AS rule ORDER BY r.rule_id
        """, shipment_id)

    def get_historical_similar_cases(self, shipment_id):
        return self._query("""
            MATCH (:Shipment {shipment_id: $shipment_id})-[:CONTAINS]->(p:Product)
            MATCH (:Shipment {shipment_id: $shipment_id})-[:SHIPPED_TO]->(country:Country)
            MATCH (cc:CustomsCase)-[:INVOLVED_PRODUCT]->(p)
            MATCH (cc)-[:OCCURRED_IN]->(country)
            OPTIONAL MATCH (cc)-[:RESOLVED_BY]->(resolution:Resolution)
            RETURN cc {.*, resolution: resolution.description} AS customs_case ORDER BY cc.case_id
        """, shipment_id)

    def get_shipment_events(self, shipment_id):
        return self._query("""
            MATCH (:Shipment {shipment_id: $shipment_id})-[:HAS_EVENT]->(e:ShipmentEvent)
            RETURN e {.*} AS event ORDER BY e.event_time, e.event_id
        """, shipment_id)

    def get_graph_context(self, shipment_id):
        """Return explainable graph evidence for customs-delay triage; no prediction is made."""
        rows = self._query("""
            MATCH (s:Shipment {shipment_id: $shipment_id})
            CALL (s) {
                MATCH (s)-[:CONTAINS]->(p:Product)-[:HAS_HS_CODE]->(h:HSCode)
                MATCH (s)-[:SHIPPED_TO]->(destination:Country)
                OPTIONAL MATCH (destination)-[:HAS_CUSTOMS_RULE]->(rule:CustomsRule)-[:APPLIES_TO]->(h)
                OPTIONAL MATCH (rule)-[:REQUIRES]->(document:Document)

                RETURN
                    p {.*, hs_code: h.code} AS product,
                    destination {.*} AS destination,
                    [item IN collect(
                        DISTINCT rule {.*, required_document: document.name}
                    ) WHERE item IS NOT NULL] AS rules
            }

            CALL (s, product, destination) {
                MATCH (customs_case:CustomsCase)
                    -[:INVOLVED_PRODUCT]->
                    (p:Product {product_id: product.product_id})

                MATCH (customs_case)-[:OCCURRED_IN]->
                    (:Country {code: destination.code})

                OPTIONAL MATCH (customs_case)-[:RESOLVED_BY]->
                    (resolution:Resolution)

                RETURN collect(
                    DISTINCT customs_case {
                        .*,
                        resolution: resolution.description
                    }
                ) AS cases
            }

            CALL (s) {
                OPTIONAL MATCH (s)-[:HAS_EVENT]->(event:ShipmentEvent)

                RETURN collect(
                    DISTINCT event {.*}
                ) AS events
            }

            RETURN
                s {.*} AS shipment,
                product,
                destination,
                rules,
                cases,
                events
        """, shipment_id)
        if not rows:
            return None
        row = rows[0]
        return {
            "shipment": row["shipment"],
            "product": row["product"],
            "destination": row["destination"],
            "applicable_customs_rules": row["rules"],
            "historical_similar_cases": row["cases"],
            "events": sorted(row["events"], key=lambda event: (event.get("event_time", ""), event.get("event_id", ""))),
        }
