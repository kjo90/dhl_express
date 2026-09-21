// Conceptual schema for this synthetic prototype. This file is documentation;
// graph/load_graph.py creates the nodes and relationships from CSV data.
// (Customer)-[:SHIPS]->(Shipment)-[:CONTAINS]->(Product)-[:HAS_HS_CODE]->(HSCode)
// (Shipment)-[:SHIPPED_FROM]->(Country)
// (Shipment)-[:SHIPPED_TO]->(Country)-[:HAS_CUSTOMS_RULE]->(CustomsRule)-[:REQUIRES]->(Document)
// (CustomsCase)-[:INVOLVED_PRODUCT]->(Product)
// (CustomsCase)-[:OCCURRED_IN]->(Country)
// (CustomsCase)-[:RESOLVED_BY]->(Resolution)
// (Shipment)-[:HAS_EVENT]->(ShipmentEvent)
