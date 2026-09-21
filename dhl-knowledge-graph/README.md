# DHL Express Knowledge Graph — Phases 1–3

This is a Neo4j prototype for the question: **“Why might this shipment be at risk of customs delay?”** It uses only synthetic data. Rule names, document requirements, cases, and outcomes are fictional examples, **not DHL policy or real customs guidance**.

## Business problem

Cross-border operations need an explainable way to connect a shipment with its product, HS code, destination, relevant requirements, and comparable historical cases. A status such as `CUSTOMS_CLEARANCE` alone does not explain what could need attention.

For the demonstration shipment `DHL-SG-10001`, the graph connects Singapore → Germany, TechCorp Singapore, a Semiconductor Component (HS `8541`), two synthetic Germany/HS-8541 requirements, and three synthetic prior semiconductor/Germany cases. Those cases show plausible evidence to check: product specification, classification evidence, and supporting documents.

## Phase 1: Graph concept and schema

Neo4j holds entities as nodes and their business context as relationships:

```text
(Customer)-[:SHIPS]->(Shipment)-[:CONTAINS]->(Product)-[:HAS_HS_CODE]->(HSCode)
                         |                    
                         +-[:SHIPPED_TO]->(Country)-[:HAS_CUSTOMS_RULE]->(CustomsRule)-[:REQUIRES]->(Document)

(CustomsCase)-[:INVOLVED_PRODUCT]->(Product)
(CustomsCase)-[:OCCURRED_IN]->(Country)
(CustomsCase)-[:RESOLVED_BY]->(Resolution)
(Shipment)-[:HAS_EVENT]->(ShipmentEvent)
```

The graph is useful because it makes multi-hop, evidence-based investigation simple: operations can see *which* synthetic requirements and analogous prior cases are connected to a specific shipment, instead of treating customs status as an isolated field.

## Phase 2: Vector search and synthetic risk model

Phase 2 preserves the Neo4j layer and consumes its existing `get_graph_context()` result. It adds:

- `data/customs_documents.jsonl`: 12 fictional customs checklists, guides, and historical case notes.
- `src/vector_store.py`: a local persistent ChromaDB collection at `.chroma/`, with `all-MiniLM-L6-v2` sentence-transformer embeddings.
- `src/risk_model.py`: an explainable logistic-regression model trained only on deterministic labels generated from the synthetic CSV dataset.
- `src/customs_delay_analysis.py`: one orchestration layer combining graph context, vector retrieval, and the synthetic score.

The first vector-indexing run downloads the sentence-transformer model. No external data, real shipment data, real DHL policies, or actual customs rules are used. The probability is a demonstration feature, not a decision, prediction, or regulatory recommendation.

## Phase 3: Dashboard and hybrid evidence

Phase 3 provides [app.py](app.py), a Streamlit dashboard that brings the prototype together:

- A decision snapshot for selected synthetic shipments.
- Hybrid evidence: graph requirements and historical cases alongside semantically retrieved Chroma documents, with source provenance retained.
- A deterministic explanation derived strictly from graph, vector, and model values—this is the default and does not require an API key.
- An optional OpenAI Responses API explanation. It is disabled unless `OPENAI_API_KEY` is explicitly set; only the existing synthetic analysis payload is sent. The optional path is constrained to label findings synthetic and cite supplied evidence IDs.

The optional integration uses the official OpenAI SDK’s Responses API pattern. See the [official OpenAI quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request) for API-key setup.

## Prerequisites

- Python 3.11+
- Docker Desktop with Docker Compose

## Setup and demo

From this project directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d
python graph/load_graph.py
PYTHONPATH=. python -c 'from src.graph_queries import DHLGraphQueries; import json; q = DHLGraphQueries(); print(json.dumps(q.get_graph_context("DHL-SG-10001"), indent=2)); q.close()'
python -m pytest -q
```

For Phase 2, once Neo4j is healthy and the graph is loaded, run:

```bash
PYTHONPATH=. python -m src.demo_phase2
```

The command indexes the synthetic documents, retrieves semantically related evidence for `DHL-SG-10001`, and prints an integrated JSON report. Re-running it is safe: document IDs are upserted into the local ChromaDB collection.

For the Phase 3 dashboard:

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (normally `http://localhost:8501`), select `DHL-SG-10001`, and choose **Analyze shipment**. To enable—not require—the LLM summary, set an API key first:

```bash
export OPENAI_API_KEY="your_api_key"
# Optional: choose an account-available model.
export OPENAI_MODEL="gpt-5"
streamlit run app.py
```

Alternatively, enter an API key in the dashboard sidebar. The field is masked and is used only for the active Streamlit session; it is not saved to this repository or written to an environment file.

Neo4j Browser is available at [http://localhost:7474](http://localhost:7474), using `neo4j` / `password` (change this outside a local demo).

To stop the local database while retaining data:

```bash
docker compose down
```

To completely recreate demo data, use `docker compose down -v`, then repeat setup.

## Python query API

`src/graph_queries.py` implements the requested read-only functions:

- `get_shipment(shipment_id)`
- `get_shipment_product(shipment_id)`
- `get_destination_country(shipment_id)`
- `get_customs_rules(shipment_id)`
- `get_historical_similar_cases(shipment_id)`
- `get_shipment_events(shipment_id)`
- `get_graph_context(shipment_id)`

The final method combines the graph evidence into a single JSON-ready response. It does not make a delay prediction or present its synthetic rules as real guidance. Phase 2 calls this method without changing it.

## Example Cypher

This is the core traversal used to discover prior similar cases for a shipment:

```cypher
MATCH (:Shipment {shipment_id: $shipment_id})-[:CONTAINS]->(product:Product)
MATCH (:Shipment {shipment_id: $shipment_id})-[:SHIPPED_TO]->(destination:Country)
MATCH (case:CustomsCase)-[:INVOLVED_PRODUCT]->(product)
MATCH (case)-[:OCCURRED_IN]->(destination)
OPTIONAL MATCH (case)-[:RESOLVED_BY]->(resolution:Resolution)
RETURN case, resolution
ORDER BY case.case_id;
```

Run the schema constraints manually if needed with `cypher-shell -a bolt://localhost:7687 -u neo4j -p password < graph/constraints.cypher`. The Python loader runs them automatically.

# How the Demo Works — End-to-End Process

The demo answers the synthetic business question: **“Why might this shipment be at risk of customs delay?”** It combines a Knowledge Graph, semantic document retrieval, and a small explainable machine-learning model. Every shipment, rule, document, case, and score in this process is synthetic.

1. **Start the local services and load synthetic data.** Docker Compose starts Neo4j. `graph/load_graph.py` reads the CSV files and creates shipments, customers, products, HS codes, countries, synthetic customs rules, historical cases, resolutions, documents, and tracking events.

2. **Select a shipment in the Streamlit dashboard.** The default interview example is `DHL-SG-10001`: a Semiconductor Component with HS code `8541`, travelling from Singapore to Germany, currently in `CUSTOMS_CLEARANCE`.

3. **Retrieve connected graph evidence.** The dashboard calls the existing `get_graph_context()` function. Neo4j traverses from the shipment to its product, HS code, destination country, applicable synthetic customs rules, required documents, shipment events, and matching historical customs cases.

4. **Build a focused vector-search query.** The product, HS code, destination, connected requirements, and historical issues are converted into a text query. This means vector search is guided by verified graph context rather than a generic natural-language prompt.

5. **Retrieve semantically related synthetic documents.** ChromaDB searches the local document collection using sentence-transformer embeddings. For the semiconductor/Germany example, this surfaces fictional technical-specification, HS-classification, and historical-resolution documents that are semantically relevant to the graph evidence.

6. **Calculate a synthetic delay-risk score.** The logistic-regression demonstration model uses only engineered synthetic features: declared value, customs-clearance status, number and severity of applicable rules, similar-case count, and average clearance days for similar cases. It produces a synthetic probability, risk band, and inspectable feature contributions.

7. **Combine evidence with source provenance.** The dashboard keeps Graph and Vector evidence separate and labeled. Graph cards show exact connected rules and cases; Vector cards show retrieved documents and their semantic distance. This makes the result explainable and avoids presenting a document match as a confirmed rule.

8. **Generate the dashboard explanation.** By default, the explanation is deterministic: it summarizes only the shipment facts, model drivers, connected graph evidence, and retrieved synthetic documents already in the result. If an `OPENAI_API_KEY` is explicitly configured, the optional LLM feature can produce a constrained narrative summary of that same synthetic payload; the graph, vector, and model evidence remain the source of truth.

9. **Present an actionable interview narrative.** For `DHL-SG-10001`, the dashboard demonstrates how an operations user could identify synthetic evidence to review—such as a technical specification or HS-classification evidence—before a potential exception becomes a delay. It does not make a real customs determination, predict a real DHL outcome, or replace operational review.

## Interactive Knowledge Graph Explorer

`Knowledge Graph Explorer` now appears directly in the main Streamlit dashboard page (under the triage section). It is a shipment-centred visual traversal experience backed by live Neo4j queries. Its initial view displays only the selected shipment’s direct neighbours. Select any node to inspect its properties, then choose **Expand selected node** to fetch its neighbours dynamically. Relationship labels, breadcrumbs, reset controls, and the graph-based **Explain shipment** action make the path from a shipment to customs requirements and historical cases visible.

Run the app as usual:

```bash
streamlit run app.py
```
