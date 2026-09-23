"""Streamlit dashboard for synthetic DHL customs-delay triage."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from src.customs_delay_analysis import CustomsDelayAnalyzer
from src.explanations import deterministic_explanation, optional_openai_explanation
from src.graph_visualization import Neo4jGraphExplorer, graph_context_explanation
from src.graph_queries import DHLGraphQueries
from src.vector_store import SyntheticCustomsVectorStore

ROOT = Path(__file__).resolve().parent
NODE_COLORS = {
    "Shipment": "#D40511", "Customer": "#334155", "Product": "#2563EB", "HSCode": "#7C3AED",
    "Country": "#0F766E", "CustomsRule": "#B45309", "Document": "#475569", "CustomsCase": "#BE123C",
    "Resolution": "#15803D", "ShipmentEvent": "#0369A1",
}

st.set_page_config(page_title="Synthetic Customs Risk Triage", page_icon="✈️", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    [data-testid="stMetricValue"] {color: #d40511;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing local graph, vector, and synthetic-model services…")
def get_analyzer():
    graph = DHLGraphQueries()
    vector_store = SyntheticCustomsVectorStore()
    vector_store.index_documents()
    return CustomsDelayAnalyzer(graph, vector_store=vector_store)


def shipment_ids() -> list[str]:
    return pd.read_csv(ROOT / "data" / "shipments.csv")["shipment_id"].tolist()


@st.cache_resource(show_spinner="Connecting to the synthetic Neo4j graph…")
def graph_explorer():
    return Neo4jGraphExplorer(DHLGraphQueries())


def rebuild_graph_from_expansions():
    nodes = dict(st.session_state.kg_base_nodes)
    edges = dict(st.session_state.kg_base_edges)
    for node_id in st.session_state.kg_expanded_nodes:
        expansion = st.session_state.kg_expansion_graphs.get(node_id)
        if expansion:
            nodes.update(expansion["nodes"])
            edges.update(expansion["edges"])
    st.session_state.kg_graph_nodes = nodes
    st.session_state.kg_graph_edges = edges
    if st.session_state.get("kg_selected_node") not in st.session_state.kg_graph_nodes:
        st.session_state.kg_selected_node = next(
            (node_id for node_id, node in nodes.items() if node["type"] == "Shipment"),
            None,
        )


def expand_selected_node(selected_node: str):
    st.session_state.kg_expansion_graphs[selected_node] = graph_explorer().expand_node(selected_node)
    st.session_state.kg_expanded_nodes.add(selected_node)
    rebuild_graph_from_expansions()


def collapse_selected_node(selected_node: str):
    st.session_state.kg_expanded_nodes.discard(selected_node)
    rebuild_graph_from_expansions()


def load_shipment_graph(shipment_id: str):
    graph = graph_explorer().shipment_neighborhood(shipment_id)
    st.session_state.kg_base_nodes = graph["nodes"]
    st.session_state.kg_base_edges = graph["edges"]
    st.session_state.kg_expansion_graphs = {}
    st.session_state.kg_expanded_nodes = set()
    st.session_state.kg_graph_nodes = dict(graph["nodes"])
    st.session_state.kg_graph_edges = dict(graph["edges"])
    st.session_state.kg_selected_node = next(
        (node_id for node_id, node in graph["nodes"].items() if node["type"] == "Shipment"),
        None,
    )
    st.session_state.kg_traversal = [st.session_state.kg_selected_node] if st.session_state.kg_selected_node else []
    st.session_state.kg_loaded_shipment = shipment_id


def display_label(node: dict) -> str:
    return f"{node['type']} · {node['display']}"


def graph_nodes_and_edges():
    nodes = [
        Node(
            id=node_id,
            label=node["display"],
            title=display_label(node),
            shape="box",
            size=24,
            color=NODE_COLORS.get(node["type"], "#64748B"),
            font={"color": "#FFFFFF", "size": 14},
        )
        for node_id, node in st.session_state.kg_graph_nodes.items()
    ]
    edges = [
        Edge(
            source=edge["source"],
            target=edge["target"],
            label=edge["label"],
            arrows="to",
            color="#94A3B8",
            font={"color": "#334155", "size": 11, "align": "middle"},
        )
        for edge in st.session_state.kg_graph_edges.values()
        if edge["source"] in st.session_state.kg_graph_nodes and edge["target"] in st.session_state.kg_graph_nodes
    ]
    return nodes, edges


def render_knowledge_graph_explorer(selected_shipment: str):
    st.header("Shipment Knowledge Graph Explorer")
    st.caption("Interactive Neo4j traversal · synthetic DHL Express prototype · every node and relationship is queried live")

    with st.sidebar:
        st.divider()
        st.header("Graph controls")
        reset_clicked = st.button("Reset graph", use_container_width=True, key="kg_reset")
        st.caption("Uses the shipment selected above. Initial view: direct shipment neighbours only. Select a node, then use Expand to retrieve its live Neo4j neighbours.")

    try:
        if reset_clicked:
            load_shipment_graph(st.session_state.get("kg_loaded_shipment", selected_shipment))
        if st.session_state.get("kg_loaded_shipment") != selected_shipment or "kg_graph_nodes" not in st.session_state:
            load_shipment_graph(selected_shipment)
    except Exception as error:
        st.error(f"Knowledge Graph Explorer unavailable: {error}")
        st.info("Confirm Neo4j is healthy with `docker compose ps`, then load the graph with `python graph/load_graph.py`.")
        return

    selected_node = st.session_state.get("kg_selected_node")
    path_nodes = [
        st.session_state.kg_graph_nodes[node_id]
        for node_id in st.session_state.get("kg_traversal", [])
        if node_id in st.session_state.kg_graph_nodes
    ]
    st.markdown("**Traversal path**")
    breadcrumb_columns = st.columns(max(1, len(path_nodes)))
    for column, node in zip(breadcrumb_columns, path_nodes):
        if column.button(display_label(node), key=f"kg-crumb-{node['id']}"):
            index = st.session_state.kg_traversal.index(node["id"])
            st.session_state.kg_traversal = st.session_state.kg_traversal[:index + 1]
            st.session_state.kg_selected_node = node["id"]
            st.rerun()

    canvas, details = st.columns([2.35, 1])
    with canvas:
        st.subheader("Live graph canvas")
        nodes, edges = graph_nodes_and_edges()
        config = Config(
            width=930,
            height=650,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#F2A900",
            collapsible=False,
        )
        clicked_node = agraph(nodes=nodes, edges=edges, config=config)
        if clicked_node and clicked_node in st.session_state.kg_graph_nodes and clicked_node != selected_node:
            st.session_state.kg_selected_node = clicked_node
            if clicked_node not in st.session_state.kg_traversal:
                st.session_state.kg_traversal.append(clicked_node)
            st.rerun()

    with details:
        st.subheader("Node details")
        if not selected_node:
            st.info("Select a graph node to inspect it.")
        else:
            detail = graph_explorer().node_details(selected_node)
            if detail:
                st.markdown(f"**{detail['type']}**")
                st.caption(detail["display"])
                st.json(detail["properties"])
                node_expanded = selected_node in st.session_state.kg_expanded_nodes
                if st.button("Expand selected node", type="primary", use_container_width=True, key="kg_expand", disabled=node_expanded):
                    expand_selected_node(selected_node)
                    st.rerun()
                if st.button("Collapse selected node", use_container_width=True, key="kg_collapse", disabled=not node_expanded):
                    collapse_selected_node(selected_node)
                    st.rerun()

    st.divider()
    left, right = st.columns([1, 2])
    with left:
        if st.button("Explain shipment", use_container_width=True, key="kg_explain"):
            graph_client = DHLGraphQueries()
            context = graph_client.get_graph_context(st.session_state.kg_loaded_shipment)
            graph_client.close()
            st.session_state.kg_shipment_explanation = (
                graph_context_explanation(context) if context else "Shipment was not found."
            )
    with right:
        st.markdown("**Why this view matters**")
        st.write("Start with one shipment, expose only direct operational context, then expand product, country, rule, document, event, and historical-case nodes as questions arise. Every displayed relationship label comes from Neo4j.")

    if st.session_state.get("kg_shipment_explanation"):
        st.info(st.session_state.kg_shipment_explanation)


def show_evidence(analysis: dict):
    evidence = analysis["hybrid_evidence"]
    graph_rows = [item for item in evidence if item["source"] == "Graph"]
    vector_rows = [item for item in evidence if item["source"] == "Vector"]
    left, right = st.columns(2)
    with left:
        st.subheader("Graph evidence")
        for item in graph_rows:
            st.markdown(f"**{item['evidence_type']} — {item['id']}**")
            st.write(item["title"])
            st.caption(item["detail"])
    with right:
        st.subheader("Vector evidence")
        for item in vector_rows:
            st.markdown(f"**{item['document_type']} — {item['id']}**")
            st.write(item["title"])
            st.caption(f"Semantic distance: {item['distance']}")


st.title("Customs-delay triage prototype")
st.caption("Graph + vector retrieval + synthetic ML · No real DHL or customs data")

with st.sidebar:
    st.header("Demo controls")
    shipment_id = st.selectbox("Shipment", shipment_ids(), index=0)
    top_k = st.slider("Vector documents", min_value=1, max_value=5, value=3)
    api_key = st.text_input(
        "OpenAI API key (optional)", type="password",
        help="Used only for this Streamlit session to generate the optional LLM explanation. It is not written to a file.",
    )
    llm_available = bool(api_key or os.getenv("OPENAI_API_KEY"))
    use_llm = st.toggle("Optional OpenAI explanation", value=False, disabled=not llm_available)
    if not llm_available:
        st.caption("Enter a key above or set `OPENAI_API_KEY` to enable the optional LLM summary.")
    analyze = st.button("Analyze shipment", type="primary", use_container_width=True)

if analyze:
    try:
        analysis = get_analyzer().analyze(shipment_id, top_k=top_k)
        if analysis is None:
            st.error("Shipment was not found in the synthetic Neo4j graph.")
            st.stop()
        st.session_state.last_analysis = analysis
        st.session_state.last_analysis_shipment_id = shipment_id
        st.session_state.last_analysis_top_k = top_k
        st.session_state.last_analysis_use_llm = use_llm
        st.session_state.last_analysis_api_key = api_key
    except Exception as error:
        st.error(f"The local prototype could not run: {error}")
        st.info("Confirm Neo4j is healthy with `docker compose ps`, then load the graph with `python graph/load_graph.py`.")
        st.stop()

analysis = st.session_state.get("last_analysis")
if analysis:
    if st.session_state.get("last_analysis_shipment_id") != shipment_id:
        st.caption(f"Showing latest analysis for {st.session_state.get('last_analysis_shipment_id')}.")

    risk = analysis["synthetic_risk_assessment"]
    graph = analysis["graph_context"]
    st.subheader("Decision snapshot")
    a, b, c, d = st.columns(4)
    a.metric("Risk band", risk["risk_band"])
    b.metric("Delay Risk Probability", f"{risk['synthetic_delay_risk_probability']:.1%}")
    c.metric("Applicable Custom Rules", len(graph["applicable_customs_rules"]))
    d.metric("Similar Cases", len(graph["historical_similar_cases"]))
    st.progress(risk["synthetic_delay_risk_probability"], text="Delay-risk score — demonstration only")

    explanation = deterministic_explanation(analysis)
    st.subheader(explanation["headline"])
    st.write(explanation["summary"])
    x, y = st.columns(2)
    with x:
        st.markdown("**Model drivers**")
        for driver in explanation["risk_drivers"]:
            st.write(f"• {driver}")
    with y:
        st.markdown("**Triage steps**")
        for next_step in explanation["synthetic_next_steps"]:
            st.write(f"• {next_step}")

    run_llm = st.session_state.get("last_analysis_use_llm", False)
    if run_llm:
        with st.spinner("Creating optional LLM summary from synthetic evidence…"):
            try:
                st.subheader("Optional LLM explanation")
                llm_api_key = st.session_state.get("last_analysis_api_key")
                st.write(optional_openai_explanation(analysis, api_key=llm_api_key or None))
            except Exception as error:
                st.warning(f"LLM explanation unavailable: {error}")

    show_evidence(analysis)
    with st.expander("Inspect graph context and model inputs"):
        st.json({"graph_context": graph, "model": risk, "vector_search_query": analysis["vector_search_query"]})
else:
    st.info("Choose a synthetic shipment and click **Analyze shipment**. The default path requires only local Neo4j, ChromaDB, and the synthetic model.")

st.divider()
render_knowledge_graph_explorer(shipment_id)
st.divider()
st.caption("All requirements, documents, cases, embeddings, retrieved evidence, and model outputs in this prototype are synthetic. They do not represent DHL policy, customs regulation, or production risk decisions.")
