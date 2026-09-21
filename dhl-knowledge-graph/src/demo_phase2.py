"""Command-line demo for Phase 2."""
import json

from src.customs_delay_analysis import CustomsDelayAnalyzer
from src.graph_queries import DHLGraphQueries
from src.vector_store import SyntheticCustomsVectorStore


if __name__ == "__main__":
    graph = DHLGraphQueries()
    store = SyntheticCustomsVectorStore()
    indexed = store.index_documents()
    print(f"Indexed {indexed} synthetic customs documents.")
    print(json.dumps(CustomsDelayAnalyzer(graph, vector_store=store).analyze("DHL-SG-10001"), indent=2))
    graph.close()
