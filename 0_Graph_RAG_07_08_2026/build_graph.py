"""
Indexing pipeline: CORD-19 metadata.csv -> Gemini extraction -> knowledge graph
-> Louvain communities -> Gemini community summaries -> persisted to disk.

Usage:
    python build_graph.py --data sample_data/cord19_sample.csv --max-docs 20
"""
import argparse

from tqdm import tqdm

import config
from data_loader import CordDataLoader
from entity_extraction import GraphExtractor
from graph_store import KnowledgeGraph
from community_summarizer import detect_communities, summarize_communities, save_communities


def build(data_path: str = None, max_docs: int = 0, skip_communities: bool = False):
    config.require_api_key()

    loader = CordDataLoader(data_path)
    docs = loader.load(max_docs=max_docs)
    print(f"Loaded {len(docs)} documents from {loader.path}")

    extractor = GraphExtractor()
    kg = KnowledgeGraph()

    for doc in tqdm(docs, desc="Extracting entities/relationships"):
        result = extractor.extract(doc)
        for e in result.entities:
            kg.add_entity(e.name, e.type, e.description, doc.doc_id)
        for r in result.relationships:
            kg.add_relationship(r.source, r.target, r.relation, r.description, doc.doc_id)

    print("Graph stats:", kg.stats())
    kg.save()
    print(f"Saved graph to {config.GRAPH_PICKLE_PATH} and {config.GRAPH_JSON_PATH}")

    if not skip_communities and kg.graph.number_of_nodes() > 0:
        print("Detecting communities...")
        partition = detect_communities(kg)
        summaries = summarize_communities(kg, partition)
        save_communities(summaries)
        kg.save()  # re-save with community ids attached to nodes
        print(f"Saved {len(summaries)} community summaries to {config.COMMUNITIES_PATH}")

    return kg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a GraphRAG knowledge graph from CORD-19.")
    parser.add_argument("--data", type=str, default=None,
                         help="Path to a CORD-19-style metadata.csv (defaults to bundled sample).")
    parser.add_argument("--max-docs", type=int, default=0,
                         help="Limit number of documents processed (0 = all).")
    parser.add_argument("--skip-communities", action="store_true",
                         help="Skip Louvain community detection + summarization.")
    args = parser.parse_args()

    build(data_path=args.data, max_docs=args.max_docs, skip_communities=args.skip_communities)
