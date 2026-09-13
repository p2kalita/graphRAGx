"""
Pure-graph knowledge store. No embeddings, no vector index — retrieval is
done entirely via fuzzy string matching to graph nodes plus graph traversal
(k-hop neighborhoods), which is what makes this "pure" GraphRAG rather than
a vector-RAG-with-a-graph-bolted-on.
"""
import json
import os
import pickle
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rapidfuzz import process, fuzz

import config


class KnowledgeGraph:
    def __init__(self):
        # MultiDiGraph: multiple distinct relations can exist between the same pair of nodes.
        self.graph = nx.MultiDiGraph()

    # ---------- Construction ----------

    def add_entity(self, name: str, entity_type: str, description: str, source_doc: str):
        name = name.strip()
        if not name:
            return
        if self.graph.has_node(name):
            node = self.graph.nodes[name]
            node["source_docs"].add(source_doc)
            # Keep the longest/most informative description seen so far.
            if len(description) > len(node.get("description", "")):
                node["description"] = description
        else:
            self.graph.add_node(
                name,
                type=entity_type or "ENTITY",
                description=description or "",
                source_docs={source_doc},
                community=None,
            )

    def add_relationship(self, source: str, target: str, relation: str,
                          description: str, source_doc: str):
        source, target = source.strip(), target.strip()
        if not source or not target:
            return
        # Ensure endpoint nodes exist even if extraction only mentioned them in a relation.
        for n in (source, target):
            if not self.graph.has_node(n):
                self.graph.add_node(n, type="ENTITY", description="",
                                     source_docs=set(), community=None)
        self.graph.add_edge(
            source, target,
            relation=relation or "RELATED_TO",
            description=description or "",
            source_doc=source_doc,
        )

    # ---------- Retrieval primitives (no embeddings) ----------

    def search_entities(self, query_term: str, top_k: int = 5,
                         threshold: int = None) -> List[Tuple[str, int]]:
        """Fuzzy string match a term against all node names."""
        threshold = threshold if threshold is not None else config.ENTITY_MATCH_THRESHOLD
        if self.graph.number_of_nodes() == 0:
            return []
        matches = process.extract(
            query_term, list(self.graph.nodes), scorer=fuzz.WRatio, limit=top_k
        )
        return [(name, score) for name, score, _ in matches if score >= threshold]

    def k_hop_subgraph(self, seed_nodes: List[str], hops: int = None) -> nx.MultiDiGraph:
        hops = hops if hops is not None else config.LOCAL_SEARCH_HOPS
        nodes: Set[str] = set(n for n in seed_nodes if self.graph.has_node(n))
        frontier = set(nodes)
        undirected = self.graph.to_undirected(as_view=True)
        for _ in range(hops):
            next_frontier = set()
            for n in frontier:
                next_frontier |= set(undirected.neighbors(n))
            frontier = next_frontier - nodes
            nodes |= next_frontier
        return self.graph.subgraph(nodes).copy()

    def edges_as_triples(self, subgraph: nx.MultiDiGraph, limit: int = None) -> List[Dict]:
        triples = []
        for u, v, data in subgraph.edges(data=True):
            triples.append({
                "source": u,
                "relation": data.get("relation", "RELATED_TO"),
                "target": v,
                "description": data.get("description", ""),
                "source_doc": data.get("source_doc", ""),
            })
        if limit:
            triples = triples[:limit]
        return triples

    # ---------- Stats / persistence ----------

    def stats(self) -> Dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    def save(self, pickle_path: str = None, json_path: str = None):
        pickle_path = pickle_path or config.GRAPH_PICKLE_PATH
        json_path = json_path or config.GRAPH_JSON_PATH
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

        with open(pickle_path, "wb") as f:
            pickle.dump(self.graph, f)

        exportable = nx.MultiDiGraph()
        for n, data in self.graph.nodes(data=True):
            d = dict(data)
            d["source_docs"] = sorted(d.get("source_docs", []))
            exportable.add_node(n, **d)
        for u, v, data in self.graph.edges(data=True):
            exportable.add_edge(u, v, **data)
        try:
            # networkx >= 3.4 renamed the "links" key to "edges"; pass explicitly
            # to silence the FutureWarning on newer versions.
            data = nx.node_link_data(exportable, edges="edges")
        except TypeError:
            data = nx.node_link_data(exportable)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, pickle_path: str = None) -> "KnowledgeGraph":
        pickle_path = pickle_path or config.GRAPH_PICKLE_PATH
        kg = cls()
        with open(pickle_path, "rb") as f:
            kg.graph = pickle.load(f)
        return kg
