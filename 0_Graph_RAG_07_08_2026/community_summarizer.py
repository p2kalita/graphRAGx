"""
Detects communities (clusters of densely-connected entities) in the
knowledge graph and asks Gemini to summarize each one. This mirrors the
"community report" step of Microsoft's GraphRAG and is what powers
*global* search (broad, corpus-level questions) as opposed to *local*
search (specific-entity questions), both purely graph-driven.
"""
import json
import os
from typing import Dict, List

import community as community_louvain  # python-louvain
import networkx as nx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
from graph_store import KnowledgeGraph

SYSTEM_PROMPT = """You summarize a cluster (community) of related entities and \
relationships from a biomedical knowledge graph built from COVID-19 research \
abstracts. Given a list of entities and the relationships between them, write:
1. A short title (5-8 words).
2. A 3-5 sentence summary of what this community is about and why the entities \
are connected.
Output ONLY valid JSON: {"title": "string", "summary": "string"}. No markdown fences.
"""


def detect_communities(kg: KnowledgeGraph) -> Dict[str, int]:
    """Returns {node_name: community_id} and annotates kg.graph in place."""
    undirected = nx.Graph(kg.graph.to_undirected())
    if undirected.number_of_nodes() == 0:
        return {}
    partition = community_louvain.best_partition(undirected, random_state=42)
    for node, comm_id in partition.items():
        kg.graph.nodes[node]["community"] = comm_id
    return partition


def _community_members(kg: KnowledgeGraph, partition: Dict[str, int]) -> Dict[int, List[str]]:
    members: Dict[int, List[str]] = {}
    for node, comm_id in partition.items():
        members.setdefault(comm_id, []).append(node)
    return members


def summarize_communities(kg: KnowledgeGraph, partition: Dict[str, int],
                           min_size: int = 2, model_name: str = None) -> List[Dict]:
    config.require_api_key()
    llm = ChatGoogleGenerativeAI(
        model=model_name or config.SUMMARY_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0,
    )

    members = _community_members(kg, partition)
    summaries = []

    for comm_id, nodes in members.items():
        if len(nodes) < min_size:
            continue
        sub = kg.graph.subgraph(nodes)

        entity_lines = [
            f"- {n} ({sub.nodes[n].get('type', 'ENTITY')}): {sub.nodes[n].get('description', '')}"
            for n in nodes
        ]
        rel_lines = [
            f"- {u} -[{data.get('relation')}]-> {v}: {data.get('description', '')}"
            for u, v, data in sub.edges(data=True)
        ]
        prompt = (
            "ENTITIES:\n" + "\n".join(entity_lines) +
            "\n\nRELATIONSHIPS:\n" + ("\n".join(rel_lines) if rel_lines else "(none)")
        )

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip().strip("`").lstrip("json").strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"title": f"Community {comm_id}", "summary": raw[:500]}

        summaries.append({
            "community_id": comm_id,
            "title": parsed.get("title", f"Community {comm_id}"),
            "summary": parsed.get("summary", ""),
            "entities": nodes,
            "size": len(nodes),
        })

    summaries.sort(key=lambda s: s["size"], reverse=True)
    return summaries


def save_communities(summaries: List[Dict], path: str = None):
    path = path or config.COMMUNITIES_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)


def load_communities(path: str = None) -> List[Dict]:
    path = path or config.COMMUNITIES_PATH
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)
