"""
Query-time GraphRAG pipeline, orchestrated with LangGraph.

Flow:
    START
      -> analyze_query        (Gemini pulls out key entities/terms + picks
                                local vs global search, no embeddings involved)
      -> local_retrieve / global_retrieve   (pure graph traversal + fuzzy match)
      -> generate_answer      (Gemini answers using only the retrieved subgraph
                                / community summaries as context)
    END

"Pure" GraphRAG here means: zero vector index, zero embedding similarity search.
All retrieval is graph-native (node lookup + k-hop traversal + community lookup).
"""
import json
from typing import Dict, List, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

import config
from graph_store import KnowledgeGraph
from community_summarizer import load_communities


class GraphRAGState(TypedDict, total=False):
    question: str
    mode: str                      # "auto" | "local" | "global"
    search_type: str                # resolved: "local" | "global"
    query_terms: List[str]
    matched_nodes: List[str]
    triples: List[Dict]
    community_context: List[Dict]
    answer: str
    sources: List[str]


QUERY_ANALYSIS_PROMPT = """You analyze a user question about COVID-19 research so \
it can be answered from a knowledge graph.

Return ONLY valid JSON:
{{
  "search_type": "local" or "global",
  "key_terms": ["term1", "term2", ...]
}}

Guidance:
- "local": the question is about specific, named entities (a drug, gene, virus, \
symptom, organization, etc.) and their direct relationships. e.g. "What does \
remdesivir target?", "How does ACE2 relate to SARS-CoV-2?"
- "global": the question is broad/thematic, asking to summarize themes across the \
corpus. e.g. "What treatment strategies have been studied?", "What are the main \
research themes in this dataset?"
- key_terms: 1-6 short noun phrases from the question to use for entity lookup \
(skip for pure global/thematic questions if none apply).

Question: {question}
"""

ANSWER_SYSTEM_PROMPT = """You are a biomedical research assistant answering questions \
using ONLY the knowledge graph context provided (entities, relationships, and/or \
community summaries extracted from CORD-19 research abstracts). 

Rules:
- Ground every claim in the provided context; do not use outside knowledge.
- If the context is insufficient to answer, say so explicitly.
- Cite source documents inline using their doc id in square brackets, e.g. [c19-0002], \
whenever you state a fact that came from a specific relationship or entity.
- Be concise and precise.
"""


class GraphRAGPipeline:
    def __init__(self, kg: KnowledgeGraph = None, communities: List[Dict] = None):
        config.require_api_key()
        self.kg = kg or KnowledgeGraph.load()
        self.communities = communities if communities is not None else load_communities()
        self.query_llm = ChatGoogleGenerativeAI(
            model=config.EXTRACTION_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0,
        )
        self.answer_llm = ChatGoogleGenerativeAI(
            model=config.ANSWER_MODEL, google_api_key=config.GOOGLE_API_KEY, temperature=0.2,
        )
        self.app = self._build_graph()

    # ---------- LangGraph nodes ----------

    def _analyze_query(self, state: GraphRAGState) -> GraphRAGState:
        if state.get("mode") in ("local", "global"):
            search_type = state["mode"]
            key_terms = [state["question"]]
        else:
            resp = self.query_llm.invoke([
                SystemMessage(content="Respond with ONLY the requested JSON, no markdown fences."),
                HumanMessage(content=QUERY_ANALYSIS_PROMPT.format(question=state["question"])),
            ])
            raw = resp.content.strip().strip("`")
            if raw.startswith("json"):
                raw = raw[4:].strip()
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"search_type": "local", "key_terms": [state["question"]]}
            search_type = parsed.get("search_type", "local")
            key_terms = parsed.get("key_terms") or [state["question"]]

        return {"search_type": search_type, "query_terms": key_terms}

    def _local_retrieve(self, state: GraphRAGState) -> GraphRAGState:
        matched_nodes = []
        for term in state["query_terms"]:
            for name, score in self.kg.search_entities(term):
                if name not in matched_nodes:
                    matched_nodes.append(name)

        if not matched_nodes:
            return {"matched_nodes": [], "triples": []}

        subgraph = self.kg.k_hop_subgraph(matched_nodes)
        triples = self.kg.edges_as_triples(subgraph, limit=config.MAX_CONTEXT_TRIPLES)
        return {"matched_nodes": matched_nodes, "triples": triples}

    def _global_retrieve(self, state: GraphRAGState) -> GraphRAGState:
        if not self.communities:
            return {"community_context": []}
        terms = " ".join(state["query_terms"]).lower()
        scored = []
        for c in self.communities:
            haystack = (c["title"] + " " + c["summary"] + " " + " ".join(c["entities"])).lower()
            overlap = sum(1 for t in terms.split() if t in haystack)
            scored.append((overlap, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for score, c in scored if score > 0][:config.MAX_CONTEXT_COMMUNITIES]
        if not top:  # thematic question with no keyword overlap: fall back to largest communities
            top = [c for _, c in scored[:config.MAX_CONTEXT_COMMUNITIES]]
        return {"community_context": top}

    def _generate_answer(self, state: GraphRAGState) -> GraphRAGState:
        context_parts = []
        sources = set()

        for t in state.get("triples", []):
            context_parts.append(
                f"- {t['source']} -[{t['relation']}]-> {t['target']}: "
                f"{t['description']} (source: {t['source_doc']})"
            )
            if t.get("source_doc"):
                sources.add(t["source_doc"])

        for c in state.get("community_context", []):
            context_parts.append(
                f"[Community: {c['title']}] {c['summary']} "
                f"(entities: {', '.join(c['entities'][:8])})"
            )

        if not context_parts:
            context_text = "(No matching entities or communities found in the graph.)"
        else:
            context_text = "\n".join(context_parts)

        resp = self.answer_llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"KNOWLEDGE GRAPH CONTEXT:\n{context_text}\n\n"
                                  f"QUESTION: {state['question']}"),
        ])
        return {"answer": resp.content, "sources": sorted(sources)}

    def _route(self, state: GraphRAGState) -> Literal["local_retrieve", "global_retrieve"]:
        return "local_retrieve" if state["search_type"] == "local" else "global_retrieve"

    # ---------- Graph assembly ----------

    def _build_graph(self):
        g = StateGraph(GraphRAGState)
        g.add_node("analyze_query", self._analyze_query)
        g.add_node("local_retrieve", self._local_retrieve)
        g.add_node("global_retrieve", self._global_retrieve)
        g.add_node("generate_answer", self._generate_answer)

        g.add_edge(START, "analyze_query")
        g.add_conditional_edges("analyze_query", self._route,
                                 {"local_retrieve": "local_retrieve",
                                  "global_retrieve": "global_retrieve"})
        g.add_edge("local_retrieve", "generate_answer")
        g.add_edge("global_retrieve", "generate_answer")
        g.add_edge("generate_answer", END)
        return g.compile()

    # ---------- Public API ----------

    def ask(self, question: str, mode: str = "auto") -> GraphRAGState:
        return self.app.invoke({"question": question, "mode": mode})
