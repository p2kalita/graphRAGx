# CORD-19 Pure GraphRAG (LangChain + LangGraph + Gemini)

A **pure** GraphRAG system: retrieval happens entirely by traversing a knowledge
graph (fuzzy entity linking + k-hop expansion + community lookup) — there is
**no vector store and no embedding similarity search anywhere in the retrieval
path**. This mirrors the design of Microsoft's GraphRAG (indexing → entity/
relationship extraction → community detection/summarization → local & global
search), applied to the CORD-19 (COVID-19 Open Research Dataset) corpus.

## Architecture

```
INDEXING (build_graph.py)
  metadata.csv (CORD-19)
    -> data_loader.py            parse docs (title + abstract)
    -> entity_extraction.py      Gemini extracts entities + relationships (JSON)
    -> graph_store.py            NetworkX MultiDiGraph knowledge graph
    -> community_summarizer.py   Louvain clustering + Gemini community summaries
    -> graph_store/*.gpickle/json + communities.json persisted to disk

QUERYING (graphrag_pipeline.py, LangGraph StateGraph)
  question
    -> analyze_query      Gemini extracts key terms, picks local vs global
    -> local_retrieve      fuzzy-match terms to graph nodes -> k-hop subgraph -> triples
       OR global_retrieve  keyword-match against community summaries
    -> generate_answer     Gemini answers from ONLY the retrieved graph context, with citations
```

- **Local search**: for specific-entity questions ("What does remdesivir target?").
  Matches query terms to graph node names, expands `k` hops, and feeds the
  resulting relationship triples to the LLM.
- **Global search**: for broad/thematic questions ("What treatment strategies
  are studied in this corpus?"). Feeds relevant community summaries instead of
  raw triples — the graph-native analogue of "map-reduce over the whole
  corpus" without embeddings.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your Gemini API key from https://aistudio.google.com/apikey
```

## Usage

```bash
# 1. Build the graph (defaults to the bundled 20-doc sample dataset)
python build_graph.py

# 2. Ask questions
python main.py --question "How does SARS-CoV-2 enter human cells?"
python main.py --question "What treatment strategies have been studied for COVID-19?" --mode global

# or interactively:
python main.py
> What does dexamethasone do?
> global: What are the main research themes in this dataset?
```

## Using the real CORD-19 dataset

The sample data in `sample_data/cord19_sample.csv` mirrors the real
`metadata.csv` schema (`cord_uid, title, abstract, authors, journal,
publish_time`) but is a small, hand-written stand-in — this sandbox has no
network access to AI2/Semantic Scholar/Kaggle, where the actual CORD-19
release is hosted. To use the real dataset:

1. Download `metadata.csv` from the CORD-19 release
   (https://github.com/allenai/cord19 has pointers to the current hosting location).
2. `python build_graph.py --data /path/to/metadata.csv --max-docs 500`
   (start with `--max-docs` since extraction makes one Gemini call per document —
   the full metadata file has 100k+ rows).
3. Everything downstream (graph store, LangGraph pipeline, CLI) is unchanged.

## Extending

- **Bigger graphs / concurrent extraction**: batch `GraphExtractor.extract` calls
  with `asyncio` + `ainvoke` for large corpora.
- **Swap the graph backend**: `graph_store.py` is the only module that touches
  NetworkX. Reimplement it against Neo4j (e.g. with `langchain-neo4j`) to scale
  past in-memory limits or use Cypher for retrieval instead of `k_hop_subgraph`.
- **Add a full-text / vector layer**: this project intentionally stays "pure"
  (graph-only). For hybrid GraphRAG, add a vector index over document chunks
  and merge its hits with `local_retrieve`'s triples before `generate_answer`.
- **Tune extraction quality**: `entity_extraction.py`'s `SYSTEM_PROMPT` and
  `config.ENTITY_TYPES` control what gets pulled into the graph — adjust for
  your subdomain (e.g. add `CLINICAL_TRIAL`, `DOSAGE` types).

## Files

| File | Purpose |
|---|---|
| `config.py` | Central settings (models, paths, thresholds) |
| `data_loader.py` | CORD-19 `metadata.csv` reader |
| `entity_extraction.py` | Gemini entity/relationship extraction per document |
| `graph_store.py` | NetworkX knowledge graph + fuzzy search + k-hop traversal |
| `community_summarizer.py` | Louvain communities + Gemini summaries |
| `build_graph.py` | Indexing CLI |
| `graphrag_pipeline.py` | LangGraph query-time pipeline (local/global search) |
| `main.py` | Query CLI |
| `sample_data/cord19_sample.csv` | 20-doc sample in the real CORD-19 schema |
