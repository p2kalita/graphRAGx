"""
Central configuration for the CORD-19 GraphRAG pipeline.

Loads settings from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
EXTRACTION_MODEL = os.environ.get("GRAPHRAG_EXTRACTION_MODEL", "gemini-2.0-flash")
ANSWER_MODEL = os.environ.get("GRAPHRAG_ANSWER_MODEL", "gemini-2.0-flash")
SUMMARY_MODEL = os.environ.get("GRAPHRAG_SUMMARY_MODEL", "gemini-2.0-flash")

# --- Data ---
DEFAULT_METADATA_PATH = os.path.join(
    os.path.dirname(__file__), "sample_data", "cord19_sample.csv"
)
MAX_DOCS = int(os.environ.get("GRAPHRAG_MAX_DOCS", "0"))  # 0 = no limit

# --- Graph store ---
GRAPH_PICKLE_PATH = os.path.join(os.path.dirname(__file__), "graph_store", "graph.gpickle")
GRAPH_JSON_PATH = os.path.join(os.path.dirname(__file__), "graph_store", "graph.json")
COMMUNITIES_PATH = os.path.join(os.path.dirname(__file__), "graph_store", "communities.json")

# --- Retrieval ---
ENTITY_MATCH_THRESHOLD = 78  # rapidfuzz score 0-100, used for linking query terms to graph nodes
LOCAL_SEARCH_HOPS = 2        # k-hop neighborhood expansion for local search
MAX_CONTEXT_TRIPLES = 60     # cap on relationship rows fed to the answer LLM
MAX_CONTEXT_COMMUNITIES = 5  # cap on community summaries fed to the answer LLM (global search)

# Entity types the extraction prompt is steered towards (biomedical CORD-19 domain).
ENTITY_TYPES = [
    "DISEASE", "PATHOGEN", "GENE_PROTEIN", "DRUG_TREATMENT",
    "VACCINE", "SYMPTOM", "ORGANIZATION", "LOCATION",
    "METHOD_TECHNIQUE", "POPULATION_GROUP",
]

def require_api_key():
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "Gemini API key from https://aistudio.google.com/apikey"
        )
