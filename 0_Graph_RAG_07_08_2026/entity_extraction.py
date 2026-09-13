"""
Uses Gemini (via langchain-google-genai) to extract a knowledge graph
(entities + relationships) from each CORD-19 document. This is the
"indexing" half of GraphRAG, analogous to Microsoft's GraphRAG extraction
step, but scoped to the biomedical CORD-19 domain.

Output per document is strict JSON:
{
  "entities": [{"name": str, "type": str, "description": str}, ...],
  "relationships": [
      {"source": str, "target": str, "relation": str, "description": str}, ...
  ]
}
"""
import json
import re
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from data_loader import Document

ENTITY_TYPES_STR = ", ".join(config.ENTITY_TYPES)

SYSTEM_PROMPT = f"""You are a biomedical information extraction system building a \
knowledge graph from COVID-19 research abstracts (the CORD-19 corpus).

Given a document, extract:
1. ENTITIES: important named concepts. Prefer these types when applicable: \
{ENTITY_TYPES_STR}. You may use another concise UPPER_SNAKE_CASE type if none fit.
2. RELATIONSHIPS: directed relationships between two extracted entities, each \
with a short relation label (e.g. "BINDS_TO", "TREATS", "CAUSES", "INHIBITS", \
"STUDIED_IN", "ASSOCIATED_WITH") and a one-sentence description grounded in the text.

Rules:
- Only extract entities/relationships that are explicitly supported by the text.
- Normalize entity names (e.g. "SARS-CoV-2", not "the virus" or "it").
- Keep entity names consistent so the same real-world thing always gets the same name.
- Return 3-12 entities and 2-15 relationships per document, depending on content.
- Output ONLY valid JSON matching the schema below. No markdown fences, no commentary.

Schema:
{{
  "entities": [{{"name": "string", "type": "string", "description": "string"}}],
  "relationships": [{{"source": "string", "target": "string", "relation": "string", "description": "string"}}]
}}
"""


class ExtractedEntity(BaseModel):
    name: str
    type: str
    description: str = ""


class ExtractedRelationship(BaseModel):
    source: str
    target: str
    relation: str
    description: str = ""


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


class GraphExtractor:
    def __init__(self, model_name: str = None):
        config.require_api_key()
        self.llm = ChatGoogleGenerativeAI(
            model=model_name or config.EXTRACTION_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0,
        )

    def extract(self, document: Document) -> ExtractionResult:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Document ID: {document.doc_id}\n\n{document.text}"),
        ]
        response = self.llm.invoke(messages)
        raw = _strip_code_fences(response.content)
        try:
            data = json.loads(raw)
            return ExtractionResult(**data)
        except (json.JSONDecodeError, TypeError) as e:
            # Fail soft: skip a document rather than crash the whole indexing run.
            print(f"[extract] Failed to parse Gemini output for {document.doc_id}: {e}")
            return ExtractionResult()
