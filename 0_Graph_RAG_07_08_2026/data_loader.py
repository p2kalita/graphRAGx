"""
Loads CORD-19 documents.

Works against the real CORD-19 metadata.csv (https://github.com/allenai/cord19,
distributed via Semantic Scholar / Kaggle) or the bundled sample with the same
columns: cord_uid, title, abstract, authors, journal, publish_time.

To use the full dataset:
    1. Download metadata.csv from the CORD-19 release.
    2. loader = CordDataLoader("/path/to/metadata.csv")
    3. docs = loader.load(max_docs=500)   # extraction is LLM-bound, start small
"""
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

import config


@dataclass
class Document:
    doc_id: str
    title: str
    abstract: str
    authors: str
    journal: str
    publish_time: str

    @property
    def text(self) -> str:
        """Text passed to the entity/relationship extraction prompt."""
        parts = [f"Title: {self.title}"]
        if self.abstract and isinstance(self.abstract, str):
            parts.append(f"Abstract: {self.abstract}")
        return "\n".join(parts)


class CordDataLoader:
    def __init__(self, path: Optional[str] = None):
        self.path = path or config.DEFAULT_METADATA_PATH

    def load(self, max_docs: int = 0) -> List[Document]:
        df = pd.read_csv(self.path, dtype=str).fillna("")
        required = {"cord_uid", "title", "abstract"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"metadata.csv is missing required columns: {missing}")

        # Drop rows with no usable text.
        df = df[(df["title"].str.strip() != "") | (df["abstract"].str.strip() != "")]

        if max_docs and max_docs > 0:
            df = df.head(max_docs)

        docs = []
        for _, row in df.iterrows():
            docs.append(
                Document(
                    doc_id=row.get("cord_uid", "") or row.name,
                    title=row.get("title", ""),
                    abstract=row.get("abstract", ""),
                    authors=row.get("authors", ""),
                    journal=row.get("journal", ""),
                    publish_time=row.get("publish_time", ""),
                )
            )
        return docs


if __name__ == "__main__":
    loader = CordDataLoader()
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {loader.path}")
    print(docs[0])
