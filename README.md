# GraphRAG with Neo4j, LangChain & Graphviz

An end-to-end Knowledge Graph Retrieval-Augmented Generation (GraphRAG) pipeline built with Neo4j, LangChain, and Graphviz visual graph rendering.

---

## 🌟 Key Features

1. **Modern LangChain & Neo4j Integration**: Built with `langchain-neo4j` and `neo4j` official drivers.
2. **Dynamic Knowledge Graph Construction**: Converts raw unstructured text into structured entity-relationship graph documents via `LLMGraphTransformer`.
3. **Graph Visualizations with Graphviz**:
   - **Text KG Visualization (Section 5.1)**: Interactive diagram of extracted entities and relationships styled by entity type (`Person`, `Organization`, `Company`, `Location`, etc.).
   - **Movie Dataset Subgraph (Section 6.1)**: Interactive diagram querying movies, cast, directors, and genres directly from Neo4j.
4. **Natural Language Cypher QA**: Uses `GraphCypherQAChain` to convert natural language queries into Cypher statements executed safely against Neo4j.
5. **Few-Shot Prompting**: Enhances Cypher generation accuracy with few-shot query examples.

---

## 🚀 Quickstart

### 1. Environment Setup

Create or verify your `.env` file in the project root:

```env
GROQ_API_KEY="your-groq-api-key"
GEMINI_API_KEY="your-gemini-api-key"

NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="your-username"
NEO4J_PASSWORD="your-password"
NEO4J_DATABASE="your-database-name"
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Notebook

Open [test.ipynb](file:///d:/graphRAG/test.ipynb) in VS Code or Jupyter and execute the cells sequentially.
