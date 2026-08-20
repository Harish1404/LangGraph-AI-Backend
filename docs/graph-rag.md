# Graph RAG

Graph RAG builds a knowledge graph of entities and relationships from the source documents, instead of (or alongside) a flat vector store. This lets retrieval follow relationships between facts — useful for multi-hop questions that plain chunk similarity search misses.

## 1. Ingestion Workflow

```mermaid
sequenceDiagram
    participant Docs as Raw Documents
    participant EX as Entity/Relation Extractor
    participant KG as Knowledge Graph Store
    participant VS as Vector Store

    Note over Docs,VS: Graph Construction Phase
    Docs->>EX: Load & Chunk Text
    EX->>EX: Extract Entities & Relationships (LLM/NLP)
    EX->>KG: Store Nodes (Entities) & Edges (Relations)
    EX->>VS: Store Chunk Embeddings (linked to graph nodes)
```

## 2. Retrieval & Generation Workflow

```mermaid
sequenceDiagram
    actor User as User
    participant Chat as Chat Service
    participant NER as Entity Recognizer
    participant KG as Knowledge Graph Store
    participant VS as Vector Store
    participant LLM as Language Model

    Note over User,LLM: Graph-Aware Query Phase
    User->>Chat: Ask Question
    Chat->>NER: Extract Entities from Query
    NER->>KG: Locate Matching Nodes
    KG->>KG: Traverse Subgraph / Community
    KG-->>Chat: Related Entities & Relationships
    Chat->>VS: Fetch Supporting Chunk Text
    VS-->>Chat: Top-K Chunks for Graph Context
    Chat->>LLM: Send Prompt + Graph Context + Chunks + Question
    LLM-->>Chat: Generate Grounded Answer
    Chat-->>User: Final Output
```

### Component Details
- **Entity/Relation Extractor**: Uses an LLM or NLP pipeline during ingestion to pull out entities (people, concepts, products) and the relationships between them.
- **Knowledge Graph Store**: Persists entities as nodes and relationships as edges, enabling traversal instead of only similarity lookup.
- **Entity Recognizer**: Identifies which entities the user's query is referring to, to anchor graph traversal.
- **Subgraph / Community Traversal**: Walks the graph outward from matched entities to gather related facts, then combines that with source chunk text for grounding.
