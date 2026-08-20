import logging

from langsmith import traceable

from app.core.tracing import drop_self, as_documents
from app.rag.data_processor import process_folder
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Retrieval half of RAG: turns a folder of documents into a searchable
    vector store backed by MongoDB Atlas, and turns a user question into the most relevant chunks.
    """

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    async def ingest(self, folder_path: str) -> int:
        """
        Reads every document in `folder_path`, splits into chunks,
        embeds ONLY NEW chunks, and stores them in MongoDB Atlas.

        Returns the number of new chunks ingested.
        """
        chunks = process_folder(folder_path)

        if not chunks:
            logger.warning(f"No documents found to ingest in '{folder_path}'.")
            return 0

        existing_ids = await self.vector_store.get_existing_ids()
        new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing_ids]

        if not new_chunks:
            logger.info(f"All {len(chunks)} chunk(s) from '{folder_path}' already exist in MongoDB. Skipping re-ingestion.")
            return 0

        logger.info(f"Found {len(new_chunks)} new chunk(s) to embed out of {len(chunks)} total.")

        texts = [chunk["text"] for chunk in new_chunks]
        embeddings = await self.embedding_model.embed_texts(texts)

        await self.vector_store.add(
            ids=[chunk["id"] for chunk in new_chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": chunk["source"]} for chunk in new_chunks],
        )

        logger.info(f"Successfully ingested {len(new_chunks)} new chunk(s) into MongoDB Atlas.")
        return len(new_chunks)

    @traceable(
        run_type="retriever",
        name="rag_retrieve",
        process_inputs=drop_self,
        process_outputs=as_documents,
    )
    async def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        """Returns the `top_k` chunks most relevant to the given query, using hybrid search."""
        query_embedding = await self.embedding_model.embed_query(query)
        return await self.vector_store.hybrid_search(query_text=query, query_embedding=query_embedding, top_k=top_k)


# One shared instance — loaded once, reused by every request.
rag_pipeline = RAGPipeline()

