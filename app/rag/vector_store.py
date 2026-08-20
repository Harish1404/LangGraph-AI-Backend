import asyncio
import logging
from pymongo import UpdateOne
from langsmith import traceable

from app.core.tracing import drop_self, drop_self_and_embedding, as_documents
from app.db.mongodb import get_vector_collection

logger = logging.getLogger("uvicorn")


class VectorStore:
    """
    Stores text chunks alongside their embeddings in MongoDB Atlas,
    and performs hybrid search (vector search + keyword search) using
    Atlas Vector Search ($vectorSearch) and Atlas Search ($search).
    """

    INDEX_NAME = "vector_index"
    KEYWORD_INDEX_NAME = "keyword_index"
    EMBEDDING_DIMENSIONS = 768

    def _get_collection(self):
        return get_vector_collection()

    async def get_existing_ids(self) -> set[str]:
        """Returns all document chunk IDs already present in MongoDB Atlas."""
        try:
            collection = self._get_collection()
            cursor = collection.find({}, {"_id": 1, "id": 1})
            docs = await cursor.to_list(length=50000)
            existing = set()
            for doc in docs:
                if "_id" in doc:
                    existing.add(str(doc["_id"]))
                if "id" in doc:
                    existing.add(str(doc["id"]))
            return existing
        except Exception as e:
            logger.error(f"Failed to fetch existing document IDs from MongoDB: {e}")
            return set()

    async def ensure_vector_index(self):
        """Automatically checks and creates the Atlas Vector Search index if missing."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()

            existing_indexes = await cursor.to_list(length=100)
            idx_names = [idx.get("name") for idx in existing_indexes]

            if self.INDEX_NAME not in idx_names:

                from pymongo.operations import SearchIndexModel

                index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": self.EMBEDDING_DIMENSIONS,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    name=self.INDEX_NAME,
                    type="vectorSearch"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Vector Search index '{self.INDEX_NAME}' on MongoDB Atlas.")
                
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def ensure_keyword_index(self):
        """Automatically checks and creates the Atlas Search (keyword) index if missing."""
        try:
            collection = self._get_collection()
            cursor = collection.list_search_indexes()
            existing_indexes = await cursor.to_list(length=100)
            idx_names = [idx.get("name") for idx in existing_indexes]
            if self.KEYWORD_INDEX_NAME not in idx_names:
                from pymongo.operations import SearchIndexModel
                index_model = SearchIndexModel(
                    definition={
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "text": {"type": "string"}
                            }
                        }
                    },
                    name=self.KEYWORD_INDEX_NAME,
                    type="search"
                )
                await collection.create_search_index(model=index_model)
                logger.info(f"✅ Automatically created Atlas Search index '{self.KEYWORD_INDEX_NAME}' on MongoDB Atlas.")
        except Exception as e:
            logger.debug(f"Atlas Search index check note: {e}")

    async def add(self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]):
        """Stores or updates a batch of document chunks in MongoDB Atlas."""
        if not ids:
            return

        collection = self._get_collection()
        operations = []

        for chunk_id, embedding, doc, meta in zip(ids, embeddings, documents, metadatas):
            doc_body = {
                "_id": chunk_id,
                "id": chunk_id,
                "text": doc,
                "embedding": embedding,
                "source": meta.get("source", "unknown"),
                "metadata": meta,
            }
            operations.append(
                UpdateOne({"_id": chunk_id}, {"$set": doc_body}, upsert=True)
            )

        if operations:
            result = await collection.bulk_write(operations)
            logger.info(f"MongoDB Vector Store updated: {result.upserted_count} inserted, {result.modified_count} modified.")
            await self.ensure_vector_index()
            await self.ensure_keyword_index()


    @traceable(
        run_type="retriever",
        name="vector_search",
        process_inputs=drop_self_and_embedding,
        process_outputs=as_documents,
    )
    async def vector_search(self, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """Finds the `top_k` chunks most similar to the query embedding using MongoDB Atlas Vector Search."""
        collection = self._get_collection()

        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(top_k * 10, 50),
                    "limit": top_k,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)

            matches = []
            for doc in results:
                matches.append({
                    "_id": doc.get("_id"),
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                    "distance": 1.0 - doc.get("score", 0.0),
                })

            return matches

        except Exception as e:
            logger.warning(
                f"MongoDB Vector Search query failed or 'vector_index' is not configured yet in Atlas UI: {e}"
            )
            return []

    @traceable(
        run_type="retriever",
        name="keyword_search",
        process_inputs=drop_self,
        process_outputs=as_documents,
    )
    async def keyword_search(self, query_text: str, top_k: int = 4) -> list[dict]:
        """Finds the `top_k` chunks whose text best matches `query_text` using MongoDB Atlas Search (keyword search)."""
        collection = self._get_collection()

        pipeline = [
            {
                "$search": {
                    "index": self.KEYWORD_INDEX_NAME,
                    "text": {
                        "query": query_text,
                        "path": "text",
                    }
                }
            },
            {"$limit": top_k},
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "source": 1,
                    "score": {"$meta": "searchScore"},
                }
            }
        ]

        try:
            cursor = collection.aggregate(pipeline)
            results = await cursor.to_list(length=top_k)
            return [
                {
                    "_id": doc.get("_id"),
                    "text": doc.get("text", ""),
                    "source": doc.get("source", "unknown"),
                    "score": doc.get("score", 0.0),
                }
                for doc in results
            ]
        except Exception as e:
            logger.warning(
                f"MongoDB Keyword Search query failed or 'keyword_index' is not configured yet in Atlas UI: {e}"
            )
            return []

    @traceable(
        run_type="chain",
        name="reciprocal_rank_fusion",
        process_inputs=drop_self,
        process_outputs=as_documents,
    )
    def _reciprocal_rank_fusion(self, result_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Combines several ranked result lists into one, using Reciprocal Rank Fusion (RRF).

        Vector search scores (cosine similarity) and keyword search scores (BM25-style)
        are on different scales, so we can't just compare them directly. Instead, RRF
        looks at each chunk's *rank position* (1st, 2nd, 3rd...) in every list it appears in:

            fused_score(chunk) = sum of 1 / (k + rank) across every list it appears in

        A chunk found near the top of either list gets a high fused score. A chunk found
        in BOTH lists gets boosted even higher. This also gives us deduplication for free,
        since each chunk id only appears once in the fused result.
        """
        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, dict] = {}

        for results in result_lists:
            for rank, chunk in enumerate(results, start=1):
                chunk_id = chunk["_id"]
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_by_id.setdefault(chunk_id, chunk)  # keep the first-seen copy for text/source

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [
            {**chunk_by_id[chunk_id], "score": fused_scores[chunk_id]}
            for chunk_id in ranked_ids
        ]

    @traceable(
        run_type="retriever",
        name="hybrid_search",
        process_inputs=drop_self_and_embedding,
        process_outputs=as_documents,
    )
    async def hybrid_search(self, query_text: str, query_embedding: list[float], top_k: int = 4) -> list[dict]:
        """
        Hybrid search: retrieves candidates from vector search AND keyword search in
        parallel, fuses the two ranked lists with Reciprocal Rank Fusion, and returns
        the best `top_k` chunks overall.
        """
        fetch_k = top_k * 5  # pull a bigger candidate pool from each ranker so RRF has enough to work with

        vector_results, keyword_results = await asyncio.gather(
            self.vector_search(query_embedding, top_k=fetch_k),
            self.keyword_search(query_text, top_k=fetch_k),
        )

        fused_results = self._reciprocal_rank_fusion([vector_results, keyword_results])
        return fused_results[:top_k]

