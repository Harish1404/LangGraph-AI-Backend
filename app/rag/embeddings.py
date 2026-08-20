from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable

from app.core.config import settings
from app.core.tracing import drop_self


class EmbeddingModel:
    """
    Turns text into vectors (lists of numbers) so we can compare how
    similar two pieces of text are by comparing their vectors.

    Uses Google's Gemini embedding API via LangChain.
    """

    def __init__(self, model_name: str = "models/gemini-embedding-001", dimension: int = 768):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            output_dimensionality=dimension
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds many chunks at once — used during ingestion."""
        if not texts:
            return []
        return await self.embeddings.aembed_documents(texts)

    @traceable(
        run_type="embedding",
        name="embed_query",
        process_inputs=drop_self,
        # A 768-float vector is not worth reading in a trace — record its size.
        process_outputs=lambda vector: {"dimensions": len(vector) if vector else 0},
    )
    async def embed_query(self, text: str) -> list[float]:
        """
        Embeds a single piece of text — used for a user's question.

        Decorated because GoogleGenerativeAIEmbeddings does not report to
        LangSmith on its own, unlike the chat models. Without this the
        embedding step is simply invisible in the trace.
        """
        return await self.embeddings.aembed_query(text)

