import logging
from typing import Literal, Optional, Sequence

from langchain_core.messages import BaseMessage
from langsmith import traceable
from pydantic import BaseModel, Field

from app.ai.models import ordered_models
from app.core.config import settings
from app.core.tracing import drop_self
from app.memory.window import as_text
from app.prompts.router_prompt import ROUTER_PROMPT

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """
    What the router works out about a single user turn.

    Both fields come back from one model call. Splitting them into two calls
    (classify, then rewrite) would double the latency on the critical path for
    no accuracy gain — the model needs the same history to do either job.
    """

    route: Literal["RAG", "TOOL", "BOTH", "DIRECT"] = Field(
        description="Which path should answer this question."
    )
    standalone_question: str = Field(
        description=(
            "The latest question rewritten to stand on its own, with every "
            "pronoun and back-reference resolved from the conversation history."
        )
    )


class QueryRouter:
    """
    Decides which path a user query should take, before any work is done.

    It also resolves the question against the conversation history. That second
    job is what makes memory actually work for retrieval: a follow-up like
    "and where does he work?" carries no searchable terms, so embedding it
    directly returns nothing useful no matter how good the vector store is.
    The rewritten question is what gets searched.
    """

    VALID_ROUTES = {"RAG", "TOOL", "BOTH", "DIRECT"}

    # If the model returns something unusable, fall back to the app's main purpose.
    DEFAULT_ROUTE = "RAG"

    # Classification output is tiny — a route name and one rewritten sentence —
    # so the same small budget suits every model here, including the reasoning
    # one. Note this is exactly why app/ai/models.py disables DeepSeek's
    # reasoning: at 300 tokens, hidden reasoning could swallow the entire
    # allowance and return no structured output at all.
    MAX_TOKENS = 300

    def __init__(self):
        # Classification must be deterministic and cheap, so these are separate
        # model instances from the ones the answering nodes use — same chain and
        # same order, but temperature 0 and a much smaller budget.
        #
        # The router runs on *every* request, ahead of any retrieval or
        # generation, so it is the single place where the primary model's
        # latency matters most.
        primary, *fallbacks = ordered_models(
            self.MAX_TOKENS, self.MAX_TOKENS, temperature=0
        )

        # Same ordering rule as bind_tools() in chat.py: structured output is
        # applied to each *model* first, because with_fallbacks() returns a
        # RunnableWithFallbacks, which has no with_structured_output() of its own.
        llm = primary.with_structured_output(RouteDecision).with_fallbacks(
            [model.with_structured_output(RouteDecision) for model in fallbacks]
        )

        # The whole router, as one LCEL pipeline
        self.chain = ROUTER_PROMPT | llm

    def _fallback(self, query: str) -> RouteDecision:
        """The safe answer: the app's main route, question left untouched."""
        return RouteDecision(route=self.DEFAULT_ROUTE, standalone_question=query)

    @traceable(run_type="chain", name="route_query", process_inputs=drop_self)
    async def route(
        self,
        query: str,
        history: Optional[Sequence[BaseMessage]] = None,
    ) -> RouteDecision:
        """
        Returns a RouteDecision. Never raises — a router failure degrades to
        RAG with the original question rather than taking the request down.

        `history` is optional so the router still works for the first message
        of a thread, and so callers that have no memory keep working unchanged.
        """
        try:
            decision = await self.chain.ainvoke({
                "question": query,
                "history": as_text(history),
            })

            # with_structured_output returns the pydantic model, but a provider
            # can hand back a plain dict; normalize before touching attributes.
            if isinstance(decision, dict):
                decision = RouteDecision(**decision)

            if decision.route in self.VALID_ROUTES:
                # An empty rewrite would search for nothing at all.
                if not decision.standalone_question.strip():
                    decision.standalone_question = query
                return decision

            logger.warning(
                f"Router returned an unrecognized route {decision.route!r}; "
                f"falling back to {self.DEFAULT_ROUTE}."
            )
        except Exception as e:
            logger.error(f"Router failed ({e}); falling back to {self.DEFAULT_ROUTE}.")

        return self._fallback(query)


# Module-level singleton, same pattern as app/rag/rag_pipeline.py
query_router = QueryRouter()
