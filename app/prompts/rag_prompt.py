from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = """You are an assistant that answers questions using ONLY the context provided below.

Rules:
- Base your answer strictly on the given context.
- If the answer isn't in the context, say "I don't know based on the information I have."
- Keep answers concise and to the point.
- mention which source the information came from.
"""

rag_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{context}\n\nQuestion: {question}")
])


def build_context_text(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a context string."""
    if not chunks:
        return "(No relevant context was found.)"
    return "\n\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk in chunks)

