import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from app.prompts.chain_prompts import STAGE_1_EXTRACT, STAGE_2_ENRICH, STAGE_3_FORMAT
from app.core.config import settings

# logger lets us print debug/error messages to the console with proper labels
logger = logging.getLogger(__name__)


def get_llm():
    """
    Initializes ChatGoogleGenerativeAI with Gemini primary and Groq fallback.
    """
    gemini_key = settings.gemini_api_key
    groq_key   = settings.groq_api_key

    primary_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=0.3
    )
    fallback_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=groq_key,
        temperature=0.3
    )

    return primary_llm.with_fallbacks([fallback_llm])


# ─────────────────────────────────────────────────────────
# SECTION 3: ChainService  (3-stage LCEL pipeline)
# ─────────────────────────────────────────────────────────

class ChainService:
    """
    Runs a 3-stage LLM pipeline on a block of raw text using LangChain LCEL:

      Stage 1 — EXTRACT : Find key concepts in the text (returns JSON via JsonOutputParser)
      Stage 2 — ENRICH  : For each concept, generate a detailed explanation
      Stage 3 — FORMAT  : Combine everything into a clean Markdown report
    """

    @classmethod
    async def run_concept_chain(cls, raw_text: str) -> dict:
        """
        Entry point for the full 3-stage chain using LCEL (|) pipelines.
        """
        logger.info("Starting 3-stage LangChain LCEL pipeline...")
        execution_history = []
        llm = get_llm()

        # ── STAGE 1: EXTRACT ──────────────────────────────
        logger.info("Stage 1: Extracting concepts from text...")

        stage1_prompt = ChatPromptTemplate.from_messages([
            ("system", STAGE_1_EXTRACT),
            ("human", "{raw_text}")
        ])
        stage1_chain = stage1_prompt | llm | JsonOutputParser()

        try:
            parsed_json = await stage1_chain.ainvoke({"raw_text": raw_text})
            concepts = parsed_json.get("concepts", []) if isinstance(parsed_json, dict) else []

            execution_history.append({
                "stage": "extract",
                "parsed_output": parsed_json,
                "status": "success"
            })

        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return {
                "success": False,
                "error": f"Stage 1 (Extract) failed: {str(e)}",
                "execution_history": execution_history
            }

        if not concepts:
            return {
                "success": False,
                "error": "Stage 1 returned no concepts. Try a more descriptive input.",
                "execution_history": execution_history
            }

        # ── STAGE 2: ENRICH ───────────────────────────────
        logger.info(f"Stage 2: Enriching {len(concepts)} concept(s)...")

        stage2_prompt = ChatPromptTemplate.from_messages([
            ("system", STAGE_2_ENRICH),
            ("human", "Concept: {name}\nContext: {context}")
        ])
        stage2_chain = stage2_prompt | llm | StrOutputParser()

        enriched_concepts = []

        for index, concept in enumerate(concepts):
            concept_name = concept.get("name", "Unknown")
            concept_context = concept.get("context", "")

            logger.info(f"  Enriching concept {index + 1}/{len(concepts)}: '{concept_name}'")

            try:
                enrichment_text = await stage2_chain.ainvoke({
                    "name": concept_name,
                    "context": concept_context
                })

                enriched_concepts.append({
                    "name": concept_name,
                    "context": concept_context,
                    "enrichment": enrichment_text
                })

            except Exception as e:
                logger.warning(f"  Failed to enrich '{concept_name}': {e}. Using placeholder.")
                enriched_concepts.append({
                    "name": concept_name,
                    "context": concept_context,
                    "enrichment": "Enrichment unavailable due to an API error."
                })

        execution_history.append({
            "stage": "enrich",
            "enriched_concepts": enriched_concepts,
            "status": "success"
        })

        # ── STAGE 3: FORMAT ───────────────────────────────
        logger.info("Stage 3: Formatting final Markdown report...")

        enriched_json_string = json.dumps(enriched_concepts, indent=2)
        stage3_prompt = ChatPromptTemplate.from_messages([
            ("system", STAGE_3_FORMAT),
            ("human", "{enriched_json}")
        ])
        stage3_chain = stage3_prompt | llm | StrOutputParser()

        try:
            final_report = await stage3_chain.ainvoke({"enriched_json": enriched_json_string})

            execution_history.append({
                "stage": "format",
                "final_report": final_report,
                "status": "success"
            })

            return {
                "success": True,
                "final_report": final_report,
                "execution_history": execution_history
            }

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            return {
                "success": False,
                "error": f"Stage 3 (Format) failed: {str(e)}",
                "execution_history": execution_history
            }




