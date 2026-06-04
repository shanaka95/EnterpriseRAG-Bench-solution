"""
HyDE (Hypothetical Document Embeddings) using a local LLM.
For each question, generate a hypothetical answer, then search with THAT.
This bridges the semantic gap between paraphrased questions and documents.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_llm = None


def get_llm():
    """Lazy-load the LLM for HyDE query expansion."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        import os
        model_path = os.environ.get(
            "HYDE_MODEL_PATH",
            "/app/models/qwen2.5-1.5b-q4_k_m.gguf"
        )
        logger.info(f"Loading HyDE LLM from {model_path}...")
        start = time.time()
        _llm = Llama(
            model_path=model_path,
            n_ctx=512,
            n_threads=4,
            n_gpu_layers=20,
            verbose=False,
        )
        logger.info(f"HyDE LLM loaded in {time.time()-start:.1f}s")
    return _llm


HYDE_PROMPT = """<|im_start|>system
You are a knowledgeable assistant. Write a brief, factual passage (2-3 sentences) that would answer the given question. Use the same technical vocabulary that would appear in a corporate knowledge base, documentation, or internal wiki. Do not hedge or say you don't know - just write a plausible answer.<|im_end|>
<|im_start|>user
Question: {query}<|im_end|>
<|im_start|>assistant
"""


def generate_hyde_passage(query: str, max_tokens: int = 120) -> str:
    """
    Generate a hypothetical answer passage for a question.
    Used as the search query for FAISS retrieval.
    """
    llm = get_llm()
    prompt = HYDE_PROMPT.format(query=query[:500])
    try:
        response = llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "\n\nQuestion:", "\nUser:"],
            echo=False,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return ""


def expand_query_with_hyde(query: str) -> list[str]:
    """
    Generate query expansion using HyDE technique.
    Returns a list of search queries to use for FAISS.
    """
    hyde_passage = generate_hyde_passage(query)
    if not hyde_passage:
        return [query]

    # Return both the original and the hypothetical passage
    return [query, hyde_passage]
