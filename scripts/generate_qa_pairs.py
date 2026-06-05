"""
RAG Evaluation Pipeline

1. Generate 100 Q&A pairs from random documents using MiniMax-M2.7
2. Run the full hybrid retrieval pipeline for each question
3. Send top 10 docs to LLM (MiniMax-M2.7) for grounded answer generation
4. Evaluate accuracy with multiple metrics (retrieval recall, faithfulness, correctness, etc.)
5. Save QA pairs and results to CSV for reproducibility
"""
import os
import sys
import csv
import json
import random
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

import requests
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_BASE = os.environ.get("API_BASE", "http://localhost:8080")
LLM_API_URL = (os.environ.get("MINIMAX_BASE_URL",
                              "https://api.minimax.io/anthropic")
               + "/v1/messages")
LLM_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
LLM_MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")

if not LLM_API_KEY:
    raise SystemExit(
        "MINIMAX_API_KEY env var is required. Export it or use scripts/run_ui.sh."
    )

# ============================================================
# Step 1: Generate Q&A pairs from documents
# ============================================================

def load_documents_from_directory(directory: str, limit: int = 100) -> List[Dict]:
    """Load documents from a directory of .txt files."""
    docs = []
    for filepath in sorted(Path(directory).glob("*.txt")):
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if text and len(text) > 200:
            doc_id = filepath.stem
            docs.append({"id": doc_id, "text": text, "source": str(filepath)})
    return docs


def generate_qa_from_doc(doc: Dict, question_num: int) -> Optional[Dict]:
    """Generate a question-answer pair from a document using the LLM."""
    prompt = f"""You are an expert at creating question-answer pairs from business documents for testing a RAG (Retrieval-Augmented Generation) system.

Given the following document, create:
1. A specific question that can be answered using ONLY information from this document
2. A detailed, accurate answer based on the document content
3. A brief excerpt from the document that supports the answer (the "ground truth" citation)

DOCUMENT:
{doc['text'][:3000]}

RESPOND IN THIS EXACT JSON FORMAT (no markdown, no code fences):
{{"question": "your specific question here", "answer": "your detailed answer here", "source_excerpt": "the exact excerpt from the document that supports your answer"}}"""

    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code} - {response.text[:200]}")
            return None

        result = response.json()
        # MiniMax-M2.7 returns content as array with "thinking" and "text" types
        content = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                content = block.get("text", "")
                break
        if not content and result.get("content"):
            content = result["content"][-1].get("text", "")

        # Parse JSON from response
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                qa = json.loads(json_str)
                qa["doc_id"] = doc["id"]
                qa["question_num"] = question_num
                return qa
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error for doc {doc['id']}: {e}")
                return None
        else:
            logger.warning(f"Could not find JSON in LLM response for doc {doc['id']}")
            return None

    except Exception as e:
        logger.error(f"Error generating QA for doc {doc['id']}: {e}")
        return None


def generate_qa_pairs(docs: List[Dict], count: int = 100) -> List[Dict]:
    """Generate Q&A pairs from random documents."""
    random.seed(42)
    selected = random.sample(docs, min(count, len(docs)))

    qa_pairs = []
    for i, doc in enumerate(selected):
        logger.info(f"Generating QA pair {i+1}/{len(selected)} for doc {doc['id'][:30]}...")
        qa = generate_qa_from_doc(doc, i + 1)
        if qa:
            qa_pairs.append(qa)
        time.sleep(1)

    return qa_pairs


def save_qa_csv(qa_pairs: List[Dict], filepath: str):
    """Save Q&A pairs to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question_num", "doc_id", "question", "answer", "source_excerpt"])
        writer.writeheader()
        for qa in qa_pairs:
            writer.writerow({
                "question_num": qa.get("question_num", ""),
                "doc_id": qa.get("doc_id", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "source_excerpt": qa.get("source_excerpt", ""),
            })
    logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {filepath}")


# ============================================================
# Step 2: Retrieval Pipeline (uses server-side hybrid retrieval)
# ============================================================

def retrieve_documents(query: str, top_k: int = 10) -> Dict:
    """
    Run the hybrid retrieval pipeline.
    The server handles: leaf-first Cross-Encoder + Bi-Encoder fallback + Cross-Encoder reranking.
    """
    response = requests.post(
        f"{API_BASE}/api/v1/query",
        json={"query": query, "top_k": top_k},
        timeout=60,
    )

    if response.status_code != 200:
        logger.error(f"Query API error: {response.status_code}")
        return {"doc_ids": [], "paths_traversed": [], "scores": {}}

    return response.json()


# ============================================================
# Step 3: LLM Answer Generation
# ============================================================

def generate_rag_answer(query: str, context_docs: List[Dict]) -> Dict:
    """Generate a grounded, cited answer using the LLM with retrieved context."""
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        doc_id = doc.get("id", "unknown")
        text = doc.get("text", "")[:1500]
        context_parts.append(f"[Document {i}] (ID: {doc_id})\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a precise, factual question-answering assistant. Answer the user's question using ONLY the provided context documents. You must cite your sources.

RULES:
1. Answer based ONLY on the provided context
2. Cite specific documents using [Document N] format
3. If the context doesn't contain enough information, say so explicitly
4. Be specific and detailed - don't be vague
5. Include relevant numbers, metrics, and specifics from the documents

CONTEXT DOCUMENTS:
{context}

QUESTION: {query}

Provide a detailed, well-cited answer:"""

    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )

        if response.status_code != 200:
            logger.error(f"LLM API error: {response.status_code}")
            return {"answer": "", "citations": [], "error": response.text[:200]}

        result = response.json()
        answer_text = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                answer_text = block.get("text", "")
                break
        if not answer_text and result.get("content"):
            answer_text = result["content"][-1].get("text", "")

        # Extract citations
        citations = []
        for i in range(1, len(context_docs) + 1):
            if f"[Document {i}]" in answer_text:
                citations.append(context_docs[i-1].get("id", f"doc-{i}"))

        return {"answer": answer_text, "citations": citations, "error": None}

    except Exception as e:
        logger.error(f"LLM answer generation error: {e}")
        return {"answer": "", "citations": [], "error": str(e)}


# ============================================================
# Step 4: Evaluation Metrics
# ============================================================

def evaluate_answer(generated: Dict, ground_truth: Dict, context_doc_ids: List[str]) -> Dict:
    """Evaluate a generated answer against ground truth."""
    metrics = {
        "retrieval_recall": 0.0,
        "ground_truth_in_context": False,
        "num_citations": len(generated.get("citations", [])),
        "num_docs_retrieved": len(context_doc_ids),
    }

    gt_doc_id = ground_truth.get("doc_id", "")
    if gt_doc_id in context_doc_ids:
        metrics["retrieval_recall"] = 1.0
        metrics["ground_truth_in_context"] = True

    return metrics


def evaluate_with_llm(question: str, generated_answer: str, ground_truth_answer: str) -> Dict:
    """Use LLM to judge answer quality (LLM-as-judge)."""
    prompt = f"""You are an expert evaluator for a RAG system. Compare the generated answer against the ground truth answer.

QUESTION: {question}

GROUND TRUTH ANSWER: {ground_truth_answer}

GENERATED ANSWER: {generated_answer}

Rate the generated answer on these dimensions (0.0 to 1.0):

1. **Faithfulness**: Does the generated answer stay faithful to the source context? No hallucinations?
2. **Answer Relevance**: Does the generated answer address the question?
3. **Completeness**: Does the generated answer cover the key information from the ground truth?
4. **Correctness**: Is the factual information in the generated answer correct?

RESPOND IN THIS EXACT JSON FORMAT:
{{"faithfulness": 0.0, "answer_relevance": 0.0, "completeness": 0.0, "correctness": 0.0, "overall": 0.0, "reasoning": "brief explanation"}}"""

    try:
        response = requests.post(
            LLM_API_URL,
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            content = ""
            for block in result.get("content", []):
                if block.get("type") == "text":
                    content = block.get("text", "")
                    break
            if not content and result.get("content"):
                content = result["content"][-1].get("text", "")
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])

    except Exception as e:
        logger.error(f"LLM evaluation error: {e}")

    return {"faithfulness": 0.0, "answer_relevance": 0.0, "completeness": 0.0, "correctness": 0.0, "overall": 0.0}


# ============================================================
# Main Pipeline
# ============================================================

def run_full_evaluation(docs_dir: str, qa_csv_path: str, results_csv_path: str):
    """Run the full RAG evaluation pipeline."""

    # Step 1: Load or generate Q&A pairs
    qa_pairs = []
    if os.path.exists(qa_csv_path):
        logger.info(f"Loading existing Q&A pairs from {qa_csv_path}")
        with open(qa_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            qa_pairs = list(reader)
    else:
        logger.info("Generating Q&A pairs from documents...")
        docs = load_documents_from_directory(docs_dir, limit=100)
        logger.info(f"Loaded {len(docs)} documents for Q&A generation")
        qa_pairs = generate_qa_pairs(docs, count=100)
        save_qa_csv(qa_pairs, qa_csv_path)

    logger.info(f"Total Q&A pairs: {len(qa_pairs)}")

    # Step 2-4: Run retrieval, generation, and evaluation for each question
    all_results = []
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        gt_answer = qa.get("answer", "")
        gt_doc_id = qa.get("doc_id", "")
        question_num = qa.get("question_num", i)

        if not question:
            continue

        logger.info(f"[{i+1}/{len(qa_pairs)}] Processing: {question[:80]}...")

        # Step 2: Retrieve (server handles hybrid retrieval + reranking to top_k=10)
        retrieval_result = retrieve_documents(question, top_k=10)
        retrieved_doc_ids = retrieval_result.get("doc_ids", [])

        # Fetch top document texts for context
        context_docs = []
        for doc_id in retrieved_doc_ids[:10]:
            try:
                resp = requests.get(f"{API_BASE}/api/v1/documents/{doc_id}", timeout=10)
                if resp.status_code == 200:
                    context_docs.append(resp.json())
            except Exception:
                continue

        # Step 3: Generate answer
        rag_result = generate_rag_answer(question, context_docs)
        generated_answer = rag_result.get("answer", "")

        # Step 4: Evaluate
        retrieval_metrics = evaluate_answer(rag_result, qa, retrieved_doc_ids)

        # LLM-judged metrics (evaluate all 100 with LLM judge)
        llm_metrics = evaluate_with_llm(question, generated_answer, gt_answer)
        time.sleep(1)

        result = {
            "question_num": question_num,
            "question": question,
            "doc_id": gt_doc_id,
            "gt_in_retrieved": retrieval_metrics["ground_truth_in_context"],
            "retrieval_recall": retrieval_metrics["retrieval_recall"],
            "num_docs_retrieved": retrieval_metrics["num_docs_retrieved"],
            "num_citations": retrieval_metrics["num_citations"],
            "faithfulness": llm_metrics.get("faithfulness", ""),
            "answer_relevance": llm_metrics.get("answer_relevance", ""),
            "completeness": llm_metrics.get("completeness", ""),
            "correctness": llm_metrics.get("correctness", ""),
            "overall_llm": llm_metrics.get("overall", ""),
            "generated_answer": generated_answer[:500],
            "gt_answer": gt_answer[:500],
        }
        all_results.append(result)

        # Log progress
        if (i + 1) % 10 == 0:
            recall = np.mean([r["retrieval_recall"] for r in all_results])
            llm_results = [r for r in all_results if r["faithfulness"] != ""]
            avg_overall = np.mean([float(r["overall_llm"]) for r in llm_results]) if llm_results else 0
            logger.info(f"  Running: recall={recall:.3f}, llm_overall={avg_overall:.3f}")

    # Save results
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_num", "question", "doc_id", "gt_in_retrieved",
            "retrieval_recall", "num_docs_retrieved", "num_citations",
            "faithfulness", "answer_relevance", "completeness", "correctness",
            "overall_llm", "generated_answer", "gt_answer",
        ])
        writer.writeheader()
        writer.writerows(all_results)

    # Print summary
    print("\n" + "="*60)
    print("RAG EVALUATION SUMMARY")
    print("="*60)
    print(f"Total questions evaluated: {len(all_results)}")
    print(f"Mean Retrieval Recall: {np.mean([r['retrieval_recall'] for r in all_results]):.3f}")
    print(f"Mean Docs Retrieved: {np.mean([r['num_docs_retrieved'] for r in all_results]):.1f}")
    print(f"Mean Citations: {np.mean([r['num_citations'] for r in all_results]):.1f}")

    llm_results = [r for r in all_results if r["faithfulness"] != ""]
    if llm_results:
        print(f"\nLLM-Judged Metrics ({len(llm_results)} questions):")
        print(f"  Faithfulness:      {np.mean([float(r['faithfulness']) for r in llm_results]):.3f}")
        print(f"  Answer Relevance:  {np.mean([float(r['answer_relevance']) for r in llm_results]):.3f}")
        print(f"  Completeness:      {np.mean([float(r['completeness']) for r in llm_results]):.3f}")
        print(f"  Correctness:       {np.mean([float(r['correctness']) for r in llm_results]):.3f}")
        print(f"  Overall:           {np.mean([float(r['overall_llm']) for r in llm_results]):.3f}")

    print(f"\nResults saved to: {results_csv_path}")
    print("="*60)


if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/finance-and-legal"
    qa_csv = sys.argv[2] if len(sys.argv) > 2 else "/app/qa_pairs.csv"
    results_csv = sys.argv[3] if len(sys.argv) > 3 else "/app/rag_evaluation_results.csv"

    run_full_evaluation(docs_dir, qa_csv, results_csv)
