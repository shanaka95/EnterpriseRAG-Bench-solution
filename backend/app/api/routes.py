"""
FastAPI API routes for the Hierarchical RAG Pipeline.
- POST /api/v1/ingest - Ingest document batches (with content-hash dedup)
- POST /api/v1/query - Multi-path query inference
- GET /api/v1/tree - Get full tree structure
- GET /api/v1/nodes/{node_id} - Get specific node
- GET /api/v1/stream - SSE stream for real-time updates
- GET /api/v1/health - Health check
"""
import json
import uuid
import hashlib
import logging
import math
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.database import get_db
from app.models.schemas import ClusterNode, Document, IngestionJob
from app.ml.embedding import encode_documents, cross_encoder_score, get_bi_encoder, get_cross_encoder
from app.core.config import settings
# colbert_reranker is imported lazily inside Phase 5 only when COLBERT_RERANK_ENABLED=True,
# so the heavy ColBERT model and LanceDB connection aren't loaded at startup unless needed.

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["rag"])


# ── Pydantic Request/Response Models ──

class DocumentInput(BaseModel):
    id: str
    text: str
    source: Optional[str] = None

class IngestRequest(BaseModel):
    documents: List[DocumentInput]

class IngestResponse(BaseModel):
    job_id: str
    status: str
    total_docs: int
    new_docs: int
    duplicate_docs: int
    message: str

class QueryRequest(BaseModel):
    query: str
    threshold: Optional[float] = None
    max_depth: Optional[int] = None
    top_k: Optional[int] = Field(default=10, ge=1, le=20000, description="Number of documents to return (1-20000)")

class QueryResponse(BaseModel):
    query: str
    doc_ids: List[str]
    paths_traversed: List[List[str]]
    scores: dict

class NodeResponse(BaseModel):
    id: str
    parent_id: Optional[str]
    medoid_doc_id: Optional[str]
    doc_count: int
    keywords: List[str]
    is_leaf: bool
    doc_ids: Optional[List[str]]
    depth: int


def _content_hash(text: str) -> str:
    """SHA-256 hash of document text for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# PostgreSQL has a 65535 parameter limit per query. Chunk large IN clauses.
_PG_MAX_PARAMS = 50000


def _bm25_score(query_terms: List[str], doc_text: str, corpus_term_counts: dict, avgdl: float, N: int, k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute a BM25-like score for a document against query terms.
    corpus_term_counts: {term: number_of_docs_containing_term}
    avgdl: average document length in tokens.
    N: total number of documents in the candidate corpus.
    """
    doc_tokens = doc_text.lower().split()
    dl = len(doc_tokens)
    if dl == 0 or avgdl == 0 or N == 0:
        return 0.0

    token_counts = {}
    for t in doc_tokens:
        token_counts[t] = token_counts.get(t, 0) + 1

    score = 0.0
    for term in query_terms:
        f = token_counts.get(term, 0)
        if f == 0:
            continue
        n = corpus_term_counts.get(term, 0)
        # Smooth idf to avoid negative values
        idf = max(0.0, math.log((N - n + 0.5) / (n + 0.5) + 1.0))
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl)))
    return score


def _chunked_in_query(db: Session, model, attr, ids: set, columns=None):
    """
    Query a SQLAlchemy model with a large IN clause, automatically chunking
    to stay under PostgreSQL's parameter limit.
    Returns a list of all matching ORM instances.
    """
    if not ids:
        return []
    id_list = list(ids)
    results = []
    query = db.query(model)
    if columns:
        query = query.with_entities(*columns)
    for i in range(0, len(id_list), _PG_MAX_PARAMS):
        chunk = id_list[i : i + _PG_MAX_PARAMS]
        results.extend(query.filter(attr.in_(chunk)).all())
    return results


# ── Health Check ──

@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Check system health including DB and model status."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    models_status = "not_loaded"
    try:
        import torch
        gpu_info = f"cuda_available={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            gpu_info += f", device={torch.cuda.get_device_name(0)}"
        models_status = gpu_info
    except Exception:
        pass

    # Check FAISS index status
    faiss_status = "not_built"
    try:
        from app.ml.faiss_index import get_index_stats
        faiss_status = get_index_stats()
    except Exception:
        pass

    return {
        "status": "ok",
        "database": db_status,
        "models": models_status,
        "faiss": faiss_status,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Ingestion Endpoint ──

@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest, db: Session = Depends(get_db)):
    """
    Accept a batch of documents, store them (with content-hash dedup),
    and kick off the recursive clustering pipeline.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided.")

    # Create ingestion job
    job = IngestionJob(
        id=uuid.uuid4(),
        status="pending",
        total_docs=len(request.documents),
    )
    db.add(job)

    # Store documents with dedup
    doc_ids = []
    new_count = 0
    dup_count = 0

    for doc_input in request.documents:
        content_hash = _content_hash(doc_input.text)

        # Check for exact content duplicate by hash
        existing_by_hash = db.query(Document).filter(
            Document.source == f"hash:{content_hash}"
        ).first()

        if existing_by_hash:
            # Exact duplicate - skip
            dup_count += 1
            doc_ids.append(existing_by_hash.id)
            continue

        existing = db.query(Document).filter(Document.id == doc_input.id).first()
        if existing:
            # Same ID but different content - update
            existing.text = doc_input.text
            existing.source = f"hash:{content_hash}"
            existing.embedding = None  # Reset embedding for re-computation
            doc_ids.append(existing.id)
            new_count += 1
        else:
            new_doc = Document(
                id=doc_input.id,
                text=doc_input.text,
                source=f"hash:{content_hash}",
            )
            db.add(new_doc)
            doc_ids.append(new_doc.id)
            new_count += 1

    db.commit()

    # Queue the clustering task - import here to avoid circular imports
    # and to ensure the Celery app is properly initialized
    try:
        from app.tasks.celery_app import celery_app
        celery_app.send_task(
            "ingest_and_build",
            args=[str(job.id), doc_ids],
            queue="gpu",
        )
        task_status = "queued"
    except Exception as e:
        logger.error(f"Failed to queue Celery task: {e}")
        task_status = f"queue_failed: {str(e)}"

    return IngestResponse(
        job_id=str(job.id),
        status=task_status,
        total_docs=len(request.documents),
        new_docs=new_count,
        duplicate_docs=dup_count,
        message=f"Ingestion: {new_count} new, {dup_count} duplicates. Task {task_status}.",
    )


# ── Dedup Cleanup Endpoint ──

@router.post("/dedup")
def remove_duplicates(db: Session = Depends(get_db)):
    """Find and remove duplicate documents (same content hash)."""
    # Process in batches to avoid loading 512K docs into RAM
    BATCH_SIZE = 50000
    seen_hashes = {}
    duplicates = []
    total_docs = 0
    offset = 0

    while True:
        batch = db.query(Document).offset(offset).limit(BATCH_SIZE).all()
        if not batch:
            break
        total_docs += len(batch)
        for doc in batch:
            content_hash = _content_hash(doc.text)
            if content_hash in seen_hashes:
                duplicates.append(doc.id)
            else:
                seen_hashes[content_hash] = doc.id
        offset += len(batch)
        if len(batch) < BATCH_SIZE:
            break

    # Remove duplicates in chunks to stay under PostgreSQL parameter limits
    for i in range(0, len(duplicates), _PG_MAX_PARAMS):
        chunk = duplicates[i : i + _PG_MAX_PARAMS]
        db.query(Document).filter(Document.id.in_(chunk)).delete(synchronize_session=False)

    db.commit()

    return {
        "total_documents": total_docs,
        "duplicates_removed": len(duplicates),
        "unique_documents": total_docs - len(duplicates),
    }


# ── Query Endpoint (Multi-Signal Hybrid Retrieval) ──

@router.post("/query", response_model=QueryResponse)
def query_tree(request: QueryRequest, db: Session = Depends(get_db)):
    """
    Multi-signal hybrid retrieval combining:
    1. Bi-Encoder cosine similarity (primary - sees full doc text embeddings)
    2. Leaf-first Cross-Encoder scoring (catches fine-grained cluster matches)
    3. BM25 sparse keyword matching (catches exact keyword matches)
    4. Multi-query expansion (rephrases query 3 ways for broader coverage)

    Fusion: Reciprocal Rank Fusion (RRF) of all signals.
    """
    import numpy as np
    from app.ml.embedding import get_bi_encoder

    threshold = request.threshold or settings.CROSS_ENCODER_THRESHOLD
    top_k = request.top_k if hasattr(request, 'top_k') and request.top_k is not None else 10

    # ── Phase 1: Multi-query expansion + Bi-Encoder (FAISS or full-scan) ──
    bi_encoder = get_bi_encoder()

    # Generate query variations for broader coverage.
    # bge-m3 supports instruction-prefixed queries for different embedding
    # spaces. Using diverse instructions dramatically increases recall.
    q = request.query
    # BGE-M3 recommended instructions for retrieval
    query_variations = [
        q,  # Original
        f"What is {q}?",
        f"Explain {q}",
        f"Details about {q}",
        f"Represent this question for searching relevant passages: {q}",
        f"Given a question, find the most relevant document: {q}",
    ]

    # Add HyDE (Hypothetical Document Embeddings) passage — bridges semantic gap
    # by searching with a hypothetical answer that uses document-like vocabulary
    try:
        from app.ml.hyde import expand_query_with_hyde
        hyde_queries = expand_query_with_hyde(q)
        for hq in hyde_queries:
            if hq and hq not in query_variations:
                query_variations.append(hq)
    except Exception as e:
        logger.warning(f"HyDE expansion failed: {e}")

    # Deduplicate while preserving order
    seen = set()
    query_variations = [v for v in query_variations if not (v in seen or seen.add(v))]

    # Try FAISS ANN search first (O(log N) for large datasets)
    from app.ml.faiss_index import is_ready as faiss_ready, search as faiss_search

    doc_max_sim = {}  # doc_id -> max similarity across all query variations
    bi_rank_map = {}  # doc_id -> rank (0 = best)
    total_docs_for_rrf = 1000000  # Default for RRF calculation

    if faiss_ready():
        # FAISS ANN search - fast for large datasets
        # Retrieve top-2000 for maximum recall, then rerank
        for qvar in query_variations:
            q_emb = bi_encoder.encode(
                [qvar[:2000]],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            results = faiss_search(q_emb, top_k=20000)
            for doc_id, sim in results:
                if doc_id not in doc_max_sim or sim > doc_max_sim[doc_id]:
                    doc_max_sim[doc_id] = sim

        # Build rank map from sorted similarities
        sorted_bi = sorted(doc_max_sim.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(sorted_bi):
            bi_rank_map[doc_id] = rank
    else:
        # Fallback: full-scan Bi-Encoder (only for small datasets)
        # For 512K docs, FAISS is required to avoid OOM and ensure speed.
        count_with_emb = db.query(Document).filter(Document.embedding.isnot(None)).count()
        if count_with_emb > 10000:
            raise HTTPException(
                status_code=503,
                detail="FAISS index required for datasets >10K documents. Build it with POST /build_faiss",
            )
        all_docs_with_emb = db.query(Document).filter(
            Document.embedding.isnot(None)
        ).all()
        total_docs_for_rrf = len(all_docs_with_emb) if all_docs_with_emb else 1

        doc_emb_array = np.array([d.embedding for d in all_docs_with_emb], dtype=np.float32) if all_docs_with_emb else None

        if doc_emb_array is not None:
            for qvar in query_variations:
                q_emb = bi_encoder.encode(
                    [qvar[:2000]],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )[0]
                sims = np.dot(doc_emb_array, q_emb)
                for i, sim in enumerate(sims):
                    doc_id = all_docs_with_emb[i].id
                    if doc_id not in doc_max_sim or float(sim) > doc_max_sim[doc_id]:
                        doc_max_sim[doc_id] = float(sim)

        # Build rank map from sorted similarities
        if doc_max_sim:
            sorted_bi = sorted(doc_max_sim.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _) in enumerate(sorted_bi):
                bi_rank_map[doc_id] = rank

    # Top-20000 candidates for combined pool (maximize recall)
    bi_encoder_doc_ids = []
    bi_encoder_scores = []
    sorted_bi = sorted(doc_max_sim.items(), key=lambda x: x[1], reverse=True)
    for doc_id, sim in sorted_bi[:20000]:
        bi_encoder_doc_ids.append(doc_id)
        bi_encoder_scores.append(sim)

    # ── Phase 2: Leaf-first Cross-Encoder scoring ──
    # FAST PATH: Skip expensive leaf cross-encoding for large-scale eval.
    # The cluster tree adds ~2-5s per query on 17K leaves. For benchmark runs,
    # pure FAISS + keyword fusion gives comparable recall with 10× speed.
    leaves = db.query(ClusterNode).filter(ClusterNode.is_leaf == True).all()
    if not leaves and not bi_encoder_doc_ids:
        raise HTTPException(status_code=404, detail="No data found. Ingest documents first.")

    leaf_doc_ids = set()
    paths_traversed = []
    ce_leaf_rank_map = {}  # doc_id -> best leaf rank
    scores_map = {}

    # Only run leaf scoring if tree exists AND request doesn't ask for fast mode
    enable_leaf = leaves and request.top_k is not None and request.top_k <= 100

    if enable_leaf:
        # Batch-load medoid documents to fix N+1 query (17K leaves = 17K round trips)
        medoid_ids = {leaf.medoid_doc_id for leaf in leaves if leaf.medoid_doc_id}
        medoid_docs = _chunked_in_query(db, Document, Document.id, medoid_ids)
        medoid_text_map = {d.id: d.text[:512] for d in medoid_docs}

        leaf_pairs = []
        for leaf in leaves:
            medoid_text = medoid_text_map.get(leaf.medoid_doc_id, "")
            keywords_str = ", ".join(leaf.keywords[:10]) if leaf.keywords else ""
            formatted_text = f"[Keywords: {keywords_str}]\n\n{medoid_text}"
            leaf_pairs.append([request.query, formatted_text])

        leaf_scores = cross_encoder_score(leaf_pairs)

        scored_leaves = list(zip(leaves, leaf_scores))
        scored_leaves.sort(key=lambda x: x[1], reverse=True)

        scores_map = {str(leaf.id): float(score) for leaf, score in scored_leaves}

        # Take top 15 leaves + threshold passes (generous to boost recall)
        top_leaves = []
        for leaf, score in scored_leaves:
            if float(score) > threshold:
                top_leaves.append(leaf)
        min_leaves = min(15, len(scored_leaves))
        for leaf, _ in scored_leaves[:min_leaves]:
            if leaf not in top_leaves:
                top_leaves.append(leaf)

        for leaf_rank, leaf in enumerate(top_leaves):
            if leaf.doc_ids:
                leaf_doc_ids.update(leaf.doc_ids)
                for did in leaf.doc_ids:
                    if did not in ce_leaf_rank_map or leaf_rank < ce_leaf_rank_map[did]:
                        ce_leaf_rank_map[did] = leaf_rank
            path = [str(leaf.id)]
            current = leaf
            while current.parent_id is not None:
                path.insert(0, str(current.parent_id))
                current = db.query(ClusterNode).filter(ClusterNode.id == current.parent_id).first()
                if current is None:
                    break
            paths_traversed.append(path)

    # ── Phase 3: PostgreSQL Full-Text Search (searches ALL docs) ──
    # This is CRITICAL for finding docs that embeddings miss.
    # FTS searches the entire 500K corpus independently of FAISS.
    fts_doc_ids = []
    fts_rank_map = {}
    try:
        # Build OR query from query terms for maximum recall
        # plainto_tsquery uses AND which is too restrictive for semantic gaps
        q_terms = request.query.lower().split()
        # Filter to meaningful terms (remove stopwords manually)
        stopwords = {'the','a','an','is','are','was','were','be','been','being',
                     'have','has','had','do','does','did','will','would','could',
                     'should','may','might','must','shall','can','need','dare',
                     'ought','used','to','of','in','for','on','with','at','by',
                     'from','as','into','through','during','before','after',
                     'above','below','between','under','again','further','then',
                     'once','here','there','when','where','why','how','all',
                     'each','few','more','most','other','some','such','no','nor',
                     'not','only','own','same','so','than','too','very','just',
                     'and','but','if','or','because','until','while','what','which',
                     'who','whom','this','that','these','those','am','it','its',
                     'our','ours','you','your','yours','he','him','his','she',
                     'her','hers','they','them','their','theirs','we','us','i',
                     'me','my','mine'}
        meaningful = [t for t in q_terms if t not in stopwords and len(t) > 2]
        if meaningful:
            or_query = " | ".join(meaningful[:20])  # OR up to 20 terms
        else:
            or_query = " | ".join(q_terms[:10])

        fts_results = db.execute(
            text("""
                SELECT id, ts_rank_cd(tsv, to_tsquery('english', :query), 32) AS rank
                FROM documents
                WHERE tsv @@ to_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT 3000
            """),
            {"query": or_query[:2000]}
        ).fetchall()
        for rank, (doc_id, score) in enumerate(fts_results):
            fts_doc_ids.append(doc_id)
            fts_rank_map[doc_id] = rank

        # Also search FTS with HyDE passage (if available) and other query variations
        # The HyDE passage contains document-like vocabulary that FTS can match
        for qvar in query_variations[1:]:  # Skip first (original already searched)
            qv_terms = qvar.lower().split()
            qv_meaningful = [t for t in qv_terms if t not in stopwords and len(t) > 2]
            if not qv_meaningful:
                continue
            qv_or = " | ".join(qv_meaningful[:15])
            try:
                qv_results = db.execute(
                    text("""
                        SELECT id, ts_rank_cd(tsv, to_tsquery('english', :query), 32) AS rank
                        FROM documents
                        WHERE tsv @@ to_tsquery('english', :query)
                        ORDER BY rank DESC
                        LIMIT 1000
                    """),
                    {"query": qv_or[:1500]}
                ).fetchall()
                # Add to fts_doc_ids if not already present
                existing = set(fts_doc_ids)
                for doc_id, score in qv_results:
                    if doc_id not in existing:
                        fts_doc_ids.append(doc_id)
                        fts_rank_map[doc_id] = len(fts_doc_ids)  # Approximate rank
                        existing.add(doc_id)
                # Cap at 5000 total
                if len(fts_doc_ids) >= 5000:
                    break
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"FTS search failed: {e}")
        fts_doc_ids = []

    # ── Phase 4: BM25-style keyword matching over candidate pool ──
    query_terms = request.query.lower().split()
    query_term_set = set(query_terms)
    keyword_doc_ids = set()
    kw_rank_map = {}
    if query_term_set and bi_encoder_doc_ids:
        # Candidate pool: top-2000 bi-encoder docs + all leaf docs + FTS docs
        candidate_ids = set(bi_encoder_doc_ids[:2000]) | leaf_doc_ids | set(fts_doc_ids[:1000])
        docs_for_kw = _chunked_in_query(db, Document, Document.id, candidate_ids)

        # Build corpus statistics for BM25 over this candidate pool
        corpus_term_counts = {}
        doc_texts_kw = {}
        total_len = 0
        for doc in docs_for_kw:
            text = doc.text[:2000].lower()
            doc_texts_kw[doc.id] = text
            tokens = text.split()
            total_len += len(tokens)
            seen = set()
            for t in tokens:
                if t not in seen:
                    corpus_term_counts[t] = corpus_term_counts.get(t, 0) + 1
                    seen.add(t)

        avgdl = total_len / len(docs_for_kw) if docs_for_kw else 1.0
        N = len(docs_for_kw)

        kw_scored = []
        for doc_id, text in doc_texts_kw.items():
            score = _bm25_score(query_terms, text, corpus_term_counts, avgdl, N)
            if score > 0:
                kw_scored.append((doc_id, score))
        kw_scored.sort(key=lambda x: x[1], reverse=True)
        for rank, (did, _) in enumerate(kw_scored):
            keyword_doc_ids.add(did)
            kw_rank_map[did] = rank

    # ── Phase 4: Union of all retrieval paths ──
    combined_doc_ids = set(leaf_doc_ids)
    combined_doc_ids.update(bi_encoder_doc_ids)
    combined_doc_ids.update(keyword_doc_ids)
    combined_doc_ids.update(fts_doc_ids)
    unique_doc_ids = sorted(list(combined_doc_ids))

    # ── Phase 5: OpenClaw-style Cross-Encoder reranking with weighted fusion ──
    # OpenClaw achieves 79% recall on EnterpriseRAG-Bench with this recipe:
    #   final_score = 0.6 * ce_score + 0.4 * hybrid_fused_score
    # CRITICAL: We must return top_k docs, not just the cross-encoded ones.
    # Cross-encoder is run on the top CE_POOL candidates (by RRF rank), then
    # the rest are filled in with their RRF fused score only.
    CE_POOL_SIZE = 1000  # How many top candidates to cross-encode (larger = slower but better)

    # Step 1: Compute hybrid fused (RRF) score across all retrieval signals
    k = 60
    total_docs = total_docs_for_rrf
    fused_scores_map = {}
    for did in unique_doc_ids:
        bi_rank = bi_rank_map.get(did, total_docs)
        kw_rank = kw_rank_map.get(did, len(keyword_doc_ids) if keyword_doc_ids else 500)
        fts_rank = fts_rank_map.get(did, len(fts_doc_ids) if fts_doc_ids else 500)
        leaf_rank = ce_leaf_rank_map.get(did, len(leaves) if leaves else 100)
        fused_scores_map[did] = (
            0.8 / (k + bi_rank) +
            1.2 / (k + kw_rank) +
            1.5 / (k + fts_rank) +
            0.3 / (k + leaf_rank)
        )

    # Step 2: Rank all candidates by RRF fused score
    fused_ranked_all = sorted(
        fused_scores_map.items(), key=lambda x: x[1], reverse=True
    )

    if len(unique_doc_ids) <= top_k:
        unique_doc_ids = [did for did, _ in fused_ranked_all]
    else:
        # Pool size: ColBERT can use its own (defaults to 1000, same as legacy CE).
        rerank_pool = settings.COLBERT_RERANK_POOL if settings.COLBERT_RERANK_ENABLED else CE_POOL_SIZE
        ce_candidates = [did for did, _ in fused_ranked_all[:rerank_pool]]

        ce_score_map: dict[str, float] = {}
        if settings.COLBERT_RERANK_ENABLED:
            # ── ColBERT (Jina-ColBERT-v2) multi-vector late-interaction rerank ──
            # No DB text fetch needed: doc embeddings live in LanceDB, keyed by id.
            try:
                from app.ml.colbert_reranker import colbert_rerank
                ranked = colbert_rerank(request.query, ce_candidates, top_k=None)
                ce_score_map = {did: float(score) for did, score in ranked}
                scored_ids = [did for did, _ in ranked]
                scores_map["reranker"] = "colbert-v2"
            except Exception as e:
                logger.error(f"ColBERT rerank failed, falling back to cross-encoder: {e!r}")
                # Fall through to the legacy cross-encoder path below
                ce_score_map = {}
                scored_ids = []

        if not ce_score_map:
            # ── Legacy cross-encoder path (BAAI/bge-reranker-v2-m3) ──
            docs = _chunked_in_query(db, Document, Document.id, set(ce_candidates))
            doc_texts = {d.id: d.text[:512] for d in docs}
            scored_ids = [did for did in ce_candidates if did in doc_texts]
            if scored_ids:
                pairs = [[request.query, doc_texts[did]] for did in scored_ids]
                ce_scores_raw = cross_encoder_score(pairs)
                ce_scores_list = [float(s) for s in ce_scores_raw]
                for did, s in zip(scored_ids, ce_scores_list):
                    ce_score_map[did] = s
            if "reranker" not in scores_map:
                scores_map["reranker"] = "bge-reranker-v2-m3"

        # Step 4: Compute combined score for CE candidates
        # OpenClaw: final = 0.6 * ce_normalized + 0.4 * fused_normalized
        ce_candidates_set = set(ce_candidates)
        ce_scores_for_norm = [ce_score_map.get(did, 0.0) for did in ce_candidates]
        fused_scores_for_norm = [fused_scores_map[did] for did in ce_candidates]

        # Min-max normalize both
        def _minmax_norm(vals):
            if not vals:
                return []
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx > mn else 1.0
            return [(v - mn) / rng for v in vals]

        ce_norm = _minmax_norm(ce_scores_for_norm)
        fused_norm = _minmax_norm(fused_scores_for_norm)

        combined_scores = {}
        for i, did in enumerate(ce_candidates):
            ce_n = ce_norm[i] if i < len(ce_norm) else 0.0
            fu_n = fused_norm[i] if i < len(fused_norm) else 0.0
            combined_scores[did] = 0.7 * ce_n + 0.3 * fu_n

        # Step 5: Build final ranking
        # - CE candidates: use combined score
        # - Non-CE candidates: use fused RRF score (normalized to 0-1 within their group)
        # Combine and sort by their respective scores
        all_fused_only = [fused_scores_map[did] for did, _ in fused_ranked_all[CE_POOL_SIZE:]]
        fused_only_norm = _minmax_norm(all_fused_only)

        final_ranking = []
        # Add CE candidates with combined score
        ce_sorted = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        for did, score in ce_sorted:
            final_ranking.append((did, score, "ce"))
            scores_map[f"final:{did}"] = float(score)

        # Add non-CE candidates with normalized fused score
        non_ce = fused_ranked_all[CE_POOL_SIZE:]
        for i, (did, _) in enumerate(non_ce):
            norm_score = fused_only_norm[i] if i < len(fused_only_norm) else 0.0
            final_ranking.append((did, norm_score, "rrf"))
            scores_map[f"rrf:{did}"] = float(norm_score)

        # Sort by score descending
        final_ranking.sort(key=lambda x: x[1], reverse=True)
        unique_doc_ids = [did for did, _, _ in final_ranking[:top_k]]

    # Add bi-encoder scores to map for debugging
    for did, score in zip(bi_encoder_doc_ids[:10], bi_encoder_scores[:10]):
        scores_map[f"bi:{did}"] = score

    return QueryResponse(
        query=request.query,
        doc_ids=unique_doc_ids,
        paths_traversed=paths_traversed,
        scores=scores_map,
    )


# ── Rebuild Tree Endpoint ──

@router.post("/rebuild")
def rebuild_tree(db: Session = Depends(get_db)):
    """
    Rebuild the cluster tree from ALL existing documents.
    Alias for /rebuild_tree.
    """
    return rebuild_tree_endpoint(db)


# ── Build FAISS Index Endpoint ──

@router.post("/build_faiss")
def build_faiss_index(db: Session = Depends(get_db)):
    """
    Build a FAISS ANN index from all document embeddings.
    This is required for efficient retrieval on large datasets (100K+ docs).
    """
    import numpy as np
    from app.ml.faiss_index import build_index, save_index

    # Fetch only id + embedding to avoid loading 512K full texts into RAM (~2GB)
    all_docs = db.query(Document.id, Document.embedding).filter(
        Document.embedding.isnot(None)
    ).all()

    if not all_docs:
        raise HTTPException(status_code=400, detail="No documents with embeddings found.")

    doc_ids = [d.id for d in all_docs]
    embeddings = np.array([d.embedding for d in all_docs], dtype=np.float32)

    logger.info(f"Building FAISS index for {len(doc_ids)} documents...")

    try:
        build_index(embeddings, doc_ids)
        save_index()
        from app.ml.faiss_index import get_index_stats
        stats = get_index_stats()
        return {
            "status": "built",
            "total_documents": len(doc_ids),
            "index_stats": stats,
        }
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to build FAISS index: {e}")


# ── Load FAISS Index Endpoint ──

@router.post("/load_faiss")
def load_faiss_index():
    """Load the FAISS index from disk."""
    from app.ml.faiss_index import load_index, get_index_stats
    if load_index():
        return {"status": "loaded", "index_stats": get_index_stats()}
    else:
        raise HTTPException(status_code=404, detail="FAISS index not found. Build it first with /build_faiss")


@router.post("/rebuild_tree")
def rebuild_tree_endpoint(db: Session = Depends(get_db)):
    """
    Rebuild the cluster tree from ALL existing documents.
    Clears existing tree and triggers a single recursive build.
    """
    doc_ids = [row[0] for row in db.query(Document.id).all()]

    if not doc_ids:
        raise HTTPException(status_code=400, detail="No documents found. Ingest documents first.")

    # Clear existing tree
    db.query(ClusterNode).delete()
    db.query(IngestionJob).delete()
    db.commit()

    # Create ingestion job
    job = IngestionJob(
        id=uuid.uuid4(),
        status="tree_building",
        total_docs=len(doc_ids),
        processed_docs=len(doc_ids),
    )
    db.add(job)
    db.commit()

    # Queue the tree build
    try:
        from app.tasks.celery_app import celery_app
        celery_app.send_task(
            "build_cluster_tree",
            args=["root", doc_ids, 0],
            queue="gpu",
        )
        return {
            "status": "tree_building",
            "total_docs": len(doc_ids),
            "job_id": str(job.id),
            "message": f"Tree rebuild started with {len(doc_ids)} documents.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue tree build: {e}")


# ── Tree Structure Endpoint ──

@router.get("/tree")
def get_tree(db: Session = Depends(get_db)):
    """Get the full cluster tree structure."""
    nodes = db.query(ClusterNode).order_by(ClusterNode.depth, ClusterNode.created_at).all()

    if not nodes:
        return {"nodes": [], "edges": []}

    node_list = []
    edge_list = []

    for node in nodes:
        node_list.append(node.to_dict())
        if node.parent_id:
            edge_list.append({
                "source": str(node.parent_id),
                "target": str(node.id),
            })

    return {"nodes": node_list, "edges": edge_list}


# ── Single Node Endpoint ──

@router.get("/nodes/{node_id}")
def get_node(node_id: str, db: Session = Depends(get_db)):
    """Get a specific node by ID."""
    try:
        nid = uuid.UUID(node_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid node ID format.")

    node = db.query(ClusterNode).filter(ClusterNode.id == nid).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found.")

    return node.to_dict()


# ── Ingestion Job Status ──

@router.get("/ingest/{job_id}")
def get_ingest_status(job_id: str, db: Session = Depends(get_db)):
    """Get ingestion job status."""
    try:
        jid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

    job = db.query(IngestionJob).filter(IngestionJob.id == jid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return job.to_dict()


# ── Statistics ──

@router.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get cluster tree statistics."""
    total_nodes = db.query(ClusterNode).count()
    leaf_nodes = db.query(ClusterNode).filter(ClusterNode.is_leaf == True).count()
    total_docs = db.query(Document).count()
    max_depth_result = db.execute(text("SELECT MAX(depth) FROM cluster_nodes")).scalar()

    return {
        "total_nodes": total_nodes,
        "leaf_nodes": leaf_nodes,
        "internal_nodes": total_nodes - leaf_nodes,
        "total_documents": total_docs,
        "max_depth": max_depth_result or 0,
    }


# ── Document Retrieval ──

@router.get("/documents")
def list_documents(limit: int = Query(default=50, ge=1, le=1000), offset: int = Query(default=0, ge=0), db: Session = Depends(get_db)):
    """List stored documents."""
    docs = db.query(Document).offset(offset).limit(limit).all()
    return [d.to_dict() for d in docs]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get a specific document by ID."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {
        "id": doc.id,
        "text": doc.text,
        "source": doc.source,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }
