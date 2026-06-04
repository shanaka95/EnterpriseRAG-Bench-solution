"""
Verify int8 quantization fidelity for the ColBERT reranker.

For 3 random docs:
  1. Re-encode locally on CPU (raw float32).
  2. Quantize to int8 with the same per-doc scale scheme used by the
     server pipeline.
  3. Compare quantized vs raw MaxSim against a fixed query.
  4. Assert relative score drift < 1%.

Run from /data/projects/rag:
  ./backend/venv/bin/python scripts/check_quant_fidelity.py
"""
from __future__ import annotations

import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.ml.colbert_reranker import (  # noqa: E402
    EMB_DIM,
    _fetch_doc_embeddings,
    _maxsim_padded,
    get_model,
    get_table,
)


def quantize(vecs: np.ndarray) -> tuple[float, np.ndarray]:
    if vecs.size == 0:
        return 1.0, np.zeros(vecs.shape, dtype=np.int8)
    max_abs = float(np.max(np.abs(vecs)))
    if max_abs == 0.0:
        return 1.0, np.zeros(vecs.shape, dtype=np.int8)
    scale = max_abs / 127.0
    q = np.clip(np.round(vecs / scale), -128, 127).astype(np.int8)
    return scale, q


def main():
    table = get_table()
    n = table.count_rows()
    print(f"LanceDB rows: {n}")

    # Sample 3 random ids
    sample = table.to_lance().to_table(columns=["id"], limit=2000)
    all_ids = sample.column("id").to_pylist()
    random.seed(7)
    picked = random.sample(all_ids, 3)

    query = "incident escalation rollback support coverage"
    model = get_model()
    print(f"Query: {query!r}\n")

    # Encode query once
    q_out = model.encode(
        sentences=[query], is_query=True, batch_size=1,
        show_progress_bar=False, convert_to_numpy=True,
        normalize_embeddings=True,
    )
    q_vecs = (q_out[0] if isinstance(q_out, list) else q_out).astype(np.float32)
    print(f"Query tokens: {q_vecs.shape[0]}, dim: {q_vecs.shape[1]}")

    # Pull stored (quantized) embeddings from LanceDB
    docs_q = _fetch_doc_embeddings(picked)

    # Locate raw text for re-encoding via the corpus directory
    corpus_root = "/data/projects/rag/data/all_documents"
    drift_pct = []
    for did in picked:
        path = os.path.join(corpus_root, did)
        if not os.path.exists(path):
            print(f"  skip (file missing): {did}")
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        # Re-encode raw
        d_out = model.encode(
            sentences=[text], is_query=False, batch_size=1,
            show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        d_raw = (d_out[0] if isinstance(d_out, list) else d_out).astype(np.float32)

        # Round-trip via quantization
        scale, qd = quantize(d_raw)
        d_dequant = qd.astype(np.float32) * scale

        s_raw = float(_maxsim_padded(q_vecs, [d_raw])[0])
        s_deq = float(_maxsim_padded(q_vecs, [d_dequant])[0])
        s_stored = float(_maxsim_padded(q_vecs, [docs_q[did]])[0]) if did in docs_q else float("nan")
        drift = abs(s_deq - s_raw) / max(abs(s_raw), 1e-9) * 100
        drift_pct.append(drift)
        print(f"  {did}")
        print(f"    raw     MaxSim: {s_raw:.4f}")
        print(f"    dequant MaxSim: {s_deq:.4f}  (drift {drift:.3f}%)")
        print(f"    stored  MaxSim: {s_stored:.4f}  (drift vs raw: "
              f"{abs(s_stored - s_raw)/max(abs(s_raw),1e-9)*100:.3f}%)")

    if drift_pct:
        avg = sum(drift_pct) / len(drift_pct)
        print(f"\navg quantization drift (round-trip): {avg:.3f}%")
        assert avg < 1.0, f"FAIL: quantization drift {avg:.3f}% > 1%"
    print("\n=== fidelity OK ===")


if __name__ == "__main__":
    main()
