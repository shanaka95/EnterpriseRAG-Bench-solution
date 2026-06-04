#!/usr/bin/env python3
"""Check if gold doc IDs from failures exist in our DB."""
import csv
import sys
sys.path.insert(0, "/app/backend")

from app.core.database import SessionLocal
from app.models.schemas import Document

db = SessionLocal()

with open("/app/eval_results/retrieval_results.csv") as f:
    rows = list(csv.DictReader(f))

failures = [r for r in rows if float(r["recall_all"]) < 1.0]
print(f"Total failures: {len(failures)}")

missing_in_db = 0
no_embedding = 0
found_with_emb = 0

for r in failures[:100]:
    # Parse expected_doc_ids from the row
    # The CSV doesn't have expected_doc_ids, so we need to load from questions.jsonl
    pass

# Load questions to get expected_doc_ids
import json
questions = {}
with open("/app/questions.jsonl") as f:
    for line in f:
        q = json.loads(line)
        questions[q["question_id"]] = q

checked = 0
for r in failures:
    qid = r["question_id"]
    q = questions.get(qid, {})
    expected = q.get("expected_doc_ids", [])
    if not expected:
        continue

    for gold_id in expected:
        checked += 1
        doc = db.query(Document).filter(Document.id == gold_id).first()
        if not doc:
            missing_in_db += 1
        elif doc.embedding is None or len(doc.embedding) == 0:
            no_embedding += 1
        else:
            found_with_emb += 1

print(f"\nChecked {checked} gold doc references from {len(failures)} failures:")
print(f"  Missing from DB entirely: {missing_in_db}")
print(f"  In DB but no embedding: {no_embedding}")
print(f"  In DB with embedding: {found_with_emb}")

# Also check some specific semantic failures
print("\nSample semantic failure gold docs:")
semantic_fails = [r for r in failures if r["question_type"] == "semantic"]
for r in semantic_fails[:5]:
    q = questions.get(r["question_id"], {})
    expected = q.get("expected_doc_ids", [])
    gold_id = expected[0] if expected else "NONE"
    doc = db.query(Document).filter(Document.id == gold_id).first() if gold_id != "NONE" else None
    status = "MISSING" if not doc else ("NO_EMB" if not doc.embedding else "OK")
    print(f"  {r['question_id']}: gold={gold_id[:40]}... status={status}")
    if doc:
        print(f"    source: {doc.source[:60] if doc.source else 'N/A'}")

db.close()
