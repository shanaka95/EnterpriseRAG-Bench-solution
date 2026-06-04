"""
Document ingestion script.
Reads .txt files from a directory and submits them to the RAG API.
"""
import os
import sys
import json
import argparse
import requests
from pathlib import Path


def load_documents(directory: str, limit: int = None) -> list:
    """Load all .txt files from a directory."""
    docs = []
    dir_path = Path(directory)

    if not dir_path.exists():
        print(f"ERROR: Directory {directory} does not exist.")
        sys.exit(1)

    for filepath in sorted(dir_path.glob("*.txt")):
        doc_id = filepath.stem
        text = filepath.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            docs.append({
                "id": doc_id,
                "text": text,
                "source": str(filepath),
            })

    if limit:
        docs = docs[:limit]

    return docs


def ingest(api_url: str, docs: list, batch_size: int = 50) -> dict:
    """Send documents to the ingestion API in batches."""
    total = len(docs)
    jobs = []

    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        print(f"Sending batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} docs)...")

        response = requests.post(
            f"{api_url}/api/v1/ingest",
            json={"documents": batch},
            headers={"Content-Type": "application/json"},
            timeout=300,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"  Job ID: {result['job_id']}, Status: {result['status']}")
            jobs.append(result)
        else:
            print(f"  ERROR: {response.status_code} - {response.text}")

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG pipeline")
    parser.add_argument("directory", help="Directory containing .txt files")
    parser.add_argument("--api-url", default="http://localhost:8080", help="API base URL")
    parser.add_argument("--batch-size", type=int, default=50, help="Documents per batch")
    parser.add_argument("--limit", type=int, default=None, help="Max documents to ingest")
    args = parser.parse_args()

    print(f"Loading documents from {args.directory}...")
    docs = load_documents(args.directory, limit=args.limit)
    print(f"Found {len(docs)} documents.")

    if not docs:
        print("No documents found. Exiting.")
        sys.exit(1)

    print(f"Ingesting to {args.api_url}...")
    jobs = ingest(args.api_url, docs, batch_size=args.batch_size)
    print(f"\nDone! Created {len(jobs)} ingestion jobs.")


if __name__ == "__main__":
    main()
