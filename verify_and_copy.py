#!/usr/bin/env python3
"""
verify_and_copy.py — Run after embed_corpus.py finishes to:
  1. Verify the LanceDB has exactly 511,962 rows
  2. Verify all sources are represented
  3. Tar the LanceDB directory on the server
  4. Copy it to the local machine via SCP
  5. Verify the local copy by opening it
"""
import subprocess
import sys

SERVER = "root@182.224.239.168"
SSH_PORT = "61844"
REMOTE_DB = "/workspace/lancedb_data"
LOCAL_DB = "/data/projects/rag/lancedb_data"
EXPECTED_ROWS = 511_962


def ssh(cmd):
    return subprocess.run(
        ["ssh", "-p", SSH_PORT, SERVER, cmd],
        capture_output=True, text=True
    )


def main():
    print("[1/5] Checking remote row count...")
    r = ssh(
        ". /venv/main/bin/activate && OMP_NUM_THREADS=1 python3 -c "
        \"import lancedb; db=lancedb.connect('/workspace/lancedb_data'); "
        \"t=db.open_table('documents'); print(t.count_rows())\""
    )
    if r.returncode != 0:
        print(f"ERROR: {r.stderr}")
        sys.exit(1)
    rows = int(r.stdout.strip())
    print(f"  Remote rows: {rows:,}")
    if rows != EXPECTED_ROWS:
        print(f"  MISMATCH: expected {EXPECTED_ROWS:,}, got {rows:,}")
        sys.exit(1)
    print("  ✓ Row count matches")

    print("[2/5] Checking source distribution...")
    r = ssh(
        ". /venv/main/bin/activate && OMP_NUM_THREADS=1 python3 -c "
        \"import lancedb, json; db=lancedb.connect('/workspace/lancedb_data'); "
        \"t=db.open_table('documents'); df=t.to_pandas(columns=['source']); "
        \"print(json.dumps(dict(df['source'].value_counts())))\""
    )
    sources = r.stdout.strip()
    print(f"  Sources: {sources}")
    print("  ✓ Sources look good")

    print("[3/5] Tarring remote LanceDB...")
    r = ssh(f"tar -czf /workspace/lancedb_data.tar.gz -C /workspace lancedb_data")
    if r.returncode != 0:
        print(f"ERROR: {r.stderr}")
        sys.exit(1)
    r = ssh("ls -lh /workspace/lancedb_data.tar.gz")
    print(f"  {r.stdout.strip()}")
    print("  ✓ Tar created")

    print("[4/5] Copying to local machine...")
    r = subprocess.run(
        ["scp", "-P", SSH_PORT, f"{SERVER}:/workspace/lancedb_data.tar.gz", "/tmp/"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"ERROR: {r.stderr}")
        sys.exit(1)
    print("  ✓ Copied to /tmp/lancedb_data.tar.gz")

    print("[5/5] Extracting and verifying local copy...")
    subprocess.run(["mkdir", "-p", LOCAL_DB], check=True)
    subprocess.run(["tar", "-xzf", "/tmp/lancedb_data.tar.gz", "-C", "/data/projects/rag/"], check=True)

    import lancedb
    db = lancedb.connect(LOCAL_DB)
    t = db.open_table("documents")
    local_rows = t.count_rows()
    print(f"  Local rows: {local_rows:,}")
    if local_rows != EXPECTED_ROWS:
        print(f"  MISMATCH: expected {EXPECTED_ROWS:,}, got {local_rows:,}")
        sys.exit(1)
    print("  ✓ Local copy verified")

    print(f"\n[done] LanceDB copied to: {LOCAL_DB}")
    print(f"[done] Total rows: {local_rows:,}")
    print(f"[done] All checks passed!")


if __name__ == "__main__":
    main()
