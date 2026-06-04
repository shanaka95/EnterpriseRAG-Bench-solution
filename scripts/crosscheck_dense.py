"""Cross-check server vs local dense LanceDB: pick 30 random doc IDs,
compute SHA256(id || float32-bytes(embedding)) on each side, compare."""
import hashlib, random, subprocess
import lancedb
import numpy as np

LOCAL = "/data/projects/rag/data/dense_index/db"
REMOTE_LANCE = "/workspace/lancedb_dense"   # resolved on server
N_SAMPLE = 30


def fp_local(table, doc_id: str) -> str:
    arr = table.search().where(f"id = '{doc_id}'").limit(1).to_arrow().to_pylist()
    if not arr:
        return None
    r = arr[0]
    h = hashlib.sha256()
    h.update(r["id"].encode())
    h.update(np.asarray(r["embedding"], dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def fp_remote(doc_id: str) -> str:
    """Run a single-id SHA on the server via SSH."""
    py = (
        "import lancedb, numpy as np, hashlib, sys\n"
        "t = lancedb.connect('/workspace/lancedb_dense').open_table('documents')\n"
        "r = t.search().where(\"id = '\" + sys.argv[1] + \"'\").limit(1).to_arrow().to_pylist()\n"
        "if not r: print('NONE'); raise SystemExit(0)\n"
        "h = hashlib.sha256()\n"
        "h.update(r[0]['id'].encode())\n"
        "h.update(np.asarray(r[0]['embedding'], dtype=np.float32).tobytes())\n"
        "print(h.hexdigest()[:16])\n"
    )
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", "-p", "31694",
         "root@182.224.239.168", f"/workspace/venv/bin/python -c $'{py}' {doc_id}"],
        capture_output=True, text=True, timeout=30,
    )
    if out.returncode != 0 or not out.stdout.strip() or out.stdout.strip() == "NONE":
        return None
    return out.stdout.strip().splitlines()[-1]


def main():
    t_local = lancedb.connect(LOCAL).open_table("documents")
    n_local = t_local.count_rows()
    print(f"local rows: {n_local:,}")

    # 30 random doc IDs from the local table
    sample = t_local.to_lance().to_table(columns=["id"], limit=n_local).column("id").to_pylist()
    random.seed(20260602)
    ids = random.sample(sample, N_SAMPLE)

    matches = 0
    misses = []
    print(f"\nChecking {N_SAMPLE} doc IDs (server ↔ local) …")
    for did in ids:
        local_h = fp_local(t_local, did)
        remote_h = fp_remote(did)
        ok = local_h == remote_h and local_h is not None
        if ok:
            matches += 1
        else:
            misses.append((did, local_h, remote_h))
        flag = "OK " if ok else "!!"
        print(f"  {flag}  local={local_h}  remote={remote_h}  {did[:80]}")

    print(f"\nResult: {matches}/{N_SAMPLE} byte-identical")
    if misses:
        print("\nMismatches:")
        for did, l, r in misses:
            print(f"  {did}\n    local = {l}\n    remote = {r}")
    if matches == N_SAMPLE:
        print("\nLOCAL DB EXACTLY MATCHES SERVER — no corruption in rsync.")
    else:
        print("\nWARNING: integrity mismatch — investigate before relying on this DB.")


if __name__ == "__main__":
    main()
