#!/usr/bin/env python3
"""Evaluate every checkpoint of the finetuning run on the 470-question benchmark.

Runs eval_ft_ckpt.py once per checkpoint dir under the experiment output dir
(skipping any whose summary already exists), then prints a comparison table.

Usage:
  sweep_eval.py [--exp-dir /workspace/rag5k/output/ft5k_lr1e-5_bs8x1_ctx2048_3ep]
                [--base]   # also (re-)eval codefuse-ai/F2LLM-v2-80M as 'base'
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ft_ckpt.py")
OUT_DIR = "/workspace/rag5k/eval_out"
KS_REPORT = [1, 5, 10, 20, 100]


def ckpt_sort_key(name: str) -> tuple:
    m = re.match(r"step_(\d+)", name)
    if m:
        return (0, int(m.group(1)))
    m = re.match(r"epoch_(\d+)", name)
    if m:
        return (1, int(m.group(1)))
    return (2, 0)


def run_eval(model: str, tag: str) -> None:
    cmd = [sys.executable, EVAL_SCRIPT, "--model", model, "--tag", tag]
    print(f"=== {tag} <- {model} ===", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir",
                    default="/workspace/rag5k/output/ft5k_lr1e-5_bs8x1_ctx2048_3ep")
    ap.add_argument("--base", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    jobs: list[tuple[str, str]] = []  # (model_path, tag)
    if args.base:
        jobs.append(("codefuse-ai/F2LLM-v2-80M", "base"))
    ckpts = sorted(
        (d for d in os.listdir(args.exp_dir)
         if re.match(r"(step|epoch)_\d+$", d)
         and os.path.isfile(os.path.join(args.exp_dir, d, "config.json"))),
        key=ckpt_sort_key)
    for c in ckpts:
        jobs.append((os.path.join(args.exp_dir, c), c))

    for model, tag in jobs:
        if os.path.isfile(f"{OUT_DIR}/{tag}_summary.json"):
            print(f"--- {tag}: summary exists, skipping", flush=True)
            continue
        run_eval(model, tag)

    # comparison table
    rows = []
    for model, tag in jobs:
        sp = f"{OUT_DIR}/{tag}_summary.json"
        if not os.path.isfile(sp):
            continue
        s = json.load(open(sp))
        row = {"tag": tag, "mrr": round(100 * s["mrr"], 2)}
        for k in KS_REPORT:
            row[f"hit@{k}"] = round(s["hit@K"][str(k)]["pct"], 2)
        rows.append(row)

    def best_first(r: dict) -> tuple:
        m = re.match(r"step_(\d+)", r["tag"])
        return (0, int(m.group(1))) if m else (1, int(re.match(r"epoch_(\d+)", r["tag"]).group(1)))

    rows.sort(key=best_first)
    cols = ["tag", "mrr", *[f"hit@{k}" for k in KS_REPORT]]
    csv_path = f"{OUT_DIR}/sweep_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    best = max(rows, key=lambda r: r["hit@10"])
    print(f"\nbest by hit@10: {best['tag']}  "
          f"(hit@1={best['hit@1']} hit@10={best['hit@10']} hit@100={best['hit@100']}, mrr={best['mrr']})")
    print(f"table -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
