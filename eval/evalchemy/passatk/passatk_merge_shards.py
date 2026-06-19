#!/usr/bin/env python3
"""Merge passatk SHARD files (passatk_shard_off*_lim*.json) for ONE cell into the whole-cell
passatk_results.json the harvest reads. Each shard carries per-problem correct `counts` (and
`flex_counts` for gsm8k) over a problem slice; the union of shards must cover all problems with no
overlap. Recomputes the unbiased pass@k estimator over the concatenated per-problem counts — i.e.
byte-identical math to a single whole-cell run, just generated in wall-bound pieces.

Usage:
  python passatk_merge_shards.py --out <CELL_OUT_DIR> --task gsm8k --model-repo <repo> \
      [--expect-problems 1319]
"""
import argparse, glob, json, os
import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    out = 1.0
    for i in range(n - c + 1, n + 1):
        out *= (1.0 - k / i)
    return 1.0 - out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="cell output dir holding the passatk_shard_*.json")
    ap.add_argument("--task", required=True)
    ap.add_argument("--model-repo", required=True)
    ap.add_argument("--expect-problems", type=int, default=0)
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(os.path.join(args.out, "passatk_shard_off*_lim*.json")))
    if not shard_paths:
        raise SystemExit(f"no shards in {args.out}")
    shards = [json.load(open(p)) for p in shard_paths]
    # order by offset, verify contiguous non-overlapping coverage from 0
    shards.sort(key=lambda s: s["offset"])
    n = shards[0]["n_samples"]
    cursor = 0
    counts, flex_counts = [], []
    for s in shards:
        assert s["n_samples"] == n, "n_samples mismatch across shards"
        assert s["offset"] == cursor, f"gap/overlap: expected offset {cursor}, got {s['offset']}"
        counts.extend(s["counts"])
        if args.task == "gsm8k":
            flex_counts.extend(s["flex_counts"])
        cursor += s["n_problems"]
    n_problems = len(counts)
    if args.expect_problems:
        assert n_problems == args.expect_problems, \
            f"coverage {n_problems} != expected {args.expect_problems}"

    ks = [1, 8, 32, 128]
    res = {"task": args.task, "model_repo": args.model_repo, "n_samples": n,
           "n_problems": n_problems, "merged_from_shards": [os.path.basename(p) for p in shard_paths]}
    for k in ks:
        if k <= n:
            res[f"pass@{k}"] = float(np.mean([pass_at_k(n, c, k) for c in counts]))
    if args.task == "gsm8k":
        for k in ks:
            if k <= n:
                res[f"pass@{k}_flex"] = float(np.mean([pass_at_k(n, c, k) for c in flex_counts]))

    with open(os.path.join(args.out, "passatk_results.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("MERGED " + json.dumps(res))


if __name__ == "__main__":
    main()
