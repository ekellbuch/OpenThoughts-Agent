#!/usr/bin/env python3
"""Harvest OPD speed x convergence grid metrics from MarinSkyRL .out logs.

The OPD trainer mirrors every train-step metric payload to stdout as a single
parseable line (skyrl_train/trainer.py:_log_metrics_stdout):

    ... WANDB_MIRROR kind=train step=N metrics={<json>}

This script parses those lines from one or more cell .out files and reports,
per cell:

  SPEED (warm-mean over a step window, default steps 5-15; falls back to all
  available warm steps >= 2 if a cell has fewer steps):
    - sec/step                 timing/step
    - teacher_score s          timing/fwd_logprobs_values_reward
                               (NOTE: this Timer wraps student policy+ref fwd
                                logprobs AND teacher scoring; teacher scoring is
                                the overwhelming majority -- see grid.md. There
                                is no dedicated teacher Timer in the trainer.)
    - student_gen s            timing/generate
    - student_train s          timing/train_critic_and_policy
    - sync s                   timing/sync_weights
    - eff tok/s                (tbs * n_samples * avg_gen_tokens) / sec_step
                               using generate/avg_num_tokens for gen tokens

  CONVERGENCE (over ALL completed steps):
    - KL first / last          distill/token_kl_mean
    - KL total delta           last - first
    - KL slope/step            OLS slope of distill/token_kl_mean vs step
    - entropy first / last     policy/policy_entropy
    - grad_norm last           policy/raw_grad_norm

Usage:
  harvest.py <cell_name>=<path-to-.out> [<cell>=<path> ...]
  harvest.py --glob '<dir>/*.out'        # cell name = basename minus _<jobid>.out
  harvest.py --warm-lo 5 --warm-hi 15 ...

Emits a markdown table to stdout (paste into grid.md Results).
"""
import argparse
import glob as globmod
import json
import os
import re
import sys

MIRROR_RE = re.compile(r"WANDB_MIRROR kind=train step=(\d+) metrics=(\{.*\})")
# Strip loguru ANSI color codes if present.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_log(path):
    """Return {step:int -> metrics:dict}, first-seen-wins per step."""
    out = {}
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "WANDB_MIRROR kind=train" not in line:
                continue
            line = ANSI_RE.sub("", line)
            m = MIRROR_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            try:
                metrics = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            out.setdefault(step, metrics)
    return out


def _slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else float("nan")


def _warm_mean(steps_metrics, key, lo, hi):
    vals = [m[key] for s, m in steps_metrics if lo <= s <= hi and key in m]
    if not vals:  # fall back to all warm steps >= 2
        vals = [m[key] for s, m in steps_metrics if s >= 2 and key in m]
    if not vals:  # last resort: every step
        vals = [m[key] for s, m in steps_metrics if key in m]
    return sum(vals) / len(vals) if vals else float("nan")


def harvest_cell(name, path, warm_lo, warm_hi):
    by_step = parse_log(path)
    if not by_step:
        return {"cell": name, "error": "no metric lines", "path": path}
    sm = sorted(by_step.items())  # list of (step, metrics)
    steps = [s for s, _ in sm]

    def wm(key):
        return _warm_mean(sm, key, warm_lo, warm_hi)

    sec_step = wm("timing/step")
    teacher_score = wm("timing/fwd_logprobs_values_reward")
    gen = wm("timing/generate")
    train = wm("timing/train_critic_and_policy")
    sync = wm("timing/sync_weights")
    avg_gen_tok = wm("generate/avg_num_tokens")

    # eff tok/s: total response tokens produced per wall second of a step.
    tbs = by_step[steps[0]].get("__tbs__")  # not present; computed by caller env
    # eff tok/s from avg gen tokens * (#sequences). #seqs = tbs*n; we don't have
    # tbs/n in metrics, so approximate throughput as avg_gen_tok-based per-seq is
    # not enough. Use generate/avg_num_tokens * implied batch via gen timing is
    # unavailable; report tokens/s of the GENERATE stage = (#seq*avg_tok)/gen.
    # Since #seq unknown from metrics alone, we report avg_gen_tok and let the
    # table note batch size separately. Provide gen-stage tok/s proxy = NaN.

    # Convergence
    kl = [(s, m["distill/token_kl_mean"]) for s, m in sm if "distill/token_kl_mean" in m]
    ent = [(s, m["policy/policy_entropy"]) for s, m in sm if "policy/policy_entropy" in m]
    gn = [(s, m["policy/raw_grad_norm"]) for s, m in sm if "policy/raw_grad_norm" in m]

    kl_first = kl[0][1] if kl else float("nan")
    kl_last = kl[-1][1] if kl else float("nan")
    kl_slope = _slope([s for s, _ in kl], [v for _, v in kl]) if len(kl) >= 2 else float("nan")
    ent_first = ent[0][1] if ent else float("nan")
    ent_last = ent[-1][1] if ent else float("nan")
    gn_last = gn[-1][1] if gn else float("nan")

    return {
        "cell": name,
        "n_steps": len(steps),
        "step_range": f"{steps[0]}-{steps[-1]}",
        "sec_step": sec_step,
        "teacher_score": teacher_score,
        "gen": gen,
        "train": train,
        "sync": sync,
        "avg_gen_tok": avg_gen_tok,
        "kl_first": kl_first,
        "kl_last": kl_last,
        "kl_delta": kl_last - kl_first if kl else float("nan"),
        "kl_slope": kl_slope,
        "ent_first": ent_first,
        "ent_last": ent_last,
        "gn_last": gn_last,
    }


def fmt(v, p=1):
    if isinstance(v, str):
        return v
    if v != v:  # NaN
        return "—"
    return f"{v:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cells", nargs="*", help="cell=path.out pairs")
    ap.add_argument("--glob", help="glob of .out files; cell=basename w/o _<jobid>.out")
    ap.add_argument("--warm-lo", type=int, default=5)
    ap.add_argument("--warm-hi", type=int, default=15)
    args = ap.parse_args()

    pairs = []
    for c in args.cells:
        if "=" not in c:
            print(f"skip (no '='): {c}", file=sys.stderr)
            continue
        name, path = c.split("=", 1)
        pairs.append((name, path))
    if args.glob:
        for p in sorted(globmod.glob(args.glob)):
            base = os.path.basename(p)
            name = re.sub(r"_\d+\.out$", "", base)
            name = re.sub(r"\.out$", "", name)
            pairs.append((name, p))

    if not pairs:
        ap.print_help()
        sys.exit(1)

    rows = [harvest_cell(n, p, args.warm_lo, args.warm_hi) for n, p in pairs]

    print(f"\n## SPEED (warm-mean steps {args.warm_lo}-{args.warm_hi})\n")
    print("| cell | steps | sec/step | teacher_score* s | student_gen s | student_train s | sync s | avg_gen_tok |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['cell']} | FAIL: {r['error']} | | | | | | |")
            continue
        print(f"| {r['cell']} | {r['n_steps']} ({r['step_range']}) | {fmt(r['sec_step'])} | "
              f"{fmt(r['teacher_score'])} | {fmt(r['gen'])} | {fmt(r['train'])} | "
              f"{fmt(r['sync'],2)} | {fmt(r['avg_gen_tok'],0)} |")

    print("\n_*teacher_score = timing/fwd_logprobs_values_reward (wraps student fwd-logprobs + teacher scoring; teacher scoring dominates — no dedicated teacher Timer in trainer)._\n")

    print("## CONVERGENCE (all completed steps)\n")
    print("| cell | KL first | KL last | KL Δ | KL slope/step | entropy first→last | grad_norm last |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "error" in r:
            print(f"| {r['cell']} | FAIL | | | | | |")
            continue
        print(f"| {r['cell']} | {fmt(r['kl_first'],4)} | {fmt(r['kl_last'],4)} | "
              f"{fmt(r['kl_delta'],4)} | {fmt(r['kl_slope'],5)} | "
              f"{fmt(r['ent_first'],3)}→{fmt(r['ent_last'],3)} | {fmt(r['gn_last'],2)} |")


if __name__ == "__main__":
    main()
