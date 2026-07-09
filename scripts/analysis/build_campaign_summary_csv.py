#!/usr/bin/env python3
"""Build a per-dataset summary CSV for a datagen (trajectory-generation) campaign.

Columns: Datagen Model | Task Source | Status | N Trials Completed |
         Mean Turns / Trace | Mean Tok / Trace | Mean Reward | HF Repo Link

Metrics are computed from the uploaded HF trace datasets, reusing the canonical
OT-Agent analysis helpers (scripts/analysis/utils.py) + the Qwen3-8B tokenizer:
  N Trials Completed  = uploaded productive rows (one row = one trace)
  Mean Turns / Trace  = mean count_turns(row)  (total conversation messages)
  Mean Tok / Trace    = mean Qwen3-8B plain token length of extract_conversation_text(row)
  Mean Reward         = Harbor-flat mean of `result` (missing/non-numeric -> 0.0),
                        matching harbor's `Mean:`; NULL if 0% of rows have a numeric
                        result (no verifier field -> a flat 0.0 would be fabricated)

Status is HF-ground-truthed: probe the repo; exists-with-rows -> COMPLETED, else the
tracker status hint (RUNNING / FAILED / NOT STARTED).

Reads are STREAMING (disk-bounded) with a pyarrow-direct fallback for repos whose
schema the HF `datasets` reader can't cast (e.g. an extra `trace_source` column).

This file is populated for the qwen3.5-122b-tt (32k) campaign. Adapt DATASETS /
PENDING / OUT_CSV / DATAGEN_MODEL for another campaign (fill from its tracker.md).
See the skill `analyze-datagen-campaign-summary`.
"""
import os, sys, csv, json
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from scripts.analysis.utils import extract_conversation_text, count_turns, extract_reward  # noqa: E402

# ---- campaign config (EDIT for a new campaign) ----------------------------
DATAGEN_MODEL = "Qwen3.5-122B-A10B-FP8"
TOK_NAME = "Qwen/Qwen3-8B"
OUT_CSV = os.path.expanduser("~/Documents/experiments/complete/qwen3.5-122b-tt/completed_datasets_summary.csv")
CKPT = os.path.join(os.path.dirname(OUT_CSV), ".completed_datasets_ckpt.jsonl")
MAX_WORKERS = 5

# (idx, task_source, candidate_hf_repo_or_None, status_hint, note)
# Copy the exact repo slug from the tracker (the slug transform is irregular).
DATASETS = [
    (1, "DCAgent/inferredbugs-sandboxes-verifier", "penfever/inferredbugs-sandboxes-verifier-qwen3.5-122b-32k-traces", "COMPLETED", "S1 rescue of hung job"),
    (2, "DCAgent/code-contests-noblock", "penfever/code-contests-noblock-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (3, "SankalpKJ/nemotron-code-oracle-filtered", "penfever/nemotron-code-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~90%"),
    (4, "DCAgent/llm-verifier-freelancer", "penfever/llm-verifier-freelancer-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~28% (host-RAM OOM)"),
    (5, "laion/exp_rpt_methods2test-large-v3", "penfever/methods2test-large-v3-qwen3.5-122b-32k-traces", "COMPLETED", "partial"),
    (6, "laion/exp_rpt_stack-junit-v6", "penfever/stack-junit-v6-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~42%"),
    (7, "DCAgent2/nl2bash-tasks-cleaned-oracle", "penfever/nl2bash-tasks-cleaned-oracle-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (8, "DCAgent/exp_rpt_curriculum-easy", "penfever/curriculum-easy-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (9, "DCAgent/exp_rpt_e2egit-v2", "penfever/e2egit-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (10, "DCAgent/exp_rpt_e2egit-large", "penfever/e2egit-large-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (11, "DCAgent/exp_rpt_nemotron-junit", "penfever/nemotron-junit-qwen3.5-122b-32k-traces", "COMPLETED", "rescued"),
    (12, "DCAgent/exp_rpt_pymethods2test-v3", "penfever/pymethods2test-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (13, "DCAgent/exp_rpt_unitsyn-python-v3", "penfever/unitsyn-python-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (14, "DCAgent/exp_rpt_unitsyn-python-large", "penfever/unitsyn-python-large-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (15, "laion/exp_rpt_ghactions-v3", "penfever/ghactions-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (16, "laion/nemotron-gym-instruction-following-structured", "penfever/nemotron-gym-if-structured-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (17, "laion/nemotron-gym-agent-calendar", "penfever/nemotron-gym-agent-calendar-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (18, "laion/exp_rpt_crosscodeeval-csharp-v4", "penfever/crosscodeeval-csharp-v4-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (19, "laion/nemotron-gym-knowledge-web-search-mcqa", "penfever/nemotron-gym-knowledge-web-search-mcqa-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (20, "laion/nemotron-gym-knowledge-mcqa", "penfever/nemotron-gym-knowledge-mcqa-qwen3.5-122b-32k-traces", "COMPLETED", "partial (marathon pool)"),
    (21, "laion/nemotron-gym-agent-workplace-v2", "penfever/nemotron-gym-agent-workplace-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (22, "laion/nemotron-gym-identity-following-v2", "penfever/nemotron-gym-identity-following-v2-qwen3.5-122b-32k-traces", "COMPLETED", "partial"),
    (23, "laion/nemotron-gym-knowledge-openqa-v2", "penfever/nemotron-gym-knowledge-openqa-v2-qwen3.5-122b-32k-traces", "COMPLETED", "partial"),
    (24, "laion/nemotron-gym-safety-v2", "penfever/nemotron-gym-safety-v2-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~22%"),
    (25, "laion/nemotron-gym-math-advanced-calculations-v3", "penfever/nemotron-gym-math-advanced-calculations-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (26, "SankalpKJ/nemotron-math-oracle-filtered", "penfever/nemotron-math-oracle-filtered-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~14%"),
    (27, "DCAgent/selfinstruct-naive-sandboxes-2-verified", "penfever/selfinstruct-naive-sandboxes-2-verified-qwen3.5-122b-32k-traces", "COMPLETED", "partial; ~6% RewardFileNotFound"),
    (28, "DCAgent/mix_h2_language_proportional", "penfever/mix-h2-language-proportional-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (29, "DCAgent/mix_h4_binary_easy", "penfever/mix-h4-binary-easy-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (30, "DCAgent/exp_rpt_pymethods2test-large", "penfever/pymethods2test-large-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (31, "laion/swegym-tasks-patched-validated-v2", None, "FAILED", "blocked/skipped (906 unique envs, snapshot-incompatible)"),
    (32, "laion/exp_rpt_stack-bash-v3", "penfever/stack-bash-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (33, "laion/exp_rpt_methods2test-large-v2", "penfever/methods2test-large-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (34, "laion/exp_rpt_codenet-python-v2", "penfever/codenet-python-v2-qwen3.5-122b-32k-traces", "COMPLETED", "QUALITY-SUSPECT: broken stdin harness, reward ~meaningless"),
    (35, "DCAgent/exp_rpt_curriculum-medium", "penfever/curriculum-medium-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (36, "DCAgent/exp_rpt_nemotron-cpp", None, "FAILED", "killed, broken verifier, NOT rescued"),
    (37, "DCAgent/exp_rpt_pr", "penfever/exp-rpt-pr-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (38, "DCAgent/exp_rpt_stack-pytest-large", "penfever/stack-pytest-large-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (39, "DCAgent/exp_rpt_stack-pytest-v2", "penfever/stack-pytest-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (40, "laion/mix_h1_struggle_zone-v2", "penfever/mix-h1-struggle-zone-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (41, "laion/mix_h2_language_balanced-v2", "penfever/mix-h2-language-balanced-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (42, "laion/mix_h8_original_tests-v2", "penfever/mix-h8-original-tests-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (43, "laion/mix_h10_reward_binary-v2", "penfever/mix-h10-reward-binary-v2-qwen3.5-122b-32k-traces", "COMPLETED", "partial ~69% (zombie kill-rescue)"),
    (44, "laion/mix_h10_reward_proportional-v2", "penfever/mix-h10-reward-proportional-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (45, "laion/mix_h10_reward_staged-v2", "penfever/mix-h10-reward-staged-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (46, "laion/mix_h11_single_skill_only-v2", "penfever/mix-h11-single-skill-only-v2-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (47, "laion/exp_rle_minimal_instructions-v3", "penfever/exp-rle-minimal-instructions-v3-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (48, "laion/nemotron-gym-instruction-following-calendar", "penfever/nemotron-gym-if-calendar-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (49, "laion/nemotron-gym-competitive-coding", "penfever/nemotron-gym-competitive-coding-qwen3.5-122b-32k-traces", "COMPLETED", ""),
    (50, "laion/nemotron-gym-instruction-following-v2", "penfever/nemotron-gym-if-v2-qwen3.5-122b-32k-traces", "RUNNING", "was running at campaign close"),
    (51, "SankalpKJ/swesmith-oracle-filtered", "penfever/swesmith-oracle-filtered-qwen3.5-122b-32k-traces", "RUNNING", "was running at campaign close; reward-dead SWE"),
    (52, "DCAgent/swe_rebench_patched_oracle", "penfever/swe-rebench-patched-oracle-qwen3.5-122b-32k-traces", "RUNNING", "reward-dead SWE; empty 0-row stub repo"),
]
PENDING = [
    "DCAgent/r2egym-patched-full-oracle", "DCAgent/mix_h6_test_quality_top25", "laion/exp_rpt_scaffold-v2",
    "DCAgent/exp_rpt_crosscodeeval-java", "laion/exp_rle_heavy_padding-v2", "laion/exp_flat25_speed_bonus-v2",
    "laion/exp_flat25_pseudocode-v2", "laion/exp_flat25_stackoverflow-v2", "laion/openswe-tasks-patched-v5-oracle-success",
    "laion/exp_rpt_stack-go-v4", "DCAgent/exp_rpt_curriculum-hard", "DCAgent/exp_rpt_issue", "DCAgent/exp_rpt_multifile",
    "DCAgent/exp_rpt_stack-dockerfile-v2", "DCAgent/exp_rpt_stack-jest-v2", "DCAgent/exp_rpt_stack-jest-large",
    "laion/exp_flat25_subtle_debug-v3", "laion/exp_rle_detailed-v3", "laion/mix_baseline_uniform-v2",
    "laion/mix_h5_skill_diverse-v2", "laion/mix_h7_raw_volume_5k-v2", "laion/mix_h8_adversarial_tests-v2",
    "laion/mix_h11_compositional_gradient-v2", "laion/exp_rle_error_report-v3", "laion/exp_rle_github_issue-v3",
    "laion/exp_rpt_defects4j-v3-v4", "laion/nemotron-gym-math-stack-overflow", "laion/nemotron-gym-math-openmathreasoning",
    "laion/exp_rpt_stack-php-v2-v6", "laion/exp_rpt_stack-php-large-v6", "laion/nemotron-gym-instruction-following-adversarial-v3",
    "DCAgent/swe_rebench_v2_patched_oracle", "laion/freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2",
    "laion/freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2", "laion/exp_rpt_taco-v2", "laion/exp_rpt_stack-bash-withtests-v2",
    "laion/exp_rpt_exercism-python-v2", "laion/exp_rpt_stack-ruby-v2", "laion/exp_rpt_stack-dockerfile-gpt5mini-v3",
    "DCAgent/exp_rle_adversarial", "laion/exp_rpt_stack-csharp-v5", "laion/exp_rpt_stack-bash-withtests-gpt5mini-v2",
    "laion/exp_rpt_pr-v2", "laion/exp_rpt_stack-rust-v2",
]
for _i, _ts in enumerate(PENDING, start=53):
    DATASETS.append((_i, _ts, None, "NOT STARTED", ""))

# ---------------------------------------------------------------------------
_TOK = None
def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOK_NAME, trust_remote_code=True)
    return _TOK


def _iter_rows(repo):
    """Yield rows; stream first, fall back to pyarrow-direct on a schema CastError."""
    from datasets import load_dataset
    try:
        for row in load_dataset(repo, split="train", streaming=True):
            yield row
        return
    except Exception:
        pass  # schema cast / streaming failure -> pyarrow-direct read below
    from huggingface_hub import HfApi, HfFileSystem
    import pyarrow.parquet as pq
    api, fs = HfApi(), HfFileSystem()
    files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    for path in files:
        with fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            for bt in pq.ParquetFile(fh).iter_batches(batch_size=256,
                                                      columns=["conversations", "result"]):
                for row in bt.to_pylist():
                    yield row


def compute_one(entry):
    idx, task_source, repo, hint, note = entry
    base = dict(idx=idx, model=DATAGEN_MODEL, task_source=task_source, status=hint,
                n_trials=None, mean_turns=None, mean_tok=None, mean_reward=None,
                hf_link="NULL", note=note)
    if not repo:
        return base
    from huggingface_hub import HfApi
    try:
        HfApi().dataset_info(repo)
    except Exception:
        return base  # repo absent -> keep tracker hint
    tok = _tok()
    n = turns = num = 0
    rew_sum = tok_sum = 0.0
    batch = []
    def flush():
        nonlocal tok_sum, batch
        if batch:
            tok_sum += sum(len(x) for x in tok(batch, add_special_tokens=False)["input_ids"])
            batch = []
    for row in _iter_rows(repo):
        n += 1
        turns += count_turns(row)
        r = extract_reward(row)
        if r is not None:
            num += 1
            rew_sum += r
        batch.append(extract_conversation_text(row))
        if len(batch) >= 128:
            flush()
    flush()
    if n == 0:
        return base  # empty/stub repo -> keep hint
    base.update(status="COMPLETED", n_trials=n,
                mean_turns=round(turns / n, 2), mean_tok=round(tok_sum / n, 1),
                # Harbor-flat (missing=0); NULL when NO row has a numeric reward
                mean_reward=(round(rew_sum / n, 4) if num > 0 else None),
                hf_link=f"https://huggingface.co/datasets/{repo}")
    if num == 0:
        base["note"] = (note + " | no verifier reward field (0% numeric result)").strip(" |")
    return base


def main():
    done = {}
    if os.path.exists(CKPT):
        for line in open(CKPT):
            if line.strip():
                r = json.loads(line)
                done[r["idx"]] = r
    todo = [e for e in DATASETS if e[0] not in done]
    print(f"total={len(DATASETS)} done={len(done)} todo={len(todo)}", flush=True)
    results = dict(done)
    with open(CKPT, "a") as ck, ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for fut in as_completed({ex.submit(compute_one, e): e[0] for e in todo}):
            r = fut.result()
            results[r["idx"]] = r
            ck.write(json.dumps(r) + "\n"); ck.flush()
            print(f"[{r['idx']:>2}] {r['status']:<11} n={r['n_trials']} turns={r['mean_turns']} "
                  f"tok={r['mean_tok']} rew={r['mean_reward']} {r['task_source']}", flush=True)
    cols = ["Datagen Model", "Task Source", "Status", "N Trials Completed",
            "Mean Turns / Trace", "Mean Tok / Trace", "Mean Reward", "HF Repo Link"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for idx in sorted(results):
            r = results[idx]
            g = lambda x: "" if x is None else x
            w.writerow([r["model"], r["task_source"], r["status"], g(r["n_trials"]),
                        g(r["mean_turns"]), g(r["mean_tok"]), g(r["mean_reward"]), r["hf_link"]])
    print(f"WROTE {OUT_CSV} ({len(results)} rows)", flush=True)


if __name__ == "__main__":
    main()
