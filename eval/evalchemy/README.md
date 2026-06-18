# eval/evalchemy — evalchemy reasoning-eval sbatch runners (Leonardo)

Generic SLURM runners for the evalchemy downstream reasoning evals (**MATH500 / AIME24 / gsm8k**) and
the **pass@k** grid, run on Leonardo under the `evalchemy-marin` conda env. These are the git-tracked,
generalized successors to the ad-hoc copies that lived (untracked) in the experiment workspace
`/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/`.

## Scripts

| Script | Purpose |
| --- | --- |
| `evalchemy_eval.sbatch` | **Generic** MATH500 + gsm8k + AIME24(×10 seeds) runner. Parameterized by `(MODEL_REPO, RUN_NAME, STAGE)`. Computes tensor-parallel size at runtime (**auto-TP**, see below). |
| `delphi_eval.sbatch` | **Thin delphi wrapper** — sets the delphi-specific defaults (delphi_v0 chat-template override + 4k context clamp) and `exec`s the generic runner. Same CLI as the old standalone delphi script. |
| `qwen3_eval.sbatch` | Qwen3 dense instruct ladder baseline runner (own ChatML template, long 32k context, thinking-on). TP defaults to 4 with a `$3` override (all Qwen3 head counts divide by 4). |
| `passatk/passatk_eval.sbatch` | pass@k grid array runner — one array task = one `(model, task)` N=128 pass@k pass, grid-driven (`grid.txt`, type-parameterized: `base` / `qwen3` / `qwen3-base` / `qwen3-moe-base` / `qwen3-moe-thinking`). |
| `passatk/passatk_smoke.sbatch` | pass@k mechanism smoke test (Qwen3-0.6B / MATH500, reduced budget; validates 128-sample generation + monotone pass@k). |
| `passatk/smoke_ext.sbatch` | second pass@k smoke (Qwen3-0.6B-Base / AIME24, no chat template). |

The pass@k scripts invoke `passatk_driver.py`, which lives in the experiment workspace
(`.../experiments/delphi-eval/passatk-sft/passatk_driver.py`) and is referenced by absolute path; it is
not vendored here. The grid file (`grid.txt`) is also experiment-local.

## auto-TP behavior (the key fix)

vLLM requires `num_attention_heads % tensor_parallel_size == 0`. The old standalone scripts hardcoded
`TP_SIZE=2`, which hard-fails for odd-head models (e.g. 9 or 11 heads). `evalchemy_eval.sbatch` now
**computes TP at runtime** after the model is cached:

1. Resolve the cached snapshot via `huggingface_hub.snapshot_download` (offline; no network).
2. Read `num_attention_heads` from the snapshot `config.json` (also checks `text_config` nesting).
   **Hard-fails** with a clear message if no `config.json` / no head count is found — never silently defaults.
3. Pick `TP = largest d in {4,3,2,1}` with `(num_heads % d == 0)` and `d <= GPUS_PER_NODE` (default 4 on A100) —
   i.e. the largest valid divisor, maximizing parallelism.

Verified mapping:

| num_heads | TP | num_heads | TP |
| --- | --- | --- | --- |
| 8  | 4 | 16 | 4 |
| 9  | 3 | 18 | 3 |
| 11 | 1 | 20 | 4 |
| 12 | 4 | 30 | 3 |
| 14 | 2 |    |   |

(18→3 and 30→3 are *more* correct than the grid's old manual "2" — largest valid divisor wins.)
This makes the per-TP copies `delphi_eval_tp{1,3,4}.sbatch` obsolete.

## Environment requirement

- conda env **`evalchemy-marin`** (the env all recent `SCORES.md` rows were produced with), `cd .../code/evalchemy-marin` (the editable-install clone the env points at; the legacy `code/evalchemy` and its `evalchemy-resume-test` worktree were removed 2026-06-18).
- Leonardo `$HOME` is **read-only** from compute and login nodes → the scripts redirect `HOME` +
  flashinfer/triton/inductor/vLLM caches to a writable work-FS `.cache` dir. Without this, no vLLM eval starts.
- `HF_HUB_OFFLINE=1`: repos + datasets must be **pre-cached on the login node** before submit.
- MATH500/AIME24 run through evalchemy `eval.eval` (chat_benchmarks, `--max_tokens` + `--verbosity INFO`
  required); gsm8k runs through the **plain `lm_eval` CLI** (single vLLM engine — avoids the double-engine OOM).

## Leonardo run convention

```bash
# generic / Qwen3
sbatch --job-name="evalchemy-eval-<RUN>" eval/evalchemy/evalchemy_eval.sbatch <MODEL_REPO> <RUN> [sft|rl|base]
sbatch --job-name="qwen3-baseline-<RUN>" eval/evalchemy/qwen3_eval.sbatch <MODEL_REPO> <RUN> [TP]

# delphi (#6279) — same CLI as before; auto-TP + delphi_v0 override + 4k clamp baked into the wrapper
RUN=delphi-9e19-p33m67-coldstart-magpie_lr1e5
sbatch --job-name="delphi-eval-$RUN" eval/evalchemy/delphi_eval.sbatch laion/$RUN $RUN sft

# pass@k grid (concurrency capped at 8)
cd .../experiments/delphi-eval/passatk-sft
sbatch --array=0-$(($(wc -l < grid.txt)-1))%8 \
  /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/eval/evalchemy/passatk/passatk_eval.sbatch
```

Submit from the repo checkout on Leonardo
(`/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent`). The experiment-dir launch path
(`experiments/delphi-eval/delphi_eval.sbatch`) symlinks to the repo wrapper.

## Excluded (left experiment-local, one-off)

- `qwen3_eval_RESUME_32b.sbatch` — hardcoded one-off resume of a specific timed-out job (46879953,
  AIME24 seeds 49/50/51, evalchemy-marin resume clone). Not generic.
- `smoke_math500_marin_limit8.sbatch` — one-off lm-eval-version gate validation pinned to a specific
  delphi model + delphi template. Superseded by the generic runner + the pass@k smokes.
- `delphi_eval_tp{1,3,4}.sbatch` — one-line `TP_SIZE` copies; obsolete now that auto-TP exists. Left in
  the experiment dir (deprecated) until their already-submitted jobs finish.

The protocol details (seeds, decoding params, the cache/TP/template gotchas) live in the experiment
workspace `EVAL_CONVENTION.md`; the codebase/resume-capability map is in
`.claude/projects/evalchemy/evalchemy.md`.
