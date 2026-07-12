# Test-suite reorganization — staged plan (SCOPING; propose-only, no code yet)

- **Date:** 2026-07-12
- **Status:** `scoped — propose-only; no code migrated yet` (gated on operator review)
- **Target repo:** `OpenThoughts-Agent` · canonical clone `/Users/benjaminfeuer/Documents/OpenThoughts-Agent` · branch `penfever/working`
- **Isolated working copy (for execution):** `/Users/benjaminfeuer/Documents/staged-work/test-suite-reorg/OpenThoughts-Agent` (rsync-clone + cut `feuer/test-suite-reorg` there — NOT on the canonical clone; see code-create-staged-plan §"Isolated working copy")
- **This doc:** parent plan + inline per-stage scopes (single-doc form; the suite change is mechanical enough not to warrant per-stage split files).
- **Pairs with:** `code-execute-staged-plan` (runs the stages gate-by-gate).

---

## Goal (testable end state)

1. **One `tests/` tree** holds every genuine, repo-level automated test, organized by subsystem, with the infra-dependency axis carried by **pytest markers** (not directories).
2. **The suite is AWARE of the infra tests:** `pytest --collect-only` (no filter) discovers *all* migrated tests including the GPU/cluster/creds ones; CI stays green by *selecting* the CI-safe subset with a marker filter (`-m "not gpu and not cluster and ..."`). Nothing is deleted.
3. **Shared primitives + conventions:** a top-level `tests/conftest.py` provides skip-guards (`skip_if_no_gpu`, env-gated creds) and extends the existing in-process Daytona/Harbor mocks, so a new test "just fits."
4. **Terminal-Bench task assets stay put** (they are in-container benchmark fixtures, not repo tests) and are explicitly excluded from collection.
5. **Ad-hoc CLI harnesses named `test_*` are moved out of the test namespace** (renamed / relocated to `scripts/`) so they never mis-collect.

**Non-goals (this scope):** changing any test's assertion logic; reformatting the whole repo; making infra tests runnable in CI. Execution is a separate, operator-gated phase.

---

## Inventory + taxonomy (git-tracked, as of 2026-07-12)

**60 files** match a naming heuristic; after removing false positives (below), **50 are real test files**: 22 already in `tests/`, 28 elsewhere.

> **False positives** (naive `grep test_` hits — NOT tests, leave alone): the 8 `data/patchers/patch_*2test_*_tasks.py` patcher scripts and `scripts/analysis/filter_latest_episodes.py` (substring `latest_` → `test_`). These are dataset patchers / analysis scripts.

### A. In `tests/` today — 22 files / 271 tests — **CI-safe unit (default marker)**
`tests/hpc/` (16) + `tests/eval/` (6). Pure unit + import-only: mock Daytona/Harbor in-process (e.g. `FakeDaytona` injected into `hpc/snapshot_manager.py`), FastAPI test clients, no GPU/torch/vllm, no cluster, no creds. This is exactly what CI runs today (`pytest tests/ -q` = 271 passed). **No move.**

### B. Genuine pytest/unittest suites OUTSIDE `tests/` — 7 files — **should migrate**
| File | Tests | Framework | Infra need | Note |
|---|---|---|---|---|
| `data/nemotron_gym/tests/test_adapter.py` | 34 | pytest | **CI-safe** | uses **relative import** `from ..adapter` — must become absolute `from data.nemotron_gym.adapter import …` on move |
| `data/nl2bash_sampled_verified/tests/test_sampler.py` | 9 | unittest | CI-safe | pure (inline skill-detect) |
| `data/nl2bash_sampled_verified/tests/test_validation.py` | 7 | unittest | CI-safe | pure |
| `data/nl2bash_sampled_verified/tests/test_failure_categorization.py` | 11 | unittest | CI-safe | pure |
| `data/nl2bash_sampled_verified/tests/test_output_normalization.py` | 13 | unittest | CI-safe | "daytona" is only test-fixture *string data*, not a dep |
| `data/negotiation/test_generate.py` | 28 | unittest | CI-safe* | lazy-imports `generate` (guards pyarrow); verifier formula inlined. *Confirm no network at collect; else mark `hf`/`net`. |
| `eval/tests/test_model_registry_resolve.py` | (script; 0 `def test_`) | prints PASS/FAIL, `sys.exit` | CI-safe | self-contained registry resolver test; **rewrite to pytest asserts** on move (currently not pytest-shaped) |

### C. Ad-hoc CLI / repro scripts named `test_*` — **NOT pytest; relocate out of test namespace** — 11 files
`main()`/argparse-driven, print-based, or live-service smoke harnesses:
- `data/all_puzzles/test_generate.py`, `data/perturbed_docker/test_generate.py`, `data/ta_rl_tasks/test_generate.py`, `data/taskmaster2/test_generate.py`, `data/trace_hints/test_generate.py` — "generate N tasks and eyeball" scripts (pull HF datasets); **0 `def test_`**.
- `data/bespoke/test_converted_tasks.py` — validates an on-disk converted-task dir; **0 `def test_`** (a `validate_task()` walker + `main()`).
- `data/freelancer_filtering/test_filters.py` — has 6 `def test_` but print-based + live LLM/network + `sys.path` hacks.
- `rl/hpc/test_hpc.py` — 4 `def test_`, but `main()`/subprocess `--dry-run`, cluster-detection; manual smoke.
- `scripts/datagen/test_reasoning_content.py` — argparse; live LiteLLM/OpenAI call.
- `scripts/beam/test_beta9_sandbox.py` — argparse; live Beta9/Beam cluster (Beam was shelved 2026-05-21).
- `scripts/s3/test_bucket.py` — live S3/MinIO creds smoke.

**Disposition:** these belong in `scripts/` (or a `scripts/manual/` / `data/<ds>/tools/`), **renamed off the `test_` prefix** (e.g. `smoke_generate.py`, `check_bucket.py`) so pytest never collects them. `rl/hpc/test_hpc.py`, `data/freelancer_filtering/test_filters.py`, and `data/bespoke/test_converted_tasks.py` are **borderline** — they *could* be refactored into real marked pytest tests (`cluster` / `net` / local-data), but that is test-*authoring*, not migration. Operator decision (see Risks).

### D. Terminal-Bench task assets — **DO NOT move; exclude from collection** — 10 files
Carry the `terminal-bench-canary GUID` header and assert against in-container paths (`/app`, `/logs/verifier`). They are copied into the Daytona/Docker container by the Harbor harness and run *there* — they are benchmark ground-truth, not repo tests:
- `data/self_instruct/few-shots/*/tests/test_outputs.py` (8: build-cython-ext, chess-best-move, configure-git-webserver, fix-code-vulnerability, polyglot-c-py, qemu-alpine-ssh, qemu-startup, regex-log)
- `data/self_instruct/few-shots/hello-world/tests/test_state.py`
- `data/frontiersmith/templates/test_state.py`

**Taxonomy totals:** 22 CI-safe (in place) · 7 migrate (6 CI-safe after import/format fixes + up-to-1 needing a marker) · 11 ad-hoc-scripts-not-tests · 10 TB task assets. (28 outside `tests/` = 7 + 11 + 10.)

---

## Target layout — subsystem subdirs + markers (minimize churn)

**Chosen: keep the flat `tests/<subsystem>/` layout and carry the infra-need axis with markers.**

```
tests/
  conftest.py            # NEW — shared fixtures, skip-guards, mock re-exports (Stage 0)
  hpc/                   # unchanged (16 files)
  eval/                  # unchanged (6) + eval/tests/test_model_registry_resolve.py → here
  data/
    nemotron_gym/test_adapter.py
    nl2bash/test_sampler.py test_validation.py test_output_normalization.py test_failure_categorization.py
    negotiation/test_generate.py
  # infra-marked tests live alongside their subsystem too; the MARKER, not a
  # dir, says what they need.
```

**Why this over `tests/{unit,integration,gpu,cluster}/`:** the "needs GPU / cluster / creds" property is *orthogonal* and composable (a test can need both `hf` and `slow`) — that is exactly what markers express and what a directory cannot. Subsystem dirs mirror the installed package under test (`data`, `eval`, `hpc`), keeping import context obvious and discovery intuitive, and — decisively — **zero churn to the 22 green tests** already in `tests/hpc` + `tests/eval`. A need-based tree would force-move those 22 and still need markers for the compound cases. Directory = "what code does this test?"; marker = "what infra does it need?".

**File → new-home map:** see the per-stage change-sets below (Stages 2–4).

---

## Marker taxonomy (register in `pyproject.toml`)

Proposed `[tool.pytest.ini_options]` block (ready to paste — Stage 0; **no-op** because CI keeps invoking `pytest tests/` with no `-m` until Stage 5):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
# TB task assets + few-shot fixtures are in-container benchmark ground-truth,
# never repo tests — keep pytest from ever recursing into them.
norecursedirs = ["data/*/few-shots", "data/*/templates", "*.egg-info", ".git"]
markers = [
    "gpu: needs a CUDA/accelerator device (torch/vllm)",
    "cluster: needs a live SLURM or iris/CoreWeave cluster (ssh/squeue/sbatch)",
    "daytona: needs live Daytona sandbox API + key",
    "hf: needs HuggingFace hub access (dataset/model download or token)",
    "supabase: needs Supabase registry creds",
    "net: needs outbound network / a live LLM or 3rd-party API (LiteLLM/OpenAI/S3)",
    "slow: long-running (minutes+); excluded from the fast CI lane",
    "integration: multi-component end-to-end (implies at least one of the above)",
]
```

- **Default (unmarked) = CI-safe unit.** Every test in group A and the migrated group-B tests stay unmarked.
- **CI selection (green lane), Stage 5:**
  `pytest -m "not gpu and not cluster and not daytona and not hf and not supabase and not net and not slow and not integration"`
- **Awareness:** `pytest --collect-only` (no `-m`) lists ALL tests incl. infra ones → the catalog is executable, not just prose.
- **Engineer run matrix (documented in `tests/README.md`, Stage 5):**

| Want to run | Command | Needs |
|---|---|---|
| Fast CI lane | `make test` → the `-m "not …"` line above | nothing (stock env) |
| A GPU suite | `pytest -m gpu` | CUDA node + `[rl]`/`[datagen]` extras |
| Cluster suite | `pytest -m cluster` | ssh creds to Leonardo/iris; run *on* login node |
| Daytona | `pytest -m daytona` | `DAYTONA_API_KEY` (+ org keys) |
| HF / creds | `pytest -m hf` / `-m supabase` / `-m net` | `HF_TOKEN` / `SUPABASE_*` / provider keys in `secrets.env` |
| Everything (see the map) | `pytest --collect-only` | nothing (collection only) |

`make` targets (`test`, `test-gpu`, `test-cluster`, `test-all-collect`) land in Stage 5.

---

## Shared primitives / conventions (`tests/conftest.py` — Stage 0)

Design (scaffold content proposed in Stage 0; no behavior change for the current green suite because it only *adds* fixtures/guards, marks nothing):

- **Skip-guards** (env/hardware-gated auto-skips so an infra test SKIPs, not ERRORs, off-target):
  - `skip_if_no_gpu` — `pytest.mark.skipif` on `torch.cuda.is_available()` unimportable/False.
  - `skip_if_no_cluster` — skip unless `$OT_CLUSTER` / known login hostname present.
  - `require_env("HF_TOKEN")`, `require_env("DAYTONA_API_KEY")`, `require_env("SUPABASE_URL")` — parametrizable env-gate returning a skip marker.
  - A `conftest` hook that auto-applies the right `skipif` to any test carrying `gpu`/`cluster`/`daytona`/`hf`/`supabase`/`net` **when its resource is absent** — so `pytest -m gpu` on a laptop skips cleanly.
- **Mock re-exports:** promote the existing in-process `FakeDaytona` factory + Harbor stubs (currently duplicated inline in `tests/hpc/test_snapshot_manager.py` et al.) into `tests/_fixtures/` fixtures (`fake_daytona`, `harbor_stub`) so migrated data-layer tests reuse them instead of re-rolling. (Re-export only in Stage 0; refactor call-sites is optional later — no forced churn.)
- **Naming/format conventions** (documented, matching current idiom + Harbor's `tests/`): files `test_<unit>.py`; functions `test_<behavior>`; module docstring stating *what is mocked and how to run*; absolute imports (`from data.… import …`, never `from ..x`); ruff-clean (tests/ is ruff-gated at 0.15.14); no top-level heavy imports (`torch`/`vllm`) in unmarked tests — import inside the test body under the marker.

---

## Stage map (dependency-ordered; each has a GO/NO-GO gate)

| Stage | Title | What | Move? | Cost | Gate (GO) |
|---|---|---|---|---|---|
| **0** | Marker + conftest scaffold | Add `[tool.pytest.ini_options]` markers/`norecursedirs`/`testpaths`; add `tests/conftest.py` skip-guards + mock re-exports; add `tests/README.md` stub + Makefile targets | none | CPU, ~1h | `pytest tests/ -q` still **271 passed**; `--collect-only` count unchanged; **no** "unknown marker" warnings; `ruff check tests/` clean. CI command unchanged ⇒ stays green. |
| **1** | Catalog / decision ledger | Commit a tracked `tests/TESTS.md` classifying **every** git-tracked `test_*`/`*_test.py` → {CI-safe / marker-X / TB-asset / ad-hoc-script} with target home. This IS the "awareness" artifact until moves land. | none (doc) | CPU, ~1h | Ledger row exists for all 50 real files + the 10 false-positive/asset callouts; peer/operator sign-off on each C/B disposition. |
| **2** | Migrate CI-safe unit suites | Move the 6 CI-safe group-B suites into `tests/data/…` (+ registry test → `tests/eval/`); fix relative→absolute imports; rewrite `test_model_registry_resolve.py` to pytest asserts | git-mv + import edits | CPU, ~3h | Each moved file **COLLECTS + PASSES** under the CI filter; total green rises (≈271 → ≈370); `ruff check/format tests/` clean; old paths gone (no dup collection). |
| **3** | Migrate + mark infra pytest tests | For any group-B/borderline test kept as pytest that needs infra, move under subsystem dir + add its marker + wire the Stage-0 skip-guard | git-mv + mark | CPU, ~2h | `pytest --collect-only` shows them (awareness); CI filter **excludes** them (green count unchanged from Stage 2); off-target run SKIPs (not ERRORs). |
| **4** | Retire ad-hoc scripts from test namespace | `git mv` the 11 group-C scripts to `scripts/…`/`data/<ds>/tools/`, **rename off `test_`**; confirm TB assets (group D) excluded via `norecursedirs` | git-mv + rename | CPU, ~2h | No `test_*`/`*_test.py` left in the repo that is a CLI harness or TB asset (audit script passes); `pytest --collect-only` = only genuine tests; CI green. |
| **5** | Flip CI to marker filter + docs | Change `pytest.yml` to the `-m "not …"` lane; add `--collect-only` awareness step; widen ruff to moved files; finalize `tests/README.md` + `make` targets | CI + docs | CPU, ~1h | CI green on the marker filter; a `collect-only` CI step lists ALL tests; `make test-gpu`/`-cluster` documented; ruff green on the widened path. |

**Critical path:** 0 → 1 → 2 → 5. Stages 3 and 4 are independent of 2 (can interleave) but both assume 0's markers/`norecursedirs`.

Each stage = one commit (or a small batch) on `feuer/test-suite-reorg`, pushed; other clones `git pull` (no cluster patching — these are pure-python editable installs, live after pull).

---

## Global invariants (assert EVERY stage)

- **Flag-off / no-op until Stage 5:** through Stages 0–4 the CI invocation stays `pytest tests/ -q`; adding markers, conftest, `norecursedirs`, and *moving files into `tests/`* must keep it green (moves only ever ADD to the green count). The behavior flip is isolated to Stage 5's one-line CI change.
- **Awareness monotonic:** after each move, `pytest --collect-only` count = previous + moved (no silent drops, no double-collection from a leftover path).
- **No assertion-logic edits.** Only: `git mv`, import-path fixes (relative→absolute), `test_model_registry_resolve.py` script→pytest reshape (Stage 2, behavior-preserving), marker decorators, and renames of group-C files. Any file needing a real logic fix is quarantined (see Risks) — not silently rewritten.
- **Minimal diff / ruff-clean:** every file that lands under `tests/` must pass the pinned ruff gate; TB assets and ad-hoc scripts never enter `tests/`.

---

## Borrow map (code anchors — reconfirm at exec time; dated read 2026-07-12)

- CI: `.github/workflows/pytest.yml` (runs `pytest tests/ -q`), `.github/workflows/ruff-format.yml` (`uvx ruff@0.15.14 check/format tests/`). Stage 5 edits pytest.yml only.
- `pyproject.toml`: **no `[tool.pytest.ini_options]` today** (Stage 0 adds it); `[tool.setuptools.packages.find]` (lines ~212–231) already installs `data`, `eval`, `hpc`, `rl`, `scripts` as packages ⇒ absolute imports from moved tests resolve. No `[tool.ruff]` config exists (ruff runs default rules on `tests/`).
- Mock idiom to promote: `tests/hpc/test_snapshot_manager.py` (injected `FakeDaytona`, patched `_discover_hash_to_env_dir`); FastAPI/`pytest.mark.parametrize` usage in `tests/hpc/test_ingress_sidecar.py`, `tests/eval/test_iris_thinking_kwarg.py`.
- Import-fix target: `data/nemotron_gym/tests/test_adapter.py` line ~21 `from ..adapter import (…)` → `from data.nemotron_gym.adapter import (…)`.

---

## Risks / decisions for the operator

1. **Ad-hoc-script vs real-test (group C, 11 files).** Recommendation: **relocate + rename off `test_`** (Stage 4) rather than delete — they are useful manual smokes. Decide per borderline file whether `rl/hpc/test_hpc.py`, `data/freelancer_filtering/test_filters.py`, `data/bespoke/test_converted_tasks.py` are worth *upgrading* into real marked pytest tests (`cluster`/`net`/local-data) vs just relocating. Upgrading = new test authoring, out of this migration's scope — flag if wanted as a follow-up.
2. **Latent-bug / ruff-red files.** The repo has ~90 pre-existing ruff findings incl. undefined-name bugs; a scattered `test_*` that is syntactically/undefined-name broken **cannot even collect**. Any group-B/C file that fails `ruff check` or import at move time is **quarantined** (left in place, NOT moved, logged in `tests/TESTS.md` with the failure) — never silently "fixed" inside a migration commit. Operator decides fix-now vs quarantine-and-file-issue. (Screen each candidate with `ruff check <file>` + `python -c "import ast; ast.parse(open(f).read())"` in Stage 1.)
3. **Import-path / conftest implications of moving.** `from ..adapter` breaks once out of the `data.nemotron_gym` package → rewrite to absolute (works, `data` is installed). `unittest.TestCase` files collect fine under pytest as-is. A new top-level `tests/conftest.py` is auto-loaded by pytest for the whole tree — a stray import error in it would break the *entire* green suite, so Stage 0's conftest must be import-safe (lazy/guarded imports of torch etc.) and is gated by re-running the 271-suite.
4. **TB task assets must never collect.** `norecursedirs` (Stage 0) plus the "no move" rule protects them; an audit in Stage 4 asserts none leaked. If Harbor ever runs these *in-container* via its own pytest, that path is unaffected (different rootdir).
5. **`negotiation/test_generate.py` (28 tests).** Confirm at Stage 2 it imports/collects with no network (lazy `generate` import suggests yes); if any test hits HF/network, mark it `hf`/`net` and it moves to Stage 3 instead. Low-risk either way.
6. **Scope discipline:** this plan does **not** widen CI to run infra tests, reformat the repo, or fix product code — only relocates/marks tests. Executing it is gated on operator approval (`code-execute-staged-plan`).

---

## Proposed no-op scaffold (NOT applied — paste at Stage 0)

The marker/`ini_options` block above and a conftest sketch below are provided so Stage 0 is turn-key. **Neither is committed by this scoping doc** — they are applied under the Stage-0 gate.

```python
# tests/conftest.py  (Stage 0 — additive only; import-safe)
import os, importlib.util, pytest

def _has(mod):   # torch/vllm present?
    return importlib.util.find_spec(mod) is not None

def _gpu():
    if not _has("torch"): return False
    import torch; return torch.cuda.is_available()

_GUARD = {
    "gpu":      (lambda: not _gpu(),                    "no CUDA device"),
    "cluster":  (lambda: not os.getenv("OT_CLUSTER"),   "no live cluster ($OT_CLUSTER)"),
    "daytona":  (lambda: not os.getenv("DAYTONA_API_KEY"), "no DAYTONA_API_KEY"),
    "hf":       (lambda: not (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")), "no HF token"),
    "supabase": (lambda: not os.getenv("SUPABASE_URL"), "no SUPABASE_URL"),
    "net":      (lambda: os.getenv("OT_NO_NET") == "1", "network disabled (OT_NO_NET=1)"),
}

def pytest_collection_modifyitems(config, items):
    for item in items:
        for name, (missing, why) in _GUARD.items():
            if item.get_closest_marker(name) and missing():
                item.add_marker(pytest.mark.skip(reason=f"{name}: {why}"))
```
