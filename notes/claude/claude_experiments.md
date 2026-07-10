# Claude experiment relaunch log

Global cross-experiment launch/relaunch journal (referenced by `.claude/ops/experiments/ops.md`). Per-experiment
status/results trackers live under `~/Documents/experiments/active|complete/<name>/`; dated failure/remediation
writeups live under `agent_logs/`. This file records the bare fact of a launch/relaunch — command + job id(s) —
so a later sweep can find "what did we submit and when" without re-deriving it from an sbatch.

## 2026-07-10 — TACC Vista: axolotl Qwen3-32B SFT relaunch (818554 → 819032 → 820376..820379)

**Why:** SFT job 818554 (axolotl-backend Qwen3-32B on `open-thoughts/OpenThoughts-Agent-SFT-10K`, config
`sft/axolotl_configs/qwen3_32b_ot_sft_10k.yaml`) died to an NCCL collective-timeout at step 91/704. Its
auto-queued resume 819032 survived that and ran 4.5h clean to step 40/704, then hit a SLURM-detected **hardware**
failure — `NODE_FAIL`, `srun: error: Node failure on c636-121` — at 05:17 UTC. The 818554→819032 `--max_restarts`
chain was exhausted (no further leg queued), and step 40 < `save_steps: 100` so no checkpoint existed → routine
relaunch from step 0, this time excluding the bad node + widening the self-heal chain. Full incident history:
`agent_logs/2026-07-10_tacc_sft_818554_nccl_timeout.md`.

**Node exclusion mechanism:** `hpc.launch` has **no per-invocation `--exclude`/`--slurm-exclude` CLI passthrough**.
The only mechanism is the per-cluster static `HPC.node_exclusion_list` field in `hpc/hpc.py` (already used this way
for Jupiter's `jpbo-*` bad-node list, with dated comments per addition) — it's forwarded into every generated
sbatch as `#SBATCH --exclude=...`. Added `c636-121` to `vista`'s list, local-clone-edit → commit `4e1f07af` →
push → `git pull` on TACC (no cluster hand-edit). Verified via `--dry_run`: rendered sbatch line
`#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c636-121`.

**Launch command** (`otagent` env — NOT `sft-axolotl`, which lacks the `harbor`/`database` modules the launcher
itself imports; `--conda_env sft-axolotl` still correctly writes `conda activate sft-axolotl` into the generated
training sbatch):
```bash
python -m hpc.launch --job_type sft --sft_backend axolotl \
  --train_config_path sft/axolotl_configs/qwen3_32b_ot_sft_10k.yaml \
  --num_nodes 16 --gpus_per_node 1 --time_limit 48:00:00 \
  --dataset open-thoughts/OpenThoughts-Agent-SFT-10K \
  --role_tag role --content_tag content \
  --conda_env sft-axolotl \
  --max_restarts 3
```
(Matches the 818554/819032 launch's config/geometry/dataset exactly, reconstructed from the rendered
`configs/*_train_config.yaml` + `sbatch/*_sft.sbatch` on TACC since no argv was captured at the time. Two
deltas from before: `--max_restarts 3` (was 1 — the chain that just got exhausted) so the afterany chain now
self-heals up to 3x, and the `c636-121` node exclusion via `hpc/hpc.py`.)

**Result:** experiment dir `experiments/sft__OpenThoughts-Agent-SFT-10K__axolotl-sft-axolotl-pinggy-False_12`.
4 jobs submitted (1 immediate + 3 chained `afterany` resume legs):
- **820376** — RUNNING immediately, 16 nodes, `TimeLimit=2-00:00:00` (48h), `ExcNodeList` includes `c636-121`
  (confirmed via `scontrol show job`), allocated nodes do NOT include it.
- **820377** — PENDING, `Dependency=afterany:820376`
- **820378** — PENDING, `Dependency=afterany:820377`
- **820379** — PENDING, `Dependency=afterany:820378`

Config confirmed current: `liger_fused_linear_cross_entropy/liger_rope/liger_rms_norm/liger_glu_activation: True`,
`auto_resume_from_checkpoints: True` (so a chain leg RESUMES from the latest checkpoint once one exists, not
restart-from-0 — the `42c1f393` passthrough fix), `save_steps: 100`, 7 epochs / 704 steps, ZeRO-3 no-offload,
`hub_model_id: None` (32B sharded path — no direct push; consolidate before upload once complete).

**Watch:** 15/30-min bring-up check (rendezvous clean, first loss line lands) + normal step-progress cadence
against the ~91-steps-in-9h pace observed on the prior attempt.
