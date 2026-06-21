# 2026-06-20 — Delphi #6279 RL math-cell entropy-explosion hparam re-tune (Leonardo)

## Context
5/6 newly-processed Delphi #6279 math RL cells DIVERGED during GRPO: `policy/policy_entropy`
EXPLODED ~0.13 → ~11.5, reward → −1.0 / pass@8 → 0 by step ~100 (D1 rlvr_math, D3 dapo_math,
D4 math500). Only D2/ifeval healthy. Failure mode = entropy EXPLOSION (NOT low-entropy collapse).
Task: design a stabilization sweep (NO KL) on max_grad_norm clip + lr + entropy_coef (swept axis),
launch a first wave on Leonardo. Source data: main_rl_evals/{SCORES.md, grid.md},
agent_logs/2026-06-19_delphi_rl_six_cells.md.

## Prior art examined
- **gsm8k accuracy_grid** (`~/Documents/experiments/gsm8k_grid_leonardo/accuracy_grid.md`): lr dominant;
  **lr1e-5 → entropy 0.04 (near-collapse)**, **lr3e-5 UNSTABLE — grad spike 1.95, rejected**, **lr3e-6
  healthy (entropy 0.115, pass@8 0.760, conservative runner-up)**. `entbonus` (use_entropy_loss=true
  coef +0.01) = the anti-COLLAPSE stabilizer there. Knob names confirmed from accuracy_grid_cells.txt.
- **gsm8k throughput grid**: layout winners (4×TP1 colocate, cudagraph on, gmu0.85) — inherited unchanged.
- **ablation_exploration_in_rl**: lrboost / seqnorm-tis-shaped / explore-tis-* confirm entropy dynamics
  are the live RL-stability axis; TIS/seqnorm reserved as a KL-free fallback trust-region tool.
- **MarinSkyRL config** (`ppo_base_config.yaml`): `trainer.policy.optimizer_config.max_grad_norm`
  (default 1.0); `trainer.algorithm.use_entropy_loss` / `entropy_loss_coef` (default 0.01); ZClip
  (`trainer.algorithm.z_clip.enabled`) adaptive grad clip available as a reserve.
- **Entropy sign** (`workers/worker.py:1045,1068-1070`): `loss = policy_loss + kl_loss_term −
  entropy_loss_term`; `entropy_loss_term = entropy × entropy_loss_coef`. POSITIVE coef = entropy BONUS
  (pushes entropy UP). → the gold +0.01 was driving the explosion; NEGATIVE coef = entropy penalty (damps).

## Ground-truth fix
The Delphi RL run scripts (`run_delphi_math_rl.sh`, `sbatch_delphi_math_rl.sh`, `rl_dataset_prep.py`)
were UNTRACKED on Leonardo only — never committed (violates the local-clone-is-SoT rule). Pulled all 3
into the local repo (`hpc/skyrl_yaml/leonardo/`) for tracking. No edit needed for the sweep: the sbatch
already forwards dotted hydra args through `"$@"`, so lr / max_grad_norm / entropy override on the
command line (last-wins). Ckpts already correctly write to `$WORK/rl_ckpts` (WRITE-PATH MANDATE OK).

## Design
Testbed = `laion/delphi-1e22-p33m67-...-wc386k_lr1e5-sft` (MATH500 45.0) × D1 rlvr_math (clean math
collapser; staged). 8-cell cross of max_grad_norm {1.0, 0.5, 0.2} × lr {1e-5, 3e-6} × entropy
{+0.01, 0, −0.001, off}, designed as isolating corners (full table + rationale in
`main_rl_evals/stabilization_grid.md`). Lead candidate gc05_lr3e6_e0 (clip 0.5 + lr 3e-6 + entropy push OFF).

## Launch log
**Wave 1 submitted 2026-06-20** (testbed = p33m67 wc386k SFT × D1 rlvr_math; 1 node × 4 A100 each,
normal QOS ≤24h; ckpts → $WORK/rl_ckpts/stab-*):
- **47447528** rl_stab-gc05_lr3e6_e0 — max_grad_norm=0.5, lr=3e-6, use_entropy_loss=false  (lead candidate)
- **47447530** rl_stab-gc02_lr3e6_e0 — max_grad_norm=0.2, lr=3e-6, use_entropy_loss=false  (hardest brake)
- **47447531** rl_stab-gc05_lr1e5_e0 — max_grad_norm=0.5, lr=1e-5, use_entropy_loss=false  (clip-only on hot lr)

All 3 PENDING(Priority) at submit — Leonardo had 9 RUNNING (8 delphi-evals + 1 passatk array), 0 other
pending; the 3 stab cells are the only new pending. They start as eval nodes free. Live count after submit:
9 RUNNING + 3 PENDING.

## Wave 2 (launch as Wave 1 frees nodes — see stabilization_grid.md)
gc05_lr3e6_eneg (clip0.5/lr3e6/entropy −0.001 penalty), gc02_lr1e5_e0 (clip0.2/hot lr), gc05_lr3e6_ekeep
(clip0.5/lr3e6/keep +0.01 — isolates the bonus), gc1_lr3e6_e0 (orig clip 1.0/lr3e6 — isolates the clip
lever), + baseline control (clip1.0/lr1e5/+0.01 — confirm explosion reproduces).
Reserve axis if all diverge: ZClip (z_clip.enabled=true) and/or DAPO eps_clip_high=0.28.

## Wave-1 HARVEST — gc05_lr3e6_e0 (47447528) COMPLETED step 101, exit 0  [2026-06-20]
**VERDICT: the lead candidate WORKS — entropy bounded + reward genuinely learned.**
Full 100-step train trajectory parsed from `…/rl_stab-gc05_lr3e6_e0_47447528.out` (100 WANDB_MIRROR lines):
- **Entropy BOUNDED** — `policy/policy_entropy` stayed in a tight 0.103–0.139 band the entire run (first 0.108
  → last 0.116, mean 0.120). NO runaway whatsoever (vs the baseline's 0.13→~11.5 explosion). The clip 0.5 +
  lr 3e-6 + entropy-push-OFF recipe fully tames the instability. (Note: supervisor spot-check cited max ~0.274;
  the train-stream `policy/policy_entropy` max is 0.139 — either window/metric difference; verdict unchanged: bounded.)
- **Reward IMPROVED (learned, not stable-flat)** — `reward/avg_raw_reward` −0.89→−0.09 (lead10 −0.61 → trail10
  −0.06); `reward/avg_pass_at_8` 0.25→0.61 (lead10 0.49 → trail10 0.68, peak 0.80 @ step ~? mid-run). Shape =
  fast climb in the first ~10 steps then a healthy plateau around pass@8 0.65–0.70. So this is NOT the
  stability-but-flat tradeoff — it stabilized AND moved the metric. (Still net-negative raw_reward because the
  reward is centered/shaped negative on this clean held-out math; the trend + pass@8 are the signal.)
- **grad_norm** tiny throughout (0.05–0.13, well under the 0.5 clip — the clip rarely even bound), confirming
  the lr 3e-6 alone keeps updates small; the tight clip is a cheap insurance, not the active brake here
  (→ exactly what gc1_lr3e6_e0 will test).

**Checkpoint preserved (HF-only, NO DB — #6279 guardrail):** step-101 ckpt is FSDP2 SHARDED_STATE_DICT
(`model_world_size_4_rank_{0..3}.pt`, DTensor Shard(dim=0)) + a `huggingface/` subdir (config/tokenizer/
chat_template/generation_config). Consolidated on a debug node inside the marinskyrl sandbox via
`$SF/consolidate_ckpt.py` (load 4 ranks → `torch.cat(local_tensors, dim=shard_dim)` per key → load into a
meta-init `Qwen3ForCausalLM` from the bundled config → `save_pretrained` safetensors). **missing=0,
unexpected=0** (incl. untied `lm_head.weight`; config `tie_word_embeddings=false`). 9.71B params / 19.43 GB bf16,
4 safetensors shards. Export at `$SF/hf_export/stab-gc05_lr3e6_e0_step101`. Uploaded PUBLIC to
**`laion/delphi-1e22-p33m67-wc386k_sft-rl-D1rlvrmath-stab-gc05lr3e6-101-9_7B`** via the Leonardo sbatch-tunnel
(login-100s-killer path; job 47469261). Arch = Qwen3 config w/ llama3 rope, 37 layers, hidden 3840.

## Wave-2 launch (after 47447528 freed its node) — 2 highest-information ISOLATORS
Now that the winner is validated with entropy OFF, the next-most-valuable cells decompose WHICH lever did it:
- **47469268** rl_stab-gc05_lr3e6_ekeep — max_grad_norm=0.5, lr=3e-6, **use_entropy_loss=true coef=+0.01**.
  Winner-identical EXCEPT keep the original +0.01 bonus → isolates whether the +0.01 bonus alone was the
  explosion driver (if this one explodes despite the tight clip+conservative lr, the bonus is confirmed guilty).
- **47469269** rl_stab-gc1_lr3e6_e0 — max_grad_norm=**1.0** (orig), lr=3e-6, use_entropy_loss=false.
  Winner-identical EXCEPT the ORIGINAL clip 1.0 → isolates whether tightening the clip was even necessary
  (grad_norm never exceeded 0.13 in the winner, so this likely also stabilizes → tightening may be optional).
Both PENDING(Priority) at submit. Considerate scheduler: only 2 queued; still RUNNING = gc02_lr3e6_e0 (47447530)
+ gc05_lr1e5_e0 (47447531) + passatk-moefast (47437951_2). Live stab count after submit: 2 RUNNING + 2 PENDING.
Deferred to a later wave: gc05_lr3e6_eneg (−0.001 penalty), gc02_lr1e5_e0, baseline control.

## Notes / unresolved
- Entropy negative-coef arm (gc05_lr3e6_eneg) is mechanistically sound (sign verified) but UNTESTED in this
  codebase — held to Wave 2 so the no-push (off) arms report first.
- The 3 untracked Delphi scripts are now pulled to local; need a commit+push to make local the SoT (the
  Leonardo copies are byte-identical to what was launched, so no re-pull needed for these runs).

## HARVEST 530/531 + D3/D4 requeue diagnose+relaunch (2026-06-21)

### (A) Harvested the 2 completed Wave-1 stab cells — 531 (lr1e-5) is the NEW BEST
Both COMPLETED step 101, exit 0; full 100-step trajectory parsed from each .out (entropy/raw_reward/pass@8).
- **47447530 gc02_lr3e6_e0** (clip0.2): entropy **bounded** 0.101–0.139 (first 0.108→last 0.113, mean 0.119);
  raw_reward −0.90→−0.14 (trail10 −0.05, max +0.13); pass@8 0.27→0.67 (trail10 0.70, peak 0.81, mean 0.65).
  **~TIES the lead 528** — the hardest clip (0.2 vs 0.5) is immaterial; lr does the work. Dominated → NOT uploaded.
- **47447531 gc05_lr1e5_e0** ⭐ (clip0.5, **lr1e-5**): entropy **bounded** 0.065–0.264 (first 0.108→last 0.078,
  mean 0.0995 — mild decline, NO runaway, NO collapse); **raw_reward −0.89→+0.18 (trail10 +0.24, peak +0.40)**;
  **pass@8 0.25→0.69 (trail10 0.75, peak 0.875, mean 0.708)**. ⭐ **NEW BEST RECIPE — BEATS prior winner 528**
  (lr3e-6: trail10 raw_reward −0.06, pass@8 0.68): the hot **lr1e-5 is SAFE once entropy-push is OFF + clip0.5**
  (the explosion was the +0.01 entropy BONUS, NOT the lr) AND it lifts reward into POSITIVE territory at higher
  pass@8. → **candidate to promote the production default lr3e-6 → lr1e-5.** (Note: the run-script GOLD defaults
  emit lr1e-5/ent+0.01 first, but the trailing CLI overrides `max_grad_norm=0.5 lr=1.0e-5 use_entropy_loss=false`
  last-win in Hydra — so this cell ran exactly clip0.5/lr1e-5/ent-off as intended.)
- **HF upload of 531:** consolidated step-101 (job 47476493, missing/unexpected=0, 410 tensors, Qwen3 37L, ~19 GB)
  → `$SF/hf_export/stab-gc05_lr1e5_e0_step101` → target `laion/delphi-1e22-p33m67-wc386k_sft-rl-D1rlvrmath-stab-gc05lr1e5-101-9_7B`.
  **Upload BLOCKED on EXPIRED step-ca SSH cert** (leonardo_daytona-cert expired 2026-06-20T23:32:51; the 528
  upload at 21:54 was inside the window). Needs supervisor browser-SSO `step ssh certificate` regen; then
  `sbatch upload_stab_gc05lr1e5.sbatch` finishes it. 530 NOT uploaded (dominated by 531).

### (B) Diagnosed + relaunched the 4 FAILED D3/D4 requeue cells (47470111-114)
**Root cause (all 4, identical):** `OSError: AF_UNIX path length cannot exceed 107 bytes` at Ray plasma_store
socket creation — `$WORK/jobtmp/<id>/ray/ray/session_<ts>_<pid>/sockets/plasma_store` = 122 B. The dotenv's LONG
`RAY_TMPDIR=$WORK/jobtmp/$ID/ray` leaked into the container because `singularity exec` runs WITHOUT `--cleanenv`.
The 3 W1 D1 cells survived only because they were submitted from a shell without the dotenv sourced (Ray fell back
to short `/tmp/ray`). NOT data/path/arg related (dapo_math + math500 parquets + all 3 SFT bases verified present).
**Fix:** (1) relaunched all 4 from a clean submit env (`env -u RAY_TMPDIR -u TMPDIR -u OT_JOB_TMP sbatch …`),
fresh `fix2-` RUN_NAME → **47476407/408/409/410 ALL RUNNING, Ray init OK, 0 AF_UNIX errors, past the 3min crash
point**; (2) durable LOCAL commit **c76af3f2** (NOT pushed) pins `RAY_TMPDIR=/tmp/ray_$SLURM_JOB_ID` (~79 B) +
passes it/TMPDIR in the singularity `--env` block so the leak can't recur. ckpts → `$WORK/rl_ckpts/fix2-…` (verified $WORK).

### Live queue at handoff (do-not-disturb cells untouched)
RUNNING: stab ekeep 47469268, gc1 47469269; D1 fix 47469765/766/767; passatk 47437951_2;
+ the 4 NEW D3/D4 relaunches 47476407-410. No DB writes. No git push. No cluster git pull.

## HARVEST: 3 D1 fix cells + 2 ablation isolators (2026-06-21, cert valid → 07:27 CEST)
All 5 target cells COMPLETED step 101, exit 0 (off-queue). Only the 4 D3/D4 fix2 cells (47476407-410)
still RUNNING (~5:53h) — untouched. Trajectories parsed from each `.out`
(`hpc/skyrl_yaml/leonardo/<name>_<jobid>.out`): entropy from the WANDB_MIRROR train stream,
reward/pass@8 from the `postprocess_generator_output` log line, 100 steps each.

### (A) The 3 D1 fix-recipe production REQUEUE cells — ALL CLEAN, ALL UPLOADED ✅
Recipe `max_grad_norm=0.5 · lr=3.0e-6 · use_entropy_loss=false · NO KL` (CLI dotted-hydra overrides).
**Verdict: the fix fully tames the D1 explosion in all 3 mixes** (vs the orig 0.13→~11.5 runaway):
- **47469765 p33m67-D1**: entropy 0.099–0.137 (first 0.108→last 0.109, mean 0.117); raw_reward −0.89→trail10 **−0.05** (max +0.14); pass@8 0.25→trail10 **0.673** (peak **0.844**, mean 0.649); grad_norm 0.049–0.095.
- **47469766 p50m50-D1**: entropy 0.110–0.138 (first 0.115→last 0.117, mean 0.120); raw_reward −0.86→trail10 **−0.12** (max +0.07); pass@8 0.34→trail10 **0.664** (peak **0.828**, mean 0.623); grad_norm 0.054–0.100.
- **47469767 p67m33-D1**: entropy 0.110–0.147 (first 0.125→last 0.116, mean 0.126); raw_reward −0.92→trail10 **−0.27** (max −0.07); pass@8 0.19→trail10 **0.603** (peak **0.750**, mean 0.562); grad_norm 0.048–0.104.
Mix-ordered as expected (p33m67 > p50m50 > p67m33). grad_norm ~0.05–0.10 throughout → the 0.5 clip rarely
bound; lr3e-6 does the work (consistent with the gc1 ablation). raw_reward net-negative (shaped) — pass@8 +
trend are the signal. **These are the production fixed-recipe D1 results.**
**HF upload (HF-only, NO DB):** each step-101 FSDP2 4-rank shard set consolidated via a generic version of
the 528/531 consolidate script (`$SF/consolidate_generic.py`: load 4 ranks → `torch.cat` per DTensor-Shard
dim → load into meta-init `Qwen3ForCausalLM(config)` → save_pretrained safetensors; **missing=0, unexpected=0**,
410 tensors, Qwen3 37L/hidden3840, 9.71B / 19.4 GB bf16, 4 safetensors + tokenizer + delphi_v0 chat_template +
generation_config). Combined consolidate+upload sbatch per cell on a compute node (boost_qos_dbg, 1 GPU) via
the Leonardo SOCKS5 sbatch-tunnel (`start_proxy_tunnel.sh` → login05); jobs 47485963 (p33m67), 47485964
(p50m50), 47486127 (p67m33) — all DONE, tunnel established inside the cert window. **All 3 verified LIVE on HF**
(hub_repo_details: 9714.7M params, qwen3, updated 2026-06-21):
- `laion/delphi-1e22-p33m67-wc386k_sft-rl-D1rlvrmath-fix-101-9_7B`
- `laion/delphi-1e22-p50m50-wc386k_sft-rl-D1rlvrmath-fix-101-9_7B`
- `laion/delphi-1e22-p67m33-wc386k_sft-rl-D1rlvrmath-fix-101-9_7B`
(dbg QOS caps submits at 2 → p67m33 launched after the first two finished; no other resource use.)

### (B) The 2 ablation ISOLATORS — both bounded, both confirm the recipe conclusions
- **47469268 gc05_lr3e6_ekeep** (clip0.5/lr3e6/**+0.01 entropy bonus**): entropy BOUNDED 0.108–0.156 (first
  0.108→last 0.146, mean 0.137 — crept up but NO runaway); raw_reward −0.89→trail10 −0.09 (max +0.14); pass@8
  0.25→trail10 0.677 (peak 0.81). **VERDICT: the +0.01 bonus ALONE does NOT explode.** With conservative lr3e-6
  + tight clip0.5, the original bonus is harmless → the orig explosion needed the *combination* lr1e-5 + bonus +
  clip1.0. lr (not the bonus alone) was the energy source.
- **47469269 gc1_lr3e6_e0** (**clip1.0**=orig default/lr3e6/ent-off): entropy BOUNDED 0.104–0.141 (first 0.108→
  last 0.114, mean 0.121); raw_reward −0.89→trail10 −0.06 (max +0.16); pass@8 0.25→trail10 0.708 (peak 0.84);
  grad_norm ≤0.11 (the 1.0 clip never even bound). **VERDICT: tightening the clip was NOT the key** — lr3e-6 +
  entropy-off suffice; clip0.5 is cheap insurance, not load-bearing.
Both analysis-only → HF-upload skipped (not production). Conclusions recorded in stabilization_grid.md.

### Net
3 D1 fix checkpoints LIVE on HF + the 2 ablation verdicts recorded; grid + this log updated. No DB, no git
push, no cluster git pull. The 4 RUNNING D3/D4 fix2 cells (47476407-410) left undisturbed for their harvest.
