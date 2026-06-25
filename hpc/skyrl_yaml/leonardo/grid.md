<!-- Generated 2026-06-18 from RL_CONVENTION.md §4 (re-picked: rows 1/5/6 promoted to 1e22; row 2 base SFT now exists).
     Consistent with RL_CONVENTION.md §4 picks table + grid. HF-upload only, NO DB (series rule). -->
# Delphi RL scaling-laws (#6279) — launch-ready RL experiment grid

The operator's actionable to-run list. **One cell = (queue row × RL dataset).** Every cell trains with
the **identical gold GRPO hparams** (RL_CONVENTION.md §3) — only `MODEL_PATH`, `DATASET`, `ENV_CLASS`,
`RUN_NAME`, `STAGE` change per cell. Picks + readiness mirror RL_CONVENTION.md §4 exactly.

## At a glance
- **12 cells launchable now** (✅): rows **1 / 5 / 6 / 2** × datasets **D1 / D3 / D4**.
- **D2 / RLVR-IFeval env NOW WIRED** (2026-06-18): `skyrl_gym/envs/ifeval/` implemented + unit-tested
  (23/23) + pulled on Leonardo (MarinSkyRL `penfever/working` @ `430e443`); validator coverage =
  24/24 func_names present in the N=500 seed-42 D2 sample. D2's held-out metric is IFEval pass-rate,
  not MATH-500.
- **2026-06-19 failure fix:** the first D1/D2 launches (D1 47283258/63/67, D2 47282615/17/18) all
  fast-failed at dataset-load. D1: RLVR-MATH prompts carried a 4-shot preamble → all 500 rows >512
  tokens → SkyRL filtered to size 0. D2: `rlvr_ifeval/` parquet was never built (missing on disk).
  Fixed `rl_dataset_prep.py` (`_strip_rlvr_math_fewshot`), rebuilt both parquets, **RELAUNCHED**
  D1 = 47290879/47290883/47290885 and D2 = 47290880/47290884/47290886 (all RUNNING). See
  `../agent_logs/2026-06-19_sweep_failures.md`.
- **1 cell held** (⏳): cell 16 (row 2 × D2) — env is ready, but row 2 is held pending the 1e22 base
  SFT eval (same hold as row 2's other cells; do not launch at 1e21 just for D2).
- **16 cells total** (4 rows × 4 datasets).
- **PICK STATUS:**
  - rows **1 & 6 FINALIZED on 1e22-`wc386k`** (2026-06-18): the last two pending 1e22-`magpie` SFT
    siblings landed (`../main_sft_evals/SCORES.md` rows 79/83) and BOTH lost — row 1 magpie 44.2 <
    wc386k 45.0 (eval 47248509), row 6 magpie 36.6 < wc386k 39.6 (eval 47225191). No pick switched →
    the already-LAUNCHED wc386k RL cells (rows 1/5/6, cells 1-12) are on the CORRECT starts; NO relaunch.
    Grid is now 100% eval-complete for the midtrained rows.
  - row **2** is picked at **1e21-base** (best DONE base) because the **1e22 base SFT is still RUNNING**
    (`../base_sft_evals/grid.md` rows 87/88). This makes row 2's scale (1e21) ≠ the midtrained rows
    (1e22). For a scale-matched base-vs-midtrained contrast, prefer to WAIT for the 1e22 base eval and
    re-pick row 2 at 1e22 (RL_CONVENTION §4 picks-table option (b), recommended). The 1e21 cells below
    are the launchable interim.

## Gold hparams reference (RL_CONVENTION §3 — applied to EVERY cell, do not vary)
GRPO · lr **1e-5** · `n_samples_per_prompt=8` · entropy bonus **on (coef 0.01)** · KL **off** ·
`max_generate_length=3584` · `max_prompt_length=512` · engine `max_model_len=4096` · temp **0.7** ·
top_p **1.0** · `train_batch_size=64` · `epochs=20 / max_steps=100` · CUDA graphs **on** ·
`gpu_memory_utilization=0.85` · placement **colocate 4×TP1** · strategy **fsdp2** ·
micro fwd/train **4/2** · `env_class` = per-dataset (`aime` for D1/D3/D4; `ifeval` TODO for D2).
Fallback (flagged separate cell only): `lr=3e-6`. Monitor: entropy (no collapse), grad_norm (no >~1
spike), reward, timings.

## Launch command shape (RL_CONVENTION §6)
```bash
# Sweep the 3 aime-ready datasets for one row's best start ckpt:
START=<MODEL_PATH from the table below>
for DS in rlvr_math dapo_math math500; do        # D1, D3, D4
  sbatch --job-name=rl_<RUN_NAME>-$DS sbatch_delphi_math_rl.sh \
    MODEL_PATH=$START \
    DATASET=$DS \
    RUN_NAME=<RUN_NAME>-$DS \
    STAGE=sft
done
# DATASET=rlvr_ifeval (D2) is BLOCKED until skyrl_gym/envs/ifeval exists.
```
Prereqs (login node, RL_CONVENTION §5): build the subsampled parquets (`rl_dataset_prep.py`, N=500
seed 42), `hf download $MODEL_PATH`, ensure `delphi_v0.jinja2` is present (the sbatch applies the
template override for `STAGE=sft`).

## DATASET → env map (RL_CONVENTION §2.2)
| DATASET token | cell | HF dataset | ENV_CLASS | held-out metric | notes |
|---|---|---|---|---|---|
| `rlvr_math` | D1 | `allenai/RLVR-MATH` | `aime` | MATH-500 / AIME24 | clean held-out math |
| `dapo_math` | D3 | `BytedTsinghua-SIA/DAPO-Math-17k` | `aime` | MATH-500 / AIME24 | clean held-out math; re-wrapped to boxed |
| `math500` | D4 | `HuggingFaceH4/MATH-500` (test) | `aime` | AIME24 (clean) | ⚠ **train-on-test** vs held-out MATH-500 — memorization ceiling, NOT generalization; judge from D1/D3 |
| `rlvr_ifeval` | D2 | `allenai/RLVR-IFeval` | `ifeval` (TODO) | IFEval pass-rate | ⏳ blocked: no `ifeval` env in MarinSkyRL |

## Starting checkpoints (resolved repo ids — RL_CONVENTION §4 picks)
| row | mix | STAGE | scale | MODEL_PATH (resolved) | SFT MATH-500 | pick status |
|---|---|---|---|---|---|---|
| 1 | p33m67 | sft | 1e22 | `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft` | 45.0 | FINAL (magpie sibling landed 44.2 < 45.0 → wc386k wins) |
| 5 | p50m50 | sft | 1e22 | `laion/delphi-1e22-p50m50-32p07b-lr0_5-ecfa99-wc386k_lr1e5-sft` | 44.6 | FINAL (both 1e22 siblings done; wc386k 44.6 > magpie 42.8) |
| 6 | p67m33 | sft | 1e22 | `laion/delphi-1e22-p67m33-32p07b-lr0_33-4e8cc7a7-wc386k_lr1e5-sft` | 39.6 | FINAL (magpie sibling landed 36.6 < 39.6 → wc386k wins) |
| 2 | base | sft | 1e21 | `laion/delphi-1e21-3.4Bparams-46.3Btokens-base-magpie_lr1e5-sft` | 3.6 | interim (1e22 base SFT still running; scale ≠ midtrained rows) |

Row 2 is the **base control** (the no-midtraining counterpart). Its checkpoint is the base-model **SFT**
output (carries the delphi template → `STAGE=sft`, same template override as rows 1/5/6) — it is NOT
RL-zero.

## The grid (16 cells = 4 rows × 4 datasets)
Ordered by row, then dataset. ✅ = launchable now · ⏳ = blocked.

| # | cell id / RUN_NAME | row | start ckpt (MODEL_PATH) | STAGE | DATASET | ENV_CLASS | readiness | status |
|---|---|---|---|---|---|---|---|---|
| 1 | `delphi-1e22-p33m67-wc386k_sft-rl-D1rlvrmath` | 1 | `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft` | sft | D1 rlvr_math | aime | ✅ launchable now | 🚀 relaunched 47290879 (47283258 failed: empty post-filter parquet, fixed) |
| 2 | `delphi-1e22-p33m67-wc386k_sft-rl-D3dapomath` | 1 | `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft` | sft | D3 dapo_math | aime | ✅ launchable now | 🚀 launched 47283259 |
| 3 | `delphi-1e22-p33m67-wc386k_sft-rl-D4math500` | 1 | `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft` | sft | D4 math500 | aime | ✅ launchable now *(train-on-test ⚠)* | 🚀 launched 47283262 |
| 4 | `delphi-1e22-p33m67-wc386k_sft-rl-D2ifeval` | 1 | `laion/delphi-1e22-p33m67-32p07b-lr0_67-54770ae7-wc386k_lr1e5-sft` | sft | D2 rlvr_ifeval | ifeval | ✅ env wired (tested + pulled) | 🚀 relaunched 47290880 (47282615 failed: rlvr_ifeval parquet missing, built) |
| 5 | `delphi-1e22-p50m50-wc386k_sft-rl-D1rlvrmath` | 5 | `laion/delphi-1e22-p50m50-32p07b-lr0_5-ecfa99-wc386k_lr1e5-sft` | sft | D1 rlvr_math | aime | ✅ launchable now | 🚀 relaunched 47290883 (47283263 failed: empty post-filter parquet, fixed) |
| 6 | `delphi-1e22-p50m50-wc386k_sft-rl-D3dapomath` | 5 | `laion/delphi-1e22-p50m50-32p07b-lr0_5-ecfa99-wc386k_lr1e5-sft` | sft | D3 dapo_math | aime | ✅ launchable now | 🚀 launched 47283264 |
| 7 | `delphi-1e22-p50m50-wc386k_sft-rl-D4math500` | 5 | `laion/delphi-1e22-p50m50-32p07b-lr0_5-ecfa99-wc386k_lr1e5-sft` | sft | D4 math500 | aime | ✅ launchable now *(train-on-test ⚠)* | 🚀 launched 47283266 |
| 8 | `delphi-1e22-p50m50-wc386k_sft-rl-D2ifeval` | 5 | `laion/delphi-1e22-p50m50-32p07b-lr0_5-ecfa99-wc386k_lr1e5-sft` | sft | D2 rlvr_ifeval | ifeval | ✅ env wired (tested + pulled) | 🚀 relaunched 47290884 (47282617 failed: rlvr_ifeval parquet missing, built) |
| 9 | `delphi-1e22-p67m33-wc386k_sft-rl-D1rlvrmath` | 6 | `laion/delphi-1e22-p67m33-32p07b-lr0_33-4e8cc7a7-wc386k_lr1e5-sft` | sft | D1 rlvr_math | aime | ✅ launchable now | 🚀 relaunched 47290885 (47283267 failed: empty post-filter parquet, fixed) |
| 10 | `delphi-1e22-p67m33-wc386k_sft-rl-D3dapomath` | 6 | `laion/delphi-1e22-p67m33-32p07b-lr0_33-4e8cc7a7-wc386k_lr1e5-sft` | sft | D3 dapo_math | aime | ✅ launchable now | 🚀 launched 47283270 |
| 11 | `delphi-1e22-p67m33-wc386k_sft-rl-D4math500` | 6 | `laion/delphi-1e22-p67m33-32p07b-lr0_33-4e8cc7a7-wc386k_lr1e5-sft` | sft | D4 math500 | aime | ✅ launchable now *(train-on-test ⚠)* | 🚀 launched 47283272 |
| 12 | `delphi-1e22-p67m33-wc386k_sft-rl-D2ifeval` | 6 | `laion/delphi-1e22-p67m33-32p07b-lr0_33-4e8cc7a7-wc386k_lr1e5-sft` | sft | D2 rlvr_ifeval | ifeval | ✅ env wired (tested + pulled) | 🚀 relaunched 47290886 (47282618 failed: rlvr_ifeval parquet missing, built) |
| 13 | `delphi-1e21-base-magpie_sft-rl-D1rlvrmath` | 2 (base) | `laion/delphi-1e21-3.4Bparams-46.3Btokens-base-magpie_lr1e5-sft` | sft | D1 rlvr_math | aime | ✅ launchable now *(scale 1e21 ≠ 1e22 ⚠)* | ⏳ pending |
| 14 | `delphi-1e21-base-magpie_sft-rl-D3dapomath` | 2 (base) | `laion/delphi-1e21-3.4Bparams-46.3Btokens-base-magpie_lr1e5-sft` | sft | D3 dapo_math | aime | ✅ launchable now *(scale 1e21 ≠ 1e22 ⚠)* | ⏳ pending |
| 15 | `delphi-1e21-base-magpie_sft-rl-D4math500` | 2 (base) | `laion/delphi-1e21-3.4Bparams-46.3Btokens-base-magpie_lr1e5-sft` | sft | D4 math500 | aime | ✅ launchable now *(train-on-test + scale ⚠)* | ⏳ pending |
| 16 | `delphi-1e21-base-magpie_sft-rl-D2ifeval` | 2 (base) | `laion/delphi-1e21-3.4Bparams-46.3Btokens-base-magpie_lr1e5-sft` | sft | D2 rlvr_ifeval | ifeval | ✅ env wired — but row 2 HELD (awaits 1e22 base eval) | ⏳ held (row 2) |

## D2 RLVR-IFeval — env wired (2026-06-18)
- The `ifeval` verifier env is implemented in MarinSkyRL (`skyrl-gym/skyrl_gym/envs/ifeval/`,
  `register(id="ifeval")`, `penfever/working` @ `430e443`): parses the RLVR-IFeval `ground_truth`
  JSON (`func_name`+kwargs) and dispatches to the canonical open-instruct `IF_FUNCTIONS_MAP` (all 25
  IFEval validators vendored). Reward = **binary all-or-nothing** per example (RLVR-IFeval = one
  constraint per row); unknown/malformed specs + validator exceptions log + score 0 (never crash the
  rollout). Unit test 23/23. D2 train set built (`$WORK/data/rl/rlvr_ifeval`, N=500 seed 42); its 24
  distinct func_names are all covered.
- **Cells 4 / 8 / 12 (rows 1/5/6) RELAUNCHED 2026-06-19** — jobids 47290880 / 47290884 / 47290886.
  (First launches 47282615/17/18 fast-failed: the `rlvr_ifeval/` parquet was never built on disk;
  built it — 500 rows, gt JSON 0 parse-fails, 24 func_names all covered — and relaunched.)
- **Cell 16 (row 2) HELD** — env ready, but row 2 awaits the 1e22 base SFT eval (do not launch at the
  interim 1e21-base scale for D2 alone).

## On completion (RL_CONVENTION §7)
Cap candidates at `max_steps=100`; held-out eval the RL output through `../EVAL_CONVENTION.md` at temp
0.7; record final MATH-500 pass@1 + reward/entropy trajectory; **HF-upload only, NO DB**; record results
in a `SCORES.md` keyed by (scale, mix, start-point) so midtrained-vs-base + SFT deltas line up against
`../main_sft_evals/SCORES.md`.
