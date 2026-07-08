# TPU node configurations

## v5p-32

v5p-32 slice composition (from iris workers table):
- 4 workers (= 4 hosts), each total_tpu_count=4 chips, chips_per_host_bounds=2,2,1 → **16 chips/slice**
  (confirms [v5p_naming_cores_not_chips]: v5p-N = N cores = N/2 chips)
- Per-host: 464.7 GB DRAM, 207 CPU cores

Per-chip v5p spec (Google datasheet):
- 95 GiB HBM2e, 2765 GB/s bandwidth
- 459 TFLOPs/s bf16 · 918 TOPS int8
- No native FP8 (a v6e feature). Qwen122B-FP8 weights dequantize to bf16 before matmul on v5p, so bf16 is the relevant compute number.

v5p-32 slice totals:

┌───────────────┬────────────┬──────────────────────────┐
│    metric     │  per chip  │        × 16 chips        │
├───────────────┼────────────┼──────────────────────────┤
│ HBM           │ 95 GiB     │ 1,520 GiB (~1.49 TiB)    │
├───────────────┼────────────┼──────────────────────────┤
│ HBM bandwidth │ 2765 GB/s  │ 44.24 TB/s aggregate     │
├───────────────┼────────────┼──────────────────────────┤
│ bf16 FLOPs/s  │ 459 TFLOPS │ 7.34 PFLOPs/s            │
├───────────────┼────────────┼──────────────────────────┤
│ int8 OPs/s    │ 918 TOPS   │ 14.69 POPS               │
├───────────────┼────────────┼──────────────────────────┤
│ host DRAM     │ —          │ ~1,859 GB across 4 hosts │
└───────────────┴────────────┴──────────────────────────┘

## v6e-8

Per-chip v6e (Trillium) spec (Google datasheet):
- 32 GiB HBM3, 1640 GB/s bandwidth
- 918 TFLOPs/s bf16 · 1836 TOPS int8
- Native FP8 at 1836 TFLOPs/s — v6e's edge over v5p (which must dequant FP8 to bf16)

v6e-8 slice totals (8 chips, single host):

┌──────────────────┬─────────────┬────────────────┐
│      metric      │  per chip   │   × 8 chips    │
├──────────────────┼─────────────┼────────────────┤
│ HBM              │ 32 GiB      │ 256 GiB        │
├──────────────────┼─────────────┼────────────────┤
│ HBM bandwidth    │ 1640 GB/s   │ 13.12 TB/s     │
├──────────────────┼─────────────┼────────────────┤
│ bf16 FLOPs/s     │ 918 TFLOPS  │ 7.34 PFLOPs/s  │
├──────────────────┼─────────────┼────────────────┤
│ FP8 / int8 OPs/s │ 1836 TFLOPS │ 14.69 PFLOPs/s │
├──────────────────┼─────────────┼────────────────┤
│ host DRAM        │ —           │ ~1,410 GiB     │
└──────────────────┴─────────────┴────────────────┘

v6e-8 vs v5p-32 — same nominal bf16 throughput, very different memory:

┌───────────────┬─────────────────────────┬───────────────────────────────┐
│               │ v6e-8 (1 host, 8 chips) │  v5p-32 (4 hosts, 16 chips)   │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ HBM total     │ 256 GiB                 │ 1,520 GiB (5.9× more)         │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ HBM bandwidth │ 13.12 TB/s              │ 44.24 TB/s (3.4× more)        │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ bf16 PFLOPs/s │ 7.34                    │ 7.34 (same)                   │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ native FP8    │ yes (14.7 PFLOPs/s)     │ no (must dequant → 7.34 bf16) │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ host count    │ 1                       │ 4 (cross-host comms cost)     │
├───────────────┼─────────────────────────┼───────────────────────────────┤
│ host DRAM     │ 1,410 GiB               │ 1,859 GiB (across 4)          │
└───────────────┴─────────────────────────┴───────────────────────────────┘

This is why 122B-FP8 fits on v5p-32 but not v6e-8 [memory: v6e8_cannot_fit_122b_fp8]: 122B weights × 1
byte ≈ 122 GiB > the 256 GiB v6e-8 budget once activations, MoE fixed footprint, and compile-time peaks
are subtracted. v5p-32 gives 6× the HBM at the cost of multi-host coordination, no native FP8, and ~3×
lower per-chip HBM bandwidth.

## Serving gotcha — prefer single-host DP=1 (marin#6136 multi-host decode bug)

⚠️ **Multi-host TPU serving hits a decode bug (marin#6136)** that forces `max_num_seqs` down (e.g.
seqs=2), crippling throughput. **Single-host `DP=1` dodges it entirely.** For 122B-FP8 @131k the
validated operating point is **v5p-8 A2: TP=4 / DP=1 / EP-on / max_num_seqs=32, fp8 kv**,
`--load-format runai_streamer`, `MODEL_IMPL_TYPE=vllm` — ~600 mean / 794 peak gen tok/s, **199
tok/s per chip** (best per-chip efficiency), decode-CLEAN. For aggregate throughput run keep-N
single-host jobs in parallel rather than one multi-host slice. (`v5p-8` = 4 chips, so TP=4 is the
per-chip ceiling — see the CORES-not-chips note in `iris_job_lifecycle.md` §1.)