# Phase 2 Colab Runbook

> Status as of 2026-05-25: Phase 2 *infrastructure* is complete and tested. The actual metric-battery sweep (Step 4 of the handoff) requires GPU and must be run on Colab. This file is the operating manual for that run.

---

## What's already done (local, on Mohamed's Mac)

- [x] `src/hybrid_arch/checkpoints.py` — `load_pythia(size, step)` and `list_checkpoints()`.
- [x] `src/scripts/run_metric_battery.py` — single-cell and `--sweep` modes, with caching.
- [x] Batched `parallel_prediction_agreement` — verified element-wise against the sequential version on a real Pythia prompt. ~3× speedup on CPU, expected ~20–30× on T4.
- [x] `notebooks/04_emergence_atlas.ipynb` — skeleton; runs end-to-end on whatever cells are present in the battery cache.
- [x] One smoke-tested battery cell already cached: `data/metric_battery/160m/step143000/wikitext/`.
- [x] 64 tests passing (`pytest -q`).

## What you need to do on Colab

The metric battery is 3 sizes × 12 checkpoints × 3 datasets = **108 cells**. With caching and the batched implementation, you can run this in a few hours of T4 time and resume after disconnects.

### 1. Open a fresh Colab T4 notebook and set up the repo

```bash
%cd /content
!git clone https://github.com/momagzoub/hybrid-architecture.git
%cd hybrid-architecture
!pip install -e . -q
```

(If the repo isn't yet on GitHub, upload the project as a zip and unzip — same effect.)

### 2. Decide the slice size

`run_metric_battery.py` defaults to `--n-tokens 512`. For Phase 2, **1024 tokens** is the recommended balance: ~1000 samples per cell for the correlation/probe analysis, manageable runtime per cell.

Estimated wall-clock per cell on T4 (1024 tokens, k=4, batched):

| Size | Forward + attn | Parallel agreement | Total |
|---|---|---|---|
| 70m | ~3 s | ~10 s | ~15 s |
| 160m | ~5 s | ~20 s | ~30 s |
| 410m | ~15 s | ~50 s | ~70 s |

Sum across 108 cells: **~2.5 hours wall-clock** on T4, dominated by the 410m cells. The Pythia downloads (first time you touch each size×step combination) add another ~30 minutes of bandwidth time.

### 3. Run the sweep

```bash
!python src/scripts/run_metric_battery.py --sweep --n-tokens 1024 --k 4
```

This enumerates all 108 cells and runs them serially. Cached cells are skipped, so if Colab disconnects you can rerun the same command and it picks up where it left off.

To run a subset (recommended for the first pass — make sure everything works on one size before launching the full sweep):

```bash
# Just Pythia-160m on WikiText, all 12 checkpoints
!python src/scripts/run_metric_battery.py --sweep \
    --sizes 160m --datasets wikitext --n-tokens 1024 --k 4
```

### 4. Persist the results

The battery writes to `data/metric_battery/`, which is gitignored — Colab will lose it when the runtime dies. Either:

- **Save to Google Drive** (recommended): before running, `from google.colab import drive; drive.mount('/content/drive')` and pass `--out-root /content/drive/MyDrive/hybrid_arch/metric_battery`.
- **Download a zip** at the end: `!zip -r metric_battery.zip data/metric_battery && from google.colab import files; files.download('metric_battery.zip')`.

### 5. Run the aggregation notebook

Once `data/metric_battery/` is populated, open `notebooks/04_emergence_atlas.ipynb` and run all cells. It produces:

- `docs/results/figures/emergence_curve.png`
- `docs/results/figures/signature_accuracy.png`
- `docs/results/figures/token_type_breakdown.png`
- `docs/results/figures/domain_shift.png`

Each plot has a guard that only saves to disk if enough cells are present (≥6 per plot is the threshold), so partial runs don't produce misleading figures.

### 6. Write the atlas writeup

After the figures exist, write `docs/results/02_emergence_atlas.md`. One paragraph per plot — describe what it shows and the takeaway. The figures should embed by relative path: `![emergence curve](figures/emergence_curve.png)`.

This writeup is your headline Phase 2 deliverable. If you can put one sentence per figure that you'd send to someone on the vLLM or TGI team, you've made the right thing.

## Things that might bite you

- **Disk pressure.** 3 sizes × 12 checkpoints × ~ a few hundred MB each = ~10–20 GB of model weights cached under `~/.cache/huggingface/`. Free Colab has ~80 GB; you have headroom, but don't also try to cache 410m × 12 checkpoints AND big dataset slices.
- **HF rate limiting.** The current script triggers an "unauthenticated requests" warning. To avoid throttling for repeated downloads, set `HF_TOKEN`: `import os; os.environ['HF_TOKEN'] = 'hf_xxx'`.
- **Pythia attention NaN bug — still real on Colab.** The `extract_attention` path in `hybrid_arch.attention` already handles this. **Do not** swap in `output_attentions=True` to "make it simpler" — the deep-layer NaN will silently poison your attention metrics. See `docs/concepts/03_attention.md`.
- **The j=0 column issue.** The battery script averages parallel-agreement over `j > 0` to drop the structurally-True j=0 column. The manifest records `parallel_agreement_excludes_j0: true`. If you change this, update the manifest format too, or the aggregation notebook's interpretations will silently shift.
- **The first 410m × step0 cell will be slow.** ~1 GB download plus initialization. Don't panic if the first cell in the 410m sweep takes 5 minutes; subsequent cells reuse the local model file.

## Then: closeout

Once the figures and writeup exist:

1. Update `PROJECT_PLAN.md` §Phase 2 status to "complete."
2. Update `CLAUDE.md` §8 with Phase 2 ✅.
3. Update `README.md` Status line.
4. Write `PHASE_3_HANDOFF.md` (the probes phase — described in `PROJECT_PLAN.md` §Phase 3).

---

End of runbook. Phase 2 infrastructure is ready; the GPU runs are yours to launch.
