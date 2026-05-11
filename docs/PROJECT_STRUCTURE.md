# Project Structure

Last organized: 2026-05-11

## Active Code

- `src/` - core Python modules, losses, datasets, model definitions, and shared
  utilities.
- `training/` - active training scripts. Current GeoWarpFiLM v6.4 entry point:
  `training/train_geowarp_film_v6_4.py`.
- `scripts/` - launchers and active operational diagnostics. Current v6.4
  launcher: `scripts/start_train_geowarp_film_v6_4.sh`.
- repo root `evaluate*.py` - evaluation entry points.
- `tests/` - automated checks.

## Active Data And Models

- `dataset/` - train/test data. `testset` is the Meta-provided test set; the
  custom static set is under `dataset/test_static`.
- `binaural_network_3blocks.net` - Meta pretrained 3-block model.
- `geowarp_film_v6_4/` - active v6.4 run directory. This is intentionally kept
  at the repo root while training scripts still write there.
- `geowarp_film/` - legacy aggregate GeoWarpFiLM evaluation outputs. Kept in
  place because existing comparison notes reference its subdirectories.
- `geowarp_film_v6_release/` - packaged v6 release snapshot.

## Documentation

- `docs/architecture/` - architecture notes grouped by research line:
  `geowarp_film/`, `dpatfnet/`, `pgcn/`, `proposals/`, and
  `infrastructure/`.
- `docs/training/` - training guides.
- `docs/analysis/` - historical cleanup and analysis reports.
- `docs/research/` - research discussions and sub-agent notes.
- `docs/reports/` - standalone reports and planning notes.
- `docs/presentations/` - slide decks and presentation assets.
- `docs/USAGE.md` - current command guide for training, monitoring, diagnostics,
  and evaluation.
- `paper/` - reference papers and manuscript notes.
- `實驗記錄/` - chronological experiment records. Start with
  `實驗記錄/實驗總表.md` and `實驗記錄/README.md`.

## Experiments

- `experiments/diagnostics/` - one-off debug and metric inspection scripts.
- `experiments/data_tools/` - data alignment, angle, and transmitter-position
  utilities.
- `experiments/training/` - obsolete quick training prototypes.
- `experiments/audio_tools/` - audio enhancement experiments.

Scripts under `experiments/` are intentionally not canonical. Verify imports and
paths before running them.

## Archive

- `archive/legacy_runs/` - old GeoWarpFiLM run directories from v4 to v6.3.
- `archive/legacy_outputs/` - old generated audio and evaluation outputs.
- `archive/original_meta_code/` - original backup of Meta code.

Do not move active run directories into `archive/` while a training process is
writing to them.

## Root Directory Rule

Keep the repo root for stable entry points, active outputs, README/license, and
large canonical assets. New notes should go into `docs/` or `實驗記錄/`; new
temporary scripts should go into the relevant `experiments/` subfolder.
