# Scripts

This folder is for active operational scripts. Legacy launchers are archived so
the root of `scripts/` only shows commands that are still expected to work.

## Active

- `start_train_geowarp_film_v6_4.sh` - current GeoWarpFiLM v6.4 training
  launcher.
- `diagnose_geowarp_film_v6_4_spectra.py` - offline spectrum diagnostics for a
  trained v6.4 checkpoint.
- `monitor_training.py` - legacy JSON training-history monitor.
- `visualize_training.py` - legacy JSON training-history plotting tool.

## Archive

- `archive/legacy_train/` - old training launchers for v4-v6.3, HybridTFNet,
  v7/v8, waveform, staged, and other obsolete experiments.
- `archive/legacy_examples/` - old examples and verification scripts.

Archived scripts are kept for reproducibility, but many contain outdated paths
such as `outputs_*`, `dataset_original`, or root-level `train_*.py`. Check and
update paths before running them.
