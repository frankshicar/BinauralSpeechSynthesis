# Experiments

This directory contains one-off research scripts and legacy utilities. These are
not the main training or evaluation entry points.

Main entry points stay at:

- `training/` for active training programs.
- `scripts/` for active shell launchers and diagnostics.
- repo root `evaluate*.py` for evaluation entry points.

## Layout

### `diagnostics/`

Debugging and analysis scripts for angle, ITD, ILD, checkpoints, and model
behavior.

- `check_90deg_ild.py`
- `check_90deg_itd.py`
- `check_before_training.py`
- `check_checkpoint.py`
- `compare_doa_methods.py`
- `debug_mono_input.py`
- `debug_warp_smooth.py`
- `diagnose_ild.py`
- `diagnose_itd_all_angles.py`
- `eval_angle_error_v4.py`
- `evaluate_enhanced.py`
- `find_best_angles.py`

### `data_tools/`

Dataset alignment, transmitter-position generation, angle cache, and calibration
helpers.

- `align_dataset.py`
- `build_angle_cache.py`
- `calibrate_75_degrees.py`
- `calibrate_angle.py`
- `count_trainset_angles.py`
- `generate_distance_variations.py`
- `generate_tx_positions.py`
- `normalize_positions.py`
- `optimize_angle_compensation.py`
- `regenerate_all_tx_raw.py`
- `regenerate_tx.py`
- `regenerate_tx_with_compensation.py`

### `training/`

Short-lived training prototypes that are no longer the canonical launch path.

- `train_static.py`
- `train_v5_quick.py`

### `audio_tools/`

Audio enhancement experiments.

- `apply_enhancement.py`
- `enhance_audio.py`

## Rule

Before reusing a script here, check imports and hard-coded paths. Some scripts
were written against older model versions and may need small updates before
running with the current v6.4 code.
