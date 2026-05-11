# Usage Guide

This guide focuses on the current GeoWarpFiLM v6.4 workflow.

Run commands from the repository root:

```bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
```

## Current Paths

- Current model code: `src/models_geowarp_film_v6_4.py`
- Current training script: `training/train_geowarp_film_v6_4.py`
- Current launcher: `scripts/start_train_geowarp_film_v6_4.sh`
- Current run directory: `geowarp_film_v6_4/`
- Training log: `geowarp_film_v6_4/train.log`
- Console log: `geowarp_film_v6_4/train_console.log`
- Training-time spectra: `geowarp_film_v6_4/training_spectra/`
- Best checkpoint: `geowarp_film_v6_4/best.net`
- Meta pretrained baseline: `binaural_network_3blocks.net`

`dataset/testset` is the Meta-provided test set. `dataset/test_static` is the
custom static-angle test set.

## Train v6.4

Foreground:

```bash
bash scripts/start_train_geowarp_film_v6_4.sh
```

Background:

```bash
mkdir -p geowarp_film_v6_4
nohup bash scripts/start_train_geowarp_film_v6_4.sh > geowarp_film_v6_4/nohup.log 2>&1 &
disown
```

Check that it is running:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'train_geowarp_film_v6_4|start_train_geowarp_film_v6_4' | grep -v grep
```

Monitor logs:

```bash
tail -f geowarp_film_v6_4/train.log
tail -f geowarp_film_v6_4/train_console.log
tail -f geowarp_film_v6_4/nohup.log
```

Stop training:

```bash
pkill -f training/train_geowarp_film_v6_4.py
```

## Inspect Training Spectra

The v6.4 training loop writes spectrum diagnostics during training:

```bash
find geowarp_film_v6_4/training_spectra -maxdepth 2 -type f | sort | tail
```

Typical files:

- `analysis.md`
- `summary.json`
- `02_spectral_error_before_after_film.png`
- `film_block_XX_band_activity.png`
- `film_block_XX_gamma_delta.png`
- `film_block_XX_beta.png`

Read the latest analysis:

```bash
latest=$(find geowarp_film_v6_4/training_spectra -maxdepth 1 -type d -name 'stage*_epoch*' | sort | tail -n 1)
sed -n '1,220p' "$latest/analysis.md"
```

## Offline Spectrum Diagnostic

Run this after a checkpoint exists:

```bash
/home/sbplab/anaconda3/bin/python scripts/diagnose_geowarp_film_v6_4_spectra.py \
  --model_file geowarp_film_v6_4/best.net \
  --dataset_directory dataset/testset \
  --subject subject4 \
  --seconds 2 \
  --output_dir geowarp_film_v6_4/spectra_diagnostics/subject4
```

The output directory contains Neural Warp layer spectra, FiLM block spectra,
FiLM gamma/beta heatmaps, and `summary.json`.

## Evaluate v6.4 On Meta Test Set

```bash
/home/sbplab/anaconda3/bin/python evaluate_geowarp_film_v6.py \
  --model_file geowarp_film_v6_4/best.net \
  --dataset_directory dataset/testset \
  --artifacts_directory geowarp_film_v6_4/eval_testset
```

This reports Meta-style weighted L2, amplitude, and phase metrics.

## Evaluate v6.4 On Static Test Set

```bash
/home/sbplab/anaconda3/bin/python evaluate_geowarp_film_v6.py \
  --model_file geowarp_film_v6_4/best.net \
  --dataset_directory dataset/test_static \
  --artifacts_directory geowarp_film_v6_4/eval_test_static
```

For angle-only evaluation:

```bash
/home/sbplab/anaconda3/bin/python evaluate_angular_v6.py \
  --model_file geowarp_film_v6_4/best.net \
  --dataset_directory dataset/test_static \
  --model_type geowarp_v4
```

`model_type geowarp_v4` is the historical name used by the script; it currently
loads `GeoWarpFiLMNet` from `src/models_geowarp_film_v6_4.py`.

## Evaluate Meta Pretrained Baseline

Meta 3-block checkpoint on Meta test set:

```bash
/home/sbplab/anaconda3/bin/python evaluate.py \
  --dataset_directory dataset/testset \
  --model_file binaural_network_3blocks.net \
  --artifacts_directory geowarp_film/meta_pretrained_eval \
  --blocks 3
```

Angle-only baseline on static test set:

```bash
/home/sbplab/anaconda3/bin/python evaluate_angular_v6.py \
  --model_file binaural_network_3blocks.net \
  --dataset_directory dataset/test_static \
  --model_type meta \
  --blocks 3
```

## Where To Put New Files

- New active training script: `training/`
- New active launcher: `scripts/`
- One-off diagnostic script: `experiments/diagnostics/`
- Architecture note: `docs/architecture/<research-line>/`
- Experiment record: `實驗記錄/`
- Finished old run: `archive/legacy_runs/`
