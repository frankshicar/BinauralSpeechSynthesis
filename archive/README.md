# Archive

This directory stores old outputs and backups that are useful for reference but
should not clutter the active repository root.

## Layout

- `legacy_runs/` - old GeoWarpFiLM training runs from v4 through v6.3.
- `legacy_outputs/` - old generated audio and evaluation outputs.
- `original_meta_code/` - original Meta code backup.

Do not move a run directory here while a training process is still writing to
it. Active runs should stay at the path expected by their training script until
training and evaluation are finished.
