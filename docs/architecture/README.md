# Architecture Index

Use this index to find architecture notes by research line.

## Current Main Line: GeoWarpFiLM

Directory: `geowarp_film/`

- `FiLM_architecture.md` - detailed GeoWarpFiLM/FiLM flow.
- `FiLM_simplified_diagram.md` - compact architecture diagram.
- `GEOMETRIC_WARP_ITD_UPGRADE.md` - geometric warp with left/right ear ITD.
- `GEOMETRIC_WARP_FIXES.md` - quaternion and geometric warp fixes.
- `geometric_warp_itd_modification.md` - ITD modification rationale.
- `WARP_SMOOTH_ISSUE_FIX.md` - warp smooth loss issue and fix.

For current training and evaluation commands, use `../USAGE.md`.

## DPATFNet

Directory: `dpatfnet/`

- `README_DPATFNet.md` - DPATFNet implementation notes.

## PGCN Proposals

Directory: `pgcn/`

These are design-stage proposals, not the current training path.

- `PGCN_README.md` - PGCN report index.
- `PGCN_Architecture_Proposal.md` - high-level architecture proposal.
- `PGCN_Module_Design.md` - module-level design.
- `PGCN_Training_Strategy.md` - loss and training plan.
- `PGCN_Implementation_Plan.md` - implementation schedule.

## Other Proposals

Directory: `proposals/`

- `BinauralTFNet_Architecture_Design.md` - BinauralTFNet design proposal.
- `architecture_improvements.md` - older architecture improvement ideas.

## Infrastructure

Directory: `infrastructure/`

- `CHECKPOINT_SYSTEM_SUMMARY.md` - checkpoint/resume system notes.

## Rule

Put new current-method docs under `geowarp_film/` unless the work starts a new
research line. Put speculative designs under `proposals/` until they are
implemented and evaluated.
