# DPATFNet Implementation

**Date**: 2026-04-27  
**Status**: Ready for training

## Quick Start

```bash
# Start training
./start_train_dpatfnet.sh

# Monitor training
tail -f train_dpatfnet.log

# Stop training
pkill -f train_dpatfnet.py
```

## Model Architecture

- **Encoder**: Conv2d (1→64→128→256)
- **DPAB**: Dual-Path Attention Block × 3
  - Intra-frame Self-Attention (across frequency)
  - Inter-frame Self-Attention (across time)
  - Position Cross-Attention
- **Decoder**: Conv2d (256→128→64→4) → real_L, imag_L, real_R, imag_R
- **Parameters**: 1.17M (channels=128, num_dpab=3)

## Training Config

- **Batch size**: 4
- **Learning rate**: 3e-4 (with ReduceLROnPlateau)
- **Loss**: Complex MSE + 10 × Time-domain L2
- **Epochs**: 100
- **Audio length**: 3 seconds (48000 samples @ 16kHz)

## Files

- `src/models_dpatfnet.py` - Model implementation
- `train_dpatfnet.py` - Training script
- `start_train_dpatfnet.sh` - Start script
- `outputs_dpatfnet/` - Checkpoints and logs

## Experiment Log

See `實驗記錄/20260427_HybridTFNet失敗分析與DPATFNet實現.md` for:
- HybridTFNet failure analysis
- All experiments and findings
- Lessons learned
