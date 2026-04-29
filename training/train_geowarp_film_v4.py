"""
GeoWarpFiLMNet v4 訓練腳本
v4 improvements:
- Temporal awareness (velocity encoding)
- Relaxed phase residual (±π)
- Increased depth (8 blocks)
- IPD anchor loss (prevent phase over-correction)
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import BinauralDataset
from src.models_geowarp_film_v4 import GeoWarpFiLMNet  # v4 model
from src.losses import L2Loss, PhaseLoss


def atomic_save(state_dict, path):
    """Save checkpoint atomically to prevent corruption on kill/disk-full."""
    import tempfile
    tmp_path = path + '.tmp'
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX

config = {
    'train_dir':      'dataset/trainset',
    'val_dir':        'dataset/testset',
    'output_dir':     'geowarp_film_v4',
    'checkpoint':     'geowarp_film_v4/best.net',
    'log_file':       'geowarp_film_v4/train.log',

    'stage1_epochs':  30,
    'stage2_epochs':  80,
    'lr':             8e-4,  # v4: higher LR for deeper network (8 blocks)
    'batch_size':     16,
    'patience':       20,
}

# Validate config
assert config['batch_size'] > 0, "batch_size must be > 0"
assert config['lr'] > 0, "learning_rate must be > 0"
assert config['stage1_epochs'] > 0 and config['stage2_epochs'] > 0, "epochs must be > 0"
assert config['patience'] > 0, "patience must be > 0"


def ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt):
    """
    IPD loss on STFT domain (avoid double round-trip).
    Uses cross-spectrum to avoid angle gradient explosion.
    Args: complex STFT tensors (B, F, T_stft)
    """
    # Guard against NaN in input
    if torch.isnan(Y_L).any() or torch.isnan(Y_R).any():
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)
    
    # Cross-spectrum approach (stable gradients)
    pred_cross = Y_L * Y_R.conj()
    gt_cross   = Y_L_gt * Y_R_gt.conj()
    
    # angle() ignores magnitude, no need to normalize
    pred_ipd = torch.angle(pred_cross)
    gt_ipd   = torch.angle(gt_cross)

    # Energy mask (compute once)
    energy = (Y_L_gt.abs() + Y_R_gt.abs()) / 2
    mask = energy > 0.1 * energy.mean()
    
    # Guard against silent input
    if mask.sum() == 0:
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)

    diff = torch.atan2(torch.sin(pred_ipd - gt_ipd), torch.cos(pred_ipd - gt_ipd))
    loss = (diff[mask] ** 2).mean()
    
    # Guard against NaN output
    if torch.isnan(loss):
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)
    
    return loss


def ipd_anchor_loss_fn(Y_L, Y_R, Y_L_init, Y_R_init, weight=0.1):
    """
    v4: IPD anchor loss to prevent phase over-correction.
    Ensures learned phase doesn't deviate too much from geometric prior.
    """
    ipd_learned = torch.angle(Y_L) - torch.angle(Y_R)
    ipd_geo = torch.angle(Y_L_init) - torch.angle(Y_R_init)
    diff = torch.atan2(torch.sin(ipd_learned - ipd_geo), torch.cos(ipd_learned - ipd_geo))
    return weight * diff.abs().mean()


def run_epoch(model, loader, optimizer, device, stage, w):
    model.train()
    l2_fn    = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    sums = {'total': 0, 'l2': 0, 'phase': 0, 'ipd': 0, 'mag_anchor': 0}
    n = 0

    for mono, binaural, view in loader:
        mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
        y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]

        y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt   = torch.cat([y_L_gt, y_R_gt], dim=1)

        # Compute GT STFT for IPD loss
        window = model.window
        Y_L_gt = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)

        l2    = l2_fn(pred, gt)
        phase = phase_fn(pred, gt)
        ipd   = ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt)

        if stage == 1:
            # Stage 1: IPD + magnitude anchor + weak phase supervision
            # Prevent "IPD correct but absolute phase wrong" trap
            mag_anchor = (torch.nn.functional.l1_loss(Y_L.abs(), Y_L_init.abs().detach()) +
                          torch.nn.functional.l1_loss(Y_R.abs(), Y_R_init.abs().detach())) / 2
            # v4: Add IPD anchor loss to prevent phase over-correction
            ipd_anchor = ipd_anchor_loss_fn(Y_L, Y_R, Y_L_init, Y_R_init, weight=0.1)
            loss = ipd + 0.1 * mag_anchor + 0.1 * phase + ipd_anchor
            sums['mag_anchor'] += mag_anchor.item()
        else:
            # v4: Add IPD anchor loss in Stage 2 as well
            ipd_anchor = ipd_anchor_loss_fn(Y_L, Y_R, Y_L_init, Y_R_init, weight=0.1)
            loss = w['l2'] * l2 + w['phase'] * phase + w['ipd'] * ipd + ipd_anchor

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums['total'] += loss.item()
        sums['l2']    += l2.item()
        sums['phase'] += phase.item()
        sums['ipd']   += ipd.item()
        n += 1

    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    l2_fn    = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    sums = {'l2': 0, 'phase': 0, 'ipd': 0}
    n = 0

    for mono, binaural, view in loader:
        mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
        y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)  # ignore Y_init
        pred = torch.cat([y_L, y_R], dim=1)
        gt   = torch.cat([y_L_gt, y_R_gt], dim=1)
        
        # Skip NaN batches
        if torch.isnan(pred).any() or torch.isnan(gt).any():
            continue

        # GT STFT
        window = model.window
        Y_L_gt = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)

        sums['l2']    += l2_fn(pred, gt).item()
        sums['phase'] += phase_fn(pred, gt).item()
        ipd_val = ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt).item()
        sums['ipd']   += ipd_val if not torch.isnan(torch.tensor(ipd_val)) else 0.0
        n += 1

    return {k: v / n for k, v in sums.items()}


def calibrate_weights(model, loader, device):
    """Run one batch to measure raw loss magnitudes, return balanced weights."""
    model.eval()
    l2_fn    = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    mono, binaural, view = next(iter(loader))
    mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
    y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]

    with torch.no_grad():
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)  # ignore Y_init
        pred = torch.cat([y_L, y_R], dim=1)
        gt   = torch.cat([y_L_gt, y_R_gt], dim=1)
        
        window = model.window
        Y_L_gt = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                            win_length=model.win_length, window=window, return_complex=True)
        
        l2    = l2_fn(pred, gt).item()
        phase = phase_fn(pred, gt).item()
        ipd   = ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt).item()

    print(f"  Raw losses → L2: {l2:.6f}  Phase: {phase:.4f}  IPD: {ipd:.4f}")

    # Normalise so each term contributes equally at init
    ref = max(phase, 1e-4)  # guard against zero phase
    w = {
        'l2':    ref / (l2    + 1e-8),
        'phase': 1.0,
        'ipd':   ref / (ipd   + 1e-8),
    }
    print(f"  Calibrated weights → L2: {w['l2']:.2f}  Phase: {w['phase']:.2f}  IPD: {w['ipd']:.2f}")
    model.train()  # restore train mode
    return w


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)

    model = GeoWarpFiLMNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_ds = BinauralDataset(config['train_dir'], chunk_size_ms=200, overlap=0.5)
    val_ds   = BinauralDataset(config['val_dir'],   chunk_size_ms=200, overlap=0.5)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Calibrate loss weights from actual magnitudes
    print("\nCalibrating loss weights...")
    w = calibrate_weights(model, train_loader, device)

    with open(config['log_file'], 'w', buffering=1) as log:
        def log_print(msg):
            print(msg)
            log.write(msg + '\n')

        best_metric = float('inf')
        patience   = 0

        # ── Stage 1: IPD only ──────────────────────────────────────────────────
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        log_print("\n=== Stage 1: IPD loss ===")

        for epoch in range(1, config['stage1_epochs'] + 1):
            tr = run_epoch(model, train_loader, optimizer, device, stage=1, w=w)
            val = evaluate(model, val_loader, device)

            log_print(
                f"[S1] Ep {epoch:3d} | "
                f"train_ipd={tr['ipd']:.4f} mag_anchor={tr['mag_anchor']:.4f} phase={tr['phase']:.4f} | "
                f"val_l2={val['l2']*1000:.3f}e-3  val_phase={val['phase']:.3f}  val_ipd={val['ipd']:.4f}"
            )

            # Stage 1: use IPD as metric
            if val['ipd'] < best_metric:
                best_metric = val['ipd']
                patience = 0
                # Save full state for resume
                atomic_save({
                    'epoch': epoch,
                    'stage': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'loss_weights': w,
                }, config['checkpoint'])
                log_print(f"  ✅ Best IPD: {best_metric:.4f}")
            else:
                patience += 1
                if patience >= config['patience']:
                    log_print("  🛑 Early stop (stage 1)")
                    break

        # ── Stage 2: L2 + Phase + IPD ──────────────────────────────────────────
        # Reload best stage 1 checkpoint
        log_print("\nReloading best Stage 1 checkpoint...")
        ckpt = torch.load(config['checkpoint'])
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Recalibrate weights after Stage 1 (loss scales have changed)
        log_print("Recalibrating loss weights after Stage 1...")
        w = calibrate_weights(model, train_loader, device)
        
        # Reset metric to phase
        best_metric = float('inf')
        patience = 0
        
        optimizer = optim.Adam(model.parameters(), lr=config['lr'] / 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        log_print("\n=== Stage 2: L2 + Phase + IPD ===")
        log_print(f"  Weights: L2={w['l2']:.2f}  Phase={w['phase']:.2f}  IPD={w['ipd']:.2f}")

        for epoch in range(1, config['stage2_epochs'] + 1):
            tr  = run_epoch(model, train_loader, optimizer, device, stage=2, w=w)
            val = evaluate(model, val_loader, device)
            
            # Guard against NaN
            import math
            if not math.isnan(val['phase']):
                scheduler.step(val['phase'])
            else:
                log_print(f"  ⚠️  NaN detected in val_phase, skipping scheduler step")

            log_print(
                f"[S2] Ep {epoch:3d} | "
                f"train={tr['total']:.4f} (l2={tr['l2']*1000:.3f}e-3 ph={tr['phase']:.3f} ipd={tr['ipd']:.4f}) | "
                f"val_l2={val['l2']*1000:.3f}e-3  val_phase={val['phase']:.3f}  val_ipd={val['ipd']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

            # Stage 2: use phase as metric
            if val['phase'] < best_metric:
                best_metric = val['phase']
                patience = 0
                atomic_save({
                    'epoch': epoch,
                    'stage': 2,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'loss_weights': w,
                }, config['checkpoint'])
                log_print(f"  ✅ Best phase: {best_metric:.3f}")
            else:
                patience += 1
                if patience >= config['patience']:
                    log_print("  🛑 Early stop (stage 2)")
                    break

        log_print(f"\nFinal best phase: {best_metric:.3f}")


if __name__ == '__main__':
    main()
