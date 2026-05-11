"""
GeoWarpFiLMNet v6.2 training script
Fine-tune from v6 best checkpoint with low-frequency cosine IPD loss.
Key change: IPD loss restricted to low-freq bins (<1500Hz, ~32 bins) using cosine distance.
This avoids high-freq phase ambiguity diluting the gradient for low-freq ITD.
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import BinauralDataset
from src.models_geowarp_film_v6 import GeoWarpFiLMNet
from src.losses import L2Loss, PhaseLoss


def atomic_save(state_dict, path):
    tmp_path = path + '.tmp'
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)


config = {
    'train_dir':      'dataset/trainset',
    'val_dir':        'dataset/testset',
    'output_dir':     'geowarp_film_v6_2',
    'checkpoint':     'geowarp_film_v6_2/best.net',
    'log_file':       'geowarp_film_v6_2/train.log',
    'pretrain':       None,

    'stage1_epochs':  30,
    'stage2_epochs':  80,
    'lr':             8e-4,
    'batch_size':     16,
    'patience':       20,

    'w_itd':          0.5,
    'low_freq_bins':  32,   # <1500Hz @ n_fft=1024, sr=48000
}


def ipd_loss_fn(Y_L, Y_R, Y_L_gt, Y_R_gt, low_freq_bins=32):
    """Low-frequency cosine IPD loss (<1500Hz only).
    Avoids high-freq phase ambiguity diluting the gradient for low-freq ITD."""
    if torch.isnan(Y_L).any() or torch.isnan(Y_R).any():
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)
    pred_ipd = torch.angle(Y_L[:, :low_freq_bins, :] * Y_R[:, :low_freq_bins, :].conj())
    gt_ipd   = torch.angle(Y_L_gt[:, :low_freq_bins, :] * Y_R_gt[:, :low_freq_bins, :].conj())
    energy = (Y_L_gt[:, :low_freq_bins, :].abs() + Y_R_gt[:, :low_freq_bins, :].abs()) / 2
    mask = energy > 0.1 * energy.mean()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=Y_L.device, requires_grad=True)
    loss = 1 - (torch.cos(pred_ipd - gt_ipd) * mask).sum() / (mask.sum() + 1e-8)
    return loss if not torch.isnan(loss) else torch.tensor(0.0, device=Y_L.device, requires_grad=True)


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

        window = model.window
        Y_L_gt_stft = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt_stft = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)

        l2    = l2_fn(pred, gt)
        phase = phase_fn(pred, gt)
        ipd   = ipd_loss_fn(Y_L, Y_R, Y_L_gt_stft, Y_R_gt_stft)

        if stage == 1:
            mag_anchor = (torch.nn.functional.l1_loss(Y_L.abs(), Y_L_init.abs().detach()) +
                          torch.nn.functional.l1_loss(Y_R.abs(), Y_R_init.abs().detach())) / 2
            loss = 0.1 * mag_anchor + 0.1 * phase
            sums['mag_anchor'] += mag_anchor.item()
        else:
            loss = w['l2'] * l2 + w['phase'] * phase + w['ipd'] * ipd

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
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt   = torch.cat([y_L_gt, y_R_gt], dim=1)

        if torch.isnan(pred).any() or torch.isnan(gt).any():
            continue

        window = model.window
        Y_L_gt_stft = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt_stft = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)

        sums['l2']    += l2_fn(pred, gt).item()
        sums['phase'] += phase_fn(pred, gt).item()
        ipd_val = ipd_loss_fn(Y_L, Y_R, Y_L_gt_stft, Y_R_gt_stft).item()
        sums['ipd']   += ipd_val if not (ipd_val != ipd_val) else 0.0
        n += 1

    return {k: v / n for k, v in sums.items()}


def calibrate_weights(model, loader, device):
    model.eval()
    l2_fn    = L2Loss()
    phase_fn = PhaseLoss(sample_rate=48000, ignore_below=0.2)

    mono, binaural, view = next(iter(loader))
    mono, binaural, view = mono.to(device), binaural.to(device), view.to(device)
    y_L_gt, y_R_gt = binaural[:, 0:1], binaural[:, 1:2]

    with torch.no_grad():
        y_L, y_R, Y_L, Y_R, _, _ = model(mono, view)
        pred = torch.cat([y_L, y_R], dim=1)
        gt   = torch.cat([y_L_gt, y_R_gt], dim=1)
        window = model.window
        Y_L_gt_stft = torch.stft(y_L_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)
        Y_R_gt_stft = torch.stft(y_R_gt.squeeze(1), n_fft=model.n_fft, hop_length=model.hop_length,
                                  win_length=model.win_length, window=window, return_complex=True)
        l2    = l2_fn(pred, gt).item()
        phase = phase_fn(pred, gt).item()
        ipd   = ipd_loss_fn(Y_L, Y_R, Y_L_gt_stft, Y_R_gt_stft).item()

    print(f"  Raw losses → L2: {l2:.6f}  Phase: {phase:.4f}  IPD: {ipd:.4f}")
    ref = max(phase, 1e-4)
    w = {
        'l2':    ref / (l2  + 1e-8),
        'phase': 1.0,
        'ipd':   ref / (ipd + 1e-8),
    }
    print(f"  Calibrated weights → L2: {w['l2']:.2f}  Phase: {w['phase']:.2f}  IPD: {w['ipd']:.2f}")
    model.train()
    return w


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(config['output_dir'], exist_ok=True)

    model = GeoWarpFiLMNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load pretrained checkpoint
    if config.get('pretrain'):
        ckpt = torch.load(config['pretrain'], map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded pretrain from {config['pretrain']}")

    train_ds = BinauralDataset(config['train_dir'], chunk_size_ms=200, overlap=0.5)
    val_ds   = BinauralDataset(config['val_dir'],   chunk_size_ms=200, overlap=0.5)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    print("\nCalibrating loss weights...")
    w = calibrate_weights(model, train_loader, device)

    with open(config['log_file'], 'w', buffering=1) as log:
        def log_print(msg):
            print(msg)
            log.write(msg + '\n')

        best_metric = float('inf')
        patience = 0

        # ── Stage 1 ────────────────────────────────────────────────────────────
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        log_print("\n=== Stage 1: mag_anchor + phase + ITD consistency ===")

        for epoch in range(1, config['stage1_epochs'] + 1):
            tr  = run_epoch(model, train_loader, optimizer, device, stage=1, w=w)
            val = evaluate(model, val_loader, device)

            log_print(
                f"[S1] Ep {epoch:3d} | "
                f"train_mag_anchor={tr['mag_anchor']:.4f} phase={tr['phase']:.4f} | "
                f"val_l2={val['l2']*1000:.3f}e-3  val_phase={val['phase']:.3f}  val_ipd={val['ipd']:.4f}"
            )

            if val['phase'] < best_metric:
                best_metric = val['phase']
                patience = 0
                atomic_save({
                    'epoch': epoch, 'stage': 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'loss_weights': w,
                }, config['checkpoint'])
                log_print(f"  ✅ Best phase: {best_metric:.4f}")
            else:
                patience += 1
                if patience >= config['patience']:
                    log_print("  🛑 Early stop (stage 1)")
                    break

        # ── Stage 2 ────────────────────────────────────────────────────────────
        if config['stage1_epochs'] > 0:
            log_print("\nReloading best Stage 1 checkpoint...")
            ckpt = torch.load(config['checkpoint'])
            model.load_state_dict(ckpt['model_state_dict'])

        log_print("Recalibrating loss weights after Stage 1...")
        w = calibrate_weights(model, train_loader, device)
        # Force IPD weight = Phase weight (calibration suppresses IPD to ~0.36, too weak)
        w['ipd'] = w['phase']  # = 1.0
        log_print(f"  Overriding IPD weight to {w['ipd']:.2f} (equal to Phase)")

        best_metric = float('inf')
        patience = 0
        optimizer = optim.Adam(model.parameters(), lr=config['lr'] / 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        log_print("\n=== Stage 2: L2 + Phase + IPD (boosted) + ITD consistency ===")
        log_print(f"  Weights: L2={w['l2']:.2f}  Phase={w['phase']:.2f}  IPD={w['ipd']:.2f}  ITD={config['w_itd']:.2f}")

        for epoch in range(1, config['stage2_epochs'] + 1):
            tr  = run_epoch(model, train_loader, optimizer, device, stage=2, w=w)
            val = evaluate(model, val_loader, device)

            import math
            if not math.isnan(val['phase']):
                scheduler.step(val['phase'])

            log_print(
                f"[S2] Ep {epoch:3d} | "
                f"train={tr['total']:.4f} (l2={tr['l2']*1000:.3f}e-3 ph={tr['phase']:.3f} ipd={tr['ipd']:.4f}) | "
                f"val_l2={val['l2']*1000:.3f}e-3  val_phase={val['phase']:.3f}  val_ipd={val['ipd']:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

            if val['phase'] < best_metric:
                best_metric = val['phase']
                patience = 0
                atomic_save({
                    'epoch': epoch, 'stage': 2,
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
