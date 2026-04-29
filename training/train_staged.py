"""
分階段訓練 HybridPhysicalLearned

Stage 1: Magnitude only (20 epochs)
Stage 2: + IPD constraint (30 epochs)
Stage 3: L2 focused (50 epochs)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from src.models_hybrid_physical import HybridPhysicalLearned
from src.dataset import BinauralDataset


def stft_transform(waveform, n_fft=1024, hop_length=64):
    """STFT"""
    window = torch.hann_window(n_fft).to(waveform.device)
    B, C, T = waveform.shape
    stft_list = []
    for c in range(C):
        stft_c = torch.stft(
            waveform[:, c, :],
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        )
        stft_list.append(stft_c)
    return torch.stack(stft_list, dim=1)


def compute_losses(pred, target, outputs, config, stage):
    """計算 loss (根據 stage 調整權重)"""
    
    # STFT
    pred_stft = stft_transform(pred, config['n_fft'], config['hop_size'])
    target_stft = stft_transform(target, config['n_fft'], config['hop_size'])
    
    # 1. Waveform L2
    loss_waveform = F.mse_loss(pred, target)
    
    # 2. Magnitude
    pred_mag = pred_stft.abs()
    target_mag = target_stft.abs()
    loss_mag = F.mse_loss(pred_mag, target_mag)
    
    # 3. IPD
    pred_ipd = torch.angle(pred_stft[:, 0] / (pred_stft[:, 1] + 1e-8))
    target_ipd = torch.angle(target_stft[:, 0] / (target_stft[:, 1] + 1e-8))
    loss_ipd = torch.mean(1 - torch.cos(pred_ipd - target_ipd))
    
    # 4. ITD 正則化
    residual_ITD = outputs['residual_ITD']
    loss_itd_reg = torch.mean(residual_ITD ** 2)
    
    # 5. Phase (監控)
    pred_phase_L = torch.angle(pred_stft[:, 0])
    pred_phase_R = torch.angle(pred_stft[:, 1])
    target_phase_L = torch.angle(target_stft[:, 0])
    target_phase_R = torch.angle(target_stft[:, 1])
    loss_phase_L = torch.mean(1 - torch.cos(pred_phase_L - target_phase_L))
    loss_phase_R = torch.mean(1 - torch.cos(pred_phase_R - target_phase_R))
    
    # Stage-specific loss weights
    if stage == 1:
        # Stage 1: Focus on magnitude
        loss = (
            10.0 * loss_mag +
            0.1 * loss_waveform +
            0.01 * loss_itd_reg
        )
    elif stage == 2:
        # Stage 2: Add IPD constraint
        loss = (
            5.0 * loss_mag +
            5.0 * loss_ipd +
            0.5 * loss_waveform +
            0.001 * loss_itd_reg
        )
    else:  # stage == 3
        # Stage 3: Focus on L2
        loss = (
            10.0 * loss_waveform +
            2.0 * loss_ipd +
            1.0 * loss_mag +
            0.0001 * loss_itd_reg
        )
    
    return {
        'loss': loss,
        'waveform': loss_waveform.item(),
        'mag': loss_mag.item(),
        'ipd': loss_ipd.item(),
        'itd_reg': loss_itd_reg.item(),
        'phase_L': loss_phase_L.item(),
        'phase_R': loss_phase_R.item(),
    }


def train_epoch(model, dataloader, optimizer, config, device, stage):
    model.train()
    
    epoch_losses = {
        'waveform': 0.0,
        'mag': 0.0,
        'ipd': 0.0,
        'itd_reg': 0.0,
        'phase_L': 0.0,
        'phase_R': 0.0,
    }
    
    for batch in dataloader:
        mono, binaural, view = batch
        mono = mono.to(device)
        binaural = binaural.to(device)
        view = view.to(device)
        
        # Forward
        pred, outputs = model(mono, view)
        
        # Loss
        loss_dict = compute_losses(pred, binaural, outputs, config, stage)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]
    
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses


def validate(model, dataloader, config, device, stage):
    model.eval()
    
    epoch_losses = {
        'waveform': 0.0,
        'mag': 0.0,
        'ipd': 0.0,
        'itd_reg': 0.0,
        'phase_L': 0.0,
        'phase_R': 0.0,
    }
    
    with torch.no_grad():
        for batch in dataloader:
            mono, binaural, view = batch
            mono = mono.to(device)
            binaural = binaural.to(device)
            view = view.to(device)
            
            pred, outputs = model(mono, view)
            loss_dict = compute_losses(pred, binaural, outputs, config, stage)
            
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
    
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses


def main():
    config = {
        'sample_rate': 48000,
        'segment_length': 0.2,
        'n_fft': 1024,
        'hop_size': 64,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'seed': 42,
        # Stage epochs
        'stage1_epochs': 20,
        'stage2_epochs': 30,
        'stage3_epochs': 50,
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = HybridPhysicalLearned(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_size=config['hop_size'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Dataset
    train_dataset = BinauralDataset(
        dataset_directory='dataset/trainset',
        chunk_size_ms=int(config['segment_length'] * 1000),
        overlap=0.5,
    )
    
    val_dataset = BinauralDataset(
        dataset_directory='dataset/testset',
        chunk_size_ms=int(config['segment_length'] * 1000),
        overlap=0.5,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    global_epoch = 0
    
    # ========== Stage 1: Magnitude only ==========
    print(f"\n{'='*60}")
    print(f"STAGE 1: Magnitude Learning ({config['stage1_epochs']} epochs)")
    print(f"{'='*60}\n")
    
    # Freeze residual_itd_net
    for param in model.residual_itd_net.parameters():
        param.requires_grad = False
    
    for epoch in range(1, config['stage1_epochs'] + 1):
        global_epoch += 1
        print(f"\n{'='*60}")
        print(f"Stage 1 - Epoch {epoch}/{config['stage1_epochs']} (Global: {global_epoch})")
        print(f"{'='*60}\n")
        
        train_losses = train_epoch(model, train_loader, optimizer, config, device, stage=1)
        val_losses = validate(model, val_loader, config, device, stage=1)
        
        print(f"Train - Waveform: {train_losses['waveform']:.6f}, Mag: {train_losses['mag']:.6f}, IPD: {train_losses['ipd']:.4f}")
        print(f"Val   - Waveform: {val_losses['waveform']:.6f}, Mag: {val_losses['mag']:.6f}, IPD: {val_losses['ipd']:.4f}")
        print(f"Phase - L: {val_losses['phase_L']:.4f}, R: {val_losses['phase_R']:.4f}")
        
        if val_losses['waveform'] < best_val_loss:
            best_val_loss = val_losses['waveform']
            torch.save({
                'epoch': global_epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_losses': val_losses,
                'config': config,
            }, 'checkpoints/staged_best.pt')
            print(f"✅ New best! Waveform: {val_losses['waveform']:.6f}")
    
    # ========== Stage 2: Add IPD ==========
    print(f"\n{'='*60}")
    print(f"STAGE 2: IPD Learning ({config['stage2_epochs']} epochs)")
    print(f"{'='*60}\n")
    
    # Unfreeze residual_itd_net
    for param in model.residual_itd_net.parameters():
        param.requires_grad = True
    
    # Reset optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] * 0.5)
    
    for epoch in range(1, config['stage2_epochs'] + 1):
        global_epoch += 1
        print(f"\n{'='*60}")
        print(f"Stage 2 - Epoch {epoch}/{config['stage2_epochs']} (Global: {global_epoch})")
        print(f"{'='*60}\n")
        
        train_losses = train_epoch(model, train_loader, optimizer, config, device, stage=2)
        val_losses = validate(model, val_loader, config, device, stage=2)
        
        print(f"Train - Waveform: {train_losses['waveform']:.6f}, Mag: {train_losses['mag']:.6f}, IPD: {train_losses['ipd']:.4f}")
        print(f"Val   - Waveform: {val_losses['waveform']:.6f}, Mag: {val_losses['mag']:.6f}, IPD: {val_losses['ipd']:.4f}")
        print(f"Phase - L: {val_losses['phase_L']:.4f}, R: {val_losses['phase_R']:.4f}, ITD_reg: {val_losses['itd_reg']:.6f}")
        
        if val_losses['waveform'] < best_val_loss:
            best_val_loss = val_losses['waveform']
            torch.save({
                'epoch': global_epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_losses': val_losses,
                'config': config,
            }, 'checkpoints/staged_best.pt')
            print(f"✅ New best! Waveform: {val_losses['waveform']:.6f}")
    
    # ========== Stage 3: L2 focused ==========
    print(f"\n{'='*60}")
    print(f"STAGE 3: L2 Optimization ({config['stage3_epochs']} epochs)")
    print(f"{'='*60}\n")
    
    # Reset optimizer with lower LR
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] * 0.1)
    
    patience = 15
    patience_counter = 0
    
    for epoch in range(1, config['stage3_epochs'] + 1):
        global_epoch += 1
        print(f"\n{'='*60}")
        print(f"Stage 3 - Epoch {epoch}/{config['stage3_epochs']} (Global: {global_epoch})")
        print(f"{'='*60}\n")
        
        train_losses = train_epoch(model, train_loader, optimizer, config, device, stage=3)
        val_losses = validate(model, val_loader, config, device, stage=3)
        
        print(f"Train - Waveform: {train_losses['waveform']:.6f}, Mag: {train_losses['mag']:.6f}, IPD: {train_losses['ipd']:.4f}")
        print(f"Val   - Waveform: {val_losses['waveform']:.6f}, Mag: {val_losses['mag']:.6f}, IPD: {val_losses['ipd']:.4f}")
        print(f"Phase - L: {val_losses['phase_L']:.4f}, R: {val_losses['phase_R']:.4f}, ITD_reg: {val_losses['itd_reg']:.6f}")
        
        if val_losses['waveform'] < best_val_loss:
            best_val_loss = val_losses['waveform']
            patience_counter = 0
            torch.save({
                'epoch': global_epoch,
                'stage': 3,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_losses': val_losses,
                'config': config,
            }, 'checkpoints/staged_best.pt')
            print(f"✅ New best! Waveform: {val_losses['waveform']:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{patience} epochs")
        
        if patience_counter >= patience:
            print(f"\n🛑 Early stopping at epoch {global_epoch}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total epochs: {global_epoch}")
    print(f"Best Val Waveform: {best_val_loss:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
