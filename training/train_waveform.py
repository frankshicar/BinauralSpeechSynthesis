"""
訓練 WaveformSpatializer

時域直接學習 + 頻域 Phase 監控
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json

from src.models_waveform import WaveformSpatializer
from src.dataset import BinauralDataset


def stft_transform(waveform, n_fft=1024, hop_length=64):
    """計算 STFT"""
    # waveform: B×C×T
    B, C, T = waveform.shape
    
    # 對每個 channel 計算 STFT
    stft_list = []
    for c in range(C):
        stft_c = torch.stft(
            waveform[:, c, :],
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft).to(waveform.device),
            return_complex=True,
        )
        stft_list.append(stft_c)
    
    return torch.stack(stft_list, dim=1)  # B×C×F×T


def compute_losses(pred, target, config):
    """
    計算多個 loss:
    1. Waveform L2 (時域)
    2. Magnitude MSE (頻域)
    3. IPD loss (相位差)
    """
    # 1. Waveform L2
    loss_waveform = F.mse_loss(pred, target)
    
    # 2. STFT
    pred_stft = stft_transform(pred, config['n_fft'], config['hop_size'])
    target_stft = stft_transform(target, config['n_fft'], config['hop_size'])
    
    # 3. Magnitude
    pred_mag = pred_stft.abs()
    target_mag = target_stft.abs()
    loss_mag = F.mse_loss(pred_mag, target_mag)
    
    # 4. IPD (Phase difference between L and R)
    pred_ipd = torch.angle(pred_stft[:, 0] / (pred_stft[:, 1] + 1e-8))
    target_ipd = torch.angle(target_stft[:, 0] / (target_stft[:, 1] + 1e-8))
    loss_ipd = torch.mean(1 - torch.cos(pred_ipd - target_ipd))
    
    # 5. Phase (individual channels)
    pred_phase_L = torch.angle(pred_stft[:, 0])
    pred_phase_R = torch.angle(pred_stft[:, 1])
    target_phase_L = torch.angle(target_stft[:, 0])
    target_phase_R = torch.angle(target_stft[:, 1])
    
    loss_phase_L = torch.mean(1 - torch.cos(pred_phase_L - target_phase_L))
    loss_phase_R = torch.mean(1 - torch.cos(pred_phase_R - target_phase_R))
    
    # Total loss
    loss = (
        config['lambda_waveform'] * loss_waveform +
        config['lambda_mag'] * loss_mag +
        config['lambda_ipd'] * loss_ipd
    )
    
    return {
        'loss': loss,
        'waveform': loss_waveform.item(),
        'mag': loss_mag.item(),
        'ipd': loss_ipd.item(),
        'phase_L': loss_phase_L.item(),
        'phase_R': loss_phase_R.item(),
    }


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    
    epoch_losses = {
        'waveform': 0.0,
        'mag': 0.0,
        'ipd': 0.0,
        'phase_L': 0.0,
        'phase_R': 0.0,
    }
    
    for batch in dataloader:
        mono, binaural, view = batch
        mono = mono.to(device)
        binaural = binaural.to(device)
        view = view.to(device)
        
        # Forward
        pred = model(mono, view)
        
        # Loss
        loss_dict = compute_losses(pred, binaural, config)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses


def validate(model, dataloader, config, device):
    model.eval()
    
    epoch_losses = {
        'waveform': 0.0,
        'mag': 0.0,
        'ipd': 0.0,
        'phase_L': 0.0,
        'phase_R': 0.0,
    }
    
    with torch.no_grad():
        for batch in dataloader:
            mono, binaural, view = batch
            mono = mono.to(device)
            binaural = binaural.to(device)
            view = view.to(device)
            
            # Forward
            pred = model(mono, view)
            
            # Loss
            loss_dict = compute_losses(pred, binaural, config)
            
            # Accumulate
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
    
    # Average
    for key in epoch_losses:
        epoch_losses[key] /= len(dataloader)
    
    return epoch_losses


def main():
    # Config
    config = {
        'sample_rate': 48000,
        'segment_length': 0.2,  # 200ms
        'n_fft': 1024,
        'hop_size': 64,
        'view_dim': 7,
        'view_hidden': 128,
        'view_output': 256,
        'filter_len': 512,
        'filter_hidden': 512,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'num_epochs': 100,
        'patience': 10,
        'seed': 42,
        # Loss weights
        'lambda_waveform': 1.0,
        'lambda_mag': 1.0,
        'lambda_ipd': 1.0,
    }
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = WaveformSpatializer(
        view_dim=config['view_dim'],
        view_hidden=config['view_hidden'],
        view_output=config['view_output'],
        filter_len=config['filter_len'],
        filter_hidden=config['filter_hidden'],
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
        dataset_directory='dataset/testset',  # 使用 testset 作為驗證集
        chunk_size_ms=int(config['segment_length'] * 1000),
        overlap=0.5,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}\n")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, config, device)
        
        # Validate
        val_losses = validate(model, val_loader, config, device)
        
        # Scheduler
        scheduler.step(val_losses['waveform'])
        
        # Print
        print(f"Train - Waveform: {train_losses['waveform']:.6f}, Mag: {train_losses['mag']:.6f}, IPD: {train_losses['ipd']:.4f}")
        print(f"Val   - Waveform: {val_losses['waveform']:.6f}, Mag: {val_losses['mag']:.6f}, IPD: {val_losses['ipd']:.4f}")
        print(f"Phase - L: {val_losses['phase_L']:.4f}, R: {val_losses['phase_R']:.4f}")
        
        # Save best
        if val_losses['waveform'] < best_val_loss:
            best_val_loss = val_losses['waveform']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_losses': val_losses,
                'config': config,
            }, 'checkpoints/waveform_best.pt')
            print(f"✅ New best model! Waveform: {val_losses['waveform']:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{config['patience']} epochs")
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
