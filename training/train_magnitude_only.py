"""
Magnitude-only Baseline

只學習 ILD，Phase 用 mono + physical ITD
作為對照實驗的上限
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


def compute_losses(pred, target, outputs, config):
    pred_stft = stft_transform(pred, config['n_fft'], config['hop_size'])
    target_stft = stft_transform(target, config['n_fft'], config['hop_size'])
    
    loss_waveform = F.mse_loss(pred, target)
    
    pred_mag = pred_stft.abs()
    target_mag = target_stft.abs()
    loss_mag = F.mse_loss(pred_mag, target_mag)
    
    pred_ipd = torch.angle(pred_stft[:, 0] / (pred_stft[:, 1] + 1e-8))
    target_ipd = torch.angle(target_stft[:, 0] / (target_stft[:, 1] + 1e-8))
    loss_ipd = torch.mean(1 - torch.cos(pred_ipd - target_ipd))
    
    pred_phase_L = torch.angle(pred_stft[:, 0])
    pred_phase_R = torch.angle(pred_stft[:, 1])
    target_phase_L = torch.angle(target_stft[:, 0])
    target_phase_R = torch.angle(target_stft[:, 1])
    loss_phase_L = torch.mean(1 - torch.cos(pred_phase_L - target_phase_L))
    loss_phase_R = torch.mean(1 - torch.cos(pred_phase_R - target_phase_R))
    
    # Only optimize magnitude
    loss = 10.0 * loss_mag + 0.1 * loss_waveform
    
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
        
        pred, outputs = model(mono, view)
        loss_dict = compute_losses(pred, binaural, outputs, config)
        
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]
    
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
            
            pred, outputs = model(mono, view)
            loss_dict = compute_losses(pred, binaural, outputs, config)
            
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
        'num_epochs': 50,
        'patience': 15,
        'seed': 42,
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Magnitude-only Baseline (Phase = mono + physical ITD)")
    
    model = HybridPhysicalLearned(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_size=config['hop_size'],
    ).to(device)
    
    # Freeze residual_itd_net (只用物理 ITD)
    for param in model.residual_itd_net.parameters():
        param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
    
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
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}\n")
        
        train_losses = train_epoch(model, train_loader, optimizer, config, device)
        val_losses = validate(model, val_loader, config, device)
        
        print(f"Train - Waveform: {train_losses['waveform']:.6f}, Mag: {train_losses['mag']:.6f}, IPD: {train_losses['ipd']:.4f}")
        print(f"Val   - Waveform: {val_losses['waveform']:.6f}, Mag: {val_losses['mag']:.6f}, IPD: {val_losses['ipd']:.4f}")
        print(f"Phase - L: {val_losses['phase_L']:.4f}, R: {val_losses['phase_R']:.4f}")
        
        if val_losses['waveform'] < best_val_loss:
            best_val_loss = val_losses['waveform']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_losses': val_losses,
                'config': config,
            }, 'checkpoints/magnitude_only_best.pt')
            print(f"✅ New best! Waveform: {val_losses['waveform']:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{config['patience']} epochs")
        
        if patience_counter >= config['patience']:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            break
    
    print(f"\n{'='*60}")
    print(f"Magnitude-only Baseline completed!")
    print(f"Best Val Waveform: {best_val_loss:.6f}")
    print(f"This is the upper bound (magnitude perfect, phase = mono + physical ITD)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
