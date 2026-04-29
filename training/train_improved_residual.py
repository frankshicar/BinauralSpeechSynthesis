import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

from src.dataset import BinauralDataset
from src.models_improved_residual import ImprovedResidualPhaseNet
from src.losses_perceptual import compute_losses


# Config
config = {
    # Model
    'model_name': 'ImprovedResidualPhaseNet',
    
    # Training
    'epochs': 100,
    'batch_size': 16,
    'learning_rate': 3e-4,
    'early_stopping_patience': 15,
    
    # Loss weights (reduced regularization)
    'loss_weights': {
        'waveform': 10.0,
        'perceptual': 5.0,
        'magnitude': 1.0,
        'residual_reg': 0.001,      # 減少 10 倍
        'temporal_smooth': 0.01     # 減少 10 倍
    },
    
    # Paths
    'train_dir': 'dataset/trainset',
    'val_dir': 'dataset/testset',
    'checkpoint_dir': 'checkpoints',
    'log_file': 'logs/train_improved_residual.log'
}


def evaluate_metrics(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    
    total_waveform = 0
    total_phase_L = 0
    total_phase_R = 0
    total_residual = 0
    num_batches = 0
    
    with torch.no_grad():
        for mono, binaural, view in dataloader:
            mono = mono.to(device)
            binaural = binaural.to(device)
            view = view.to(device)
            
            # Forward
            y_L, y_R, outputs = model(mono, view)
            
            # Ground truth
            y_L_gt = binaural[:, 0:1]
            y_R_gt = binaural[:, 1:2]
            
            # Waveform L2
            waveform_loss = (
                torch.mean((y_L - y_L_gt) ** 2) +
                torch.mean((y_R - y_R_gt) ** 2)
            ) / 2
            
            # Phase correlation
            mono_phase = outputs['mono_phase']
            phase_L = outputs['phase_L']
            phase_R = outputs['phase_R']
            
            # Correlation with mono phase (should be < 1.0 if learning)
            phase_L_corr = torch.corrcoef(torch.stack([
                mono_phase.flatten(),
                phase_L.flatten()
            ]))[0, 1]
            
            phase_R_corr = torch.corrcoef(torch.stack([
                mono_phase.flatten(),
                phase_R.flatten()
            ]))[0, 1]
            
            # Residual magnitude
            residual = outputs['residual']
            residual_mag = torch.mean(torch.abs(residual))
            
            total_waveform += waveform_loss.item()
            total_phase_L += phase_L_corr.item()
            total_phase_R += phase_R_corr.item()
            total_residual += residual_mag.item()
            num_batches += 1
    
    return {
        'waveform': total_waveform / num_batches,
        'phase_L': total_phase_L / num_batches,
        'phase_R': total_phase_R / num_batches,
        'residual_mag': total_residual / num_batches
    }


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    loss_components = {
        'waveform': 0,
        'perceptual': 0,
        'magnitude': 0,
        'residual_reg': 0,
        'temporal_smooth': 0
    }
    num_batches = 0
    
    for mono, binaural, view in dataloader:
        mono = mono.to(device)
        binaural = binaural.to(device)
        view = view.to(device)
        
        # Forward
        y_L, y_R, outputs = model(mono, view)
        
        # Ground truth
        y_L_gt = binaural[:, 0:1]
        y_R_gt = binaural[:, 1:2]
        
        # Compute loss
        loss, loss_dict = compute_losses(
            (y_L, y_R),
            (y_L_gt, y_R_gt),
            outputs,
            config
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        num_batches += 1
    
    # Average
    avg_loss = total_loss / num_batches
    for key in loss_components:
        loss_components[key] /= num_batches
    
    return avg_loss, loss_components


def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['log_file']), exist_ok=True)
    
    # Model
    model = ImprovedResidualPhaseNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Dataset
    train_dataset = BinauralDataset(
        dataset_directory=config['train_dir'],
        chunk_size_ms=200,
        overlap=0.5
    )
    
    val_dataset = BinauralDataset(
        dataset_directory=config['val_dir'],
        chunk_size_ms=200,
        overlap=0.5
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
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
    
    log_file = open(config['log_file'], 'w')
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}\n")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device, config
        )
        
        # Validate
        val_metrics = evaluate_metrics(model, val_loader, device)
        val_loss = val_metrics['waveform']
        
        # Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        log_msg = (
            f"Epoch {epoch}\n"
            f"Train - Total: {train_loss:.6f}, "
            f"Waveform: {train_components['waveform']:.6f}, "
            f"Perceptual: {train_components['perceptual']:.6f}, "
            f"Mag: {train_components['magnitude']:.6f}, "
            f"Residual: {train_components['residual_reg']:.6f}, "
            f"Temporal: {train_components['temporal_smooth']:.6f}\n"
            f"Val   - Waveform: {val_metrics['waveform']:.6f}, "
            f"Phase L: {val_metrics['phase_L']:.4f}, "
            f"Phase R: {val_metrics['phase_R']:.4f}, "
            f"Residual: {val_metrics['residual_mag']:.4f}\n"
            f"LR: {current_lr:.6f}\n"
        )
        
        print(log_msg)
        log_file.write(log_msg + '\n')
        log_file.flush()
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save as .pt
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, os.path.join(config['checkpoint_dir'], 'improved_residual_best.pt'))
            
            # Save as .net (Meta format)
            torch.save(
                model.state_dict(),
                os.path.join(config['checkpoint_dir'], 'improved_residual_best.net')
            )
            
            print(f"✅ New best! Waveform: {val_loss:.6f}")
            log_file.write(f"✅ New best! Waveform: {val_loss:.6f}\n\n")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{config['early_stopping_patience']} epochs")
            log_file.write(f"⚠️  No improvement for {patience_counter}/{config['early_stopping_patience']} epochs\n\n")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping at epoch {epoch}")
            log_file.write(f"\n🛑 Early stopping at epoch {epoch}\n")
            break
    
    # Final summary
    summary = (
        f"\n{'='*60}\n"
        f"Training completed!\n"
        f"Best Val Waveform: {best_val_loss:.6f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    log_file.write(summary)
    log_file.close()


if __name__ == '__main__':
    main()
