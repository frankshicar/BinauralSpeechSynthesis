"""
訓練 HybridTFNet-IPD

改進：
1. 只學習 IPD (Interaural Phase Difference)
2. 使用 gradient checkpointing
3. 加入 random seed
4. 加入 early stopping

作者：AI Engineer Agent
日期：2026-04-27
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json

from src.models_hybrid_ipd import HybridTFNetIPD
from src.dataset import BinauralDataset


# ==================== Config ====================

config = {
    # Model
    'sample_rate': 48000,
    'n_fft': 1024,
    'hop_size': 64,
    'tf_channels': 128,
    'tf_blocks': 4,
    'use_checkpointing': True,
    
    # Training
    'batch_size': 16,
    'learning_rate': 3e-4,
    'num_epochs': 100,
    'gradient_clip': 5.0,
    
    # Data
    'chunk_size_ms': 200,
    'num_workers': 4,
    
    # Loss weights
    'lambda_l2': 10.0,
    'lambda_ipd': 1.0,
    'lambda_mag': 1.0,
    
    # Early stopping
    'patience': 10,
    'min_delta': 1e-5,
    
    # Random seed
    'seed': 42,
    
    # Paths
    'output_dir': 'outputs_hybrid_ipd',
    'checkpoint_dir': 'outputs_hybrid_ipd/checkpoints',
}


# ==================== Set Random Seed ====================

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==================== Loss Functions ====================

def wrapped_mse(pred, target):
    """Wrapped MSE for phase/IPD"""
    diff = pred - target
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return (diff ** 2).mean()


def compute_losses(y_pred, y_gt, outputs, mono, config):
    """
    計算所有 loss
    
    Args:
        y_pred: B×2×T
        y_gt: B×2×T
        outputs: dict with IPD, Phase_L, Phase_R, Mag_L, Mag_R
        mono: B×1×T
        config: dict
    
    Returns:
        loss_dict: dict with all losses
    """
    device = y_pred.device
    B, _, T = mono.shape
    
    # STFT parameters
    n_fft = config['n_fft']
    hop_size = config['hop_size']
    window = torch.hann_window(n_fft).to(device)
    
    # GT STFT
    Y_L_gt = torch.stft(y_gt[:, 0, :], n_fft, hop_size, window=window, return_complex=True)
    Y_R_gt = torch.stft(y_gt[:, 1, :], n_fft, hop_size, window=window, return_complex=True)
    Y_mono = torch.stft(mono.squeeze(1), n_fft, hop_size, window=window, return_complex=True)
    
    # GT Phase and Magnitude
    Phase_L_gt = torch.angle(Y_L_gt)
    Phase_R_gt = torch.angle(Y_R_gt)
    Mag_L_gt = torch.abs(Y_L_gt)
    Mag_R_gt = torch.abs(Y_R_gt)
    
    # GT IPD
    IPD_gt = Phase_L_gt - Phase_R_gt
    
    # 1. L2 Loss (time-domain)
    loss_l2 = F.mse_loss(y_pred, y_gt)
    
    # 2. IPD Loss (wrapped MSE)
    loss_ipd = wrapped_mse(outputs['IPD'], IPD_gt)
    
    # 3. Magnitude Loss
    loss_mag_L = F.mse_loss(outputs['Mag_L'], Mag_L_gt)
    loss_mag_R = F.mse_loss(outputs['Mag_R'], Mag_R_gt)
    loss_mag = (loss_mag_L + loss_mag_R) / 2
    
    # Total loss
    loss = (config['lambda_l2'] * loss_l2 + 
            config['lambda_ipd'] * loss_ipd + 
            config['lambda_mag'] * loss_mag)
    
    return {
        'loss': loss,
        'l2': loss_l2.item(),
        'ipd': loss_ipd.item(),
        'mag': loss_mag.item(),
        'mag_L': loss_mag_L.item(),
        'mag_R': loss_mag_R.item(),
    }


# ==================== Metrics ====================

def compute_metrics(y_pred, y_gt, outputs, mono, config):
    """計算評估指標"""
    device = y_pred.device
    
    # STFT parameters
    n_fft = config['n_fft']
    hop_size = config['hop_size']
    window = torch.hann_window(n_fft).to(device)
    
    # GT STFT
    Y_L_gt = torch.stft(y_gt[:, 0, :], n_fft, hop_size, window=window, return_complex=True)
    Y_R_gt = torch.stft(y_gt[:, 1, :], n_fft, hop_size, window=window, return_complex=True)
    
    # GT Phase and Magnitude
    Phase_L_gt = torch.angle(Y_L_gt)
    Phase_R_gt = torch.angle(Y_R_gt)
    Mag_L_gt = torch.abs(Y_L_gt)
    Mag_R_gt = torch.abs(Y_R_gt)
    
    # GT IPD
    IPD_gt = Phase_L_gt - Phase_R_gt
    
    # Compute errors
    ipd_err = torch.sqrt(wrapped_mse(outputs['IPD'], IPD_gt))
    phase_L_err = torch.sqrt(wrapped_mse(outputs['Phase_L'], Phase_L_gt))
    phase_R_err = torch.sqrt(wrapped_mse(outputs['Phase_R'], Phase_R_gt))
    mag_L_err = torch.sqrt(F.mse_loss(outputs['Mag_L'], Mag_L_gt))
    mag_R_err = torch.sqrt(F.mse_loss(outputs['Mag_R'], Mag_R_gt))
    l2_err = torch.sqrt(F.mse_loss(y_pred, y_gt))
    
    return {
        'ipd': ipd_err.item(),
        'phase_L': phase_L_err.item(),
        'phase_R': phase_R_err.item(),
        'mag_L': mag_L_err.item(),
        'mag_R': mag_R_err.item(),
        'l2': l2_err.item(),
    }


# ==================== Training ====================

def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    
    epoch_losses = []
    epoch_metrics = []
    
    for batch in dataloader:
        mono, binaural, view = batch  # Dataset returns tuple
        mono = mono.to(device)
        binaural = binaural.to(device)
        view = view.to(device)
        
        # Forward
        y_pred, outputs = model(mono, view)
        
        # Loss
        loss_dict = compute_losses(y_pred, binaural, outputs, mono, config)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(y_pred, binaural, outputs, mono, config)
        
        epoch_losses.append({k: v for k, v in loss_dict.items() if k != 'loss'})
        epoch_metrics.append(metrics)
    
    # Average
    avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}
    avg_metrics = {k: np.mean([d[k] for d in epoch_metrics]) for k in epoch_metrics[0]}
    
    return avg_losses, avg_metrics


def validate(model, dataloader, config, device):
    model.eval()
    
    epoch_losses = []
    epoch_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            mono, binaural, view = batch  # Dataset returns tuple
            mono = mono.to(device)
            binaural = binaural.to(device)
            view = view.to(device)
            
            # Forward
            y_pred, outputs = model(mono, view)
            
            # Loss
            loss_dict = compute_losses(y_pred, binaural, outputs, mono, config)
            
            # Metrics
            metrics = compute_metrics(y_pred, binaural, outputs, mono, config)
            
            epoch_losses.append({k: v for k, v in loss_dict.items() if k != 'loss'})
            epoch_metrics.append(metrics)
    
    # Average
    avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0]}
    avg_metrics = {k: np.mean([d[k] for d in epoch_metrics]) for k in epoch_metrics[0]}
    
    return avg_losses, avg_metrics


# ==================== Main ====================

def main():
    # Set seed
    set_seed(config['seed'])
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = HybridTFNetIPD(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_size=config['hop_size'],
        tf_channels=config['tf_channels'],
        tf_blocks=config['tf_blocks'],
        use_checkpointing=config['use_checkpointing'],
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Dataset
    train_dataset = BinauralDataset(
        dataset_directory='dataset/trainset',
        chunk_size_ms=config['chunk_size_ms'],
        overlap=0.5
    )
    
    val_dataset = BinauralDataset(
        dataset_directory='dataset/testset',
        chunk_size_ms=config['chunk_size_ms'],
        overlap=0.5
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_losses, train_metrics = train_epoch(model, train_loader, optimizer, config, device)
        
        # Validate
        val_losses, val_metrics = validate(model, val_loader, config, device)
        
        # Scheduler
        scheduler.step(val_losses['l2'])
        
        # Print
        print(f"\nTrain - L2: {train_losses['l2']:.6f}, IPD: {train_losses['ipd']:.4f}, Mag: {train_losses['mag']:.6f}")
        print(f"Val   - L2: {val_losses['l2']:.6f}, IPD: {val_losses['ipd']:.4f}, Mag: {val_losses['mag']:.6f}")
        print(f"\nMetrics - IPD: {val_metrics['ipd']:.4f}, Phase_L: {val_metrics['phase_L']:.4f}, Phase_R: {val_metrics['phase_R']:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
            'metrics': val_metrics,
        })
        
        with open(os.path.join(config['output_dir'], 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'epoch_{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'history': history,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if val_losses['l2'] < best_val_loss - config['min_delta']:
            best_val_loss = val_losses['l2']
            patience_counter = 0
            
            # Save best model
            best_path = os.path.join(config['checkpoint_dir'], 'best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'history': history,
            }, best_path)
            print(f"✅ New best model! L2: {best_val_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter}/{config['patience']} epochs")
            
            if patience_counter >= config['patience']:
                print(f"\n🛑 Early stopping triggered!")
                break
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best val L2: {best_val_loss:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
