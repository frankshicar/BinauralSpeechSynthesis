"""
Training script for DPATFNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import BinauralDataset
from src.models_dpatfnet import DPATFNet
import os
import json


def complex_mse_loss(pred_complex, target_complex):
    """MSE loss for complex tensors"""
    loss_real = F.mse_loss(pred_complex.real, target_complex.real)
    loss_imag = F.mse_loss(pred_complex.imag, target_complex.imag)
    return loss_real + loss_imag


def train_epoch(model, dataloader, optimizer, device, n_fft, hop_size):
    model.train()
    total_loss = 0
    
    for mono, target, view in dataloader:
        mono = mono.to(device)
        target = target.to(device)
        view = view.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred, outputs = model(mono, view)
        
        # GT complex spectrum
        window = torch.hann_window(n_fft).to(device)
        Y_L_gt = torch.stft(target[:, 0, :], n_fft, hop_size, window=window, return_complex=True)
        Y_R_gt = torch.stft(target[:, 1, :], n_fft, hop_size, window=window, return_complex=True)
        
        # Loss: Complex MSE + Time-domain L2
        loss_complex = complex_mse_loss(outputs['Y_L'], Y_L_gt) + \
                      complex_mse_loss(outputs['Y_R'], Y_R_gt)
        loss_time = F.mse_loss(pred, target)
        
        loss = loss_complex + 10 * loss_time  # Weight time-domain loss higher
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, n_fft, hop_size):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mono, target, view in dataloader:
            mono = mono.to(device)
            target = target.to(device)
            view = view.to(device)
            
            pred, outputs = model(mono, view)
            
            window = torch.hann_window(n_fft).to(device)
            Y_L_gt = torch.stft(target[:, 0, :], n_fft, hop_size, window=window, return_complex=True)
            Y_R_gt = torch.stft(target[:, 1, :], n_fft, hop_size, window=window, return_complex=True)
            
            loss_complex = complex_mse_loss(outputs['Y_L'], Y_L_gt) + \
                          complex_mse_loss(outputs['Y_R'], Y_R_gt)
            loss_time = F.mse_loss(pred, target)
            
            loss = loss_complex + 10 * loss_time
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Config - 200ms 音訊只能用 batch_size=8
    config = {
        'batch_size': 8,  # 16 → 8（Dual-Path Attention 記憶體需求大）
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'sample_rate': 48000,
        'n_fft': 1024,
        'hop_size': 64,
        'channels': 64,
        'num_dpab': 2,
        'num_heads': 4,
        'save_interval': 10
    }
    
    print("Configuration (Memory-constrained):")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Dataset - 使用 200ms 音訊（與 v8.3 一致）
    print("Loading datasets...")
    train_dataset = BinauralDataset('dataset/trainset', chunk_size_ms=200)
    val_dataset = BinauralDataset('dataset/testset', chunk_size_ms=200)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Model
    print("Initializing model...")
    model = DPATFNet(
        n_fft=config['n_fft'],
        hop_size=config['hop_size'],
        channels=config['channels'],
        num_dpab=config['num_dpab'],
        num_heads=config['num_heads']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Output directory
    os.makedirs('outputs_dpatfnet/checkpoints', exist_ok=True)
    
    # Training
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                config['n_fft'], config['hop_size'])
        val_loss = validate(model, val_loader, device,
                           config['n_fft'], config['hop_size'])
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 計算詳細指標
        model.eval()
        with torch.no_grad():
            mono_sample, target_sample, view_sample = next(iter(val_loader))
            mono_sample = mono_sample.to(device)
            target_sample = target_sample.to(device)
            view_sample = view_sample.to(device)
            
            pred_sample, outputs_sample = model(mono_sample, view_sample)
            
            # GT complex spectrum
            window = torch.hann_window(config['n_fft']).to(device)
            Y_L_gt = torch.stft(target_sample[:, 0, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            Y_R_gt = torch.stft(target_sample[:, 1, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            
            # 各項指標
            l2_loss = F.mse_loss(pred_sample, target_sample).item()
            complex_L_loss = complex_mse_loss(outputs_sample['Y_L'], Y_L_gt).item()
            complex_R_loss = complex_mse_loss(outputs_sample['Y_R'], Y_R_gt).item()
            
            # Magnitude and Phase error
            mag_L_err = F.mse_loss(torch.abs(outputs_sample['Y_L']), torch.abs(Y_L_gt)).item()
            mag_R_err = F.mse_loss(torch.abs(outputs_sample['Y_R']), torch.abs(Y_R_gt)).item()
            
            phase_L_err = F.mse_loss(torch.angle(outputs_sample['Y_L']), torch.angle(Y_L_gt)).item()
            phase_R_err = F.mse_loss(torch.angle(outputs_sample['Y_R']), torch.angle(Y_R_gt)).item()
        
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']}: "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")
        print(f"  L2: {l2_loss:.6f} | Complex_L: {complex_L_loss:.6f} | Complex_R: {complex_R_loss:.6f}")
        print(f"  Mag_L: {mag_L_err:.6f} | Mag_R: {mag_R_err:.6f} | Phase_L: {phase_L_err:.4f} | Phase_R: {phase_R_err:.4f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'l2': l2_loss,
            'complex_L': complex_L_loss,
            'complex_R': complex_R_loss,
            'mag_L': mag_L_err,
            'mag_R': mag_R_err,
            'phase_L': phase_L_err,
            'phase_R': phase_R_err
        })
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = f'outputs_dpatfnet/checkpoints/epoch_{epoch+1:03d}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, 'outputs_dpatfnet/best_model.pt')
    
    # Save history
    with open('outputs_dpatfnet/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
