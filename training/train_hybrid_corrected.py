"""
HybridTFNet 訓練 - 修正 sample_rate=48000
最終版本：STFT domain + FiLM + Phase difference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.dataset import BinauralDataset
from src.models_hybrid import HybridTFNet
import os
import json


def wrapped_mse(pred, target):
    """Wrapped MSE for phase"""
    diff = pred - target
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return (diff ** 2).mean()


def train_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0
    
    for mono, target, view in dataloader:
        mono = mono.to(device)
        target = target.to(device)
        view = view.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        pred, outputs = model(mono, view)
        
        # GT phase difference
        window = torch.hann_window(config['n_fft']).to(device)
        Y_mono = torch.stft(mono.squeeze(1), config['n_fft'], config['hop_size'],
                           window=window, return_complex=True)
        Y_L_gt = torch.stft(target[:, 0, :], config['n_fft'], config['hop_size'],
                           window=window, return_complex=True)
        Y_R_gt = torch.stft(target[:, 1, :], config['n_fft'], config['hop_size'],
                           window=window, return_complex=True)
        
        Phase_diff_L_gt = torch.angle(Y_L_gt / (Y_mono + 1e-8))
        Phase_diff_R_gt = torch.angle(Y_R_gt / (Y_mono + 1e-8))
        
        # Loss: Phase difference + L2
        loss_phase = wrapped_mse(outputs['Phase_L'], Phase_diff_L_gt) + \
                    wrapped_mse(outputs['Phase_R'], Phase_diff_R_gt)
        loss_l2 = F.mse_loss(pred, target)
        
        loss = loss_phase + 10 * loss_l2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for mono, target, view in dataloader:
            mono = mono.to(device)
            target = target.to(device)
            view = view.to(device)
            
            pred, outputs = model(mono, view)
            
            window = torch.hann_window(config['n_fft']).to(device)
            Y_mono = torch.stft(mono.squeeze(1), config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            Y_L_gt = torch.stft(target[:, 0, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            Y_R_gt = torch.stft(target[:, 1, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            
            Phase_diff_L_gt = torch.angle(Y_L_gt / (Y_mono + 1e-8))
            Phase_diff_R_gt = torch.angle(Y_R_gt / (Y_mono + 1e-8))
            
            loss_phase = wrapped_mse(outputs['Phase_L'], Phase_diff_L_gt) + \
                        wrapped_mse(outputs['Phase_R'], Phase_diff_R_gt)
            loss_l2 = F.mse_loss(pred, target)
            
            loss = loss_phase + 10 * loss_l2
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Config - 修正 sample_rate
    config = {
        'batch_size': 16,
        'learning_rate': 3e-4,
        'num_epochs': 100,
        'sample_rate': 48000,  # 修正！
        'n_fft': 1024,
        'hop_size': 64,
        'tf_channels': 128,
        'tf_blocks': 4,
        'save_interval': 10
    }
    
    print("HybridTFNet Training (Corrected sample_rate=48000):")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # Dataset
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
    print("Initializing HybridTFNet...")
    model = HybridTFNet(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        hop_size=config['hop_size'],
        tf_channels=config['tf_channels'],
        tf_blocks=config['tf_blocks'],
        use_cuda=True
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Output directory
    os.makedirs('outputs_hybrid/checkpoints', exist_ok=True)
    
    # Training
    print("="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        val_loss = validate(model, val_loader, device, config)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 詳細指標
        model.eval()
        with torch.no_grad():
            mono_sample, target_sample, view_sample = next(iter(val_loader))
            mono_sample = mono_sample.to(device)
            target_sample = target_sample.to(device)
            view_sample = view_sample.to(device)
            
            pred_sample, outputs_sample = model(mono_sample, view_sample)
            
            window = torch.hann_window(config['n_fft']).to(device)
            Y_mono = torch.stft(mono_sample.squeeze(1), config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            Y_L_gt = torch.stft(target_sample[:, 0, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            Y_R_gt = torch.stft(target_sample[:, 1, :], config['n_fft'], config['hop_size'],
                               window=window, return_complex=True)
            
            Phase_diff_L_gt = torch.angle(Y_L_gt / (Y_mono + 1e-8))
            Phase_diff_R_gt = torch.angle(Y_R_gt / (Y_mono + 1e-8))
            
            l2_loss = F.mse_loss(pred_sample, target_sample).item()
            phase_L_err = wrapped_mse(outputs_sample['Phase_L'], Phase_diff_L_gt).item()
            phase_R_err = wrapped_mse(outputs_sample['Phase_R'], Phase_diff_R_gt).item()
            
            mag_L_err = F.mse_loss(outputs_sample['Mag_L'], torch.abs(Y_L_gt)).item()
            mag_R_err = F.mse_loss(outputs_sample['Mag_R'], torch.abs(Y_R_gt)).item()
        
        print(f"Epoch {epoch+1:3d}/{config['num_epochs']}: "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")
        print(f"  L2: {l2_loss:.6f} | Phase_L: {phase_L_err:.4f} | Phase_R: {phase_R_err:.4f}")
        print(f"  Mag_L: {mag_L_err:.6f} | Mag_R: {mag_R_err:.6f}")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'l2': l2_loss,
            'phase_L': phase_L_err,
            'phase_R': phase_R_err,
            'mag_L': mag_L_err,
            'mag_R': mag_R_err
        })
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = f'outputs_hybrid/checkpoints/epoch_{epoch+1:03d}.pt'
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
            }, 'outputs_hybrid/best_model.pt')
    
    # Save history
    with open('outputs_hybrid/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
