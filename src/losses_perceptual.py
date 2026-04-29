import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for perceptual audio quality"""
    
    def __init__(self, resolutions=None, sample_rate=48000):
        super().__init__()
        
        if resolutions is None:
            # Default: 3 resolutions
            resolutions = [
                {'n_fft': 512, 'hop': 128, 'win': 512},
                {'n_fft': 1024, 'hop': 256, 'win': 1024},
                {'n_fft': 2048, 'hop': 512, 'win': 2048}
            ]
        
        self.resolutions = resolutions
        self.sample_rate = sample_rate
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, L) or (B, 1, L)
            target: (B, L) or (B, 1, L)
        """
        # Remove channel dim if present
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)
        
        total_loss = 0
        
        for res in self.resolutions:
            n_fft = res['n_fft']
            hop = res['hop']
            win = res['win']
            
            # STFT
            window = torch.hann_window(win).to(pred.device)
            
            pred_stft = torch.stft(
                pred,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                return_complex=True
            )
            
            target_stft = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                return_complex=True
            )
            
            # Magnitude
            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()
            
            # Magnitude loss (L1)
            mag_loss = F.l1_loss(pred_mag, target_mag)
            
            # Phase (weighted by magnitude)
            pred_phase = torch.angle(pred_stft)
            target_phase = torch.angle(target_stft)
            
            # Magnitude weighting: phase matters more where magnitude is large
            mag_weight = target_mag / (target_mag.max() + 1e-8)
            
            # Phase loss (using sin/cos to handle wrapping)
            phase_loss = (
                F.l1_loss(torch.sin(pred_phase) * mag_weight, 
                         torch.sin(target_phase) * mag_weight) +
                F.l1_loss(torch.cos(pred_phase) * mag_weight,
                         torch.cos(target_phase) * mag_weight)
            )
            
            # Combine
            total_loss += mag_loss + 0.1 * phase_loss
        
        return total_loss / len(self.resolutions)


def temporal_smoothness_loss(phase):
    """
    Penalize large phase jumps between adjacent frames
    
    Args:
        phase: (B, F, T) - phase values
    
    Returns:
        loss: scalar
    """
    # Phase difference between adjacent frames
    phase_diff = phase[:, :, 1:] - phase[:, :, :-1]
    
    # Wrap to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    # Penalize large jumps
    return torch.mean(phase_diff ** 2)


def residual_regularization(residual):
    """
    L2 regularization on residual ITD
    
    Args:
        residual: (B, F, T) - residual ITD values
    
    Returns:
        loss: scalar
    """
    return torch.mean(residual ** 2)


def compute_losses(pred, target, outputs, config):
    """
    Compute all losses for improved residual phase model
    
    Args:
        pred: tuple of (y_L, y_R) - predicted binaural audio
        target: tuple of (y_L_gt, y_R_gt) - ground truth binaural audio
        outputs: dict with intermediate results
        config: dict with loss weights
    
    Returns:
        total_loss: scalar
        loss_dict: dict with individual losses
    """
    y_L_pred, y_R_pred = pred
    y_L_gt, y_R_gt = target
    
    # 1. Waveform L2
    waveform_loss = (
        F.mse_loss(y_L_pred, y_L_gt) +
        F.mse_loss(y_R_pred, y_R_gt)
    ) / 2
    
    # 2. Perceptual loss (multi-resolution STFT)
    perceptual_loss_fn = MultiResolutionSTFTLoss()
    perceptual_loss = (
        perceptual_loss_fn(y_L_pred, y_L_gt) +
        perceptual_loss_fn(y_R_pred, y_R_gt)
    ) / 2
    
    # 3. Residual regularization
    residual = outputs['residual']
    residual_reg = residual_regularization(residual)
    
    # 4. Temporal smoothness
    phase_L = outputs['phase_L']
    phase_R = outputs['phase_R']
    temporal_smooth = (
        temporal_smoothness_loss(phase_L) +
        temporal_smoothness_loss(phase_R)
    ) / 2
    
    # 5. Magnitude loss (auxiliary)
    mag_L_pred = outputs['mag_L']
    mag_R_pred = outputs['mag_R']
    
    # Compute ground truth magnitude
    y_L_gt_squeeze = y_L_gt.squeeze(1) if y_L_gt.dim() == 3 else y_L_gt
    y_R_gt_squeeze = y_R_gt.squeeze(1) if y_R_gt.dim() == 3 else y_R_gt
    
    window = torch.hann_window(1024).to(y_L_gt.device)
    
    gt_L_stft = torch.stft(
        y_L_gt_squeeze,
        n_fft=1024,
        hop_length=64,
        win_length=1024,
        window=window,
        return_complex=True
    )
    gt_R_stft = torch.stft(
        y_R_gt_squeeze,
        n_fft=1024,
        hop_length=64,
        win_length=1024,
        window=window,
        return_complex=True
    )
    
    mag_L_gt = gt_L_stft.abs()
    mag_R_gt = gt_R_stft.abs()
    
    mag_loss = (
        F.l1_loss(mag_L_pred, mag_L_gt) +
        F.l1_loss(mag_R_pred, mag_R_gt)
    ) / 2
    
    # Get loss weights
    w = config.get('loss_weights', {})
    w_waveform = w.get('waveform', 10.0)
    w_perceptual = w.get('perceptual', 5.0)
    w_magnitude = w.get('magnitude', 1.0)
    w_residual = w.get('residual_reg', 0.01)
    w_temporal = w.get('temporal_smooth', 0.1)
    
    # Total loss
    total_loss = (
        w_waveform * waveform_loss +
        w_perceptual * perceptual_loss +
        w_magnitude * mag_loss +
        w_residual * residual_reg +
        w_temporal * temporal_smooth
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'waveform': waveform_loss.item(),
        'perceptual': perceptual_loss.item(),
        'magnitude': mag_loss.item(),
        'residual_reg': residual_reg.item(),
        'temporal_smooth': temporal_smooth.item()
    }
    
    return total_loss, loss_dict


if __name__ == '__main__':
    # Test
    print("Testing MultiResolutionSTFTLoss...")
    
    loss_fn = MultiResolutionSTFTLoss()
    
    pred = torch.randn(2, 9600)
    target = torch.randn(2, 9600)
    
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.6f}")
    
    print("\nTesting temporal_smoothness_loss...")
    phase = torch.randn(2, 513, 150)
    smooth_loss = temporal_smoothness_loss(phase)
    print(f"Smoothness loss: {smooth_loss.item():.6f}")
    
    print("\nTesting residual_regularization...")
    residual = torch.randn(2, 513, 150) * 0.1
    reg_loss = residual_regularization(residual)
    print(f"Regularization loss: {reg_loss.item():.6f}")
    
    print("\n✅ All loss functions test passed!")
