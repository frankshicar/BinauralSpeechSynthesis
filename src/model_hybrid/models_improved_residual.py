import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViewEncoder(nn.Module):
    def __init__(self, view_dim=7, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, view):
        # view: (B, 7, T) -> (B, T, 7)
        view = view.transpose(1, 2)
        # Take mean over time
        view = view.mean(dim=1)  # (B, 7)
        return self.net(view)  # (B, 256)


class MagnitudeNet(nn.Module):
    def __init__(self, freq_bins=513, view_dim=256, hidden_dim=512):
        super().__init__()
        self.freq_bins = freq_bins
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(freq_bins, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # View fusion
        self.view_proj = nn.Linear(view_dim, hidden_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, freq_bins * 2, 3, padding=1)
        )
    
    def forward(self, mono_mag, view_feat):
        # mono_mag: (B, F, T)
        # view_feat: (B, 256)
        
        # Encode
        x = self.encoder(mono_mag)  # (B, hidden, T)
        
        # Add view information
        view_feat = self.view_proj(view_feat)  # (B, hidden)
        view_feat = view_feat.unsqueeze(-1)  # (B, hidden, 1)
        x = x + view_feat  # Broadcast
        
        # Decode
        out = self.decoder(x)  # (B, F*2, T)
        
        # Split L/R
        mag_L = out[:, :self.freq_bins]
        mag_R = out[:, self.freq_bins:]
        
        # Ensure positive
        mag_L = F.softplus(mag_L)
        mag_R = F.softplus(mag_R)
        
        return mag_L, mag_R


class ResidualITDNet(nn.Module):
    """Smaller network for learning residual ITD"""
    def __init__(self, freq_bins=513, view_dim=256, hidden_dim=128, num_layers=2):
        super().__init__()
        self.freq_bins = freq_bins
        
        # Encoder
        layers = []
        in_dim = freq_bins
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        
        # View fusion
        self.view_proj = nn.Linear(view_dim, hidden_dim)
        
        # Decoder (output per-frequency residual ITD)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, freq_bins, 3, padding=1)
        )
    
    def forward(self, mono_mag, view_feat):
        # mono_mag: (B, F, T)
        # view_feat: (B, 256)
        
        # Encode
        x = self.encoder(mono_mag)  # (B, hidden, T)
        
        # Add view
        view_feat = self.view_proj(view_feat)
        view_feat = view_feat.unsqueeze(-1)
        x = x + view_feat
        
        # Decode
        residual = self.decoder(x)  # (B, F, T)
        
        return residual


class ImprovedResidualPhaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Components
        self.view_encoder = ViewEncoder(7, 128, 256)
        self.magnitude_net = MagnitudeNet(513, 256, 512)
        self.residual_itd_net = ResidualITDNet(513, 256, 128, 2)
        
        # Physical parameters
        self.head_radius = 0.0875  # 8.75cm
        self.sound_speed = 343.0   # m/s
        
        # Frequency mask: low freq (1.0) -> high freq (0.1)
        self.register_buffer(
            'freq_mask',
            torch.linspace(1.0, 0.1, 513)
        )
        
        # STFT parameters
        self.n_fft = 1024
        self.hop_length = 64
        self.win_length = 1024
    
    def compute_physical_itd(self, view):
        """Compute physical ITD from view (geometric)"""
        # view: (B, 7, T) - [x, y, z, qx, qy, qz, qw]
        # Use x, y for azimuth
        x = view[:, 0].mean(dim=-1)  # (B,)
        y = view[:, 1].mean(dim=-1)  # (B,)
        
        azimuth = torch.atan2(y, x)  # (B,)
        
        # Woodworth formula
        itd = (self.head_radius / self.sound_speed) * (
            torch.sin(azimuth) + azimuth
        )
        
        return itd  # (B,)
    
    def forward(self, mono, view):
        """
        Args:
            mono: (B, 1, L) - mono waveform
            view: (B, 7, T) - view parameters
        
        Returns:
            y_L, y_R: (B, 1, L) - binaural waveforms
            outputs: dict with intermediate results
        """
        # Remove channel dim
        mono = mono.squeeze(1)  # (B, L)
        
        # STFT
        mono_stft = torch.stft(
            mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(mono.device),
            return_complex=True
        )  # (B, F, T)
        
        mono_mag = mono_stft.abs()
        mono_phase = torch.angle(mono_stft)
        
        # View encoding
        view_feat = self.view_encoder(view)  # (B, 256)
        
        # 1. Magnitude (learned)
        mag_L, mag_R = self.magnitude_net(mono_mag, view_feat)
        
        # 2. Physical ITD
        physical_itd = self.compute_physical_itd(view)  # (B,)
        
        # 3. Learned residual (constrained)
        residual = self.residual_itd_net(mono_mag, view_feat)  # (B, F, T)
        
        # Constrain residual
        residual = torch.tanh(residual) * 0.5  # Limit to [-0.5, 0.5]
        
        # Apply frequency mask
        freq_mask = self.freq_mask.view(1, -1, 1)  # (1, F, 1)
        residual = residual * freq_mask  # Low freq large, high freq small
        
        # 4. Total ITD (per frequency)
        physical_itd = physical_itd.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        total_itd = physical_itd + residual  # (B, F, T)
        
        # 5. Phase shift
        freq_bins = torch.fft.rfftfreq(self.n_fft, 1/48000).to(mono.device)
        freq_bins = freq_bins.view(1, -1, 1)  # (1, F, 1)
        phase_shift = 2 * np.pi * freq_bins * total_itd  # (B, F, T)
        
        # 6. Binaural phase
        phase_L = mono_phase + phase_shift / 2
        phase_R = mono_phase - phase_shift / 2
        
        # Wrap to [-π, π]
        phase_L = torch.atan2(torch.sin(phase_L), torch.cos(phase_L))
        phase_R = torch.atan2(torch.sin(phase_R), torch.cos(phase_R))
        
        # 7. Reconstruct complex STFT
        Y_L = mag_L * torch.exp(1j * phase_L)
        Y_R = mag_R * torch.exp(1j * phase_R)
        
        # 8. iSTFT
        y_L = torch.istft(
            Y_L,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(mono.device),
            length=mono.shape[-1]
        )
        
        y_R = torch.istft(
            Y_R,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(mono.device),
            length=mono.shape[-1]
        )
        
        # Add channel dim back
        y_L = y_L.unsqueeze(1)  # (B, 1, L)
        y_R = y_R.unsqueeze(1)  # (B, 1, L)
        
        outputs = {
            'mag_L': mag_L,
            'mag_R': mag_R,
            'phase_L': phase_L,
            'phase_R': phase_R,
            'residual': residual,
            'physical_itd': physical_itd.squeeze(),
            'mono_phase': mono_phase
        }
        
        return y_L, y_R, outputs


if __name__ == '__main__':
    # Test
    model = ImprovedResidualPhaseNet()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward
    batch_size = 2
    length = 9600  # 200ms at 48kHz
    
    mono = torch.randn(batch_size, 1, length)
    view = torch.randn(batch_size, 7, 24)
    
    y_L, y_R, outputs = model(mono, view)
    
    print(f"\nInput shape: {mono.shape}")
    print(f"Output L shape: {y_L.shape}")
    print(f"Output R shape: {y_R.shape}")
    print(f"Residual shape: {outputs['residual'].shape}")
    print(f"Residual range: [{outputs['residual'].min():.3f}, {outputs['residual'].max():.3f}]")
    print(f"Physical ITD: {outputs['physical_itd']}")
    
    print("\n✅ Model test passed!")
