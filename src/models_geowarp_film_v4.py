"""
GeoWarpFiLMNet：GeometricWarper + Fourier Position Encoder + 64-band FiLM ResStack

架構：
  1. GeometricWarper（Meta 完整幾何 ITD，含 quaternion）→ y_init
  2. STFT(y_init) → 左右耳頻域特徵 concat
  3. Fourier Position Encoder（L=8）→ pos_feat
  4. Conv1d Encoder
  5. FiLM ResStack × N（每層注入 64-band FiLM）
  6. Conv1 + Tanh → y_L, y_R
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import GeometricWarper


# ---------------------------------------------------------------------------
# Fourier Position Encoder
# ---------------------------------------------------------------------------

class FourierPositionEncoder(nn.Module):
    """
    Multi-scale Fourier Features with temporal awareness (v4).
    Captures motion patterns (velocity/acceleration) via 1D Conv.
    Input:  view (B, 7, K)
    Output: pos_feat (B, output_dim)
    """
    def __init__(self, input_dim=7, L=8, output_dim=256):
        super().__init__()
        self.L = L
        freqs = (2.0 ** torch.arange(L)) * torch.pi   # (L,)
        self.register_buffer('freqs', freqs)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2 * L + input_dim, 512),  # +input_dim for velocity
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, view):
        # view: (B, 7, K)
        B, C, K = view.shape
        
        # Temporal features (motion) - use actual velocity
        if K > 1:
            velocity = (view[:, :, 1:] - view[:, :, :-1]).mean(dim=2)  # (B, 7) - actual velocity
        else:
            velocity = torch.zeros(B, C, device=view.device)
        
        # Static position (for Fourier)
        pos = view.mean(dim=2)  # (B, 7)
        
        # Fourier encoding
        scaled = pos.unsqueeze(2) * self.freqs.view(1, 1, -1)
        feat = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=2)  # (B, 7, 2L)
        feat = feat.flatten(1)  # (B, 7*2L)
        
        # Combine Fourier + velocity
        combined = torch.cat([feat, velocity], dim=1)  # (B, 7*2L + 7)
        return self.mlp(combined)  # (B, output_dim)


# ---------------------------------------------------------------------------
# 64-band FiLM modulation
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation per frequency band.
    Splits F frequency bins into num_bands bands, applies independent gamma/beta.
    """
    def __init__(self, channels, pos_dim=256, num_bands=64):
        super().__init__()
        self.num_bands = num_bands
        self.channels = channels
        # predict gamma and beta for each band
        self.film_net = nn.Linear(pos_dim, num_bands * 2)
        
        # Precompute band assignment (will be set in first forward)
        self.band_ids = None

    def forward(self, x, pos_feat):
        """
        x:        (B, C, F, T)  — frequency-domain feature
        pos_feat: (B, pos_dim)
        """
        B, C, F, T = x.shape
        
        # Lazy init band_ids on first forward (now we know F)
        if self.band_ids is None:
            band_edges = [round(F * i / self.num_bands) for i in range(self.num_bands + 1)]
            band_ids = torch.zeros(F, dtype=torch.long, device=x.device)
            for i in range(self.num_bands):
                band_ids[band_edges[i]:band_edges[i+1]] = i
            self.band_ids = band_ids  # (F,)
        
        params = self.film_net(pos_feat)                # (B, num_bands*2)
        params = params.view(B, self.num_bands, 2)
        
        # Vectorized gather: (B, num_bands, 2) → (B, F, 2) via band_ids
        gamma = params[:, self.band_ids, 0].unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)
        beta  = params[:, self.band_ids, 1].unsqueeze(1).unsqueeze(-1)  # (B, 1, F, 1)

        return x * gamma + beta  # no clone, single fused kernel


# ---------------------------------------------------------------------------
# FiLM ResStack block
# ---------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    def __init__(self, channels, pos_dim=256, num_bands=64, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3),
                      padding=(dilation, 1), dilation=(dilation, 1)),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(channels),
        )
        self.film = FiLMLayer(channels, pos_dim, num_bands)
        self.relu = nn.ReLU()

    def forward(self, x, pos_feat):
        residual = x
        x = self.conv(x)
        x = self.film(x, pos_feat)
        return self.relu(x + residual)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GeoWarpFiLMNet(nn.Module):
    """
    v4: GeometricWarper + Fourier Encoder + 64-band FiLM ResStack
    Improvements:
    - Temporal awareness (1D Conv in Position Encoder)
    - Relaxed phase residual (±π instead of ±π/2)
    - Increased depth (8 blocks instead of 6)
    """
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 channels=128,
                 num_blocks=8,  # v4: 6 → 8
                 fourier_L=8,
                 num_bands=64,
                 pos_dim=256):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        n_freq = n_fft // 2 + 1   # 513

        # 1. Geometric warper (Meta, full quaternion)
        self.geo_warper = GeometricWarper(sampling_rate=48000)

        # 2. Fourier position encoder
        self.pos_encoder = FourierPositionEncoder(7, fourier_L, pos_dim)

        # 3. Encoder: 2 channels (L+R concat) → channels
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # 4. FiLM ResStack
        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(channels, pos_dim, num_bands, dilations[i % len(dilations)])
            for i in range(num_blocks)
        ])

        # 5. Output head → 2 channels (L, R magnitude) + phase residual
        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Softplus(),   # ensure positive magnitude
        )
        
        # Phase residual head (learn phase correction)
        self.phase_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Tanh(),  # [-1, 1] → scale to [-π, π]
        )
        
        # Register window as buffer
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, mono, view):
        """
        Args:
            mono: (B, 1, L)
            view: (B, 7, K)
        Returns:
            y_L, y_R: (B, 1, L)
            Y_L, Y_R: (B, F, T_stft) complex STFT (for IPD loss)
            Y_L_init, Y_R_init: (B, F, T_stft) geometric warp STFT (for mag_anchor)
        """
        B, _, T = mono.shape
        device = mono.device
        
        # Input validation
        assert T >= self.n_fft, f"Input length {T} < n_fft {self.n_fft}"
        assert view.shape[2] > 0, f"view has K=0 frames"

        # 1. Geometric warp → y_init (B, 2, T)
        y_init = self.geo_warper(mono, view)            # (B, 2, T)

        # 2. STFT of warped L/R
        def stft(x):
            return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=self.window,
                               return_complex=True)

        Y_L_init = stft(y_init[:, 0])   # (B, F, T_stft)
        Y_R_init = stft(y_init[:, 1])   # (B, F, T_stft)

        # Magnitude of warped signal as input feature
        mag_L = Y_L_init.abs()          # (B, F, T_stft)
        mag_R = Y_R_init.abs()
        x = torch.stack([mag_L, mag_R], dim=1)   # (B, 2, F, T_stft)

        # 3. Position encoding
        pos_feat = self.pos_encoder(view)           # (B, pos_dim)

        # 4. Encode
        x = self.encoder(x)                         # (B, C, F, T_stft)

        # 5. FiLM ResStack
        for block in self.res_blocks:
            x = block(x, pos_feat)

        # 6. Output magnitude correction
        mag_out = self.output_head(x)               # (B, 2, F, T_stft)
        mag_L_out = mag_out[:, 0].clamp(min=1e-6)   # (B, F, T_stft), prevent zero magnitude
        mag_R_out = mag_out[:, 1].clamp(min=1e-6)
        
        # 7. Output phase residual (v4: relax to ±π for more correction freedom)
        phase_res = self.phase_head(x) * torch.pi   # (B, 2, F, T_stft), scale to [-π, π]
        phase_L_res = phase_res[:, 0]
        phase_R_res = phase_res[:, 1]

        # 8. Combine geometric phase + learned residual (detach geo phase to prevent gradient explosion)
        phase_L_geo = torch.angle(Y_L_init).detach()
        phase_R_geo = torch.angle(Y_R_init).detach()
        phase_L = phase_L_geo + phase_L_res
        phase_R = phase_R_geo + phase_R_res
        
        # Wrap to [-π, π]
        phase_L = torch.atan2(torch.sin(phase_L), torch.cos(phase_L))
        phase_R = torch.atan2(torch.sin(phase_R), torch.cos(phase_R))

        # 9. Reconstruct complex STFT
        Y_L = mag_L_out * torch.exp(1j * phase_L)
        Y_R = mag_R_out * torch.exp(1j * phase_R)
        
        # Shape assertion
        assert Y_L.shape == Y_L_init.shape, f"Shape mismatch: {Y_L.shape} vs {Y_L_init.shape}"

        # 10. iSTFT
        y_L = torch.istft(Y_L, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window, length=T)
        y_R = torch.istft(Y_R, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window, length=T)

        return y_L.unsqueeze(1), y_R.unsqueeze(1), Y_L, Y_R, Y_L_init, Y_R_init


if __name__ == '__main__':
    model = GeoWarpFiLMNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    mono = torch.randn(2, 1, 9600)
    view = torch.randn(2, 7, 24)
    # GeometricWarper needs real-ish quaternions (non-zero norm)
    view[:, 3:, :] = F.normalize(view[:, 3:, :], dim=1)

    y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init = model(mono, view)
    print(f"Output L: {y_L.shape}, R: {y_R.shape}")
    print(f"STFT L: {Y_L.shape}, R: {Y_R.shape}")
    print(f"Init STFT L: {Y_L_init.shape}, R: {Y_R_init.shape}")
    print("✅ Forward pass OK")
