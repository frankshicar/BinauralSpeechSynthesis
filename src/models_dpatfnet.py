"""
DPATFNet: Dual-Path Attention TF-domain Network for Binaural Speech Synthesis
Based on ICASSP 2025 paper

Architecture:
- Encoder: Conv2d layers on STFT magnitude
- DPAB (Dual-Path Attention Block): Intra-frame + Inter-frame Self-Attention + Position Cross-Attention
- Decoder: Conv2d layers to predict complex spectrum
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathAttentionBlock(nn.Module):
    """
    Dual-Path Attention Block (DPAB)
    - Intra-frame: Self-Attention across frequency bins (within each time frame)
    - Inter-frame: Self-Attention across time frames (within each frequency bin)
    - Position: Cross-Attention with position encoding
    """
    def __init__(self, channels=256, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        # Intra-frame (frequency dimension)
        self.intra_norm = nn.LayerNorm(channels)
        self.intra_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # Inter-frame (time dimension)
        self.inter_norm = nn.LayerNorm(channels)
        self.inter_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # Position cross-attention
        self.pos_norm = nn.LayerNorm(channels)
        self.pos_cross_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # Feed-forward
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels)
        )
    
    def forward(self, x, pos_encoding):
        """
        Args:
            x: B×C×F×T (feature map)
            pos_encoding: B×C (position encoding)
        Returns:
            x: B×C×F×T (processed feature map)
        """
        B, C, F, T = x.shape
        
        # Intra-frame attention (across frequency)
        x_intra = x.permute(0, 3, 2, 1)  # B×T×F×C
        x_intra = x_intra.reshape(B * T, F, C)
        x_intra = self.intra_norm(x_intra)
        x_intra_out, _ = self.intra_attn(x_intra, x_intra, x_intra)
        x_intra_out = x_intra_out.reshape(B, T, F, C).permute(0, 3, 2, 1)  # B×C×F×T
        x = x + x_intra_out
        
        # Inter-frame attention (across time)
        x_inter = x.permute(0, 2, 3, 1)  # B×F×T×C
        x_inter = x_inter.reshape(B * F, T, C)
        x_inter = self.inter_norm(x_inter)
        x_inter_out, _ = self.inter_attn(x_inter, x_inter, x_inter)
        x_inter_out = x_inter_out.reshape(B, F, T, C).permute(0, 3, 1, 2)  # B×C×F×T
        x = x + x_inter_out
        
        # Position cross-attention
        x_pos = x.permute(0, 2, 3, 1).reshape(B, F * T, C)  # B×(F*T)×C
        x_pos = self.pos_norm(x_pos)
        pos_query = pos_encoding.unsqueeze(1)  # B×1×C
        x_pos_out, _ = self.pos_cross_attn(pos_query, x_pos, x_pos)
        x_pos_out = x_pos_out.squeeze(1).unsqueeze(2).unsqueeze(3).expand(B, C, F, T)
        x = x + x_pos_out
        
        # Feed-forward
        x_ff = x.permute(0, 2, 3, 1)  # B×F×T×C
        x_ff = self.ff_norm(x_ff)
        x_ff = self.ff(x_ff)
        x_ff = x_ff.permute(0, 3, 1, 2)  # B×C×F×T
        x = x + x_ff
        
        return x


class PositionEncoder(nn.Module):
    """Encode position (view) to feature vector"""
    def __init__(self, view_dim=7, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(view_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, view):
        """
        Args:
            view: B×7 or B×7×K
        Returns:
            pos_encoding: B×output_dim
        """
        if view.dim() == 3:
            view = view.mean(dim=2)  # B×7×K → B×7
        return self.encoder(view)


class DPATFNet(nn.Module):
    """
    DPATFNet: Dual-Path Attention TF-domain Network
    """
    def __init__(self, 
                 n_fft=1024,
                 hop_size=256,
                 channels=256,
                 num_dpab=4,
                 num_heads=8,
                 use_cuda=True):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.n_freq = n_fft // 2 + 1
        
        # Position encoder
        self.position_encoder = PositionEncoder(view_dim=7, output_dim=channels)
        
        # Encoder: STFT magnitude → feature map
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # DPAB stack
        self.dpab_blocks = nn.ModuleList([
            DualPathAttentionBlock(channels, num_heads)
            for _ in range(num_dpab)
        ])
        
        # Decoder: feature map → complex spectrum (real + imag)
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, padding=1)  # 4 channels: real_L, imag_L, real_R, imag_R
        )
        
        if use_cuda and torch.cuda.is_available():
            self.cuda()
    
    def forward(self, mono, view):
        """
        Args:
            mono: B×1×T
            view: B×7 or B×7×K
        Returns:
            y_binaural: B×2×T
            outputs: dict with Y_L, Y_R (complex)
        """
        B, _, T = mono.shape
        device = mono.device
        
        # 1. STFT
        window = torch.hann_window(self.n_fft).to(device)
        Y_mono = torch.stft(mono.squeeze(1), self.n_fft, self.hop_size,
                           window=window, return_complex=True)  # B×F×T_stft
        
        # 2. Magnitude as input
        Mag_mono = torch.abs(Y_mono).unsqueeze(1)  # B×1×F×T_stft
        
        # 3. Position encoding
        pos_encoding = self.position_encoder(view)  # B×channels
        
        # 4. Encoder
        x = self.encoder(Mag_mono)  # B×channels×F×T_stft
        
        # 5. DPAB stack
        for dpab in self.dpab_blocks:
            x = dpab(x, pos_encoding)
        
        # 6. Decoder
        out = self.decoder(x)  # B×4×F×T_stft
        
        # 7. Split to real and imaginary parts
        real_L = out[:, 0, :, :]
        imag_L = out[:, 1, :, :]
        real_R = out[:, 2, :, :]
        imag_R = out[:, 3, :, :]
        
        Y_L = torch.complex(real_L, imag_L)
        Y_R = torch.complex(real_R, imag_R)
        
        # 8. iSTFT
        y_L = torch.istft(Y_L, self.n_fft, self.hop_size, window=window, length=T)
        y_R = torch.istft(Y_R, self.n_fft, self.hop_size, window=window, length=T)
        
        y_binaural = torch.stack([y_L, y_R], dim=1)  # B×2×T
        
        return y_binaural, {
            'Y_L': Y_L,
            'Y_R': Y_R
        }


if __name__ == "__main__":
    # Test
    model = DPATFNet(channels=128, num_dpab=2, use_cuda=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    mono = torch.randn(2, 1, 48000)
    view = torch.randn(2, 7)
    
    y, outputs = model(mono, view)
    print(f"Output shape: {y.shape}")
    print(f"Y_L shape: {outputs['Y_L'].shape}")
