"""
HybridPhysicalLearned: 物理 ITD + 學習 ILD + 殘差修正

結合物理模型和神經網路
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ViewEncoder(nn.Module):
    """視角編碼"""
    def __init__(self, view_dim=7, hidden_dim=128, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(view_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, view):
        # view: B×7×K → B×7
        view_mean = view.mean(dim=-1)
        return self.net(view_mean)


class MagnitudeNet(nn.Module):
    """學習 ILD (magnitude)"""
    def __init__(self, freq_dim=513, view_dim=256, hidden_dim=512):
        super().__init__()
        
        self.freq_encoder = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.view_proj = nn.Linear(view_dim, hidden_dim)
        
        self.decoder_L = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim),
            nn.Softplus(),  # 保證正值
        )
        
        self.decoder_R = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim),
            nn.Softplus(),
        )
    
    def forward(self, mono_mag, view_feat):
        # mono_mag: B×F×T
        # view_feat: B×view_dim
        
        B, F, T = mono_mag.shape
        
        # 時間平均
        mono_mag_mean = mono_mag.mean(dim=-1)  # B×F
        
        # 編碼
        freq_feat = self.freq_encoder(mono_mag_mean)  # B×hidden
        view_feat_proj = self.view_proj(view_feat)    # B×hidden
        
        # 融合
        combined = freq_feat + view_feat_proj  # B×hidden
        
        # 解碼
        mag_L = self.decoder_L(combined).unsqueeze(-1)  # B×F×1
        mag_R = self.decoder_R(combined).unsqueeze(-1)  # B×F×1
        
        # 廣播到時間維度
        mag_L = mag_L.expand(-1, -1, T) * mono_mag  # B×F×T
        mag_R = mag_R.expand(-1, -1, T) * mono_mag  # B×F×T
        
        return mag_L, mag_R


class ResidualITDNet(nn.Module):
    """學習殘差 ITD (頻率相關)"""
    def __init__(self, freq_dim=513, view_dim=256, hidden_dim=256):
        super().__init__()
        
        self.view_proj = nn.Linear(view_dim, hidden_dim)
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim),
            nn.Tanh(),  # 限制範圍
        )
    
    def forward(self, view_feat):
        # view_feat: B×view_dim
        feat = self.view_proj(view_feat)
        residual = self.net(feat) * 0.0001  # 小的修正 (±0.1ms)
        return residual  # B×F


class HybridPhysicalLearned(nn.Module):
    """
    混合物理學習模型
    
    物理 ITD + 學習 ILD + 殘差 ITD
    """
    def __init__(
        self,
        sample_rate=48000,
        n_fft=1024,
        hop_size=64,
        view_dim=7,
        view_hidden=128,
        view_output=256,
        mag_hidden=512,
        itd_hidden=256,
    ):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.freq_dim = n_fft // 2 + 1
        
        # 物理參數
        self.head_radius = 0.0875  # 8.75cm
        self.sound_speed = 343.0   # m/s
        
        # 網路模組
        self.view_encoder = ViewEncoder(view_dim, view_hidden, view_output)
        self.magnitude_net = MagnitudeNet(self.freq_dim, view_output, mag_hidden)
        self.residual_itd_net = ResidualITDNet(self.freq_dim, view_output, itd_hidden)
        
        # 頻率軸
        self.register_buffer(
            'freqs',
            torch.linspace(0, sample_rate/2, self.freq_dim)
        )
    
    def compute_physical_ITD(self, azimuth):
        """
        計算物理 ITD (Woodworth 公式)
        ITD = (r/c) × (sin(θ) + θ)
        """
        theta = azimuth * np.pi / 180  # 轉弧度
        ITD = (self.head_radius / self.sound_speed) * (
            torch.sin(theta) + theta
        )
        return ITD  # B
    
    def forward(self, mono, view):
        """
        Args:
            mono: B×1×T (時域)
            view: B×7×K
        
        Returns:
            binaural: B×2×T (時域)
            outputs: dict (中間結果)
        """
        B = mono.size(0)
        
        # STFT
        window = torch.hann_window(self.n_fft).to(mono.device)
        mono_stft = torch.stft(
            mono.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=window,
            return_complex=True,
        )  # B×F×T
        
        mono_mag = mono_stft.abs()
        mono_phase = torch.angle(mono_stft)
        
        # 編碼視角
        view_feat = self.view_encoder(view)  # B×view_output
        
        # === Magnitude 分支 (學習 ILD) ===
        mag_L, mag_R = self.magnitude_net(mono_mag, view_feat)
        
        # === Phase 分支 (物理 + 殘差) ===
        # 1. 物理 ITD
        azimuth = view[:, 0].mean(dim=-1)  # B (假設第一維是方位角)
        physical_ITD = self.compute_physical_ITD(azimuth)  # B
        
        # 2. 殘差 ITD
        residual_ITD = self.residual_itd_net(view_feat)  # B×F
        
        # 3. 總 ITD (頻率相關)
        total_ITD = physical_ITD.unsqueeze(1) + residual_ITD  # B×F
        
        # 4. 計算相位偏移
        F, T_stft = mono_phase.shape[1], mono_phase.shape[2]
        phase_shift = 2 * np.pi * self.freqs[:F].unsqueeze(0).unsqueeze(-1) * total_ITD[:, :F].unsqueeze(-1)  # B×F×1
        phase_shift = phase_shift.expand(-1, -1, T_stft)  # B×F×T
        
        # 5. 應用到 mono phase
        phase_L = mono_phase + phase_shift / 2
        phase_R = mono_phase - phase_shift / 2
        
        # === 合成 ===
        Y_L = mag_L * torch.exp(1j * phase_L)
        Y_R = mag_R * torch.exp(1j * phase_R)
        
        # ISTFT
        binaural_L = torch.istft(
            Y_L,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=window,
            length=mono.size(-1),
        )
        binaural_R = torch.istft(
            Y_R,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=window,
            length=mono.size(-1),
        )
        
        binaural = torch.stack([binaural_L, binaural_R], dim=1)  # B×2×T
        
        outputs = {
            'mag_L': mag_L,
            'mag_R': mag_R,
            'phase_L': phase_L,
            'phase_R': phase_R,
            'physical_ITD': physical_ITD,
            'residual_ITD': residual_ITD,
            'total_ITD': total_ITD,
            'Y_L': Y_L,
            'Y_R': Y_R,
        }
        
        return binaural, outputs


def test_model():
    """測試模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HybridPhysicalLearned(
        sample_rate=48000,
        n_fft=1024,
        hop_size=64,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 測試
    B, T, K = 4, 9600, 24
    mono = torch.randn(B, 1, T).to(device)
    view = torch.randn(B, 7, K).to(device)
    
    print(f"\nInput shapes:")
    print(f"  mono: {mono.shape}")
    print(f"  view: {view.shape}")
    
    with torch.no_grad():
        binaural, outputs = model(mono, view)
    
    print(f"\nOutput shapes:")
    print(f"  binaural: {binaural.shape}")
    print(f"  physical_ITD: {outputs['physical_ITD'].shape}")
    print(f"  residual_ITD: {outputs['residual_ITD'].shape}")
    print(f"  mag_L: {outputs['mag_L'].shape}")
    
    print("\n✅ Model test passed!")


if __name__ == '__main__':
    test_model()
