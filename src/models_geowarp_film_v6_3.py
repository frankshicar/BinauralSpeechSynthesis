"""
GeoWarpFiLMNet v6.3

v6.3 改進（相對 v6）：
1. 對數頻帶分配：64 個頻帶改為對數等分，低頻（0~1500Hz）從 4 個頻帶增加到 ~20 個
2. NeuralWarpCorrector 加深：4 層 → 6 層，更大感受野
3. IPD loss 改為低頻 sin²(diff/2)（在訓練腳本中）
"""
import sys
sys.path.insert(0, '/home/sbplab/frank/BinauralSpeechSynthesis')

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import GeometricWarper
from src.warping import MonotoneTimeWarper


# ---------------------------------------------------------------------------
# 1. NeuralWarpCorrector（仿 Meta Warpnet.neural_warpfield）
# ---------------------------------------------------------------------------

class NeuralWarpCorrector(nn.Module):
    """
    學習修正 geometric warpfield 的 ITD 低估。
    輸入：view (B, 7, K)，geometric_warpfield (B, 2, T)
    輸出：修正後的 warped audio (B, 2, T)

    設計：4 層 causal Conv1d（與 Meta Warpnet 相同），輸出 delta warpfield，
    最終 warpfield = geometric + delta，確保 causality 後用 MonotoneTimeWarper 施加。
    """
    def __init__(self, view_dim=7, channels=64, layers=6):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2)
            for l in range(layers)
        ])
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()

    def forward(self, mono, view, geometric_warpfield):
        """
        mono:               (B, 1, T)
        view:               (B, 7, K)
        geometric_warpfield:(B, 2, T)
        returns:            warped (B, 2, T)
        """
        x = view
        for conv in self.convs:
            x = F.pad(x, [1, 0])   # causal padding
            x = F.relu(conv(x))
        delta = self.linear(x)                                      # (B, 2, K)
        delta = F.interpolate(delta, size=mono.shape[-1])           # (B, 2, T)

        warpfield = geometric_warpfield + delta
        warpfield = -F.relu(-warpfield)                             # causality: no look-ahead

        mono_stereo = torch.cat([mono, mono], dim=1)                # (B, 2, T)
        return self.neural_warper(mono_stereo, warpfield)           # (B, 2, T)


# ---------------------------------------------------------------------------
# 2. TemporalPositionEncoder（per-frame conditioning）
# ---------------------------------------------------------------------------

class TemporalPositionEncoder(nn.Module):
    """
    輸出 per-frame conditioning，保留時間維度。
    v5b 的 FourierPositionEncoder 把整段壓成一個向量（全段平均），
    v6 輸出 (B, output_dim, T_stft)，每個 STFT frame 有獨立的 conditioning。

    輸入：view (B, 7, K)
    輸出：pos_feat (B, output_dim, T_stft)
    """
    def __init__(self, input_dim=7, L=8, output_dim=256):
        super().__init__()
        self.L = L
        freqs = (2.0 ** torch.arange(L)) * torch.pi
        self.register_buffer('freqs', freqs)

        # 輸入：pos Fourier + vel Fourier + raw view = input_dim*4L + input_dim
        in_ch = input_dim * 4 * L + input_dim
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def _fourier(self, x):
        """x: (B, C, K) → (B, C*2L, K)"""
        scaled = x.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)  # (B, C, K, L)
        enc = torch.cat([scaled.sin(), scaled.cos()], dim=-1)     # (B, C, K, 2L)
        return enc.permute(0, 1, 3, 2).flatten(1, 2)             # (B, C*2L, K)

    def forward(self, view, T_stft):
        """
        view:   (B, 7, K)
        T_stft: int，目標時間維度
        returns:(B, output_dim, T_stft)
        """
        # 速度（差分）
        vel = torch.zeros_like(view)
        vel[:, :, 1:] = view[:, :, 1:] - view[:, :, :-1]

        pos_enc = self._fourier(view)   # (B, 7*2L, K)
        vel_enc = self._fourier(vel)    # (B, 7*2L, K)
        feat = torch.cat([pos_enc, vel_enc, view], dim=1)  # (B, 7*4L+7, K)

        feat = self.mlp(feat)           # (B, output_dim, K)
        return F.interpolate(feat, size=T_stft, mode='linear', align_corners=False)


# ---------------------------------------------------------------------------
# 3. FiLMLayer（per-frame）
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """
    64-band FiLM，接受 per-frame conditioning。
    x:        (B, C, F, T)
    pos_feat: (B, pos_dim, T)
    """
    def __init__(self, channels, pos_dim=256, num_bands=64):
        super().__init__()
        self.num_bands = num_bands
        # Conv1d 在時間維度上預測每幀的 gamma/beta
        self.film_net = nn.Conv1d(pos_dim, num_bands * 2, kernel_size=1)
        self.band_ids = None

    def _init_band_ids(self, F, device):
        # 對數頻帶分配：低頻密集，高頻稀疏（模擬人耳感知）
        # 從 bin 1 開始對數等分（bin 0 是直流，單獨歸入第一個頻帶）
        import numpy as np
        edges = np.unique(np.round(
            np.logspace(0, np.log10(F), self.num_bands + 1)
        ).astype(int).clip(0, F))
        # 確保有足夠的邊界點
        while len(edges) < self.num_bands + 1:
            edges = np.unique(np.append(edges, edges[:-1] + 1))
        edges = edges[:self.num_bands + 1]

        band_ids = torch.zeros(F, dtype=torch.long, device=device)
        for i in range(len(edges) - 1):
            band_ids[edges[i]:edges[i+1]] = i
        self.band_ids = band_ids

    def forward(self, x, pos_feat):
        B, C, F, T = x.shape
        if self.band_ids is None or self.band_ids.device != x.device:
            self._init_band_ids(F, x.device)

        params = self.film_net(pos_feat)                    # (B, num_bands*2, T)
        params = params.view(B, self.num_bands, 2, T)

        gamma = params[:, self.band_ids, 0, :]              # (B, F, T)
        beta  = params[:, self.band_ids, 1, :]              # (B, F, T)

        return x * gamma.unsqueeze(1) + beta.unsqueeze(1)  # (B, C, F, T)


# ---------------------------------------------------------------------------
# 4. FiLM ResBlock
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
# 5. GeoWarpFiLMNet v6
# ---------------------------------------------------------------------------

class GeoWarpFiLMNet(nn.Module):
    """
    v6：GeometricWarper + NeuralWarpCorrector + TemporalPositionEncoder + per-frame FiLM ResStack

    前向流程：
      mono + view
        ↓
      GeometricWarper._warpfield()     → geometric_warpfield (B, 2, T)
        ↓
      NeuralWarpCorrector(view)        → delta，warpfield = geo + delta → y_init (B, 2, T)
        ↓
      STFT(y_init)                     → 頻域特徵
        ↓
      TemporalPositionEncoder(view)    → pos_feat (B, pos_dim, T_stft)  [per-frame]
        ↓
      FiLM ResStack (64-band, per-frame)
        ↓
      輸出 (B, 2, T)
    """
    def __init__(self,
                 n_fft=1024,
                 hop_length=256,
                 channels=128,
                 num_blocks=8,
                 fourier_L=8,
                 num_bands=64,
                 pos_dim=256,
                 warp_channels=64,
                 warp_layers=4):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        n_freq = n_fft // 2 + 1  # 513

        # 1. Geometric warper
        self.geo_warper = GeometricWarper(sampling_rate=48000)

        # 2. Neural warp corrector（新增）
        self.neural_warp = NeuralWarpCorrector(7, warp_channels, warp_layers)

        # 3. Temporal position encoder（per-frame，取代 FourierPositionEncoder）
        self.pos_encoder = TemporalPositionEncoder(7, fourier_L, pos_dim)

        # 4. Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # 5. FiLM ResStack
        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(channels, pos_dim, num_bands, dilations[i % len(dilations)])
            for i in range(num_blocks)
        ])

        # 6. Output heads
        self.output_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Softplus(),
        )
        self.phase_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, kernel_size=(1, 1)),
            nn.Tanh(),
        )

        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, mono, view):
        """
        mono: (B, 1, T)
        view: (B, 7, K)
        returns: y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init
        """
        B, _, T = mono.shape

        # 1. Geometric warpfield
        geo_wf = self.geo_warper._warpfield(view, T)        # (B, 2, T)

        # 2. Neural warp correction → y_init
        y_init = self.neural_warp(mono, view, geo_wf)       # (B, 2, T)

        # 3. STFT
        def stft(x):
            return torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=self.window,
                               return_complex=True)

        Y_L_init = stft(y_init[:, 0])   # (B, F, T_stft)
        Y_R_init = stft(y_init[:, 1])

        x = torch.stack([Y_L_init.abs(), Y_R_init.abs()], dim=1)  # (B, 2, F, T_stft)
        T_stft = x.shape[-1]

        # 4. Per-frame position encoding
        pos_feat = self.pos_encoder(view, T_stft)           # (B, pos_dim, T_stft)

        # 5. Encode + FiLM ResStack
        x = self.encoder(x)
        for block in self.res_blocks:
            x = block(x, pos_feat)

        # 6. Magnitude + phase residual
        mag_out = self.output_head(x)
        mag_L_out = mag_out[:, 0].clamp(min=1e-6)
        mag_R_out = mag_out[:, 1].clamp(min=1e-6)

        phase_res = self.phase_head(x) * torch.pi          # (B, 2, F, T_stft)
        phase_L_geo = torch.angle(Y_L_init).detach()
        phase_R_geo = torch.angle(Y_R_init).detach()
        phase_L = torch.atan2(torch.sin(phase_L_geo + phase_res[:, 0]),
                               torch.cos(phase_L_geo + phase_res[:, 0]))
        phase_R = torch.atan2(torch.sin(phase_R_geo + phase_res[:, 1]),
                               torch.cos(phase_R_geo + phase_res[:, 1]))

        # 7. Reconstruct
        Y_L = mag_L_out * torch.exp(1j * phase_L)
        Y_R = mag_R_out * torch.exp(1j * phase_R)

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
    view[:, 3:, :] = F.normalize(view[:, 3:, :], dim=1)

    y_L, y_R, Y_L, Y_R, Y_L_init, Y_R_init = model(mono, view)
    print(f"Output L: {y_L.shape}, R: {y_R.shape}")
    print(f"STFT L: {Y_L.shape}")
