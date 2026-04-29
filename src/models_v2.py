"""
BinauralTFNet：時頻域分離雙耳音訊合成架構
- Stage 1 (CommonBranch)：時域，學兩耳共同部分，用 L2 + DifferentiableITD 監督
- Stage 2 (SpecificBranch)：頻域，學左右耳差異部分，用 Phase + IPD 監督
- Stage 3：端對端 fine-tune
"""

import numpy as np
import scipy.linalg
from scipy.spatial.transform import Rotation as R

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from src.hyperconv import HyperConvBlock
from src.warping import GeometricTimeWarper, MonotoneTimeWarper
from src.utils import Net


# ─────────────────────────────────────────────
# 複用 models.py 的 Warpnet / HyperConvWavenet
# ─────────────────────────────────────────────

class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)

    def _transmitter_mouth(self, view):
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = th.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        left_ear_offset  = th.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, -0.08, -0.22])
        right_ear_offset = th.Tensor([0,  0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0,  0.08, -0.22])
        displacement_left  = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        return th.stack([displacement_left, displacement_right], dim=1)

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)


class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        )
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()

    def neural_warpfield(self, view, seq_length):
        warpfield = view
        for layer in self.layers:
            warpfield = F.pad(warpfield, pad=[1, 0])
            warpfield = F.relu(layer(warpfield))
        warpfield = self.linear(warpfield)
        return F.interpolate(warpfield, size=seq_length)

    def forward(self, mono, view):
        geo_wf    = self.geometric_warper._warpfield(view, mono.shape[-1])
        neural_wf = self.neural_warpfield(view, mono.shape[-1])
        warpfield = -F.relu(-(geo_wf + neural_wf))
        return self.neural_warper(th.cat([mono, mono], dim=1), warpfield), warpfield


class HyperConvWavenet(nn.Module):
    def __init__(self, z_dim, channels=64, blocks=3, layers_per_block=10, conv_len=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for b in range(blocks):
            for l in range(layers_per_block):
                self.layers.append(HyperConvBlock(channels, channels, z_dim, kernel_size=conv_len, dilation=2**l))

    def forward(self, x, z):
        skips = []
        for layer in self.layers:
            x, skip = layer(x, z)
            skips.append(skip)
        return x, skips


class WaveoutBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first  = nn.Conv1d(channels, channels, kernel_size=1)
        self.second = nn.Conv1d(channels, 1, kernel_size=1)
        self.first.weight.data.uniform_(-np.sqrt(6.0 / channels), np.sqrt(6.0 / channels))

    def forward(self, x):
        return self.second(th.sin(self.first(x)))


# ─────────────────────────────────────────────
# Stage 1：CommonBranch
# ─────────────────────────────────────────────

class CommonBranch(nn.Module):
    def __init__(self, view_dim=7, warpnet_layers=4, warpnet_channels=64,
                 wavenet_blocks=3, layers_per_block=10, wavenet_channels=64):
        super().__init__()
        # 保留 warper（給 Stage 2 用，但 Stage 1 不用）
        self.warper = Warpnet(warpnet_layers, warpnet_channels, view_dim)
        
        # v8.3 最終修正：模仿 BinauralGrad
        # 1. 空間編碼器（處理位置/朝向）
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(view_dim, wavenet_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(wavenet_channels, wavenet_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 2. 音訊編碼器（處理單聲道）
        self.audio_encoder = nn.Conv1d(1, wavenet_channels, kernel_size=1)
        self.audio_encoder.weight.data.uniform_(-np.sqrt(6.0 / 1), np.sqrt(6.0 / 1))
        
        # 3. WaveNet（處理拼接後的特徵）
        # 輸入是 wavenet_channels*2（spatial + audio）
        self.input_proj = nn.Conv1d(wavenet_channels * 2, wavenet_channels, kernel_size=1)
        self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels, wavenet_blocks, layers_per_block)
        self.output_net = WaveoutBlock(wavenet_channels)

    def forward(self, mono, view):
        # 完整左右耳 warp（給 Stage 2 使用）
        warped, warpfield = self.warper(mono, view)
        
        # v8.3 最終修正：從 mono 開始，不用 warped_common
        # 1. 處理空間資訊（位置/朝向）
        spatial_feat = self.spatial_encoder(view)  # B×7×K → B×64×K
        
        # 2. 插值到音訊長度
        T = mono.shape[-1]
        spatial_feat = F.interpolate(spatial_feat, size=T, mode='linear', align_corners=False)  # B×64×T
        
        # 3. 處理音訊
        audio_feat = self.audio_encoder(mono)  # B×1×T → B×64×T
        
        # 4. 拼接（BinauralGrad 的方式）
        combined = th.cat([spatial_feat, audio_feat], dim=1)  # B×128×T
        
        # 5. 投影到 wavenet_channels（HyperConvWavenet 期望 B×64×T）
        x = self.input_proj(combined)  # B×128×T → B×64×T
        
        # 6. WaveNet 合成（用全零 view，避免重複使用空間資訊）
        B, _, K = view.shape
        dummy_view = th.zeros(B, 7, K, device=view.device)
        _, skips = self.hyperconv_wavenet(x, dummy_view)  # x: B×64×T
        y_common = self.output_net(th.mean(th.stack(skips, dim=0), dim=0))  # B×1×T

        return y_common, warped


# ─────────────────────────────────────────────
# Stage 2：SpecificBranch
# ─────────────────────────────────────────────

class ResBlock(nn.Module):
    """頻域 1D 殘差 block，F 維度 flatten 進 channel"""
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3,
                              dilation=dilation, padding=dilation)
        self.cond = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, cond):
        # x: B×C×T_stft, cond: B×C×T_stft
        return x + th.tanh(self.conv(x) + self.cond(cond))


class SimpleDPAB(nn.Module):
    """
    簡化版位置編碼，用 Conv1d 處理 view
    移除 Cross-attention，減少複雜度
    """
    def __init__(self, view_dim=7, cond_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(view_dim, cond_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
    
    def forward(self, view, t_stft):
        # view: B×7×K
        x = F.relu(self.conv1(view))      # B×cond_dim×K
        x = F.relu(self.conv2(x))         # B×cond_dim×K
        x = self.conv3(x)                 # B×cond_dim×K
        
        # 插值到 T_stft
        x = F.interpolate(x, size=t_stft, mode='linear', align_corners=False)
        return x  # B×cond_dim×T_stft


class TFResStack(nn.Module):
    """
    頻域殘差堆疊
    輸入：Y_warp_L 和 Y_warp_R 的 real/imag，flatten F 進 channel
    輸出：Y_delta_L, Y_delta_R（同樣 shape）
    """
    def __init__(self, freq_bins, in_ch=4, channels=128, cond_dim=64, num_blocks=4):
        super().__init__()
        # in_ch=4: [Y_warp_L_real, Y_warp_L_imag, Y_warp_R_real, Y_warp_R_imag]
        # 把 F 維度 flatten 進 channel: in_channels = freq_bins * in_ch
        in_channels = freq_bins * in_ch
        out_channels = freq_bins * 4  # delta_L (real/imag) + delta_R (real/imag)

        self.input_conv = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.cond_proj  = nn.Conv1d(cond_dim, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            ResBlock(channels, dilation=2**i) for i in range(num_blocks)
        ])

        self.output_conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, Y_warp_L, Y_warp_R, cond):
        # Y_warp_L/R: B×F×T_stft×2 (real/imag)
        B, F, T_stft, _ = Y_warp_L.shape

        # flatten: B × (F*4) × T_stft
        x = th.cat([
            Y_warp_L.permute(0, 1, 3, 2).reshape(B, F * 2, T_stft),
            Y_warp_R.permute(0, 1, 3, 2).reshape(B, F * 2, T_stft),
        ], dim=1)

        x = self.input_conv(x)
        c = self.cond_proj(cond)

        for block in self.blocks:
            x = block(x, c)

        out = self.output_conv(x)  # B × (F*4) × T_stft

        # 拆回 delta_L, delta_R
        delta_L = out[:, :F*2, :].reshape(B, F, 2, T_stft).permute(0, 1, 3, 2)  # B×F×T_stft×2
        delta_R = out[:, F*2:, :].reshape(B, F, 2, T_stft).permute(0, 1, 3, 2)  # B×F×T_stft×2
        return delta_L, delta_R


class SpecificBranch(nn.Module):
    def __init__(self, fft_size=1024, hop_size=256,
                 cond_dim=64, tf_channels=256, tf_blocks=8):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        freq_bins = fft_size // 2 + 1  # 513 for fft_size=1024

        self.dpab = SimpleDPAB(view_dim=7, cond_dim=cond_dim)
        self.tf_resstack = TFResStack(freq_bins, in_ch=4, channels=tf_channels,
                                      cond_dim=cond_dim, num_blocks=tf_blocks)

        win = th.hann_window(fft_size)
        self.register_buffer('window', win)

    def _stft(self, x):
        # x: B×1×T → B×F×T_stft×2
        spec = th.stft(x.squeeze(1), n_fft=self.fft_size, hop_length=self.hop_size,
                       win_length=self.fft_size, window=self.window,
                       center=True, return_complex=True)  # B×F×T_stft (complex)
        return th.view_as_real(spec)  # B×F×T_stft×2

    def _istft(self, spec, length):
        # spec: B×F×T_stft×2 → B×T
        B = spec.shape[0]
        spec_c = th.view_as_complex(spec.contiguous())  # B×F×T_stft
        wav = th.istft(spec_c, n_fft=self.fft_size, hop_length=self.hop_size,
                       win_length=self.fft_size, window=self.window,
                       center=True, length=length)
        return wav.unsqueeze(1)  # B×1×T

    def forward(self, y_common, view, seq_length):
        # y_common: B×1×T（CommonBranch 的輸出）
        Y_common = self._stft(y_common)  # B×F×T_stft×2
        
        # 複製成左右聲道（作為 TFResStack 的輸入）
        Y_common_L = Y_common
        Y_common_R = Y_common

        T_stft = Y_common.shape[2]
        cond = self.dpab(view, T_stft)  # B×cond_dim×T_stft

        delta_L, delta_R = self.tf_resstack(Y_common_L, Y_common_R, cond)

        # v8.3 關鍵修改：從 y_common 出發，SpecificBranch 負責全部的左右差異
        Y_L = Y_common + delta_L
        Y_R = Y_common + delta_R

        y_L = self._istft(Y_L, seq_length)  # B×1×T
        y_R = self._istft(Y_R, seq_length)

        return th.cat([y_L, y_R], dim=1)  # B×2×T



# ─────────────────────────────────────────────
# BinauralTFNet：整合兩個 Branch
# ─────────────────────────────────────────────

class BinauralTFNet(Net):
    def __init__(self,
                 view_dim=7,
                 warpnet_layers=4,
                 warpnet_channels=64,
                 wavenet_blocks=3,
                 layers_per_block=10,
                 wavenet_channels=64,
                 fft_size=1024,      # v8.2: 512 → 1024
                 hop_size=256,       # v8.2: 128 → 256
                 cond_dim=64,
                 tf_channels=256,    # v8.2: 128 → 256
                 tf_blocks=8,        # v8.2: 4 → 8
                 model_name='binaural_tfnet',
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.common   = CommonBranch(view_dim, warpnet_layers, warpnet_channels,
                                     wavenet_blocks, layers_per_block, wavenet_channels)
        self.specific = SpecificBranch(fft_size, hop_size,
                                       cond_dim, tf_channels, tf_blocks)
        if use_cuda:
            self.cuda()

    def forward(self, mono, view):
        """
        :param mono:  B×1×T
        :param view:  B×7×K
        :return: dict with 'output' (B×2×T), 'y_common' (B×1×T), 'warped' (B×2×T)
        """
        y_common, warped = self.common(mono, view)
        output = self.specific(y_common, view, mono.shape[-1])  # v8.3: 傳 y_common
        return {"output": output, "y_common": y_common, "warped": warped}
