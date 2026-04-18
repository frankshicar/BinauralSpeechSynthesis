"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
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


class GeometricWarper(nn.Module):
    def __init__(self, sampling_rate=48000):
        super().__init__()
        self.warper = GeometricTimeWarper(sampling_rate=sampling_rate)
        self.ear_offset = 0.0875  # 單耳到頭部中心距離，約 8.75cm

    def _transmitter_mouth(self, view):
        # offset between tracking markers and real mouth position in the dataset
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        # make sure zero-padded values are set to non-zero values (else scipy raises an exception)
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = th.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _quaternion_to_right_vector(self, quat):
        """
        從 quaternion 計算右向量（+Y 方向）
        :param quat: quaternion as numpy array (N x 4) in [x, y, z, w] format
        :return: right vector as numpy array (N x 3)
        """
        # scipy 使用 [x, y, z, w] 格式
        rot_mat = R.from_quat(quat)
        # 右向量是 Y 軸方向 [0, 1, 0]
        right_vector = rot_mat.apply([0, 1, 0])
        return right_vector

    def _listener_ear_positions(self, view):
        """
        計算左右耳的實際 3D 位置（在世界坐標系中）
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K
                     view[:, 0:3, :] = transmitter position
                     view[:, 3:7, :] = receiver (listener) orientation quaternion (qx, qy, qz, qw)
        :return: left_ear_pos, right_ear_pos as tensors of shape B x 3 x K
        
        注意：在此數據集中，receiver (listener) 固定在原點 (0, 0, 0)
        但為了通用性，此方法支持 listener 在任意位置
        """
        # 提取 listener 位置（在此數據集中為原點，但保留通用性）
        # 注意：view[:, 0:3, :] 是 transmitter 位置，不是 listener 位置
        # 在此數據集中 listener 固定在原點
        listener_pos = th.zeros_like(view[:, 0:3, :])  # B x 3 x K，全為 0
        
        # 提取 listener 的 quaternion 方向 (qx, qy, qz, qw)
        listener_quat = view[:, 3:7, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()  # (B*K) x 4
        
        # 確保 quaternion 非零（scipy 要求）
        norms = scipy.linalg.norm(listener_quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        listener_quat = listener_quat + eps_val[:, None]
        
        # 計算右向量
        # scipy R.from_quat 使用 [x, y, z, w] 格式，與數據集的 (qx, qy, qz, qw) 一致
        right_vector = self._quaternion_to_right_vector(listener_quat)  # (B*K) x 3
        right_vector = th.Tensor(right_vector).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()  # B x 3 x K
        
        if view.is_cuda:
            right_vector = right_vector.cuda()
            listener_pos = listener_pos.cuda()
        
        # 計算左右耳位置
        # 座標系統：X+ = 前方, Y+ = 右方, Z+ = 上方
        # 左耳 = listener_pos - ear_offset * right_vector (Y- 方向)
        # 右耳 = listener_pos + ear_offset * right_vector (Y+ 方向)
        left_ear_pos = listener_pos - self.ear_offset * right_vector
        right_ear_pos = listener_pos + self.ear_offset * right_vector
        
        return left_ear_pos, right_ear_pos

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        # offset between tracking markers and ears in the dataset (receiver is fixed at origin)
        left_ear_offset = th.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, -0.08, -0.22])
        right_ear_offset = th.Tensor([0, 0.08, -0.22]).cuda() if view.is_cuda else th.Tensor([0, 0.08, -0.22])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        displacement = th.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def _warpfield(self, view, seq_length):
        return self.warper.displacements2warpfield(self._3d_displacements(view), seq_length)

    def forward(self, mono, view):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
        '''
        return self.warper(th.cat([mono, mono], dim=1), self._3d_displacements(view))


class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        # 修改 A: 左右耳獨立的可學習 output scale，初始值 10.0
        # 對應約 10/48000*343 ≈ 0.071m 的 ITD 修正量，接近理論耳間距量級
        self.output_scale = nn.Parameter(th.full((2, 1), 10.0))
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()

    @staticmethod
    def _view_to_features(view):
        """
        修改 B: 將原始 7 維位置向量轉換為幾何意義更明確的特徵
        輸入: view B x 7 x K  (x, y, z, qx, qy, qz, qw)
        輸出: feat B x 7 x K  (azimuth, elevation, distance, sin_az, cos_az, sin_el, cos_el)
        改善左右對稱性：azimuth 對左右是奇函數，sin/cos 展開讓網路更容易學到對稱映射
        """
        x, y, z = view[:, 0], view[:, 1], view[:, 2]
        dist = th.sqrt(x**2 + y**2 + z**2 + 1e-8)
        az = th.atan2(-y, x)          # Y+ = 右 → 負角度為右側，與角度定義一致
        el = th.asin((z / dist).clamp(-1, 1))
        feat = th.stack([az, el, dist, th.sin(az), th.cos(az), th.sin(el), th.cos(el)], dim=1)
        return feat

    def neural_warpfield(self, view, seq_length):
        warpfield = self._view_to_features(view)   # 修改 B: 特徵轉換
        for layer in self.layers:
            warpfield = F.pad(warpfield, pad=[1, 0])
            warpfield = F.relu(layer(warpfield))
        warpfield = self.linear(warpfield)
        warpfield = warpfield * self.output_scale   # 修改 A: 可學習 scale
        warpfield = F.interpolate(warpfield, size=seq_length)
        return warpfield

    def forward(self, mono, view, return_warpfields=False):
        '''
        :param mono: input signal as tensor of shape B x 1 x T
        :param view: rx/tx position/orientation as tensor of shape B x 7 x K (K = T / 400)
        :param return_warpfields: if True, return geometric and neural warpfields for loss computation
        :return: warped: warped left/right ear signal as tensor of shape B x 2 x T
                 (optional) warpfields: dict with 'geometric' and 'neural' warpfields
        '''
        geometric_warpfield = self.geometric_warper._warpfield(view, mono.shape[-1])
        neural_warpfield = self.neural_warpfield(view, mono.shape[-1])
        warpfield = geometric_warpfield + neural_warpfield
        # ensure causality
        warpfield = -F.relu(-warpfield)
        warped = self.neural_warper(th.cat([mono, mono], dim=1), warpfield)
        
        if return_warpfields:
            return warped, {
                'geometric': geometric_warpfield,
                'neural': neural_warpfield,
                'total': warpfield
            }
        return warped


class HyperConvWavenet(nn.Module):
    def __init__(self, z_dim, channels=64, blocks=3, layers_per_block=10, conv_len=2):
        super().__init__()
        self.layers = []
        self.rectv_field = 1
        for b in range(blocks):
            for l in range(layers_per_block):
                self.layers += [HyperConvBlock(channels, channels, z_dim, kernel_size=conv_len, dilation=2**l)]
                self.rectv_field += self.layers[-1].receptive_field() - 1
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, z):
        '''
        :param x: input signal as a B x channels x T tensor
        :param z: weight-generating input as a B x z_dim z K tensor (K = T / 400)
        :return: x: output signal as a B x channels x T tensor
                 skips: skip signal for each layer as a list of B x channels x T tensors
        '''
        skips = []
        for layer in self.layers:
            x, skip = layer(x, z)
            skips += [skip]
        return x, skips

    def receptive_field(self):
        return self.rectv_field


class WaveoutBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size=1)
        self.first.weight.data.uniform_(-np.sqrt(6.0 / channels), np.sqrt(6.0 / channels))
        self.second = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, x):
        x = th.sin(self.first(x))
        return self.second(x)


class BinauralNetwork(Net):
    def __init__(self,
                 view_dim=7,
                 warpnet_layers=4,
                 warpnet_channels=64,
                 wavenet_blocks=3,
                 layers_per_block=10,
                 wavenet_channels=64,
                 model_name='binaural_network',
                 use_cuda=True):
        super().__init__(model_name, use_cuda)
        self.warper = Warpnet(warpnet_layers, warpnet_channels)
        self.input = nn.Conv1d(2, wavenet_channels, kernel_size=1)
        self.input.weight.data.uniform_(-np.sqrt(6.0 / 2), np.sqrt(6.0 / 2))
        self.hyperconv_wavenet = HyperConvWavenet(view_dim, wavenet_channels, wavenet_blocks, layers_per_block)
        self.output_net = nn.ModuleList([WaveoutBlock(wavenet_channels)
                                        for _ in range(wavenet_blocks*layers_per_block)])
        if self.use_cuda:
            self.cuda()

    def forward(self, mono, view, return_warpfields=False):
        '''
        :param mono: the input signal as a B x 1 x T tensor
        :param view: the receiver/transmitter position as a B x 7 x T tensor
        :param return_warpfields: if True, return warpfields for loss computation
        :return: out: the binaural output produced by the network
                 intermediate: a two-channel audio signal obtained from the output of each intermediate layer
                               as a list of B x 2 x T tensors
                 (optional) warpfields: dict with 'geometric' and 'neural' warpfields
        '''
        if return_warpfields:
            warped, warpfields = self.warper(mono, view, return_warpfields=True)
        else:
            warped = self.warper(mono, view)
            warpfields = None
            
        x = self.input(warped)
        _, skips = self.hyperconv_wavenet(x, view)
        # collect output and skips after each layer
        x = []
        for k in range(len(skips), 0, -1):
            y = th.mean(th.stack(skips[:k], dim=0), dim=0)
            y = self.output_net[k-1](y)
            x += [y]
        x += [warped]
        
        result = {"output": x[0], "intermediate": x[1:]}
        if return_warpfields:
            result["warpfields"] = warpfields
        
        return result

    def receptive_field(self):
        return self.hyperconv_wavenet.receptive_field()
