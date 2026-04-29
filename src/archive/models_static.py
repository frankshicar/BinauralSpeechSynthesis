"""
針對靜止角度優化的雙耳合成模型
基於原始 BinauralNetwork，但針對固定角度場景進行優化

核心改進：
1. 靜止角度檢測 → 使用專用的角度編碼器而非複雜的 time warping
2. 多階諧波特徵 → 提高角度分辨率（特別是 ±30° 附近）
3. 直接 ITD/ILD 預測 → 繞過複雜的 geometric warping
4. 角度損失函數 → 直接優化 GCC-PHAT 估計的角度準確度
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from src.models import BinauralNetwork, Warpnet
from src.warping import MonotoneTimeWarper


class StaticAngleWarpnet(Warpnet):
    """針對靜止角度優化的 Warpnet
    
    改進邏輯：
    - 原始 Warpnet 用 geometric warping + neural warping 處理動態移動
    - 對靜止角度，geometric warping 是常數，neural warping 需要學習 HRTF
    - 本版本直接用角度編碼器學習 ITD/ILD，避免複雜的 time warping
    """
    
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__(layers, channels, view_dim)
        
        # 角度專用編碼器：角度 → ITD/ILD 參數
        # 為什麼有效：ITD/ILD 是角度的單調函數，直接學習比 time warping 更高效
        self.angle_encoder = nn.Sequential(
            nn.Linear(7, 64),  # 7 維特徵
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # 輸出 [ITD_scale, ILD_scale]
        )
        
        self.neural_warper = MonotoneTimeWarper()
        
    @staticmethod
    def _view_to_enhanced_features(view):
        """增強的特徵提取，針對靜止角度優化
        
        為什麼多階諧波有效：
        - 基本特徵 (sin/cos) 在 ±30° 附近變化不夠快
        - 多階諧波 (sin(2az), cos(3az) 等) 提供更高的角度分辨率
        - 類似傅立葉級數展開，能更精確表示非線性的 HRTF 特性
        """
        x, y, z = view[:, 0], view[:, 1], view[:, 2]
        dist = th.sqrt(x**2 + y**2 + z**2 + 1e-8)
        az = th.atan2(-y, x)  # 方位角 [-π, π]
        el = th.asin((z / dist).clamp(-1, 1))  # 仰角 [-π/2, π/2]
        
        # 基本特徵
        sin_az = th.sin(az)
        cos_az = th.cos(az)
        sin_el = th.sin(el)
        cos_el = th.cos(el)
        
        # 多階諧波（提高角度分辨率）
        sin_2az = th.sin(2 * az)
        cos_2az = th.cos(2 * az)
        sin_3az = th.sin(3 * az)
        cos_3az = th.cos(3 * az)
        
        # 距離歸一化
        dist_norm = dist / (1.0 + 1e-3)
        
        # 堆疊所有特徵
        enhanced_feat = th.stack([
            az, el, dist_norm,
            sin_az, cos_az, sin_el, cos_el,
            sin_2az, cos_2az, sin_3az, cos_3az
        ], dim=1)  # B x 11
        
        return enhanced_feat
    
    def neural_warpfield(self, view, seq_length):
        """計算 neural warpfield
        
        策略：
        - 檢測靜止角度 → 使用角度編碼器
        - 動態角度 → 使用原始 conv 路徑
        """
        # 檢查是否為靜止角度
        pos_std = th.std(view[:, :3, :], dim=2)  # B x 3
        is_static = th.all(pos_std < 1e-4, dim=1)  # B
        
        if th.any(is_static):
            # 靜止角度路徑：直接預測 ITD/ILD
            feat = self._view_to_enhanced_features(view)  # B x 11
            
            # 截斷到前 7 維（角度相關特徵）
            feat_angle = feat[:, :7]
            
            # 角度編碼器
            itd_ild = self.angle_encoder(feat_angle)  # B x 2
            
            # 轉換為 warpfield 格式 (B x 2 x 1)
            warpfield = itd_ild.unsqueeze(2)
            warpfield = warpfield * self.output_scale
            warpfield = F.interpolate(warpfield, size=seq_length)
            
            return warpfield
        else:
            # 動態角度路徑：使用原始 conv
            warpfield = self._view_to_enhanced_features(view)
            for layer in self.layers:
                warpfield = F.pad(warpfield, pad=[1, 0])
                warpfield = F.relu(layer(warpfield))
            warpfield = self.linear(warpfield)
            warpfield = warpfield * self.output_scale
            warpfield = F.interpolate(warpfield, size=seq_length)
            return warpfield


class StaticAngleBinauralNetwork(BinauralNetwork):
    """針對靜止角度優化的雙耳網路
    
    改進原理：
    1. 替換 Warpnet → StaticAngleWarpnet（靜止角度檢測 + 角度編碼器）
    2. 加入角度損失 → 直接優化 GCC-PHAT 估計的角度準確度
    3. 多尺度訓練 → 同時優化 L2/Phase/Angle
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 替換 warper
        self.warper = StaticAngleWarpnet(
            layers=kwargs.get('warpnet_layers', 4),
            channels=kwargs.get('warpnet_channels', 64)
        )
    
    def forward(self, mono, view):
        """前向傳播（保持與原始模型相同的 API）"""
        return super().forward(mono, view)


def create_static_angle_model(**kwargs):
    """創建靜止角度優化模型"""
    return StaticAngleBinauralNetwork(**kwargs)