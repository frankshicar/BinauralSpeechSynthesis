"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch as th
from src.utils import FourierTransform


class Loss(th.nn.Module):
    def __init__(self, mask_beginning=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning

    def forward(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data[..., self.mask_beginning:]
        target = target[..., self.mask_beginning:]
        return self._loss(data, target)

    def _loss(self, data, target):
        pass


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return th.mean((data - target).pow(2))


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        data = th.sum(data**2, dim=-1) ** 0.5
        target = th.sum(target**2, dim=-1) ** 0.5
        return th.mean(th.abs(data - target))


class PhaseLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2)
        # ignore low energy components for numerical stability
        target_energy = th.sum(th.abs(target), dim=-1)
        pred_energy = th.sum(th.abs(data.detach()), dim=-1)
        target_mask = target_energy > self.ignore_below * th.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * th.mean(target_energy)
        indices = th.nonzero(target_mask * pred_mask).view(-1)
        data, target = th.index_select(data, 0, indices), th.index_select(target, 0, indices)
        # compute actual phase loss in angular space
        data_angles, target_angles = th.atan2(data[:, 0], data[:, 1]), th.atan2(target[:, 0], target[:, 1])
        loss = th.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - th.abs(loss - np.pi)
        return th.mean(loss)


class AngularError(PhaseLoss):
    def _loss(self, data, target):
        '''
        計算預測訊號與目標訊號之間的角度誤差（以度為單位）
        Compute the angular error between predicted and target signals (in Degrees).
        :param data: 預測的波形訊號 (predicted wave signals), 大小為 B x channels x T
        :param target: 目標波形訊號 (target wave signals), 大小為 B x channels x T
        :return: 純量誤差值 (a scalar loss value), 單位為「度」 (Degrees)
        '''
        # 使用父類別 PhaseLoss 的邏輯先計算弧度誤差
        # Use parent class logic to compute radian error first
        radian_loss = super()._loss(data, target)
        
        # 將弧度轉換為角度 (0 ~ 180 度)
        # Convert radians to degrees
        degree_loss = radian_loss * 180 / np.pi
        return degree_loss


class ITDLoss(Loss):
    def __init__(self, sample_rate=48000, max_shift_ms=1.0, mask_beginning=0):
        '''
        計算預測與目標信號之間的 ITD (Interaural Time Difference) 誤差
        Compute ITD error between predicted and target binaural signals
        
        :param sample_rate: (int) 音訊採樣率 / audio sample rate
        :param max_shift_ms: (float) 最大搜索延遲（毫秒）/ maximum search delay in milliseconds
                             人類最大 ITD 約 0.8ms / human max ITD is ~0.8ms
        :param mask_beginning: (int) 開頭遮罩樣本數 / number of samples to mask at beginning
        '''
        super().__init__(mask_beginning)
        self.sample_rate = sample_rate
        self.max_shift_ms = max_shift_ms
        self.max_shift_samples = int(max_shift_ms * sample_rate / 1000)
    
    def _compute_itd(self, stereo_signal):
        '''
        使用互相關方法計算雙聲道信號的 ITD
        Compute ITD of stereo signal using cross-correlation method
        
        :param stereo_signal: 雙聲道信號 (B x 2 x T) / stereo signal tensor
        :return: ITD 值（微秒）/ ITD value in microseconds
        '''
        # 確保輸入是 (B, 2, T) 格式
        # Ensure input is in (B, 2, T) format
        if stereo_signal.shape[1] != 2:
            raise ValueError(f"Expected 2 channels, got {stereo_signal.shape[1]}")
        
        left = stereo_signal[:, 0, :]   # (B, T)
        right = stereo_signal[:, 1, :]  # (B, T)
        
        batch_size = left.shape[0]
        itds = []
        
        for b in range(batch_size):
            # 對每個樣本計算互相關
            # Compute cross-correlation for each sample
            left_b = left[b]
            right_b = right[b]
            
            # 使用簡單的滑動窗口方法計算互相關
            # Use simple sliding window method to compute cross-correlation
            correlations = []
            
            for shift in range(-self.max_shift_samples, self.max_shift_samples + 1):
                if shift < 0:
                    # 左聲道提前（右聲道延遲）
                    # Left channel leads (right channel delayed)
                    l_slice = left_b[:shift]
                    r_slice = right_b[-shift:]
                elif shift > 0:
                    # 右聲道提前（左聲道延遲）
                    # Right channel leads (left channel delayed)
                    l_slice = left_b[shift:]
                    r_slice = right_b[:-shift]
                else:
                    # 沒有延遲
                    # No delay
                    l_slice = left_b
                    r_slice = right_b
                
                # 計算正規化的互相關
                # Compute normalized cross-correlation
                if len(l_slice) > 0:
                    corr = th.sum(l_slice * r_slice) / len(l_slice)
                else:
                    corr = th.tensor(0.0)
                correlations.append(corr)
            
            # 找到最大相關對應的延遲
            # Find delay corresponding to maximum correlation
            correlations = th.stack(correlations)
            max_idx = th.argmax(correlations)
            itd_samples = max_idx - self.max_shift_samples
            itd_seconds = itd_samples.float() / self.sample_rate
            itd_microseconds = itd_seconds * 1e6
            itds.append(itd_microseconds)
        
        # 返回平均 ITD
        # Return mean ITD across batch
        return th.stack(itds).mean()

    
    def _loss(self, data, target):
        '''
        計算預測與目標信號的 ITD 誤差
        Compute ITD error between predicted and target signals
        
        :param data: 預測的波形訊號 (predicted wave signals), 大小為 B x 2 x T
        :param target: 目標波形訊號 (target wave signals), 大小為 B x 2 x T
        :return: ITD 誤差（微秒）/ ITD error in microseconds
        '''
        # 計算預測和目標的 ITD
        # Compute ITD for both predicted and target
        pred_itd = self._compute_itd(data)
        target_itd = self._compute_itd(target)
        
        # 返回絕對誤差
        # Return absolute error
        return th.abs(pred_itd - target_itd)


class ILDLoss(Loss):
    def __init__(self, sample_rate=48000, freq_bands=None, mask_beginning=0):
        '''
        計算預測與目標信號之間的 ILD (Interaural Level Difference) 誤差
        Compute ILD error between predicted and target binaural signals
        
        :param sample_rate: (int) 音訊採樣率 / audio sample rate
        :param freq_bands: (list of tuples) 頻帶列表 [(low, high), ...] / frequency bands
                          預設為高頻段 / defaults to high-frequency bands
        :param mask_beginning: (int) 開頭遮罩樣本數 / number of samples to mask at beginning
        '''
        super().__init__(mask_beginning)
        self.sample_rate = sample_rate
        self.fft = FourierTransform(sample_rate=sample_rate)
        
        # 預設使用高頻段，因為 ILD 在高頻最有效
        # Default to high-frequency bands where ILD is most effective
        if freq_bands is None:
            self.freq_bands = [(1000, 4000), (4000, 8000)]
        else:
            self.freq_bands = freq_bands
    
    def _compute_ild(self, stereo_signal):
        '''
        使用頻域能量分析計算雙聲道信號的 ILD
        Compute ILD of stereo signal using spectral energy analysis
        
        :param stereo_signal: 雙聲道信號 (B x 2 x T) / stereo signal tensor
        :return: ILD 值（dB）/ ILD value in decibels
        '''
        # 確保輸入是 (B, 2, T) 格式
        if stereo_signal.shape[1] != 2:
            raise ValueError(f"Expected 2 channels, got {stereo_signal.shape[1]}")
        
        batch_size = stereo_signal.shape[0]
        ilds = []
        
        for b in range(batch_size):
            # 對每個聲道進行 STFT
            # Apply STFT to each channel
            left_signal = stereo_signal[b:b+1, 0, :].unsqueeze(1)  # (1, 1, T)
            right_signal = stereo_signal[b:b+1, 1, :].unsqueeze(1)  # (1, 1, T)
            
            left_stft = self.fft.stft(left_signal.squeeze(1))  # STFT output
            right_stft = self.fft.stft(right_signal.squeeze(1))
            
            # 計算能量譜
            # Compute power spectrum
            left_power = th.sum(left_stft**2, dim=-1)  # Sum over real/imag
            right_power = th.sum(right_stft**2, dim=-1)
            
            # 在指定頻帶內累積能量
            # Accumulate energy in specified frequency bands
            # STFT 輸出的頻率解析度
            # Frequency resolution of STFT output
            fft_bins = self.fft.fft_bins
            freq_resolution = self.sample_rate / fft_bins
            
            total_left_energy = 0
            total_right_energy = 0
            
            for low_freq, high_freq in self.freq_bands:
                # 轉換頻率到 bin index
                # Convert frequency to bin index
                low_bin = int(low_freq / freq_resolution)
                high_bin = int(high_freq / freq_resolution)
                
                # 確保 bin 在有效範圍內
                # Ensure bins are within valid range
                low_bin = max(0, low_bin)
                high_bin = min(left_power.shape[1], high_bin)
                
                # 累積能量
                # Accumulate energy
                total_left_energy = total_left_energy + th.sum(left_power[:, low_bin:high_bin])
                total_right_energy = total_right_energy + th.sum(right_power[:, low_bin:high_bin])
            
            # 計算 ILD (dB)
            # Compute ILD in decibels
            # ILD = 10 * log10(E_left / E_right)
            eps = 1e-10  # 避免除以零 / avoid division by zero
            ild = 10 * th.log10((total_left_energy + eps) / (total_right_energy + eps))
            ilds.append(ild)
        
        # 返回平均 ILD
        # Return mean ILD across batch
        return th.stack(ilds).mean()
    
    def _loss(self, data, target):
        '''
        計算預測與目標信號的 ILD 誤差
        Compute ILD error between predicted and target signals
        
        :param data: 預測的波形訊號 (predicted wave signals), 大小為 B x 2 x T
        :param target: 目標波形訊號 (target wave signals), 大小為 B x 2 x T
        :return: ILD 誤差（dB）/ ILD error in decibels
        '''
        # 計算預測和目標的 ILD
        # Compute ILD for both predicted and target
        pred_ild = self._compute_ild(data)
        target_ild = self._compute_ild(target)
        
        # 返回絕對誤差
        # Return absolute error
        return th.abs(pred_ild - target_ild)

