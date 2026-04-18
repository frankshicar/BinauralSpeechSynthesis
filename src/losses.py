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
        data_angles, target_angles = th.atan2(data[:, 1], data[:, 0]), th.atan2(target[:, 1], target[:, 0])
        loss = th.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - th.abs(loss - np.pi)
        return th.mean(loss)


class IPDLoss(Loss):
    def __init__(self, sample_rate=48000, mask_beginning=0, ignore_below=0.1):
        '''
        計算預測與目標信號之間的 IPD (Interaural Phase Difference) 誤差
        Compute IPD error between predicted and target binaural signals
        
        :param sample_rate: (int) 音訊採樣率 / audio sample rate
        :param mask_beginning: (int) 開頭遮罩樣本數 / number of samples to mask at beginning
        :param ignore_below: (float) 忽略低能量成分的閾值 / threshold for ignoring low energy components
        '''
        super().__init__(mask_beginning)
        self.fft = FourierTransform(sample_rate=sample_rate)
        self.ignore_below = ignore_below
    
    def _loss(self, data, target):
        '''
        計算預測與目標信號的 IPD 誤差
        Compute IPD error between predicted and target signals
        
        :param data: 預測的波形訊號 (predicted wave signals), 大小為 B x 2 x T
        :param target: 目標波形訊號 (target wave signals), 大小為 B x 2 x T
        :return: IPD 誤差（弧度）/ IPD error in radians
        '''
        # 確保輸入是 (B, 2, T) 格式
        if data.shape[1] != 2:
            raise ValueError(f"Expected 2 channels, got {data.shape[1]}")
        
        # 對左右聲道分別做 STFT，然後 flatten
        # STFT 返回 (B, freq_bins, time_steps, 2)，我們 flatten 成 (B*freq_bins*time_steps, 2)
        pred_left_stft = self.fft.stft(data[:, 0, :]).view(-1, 2)    # (N, 2)
        pred_right_stft = self.fft.stft(data[:, 1, :]).view(-1, 2)   # (N, 2)
        target_left_stft = self.fft.stft(target[:, 0, :]).view(-1, 2)
        target_right_stft = self.fft.stft(target[:, 1, :]).view(-1, 2)
        
        # 計算能量，用於過濾低能量頻率
        target_left_energy = th.sum(th.abs(target_left_stft), dim=-1)
        target_right_energy = th.sum(th.abs(target_right_stft), dim=-1)
        pred_left_energy = th.sum(th.abs(pred_left_stft.detach()), dim=-1)
        pred_right_energy = th.sum(th.abs(pred_right_stft.detach()), dim=-1)
        
        # 能量遮罩（與 PhaseLoss 類似的方式）
        target_mask = (target_left_energy > self.ignore_below * th.mean(target_left_energy)) & \
                      (target_right_energy > self.ignore_below * th.mean(target_right_energy))
        pred_mask = (pred_left_energy > self.ignore_below * th.mean(target_left_energy)) & \
                    (pred_right_energy > self.ignore_below * th.mean(target_right_energy))
        
        indices = th.nonzero(target_mask * pred_mask).view(-1)
        
        # 如果有效頻率太少，使用所有頻率
        if indices.numel() < 10:
            indices = th.arange(pred_left_stft.shape[0], device=data.device)
        
        # 選擇有效頻率
        pred_left_stft = th.index_select(pred_left_stft, 0, indices)
        pred_right_stft = th.index_select(pred_right_stft, 0, indices)
        target_left_stft = th.index_select(target_left_stft, 0, indices)
        target_right_stft = th.index_select(target_right_stft, 0, indices)
        
        # 計算相位
        pred_left_phase = th.atan2(pred_left_stft[:, 0], pred_left_stft[:, 1])
        pred_right_phase = th.atan2(pred_right_stft[:, 0], pred_right_stft[:, 1])
        target_left_phase = th.atan2(target_left_stft[:, 0], target_left_stft[:, 1])
        target_right_phase = th.atan2(target_right_stft[:, 0], target_right_stft[:, 1])
        
        # 計算 IPD (左耳相位 - 右耳相位)
        pred_ipd = pred_left_phase - pred_right_phase
        target_ipd = target_left_phase - target_right_phase
        
        # 計算 IPD 誤差（與 PhaseLoss 相同的方式處理週期性）
        ipd_diff = th.abs(pred_ipd - target_ipd)
        ipd_diff = np.pi - th.abs(ipd_diff - np.pi)
        
        return th.mean(ipd_diff)


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


class AngularErrorLoss(Loss):
    def __init__(self, ground_truth_angle, sample_rate=48000, mask_beginning=0):
        '''
        計算預測雙耳音訊與目標角度之間的角度誤差
        Compute angular error between predicted binaural audio and ground truth angle
        
        使用 GCC-PHAT 算法從預測音訊中估計方位角，並與真實角度比較
        Uses GCC-PHAT algorithm to estimate azimuth from predicted audio and compare with ground truth
        
        :param ground_truth_angle: (float) 目標方位角（度）/ ground truth azimuth angle (degrees)
                                   範圍 [-90, +90]: 0° = 正前方, +angle = 左方, -angle = 右方
        :param sample_rate: (int) 音訊採樣率 / audio sample rate
        :param mask_beginning: (int) 開頭遮罩樣本數 / number of samples to mask at beginning
        '''
        super().__init__(mask_beginning)
        self.ground_truth_angle = ground_truth_angle
        self.sample_rate = sample_rate
    
    def _loss(self, data, target):
        '''
        計算角度誤差
        Compute angular error
        
        :param data: 預測的雙耳訊號 (B x 2 x T) / predicted binaural signals
        :param target: 參考訊號（此處不使用，僅為符合接口）/ reference signal (not used, for interface compatibility)
        :return: 角度誤差（度）/ angular error in degrees
        '''
        from src.doa import gcc_phat_estimate
        
        batch_errors = []
        for b in range(data.shape[0]):
            # 估計預測音訊的方位角
            # Estimate azimuth from predicted audio
            pred_angle = gcc_phat_estimate(
                data[b].cpu().numpy(), 
                sample_rate=self.sample_rate
            )
            # 計算與真實角度的誤差
            # Compute error with ground truth angle
            error = abs(pred_angle - self.ground_truth_angle)
            batch_errors.append(error)
        
        # 返回批次平均誤差
        # Return mean error across batch
        return th.tensor(batch_errors).mean()


class STFTLoss(th.nn.Module):
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(th, window)(win_length)

    def forward(self, x, y):
        x = x.view(-1, x.size(-1))
        y = y.view(-1, y.size(-1))
        
        # Ensure window is on same device
        if self.window.device != x.device:
            self.window = self.window.to(x.device)

        x_stft = th.stft(x, self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True)
        y_stft = th.stft(y, self.fft_size, self.shift_size, self.win_length, self.window, return_complex=True)

        x_mag = th.sqrt(th.clamp((x_stft.real ** 2) + (x_stft.imag ** 2), min=1e-8))
        y_mag = th.sqrt(th.clamp((y_stft.real ** 2) + (y_stft.imag ** 2), min=1e-8))

        sc_loss = th.norm(y_mag - x_mag, p="fro") / th.norm(y_mag, p="fro")
        mag_loss = th.nn.L1Loss()(y_mag, x_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(th.nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.5, factor_mag=0.5):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = th.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses.append(STFTLoss(fs, ss, wl, window))
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc * sc_loss + self.factor_mag * mag_loss


class WarpLoss(th.nn.Module):
    def __init__(self, lambda_warp=1.0):
        '''
        Warp Loss - 懲罰 neural warp 偏離 geometric warp 太遠
        Penalizes neural warp from deviating too far from geometric warp
        
        這個 loss 確保 neural warp 只做細微修正，而不是完全抵消 geometric warp
        This loss ensures neural warp only makes fine corrections, not completely canceling geometric warp
        
        :param lambda_warp: (float) warp loss 的權重 / weight for warp loss
        '''
        super().__init__()
        self.lambda_warp = lambda_warp
    
    def forward(self, neural_warpfield, geometric_warpfield):
        '''
        計算 neural warp 與 geometric warp 的偏差
        Compute deviation between neural warp and geometric warp
        
        :param neural_warpfield: neural network 預測的 warpfield (B x 2 x T)
        :param geometric_warpfield: 物理模型計算的 warpfield (B x 2 x T)
        :return: warp loss (scalar)
        '''
        # L2 loss between neural and geometric warpfields
        # 我們希望 neural_warpfield 接近 0（即不偏離 geometric warp 太多）
        # We want neural_warpfield to be close to 0 (i.e., not deviate too much from geometric warp)
        
        # 計算 neural warp 的 L2 norm
        # Compute L2 norm of neural warp
        warp_deviation = th.mean(neural_warpfield ** 2)
        
        return self.lambda_warp * warp_deviation


class WarpSmoothnessLoss(th.nn.Module):
    def __init__(self, lambda_smooth=0.1):
        '''
        Warp Smoothness Loss - 懲罰 warpfield 的時間不連續性
        Penalizes temporal discontinuities in warpfield
        
        確保 warpfield 在時間上平滑變化
        Ensures warpfield changes smoothly over time
        
        :param lambda_smooth: (float) smoothness loss 的權重 / weight for smoothness loss
        '''
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def forward(self, warpfield):
        '''
        計算 warpfield 的時間導數
        Compute temporal derivative of warpfield
        
        :param warpfield: warpfield tensor (B x 2 x T)
        :return: smoothness loss (scalar)
        '''
        # 計算相鄰時間步的差異
        # Compute difference between adjacent time steps
        temporal_diff = warpfield[:, :, 1:] - warpfield[:, :, :-1]
        
        # L2 norm of temporal differences
        smoothness_loss = th.mean(temporal_diff ** 2)
        
        return self.lambda_smooth * smoothness_loss

