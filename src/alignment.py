"""
時間對齊工具模組 (Temporal Alignment Utilities)
用於雙耳語音合成 (Binaural Speech Synthesis)

此模組提供函數來偵測並修正 mono 和 binaural 音訊信號之間的時間偏移
使用互相關 (cross-correlation) 方法

作者: Antigravity AI Assistant
日期: 2026-01-24
目的: 診斷並修正 Phase Error 高的問題
"""

import numpy as np
import torch as th
from scipy import signal


def find_alignment_offset(mono, binaural, max_shift=2400, sample_rate=48000):
    """
    使用「振幅包絡 (Amplitude Envelope)」的互相關找到 mono 和 binaural 信號之間的時間偏移
    Find the temporal offset between mono and binaural signals using cross-correlation of envelopes.
    
    參數 Args:
        mono: Mono 音訊信號，形狀 (1, T) 或 (T,)
        binaural: Binaural 音訊信號，形狀 (2, T) 或 (T,)
        max_shift: 最大搜尋範圍（samples），預設 2400 = 50ms @ 48kHz
        sample_rate: 採樣率 Hz，預設 48000
    
    回傳 Returns:
        offset: 需要移動的 samples 數（正數 = mono 延遲）
        correlation: 最大互相關係數 (基於包絡)
    
    2026-01-26: Updated to use Envelope Correlation for robustness against timbre mismatch
    """
    # 轉換為 numpy (如果需要) - Convert to numpy if needed
    if isinstance(mono, th.Tensor):
        mono = mono.cpu().numpy()
    if isinstance(binaural, th.Tensor):
        binaural = binaural.cpu().numpy()
    
    # 確保 mono 是 1D - Ensure mono is 1D
    if mono.ndim > 1:
        mono = mono.flatten()
    
    # 使用 binaural 的左聲道作為參考 - Use left channel of binaural for correlation
    if binaural.ndim > 1:
        binaural_ref = binaural[0, :] if binaural.shape[0] == 2 else binaural[:, 0]
    else:
        binaural_ref = binaural
    
    # 裁剪到相同長度 - Trim to same length for correlation
    min_len = min(len(mono), len(binaural_ref))
    mono = mono[:min_len]
    binaural_ref = binaural_ref[:min_len]
    
    # 只使用前 10 秒的信號以加快速度 - Use only first 10 seconds for speed
    max_samples = min(min_len, sample_rate * 10)
    mono_segment = mono[:max_samples]
    binaural_segment = binaural_ref[:max_samples]

    # --- 2026-01-26: 計算振幅包絡 (Amplitude Envelope) ---
    # 使用 Hilbert Transform 提取包絡，這對音色不同但內容相同的訊號對齊更有效
    try:
        mono_envelope = np.abs(signal.hilbert(mono_segment))
        binaural_envelope = np.abs(signal.hilbert(binaural_segment))
    except Exception as e:
        print(f"  [ALIGNMENT] Hilbert transform failed: {e}, falling back to raw signal")
        mono_envelope = np.abs(mono_segment) # Fallback to rectification
        binaural_envelope = np.abs(binaural_segment)

    # 正規化包絡（零均值、單位方差）- Normalize envelopes
    mono_envelope = (mono_envelope - np.mean(mono_envelope)) / (np.std(mono_envelope) + 1e-8)
    binaural_envelope = (binaural_envelope - np.mean(binaural_envelope)) / (np.std(binaural_envelope) + 1e-8)
    
    # 透過 FFT 進行互相關 - Cross-correlation via FFT
    # Note: We are correlating ENVELOPES now
    correlation = signal.correlate(binaural_envelope, mono_envelope, mode='same', method='fft')
    
    # 在搜尋範圍內找到最大相關性的 lag - Find the lag with maximum correlation within search range
    center = len(correlation) // 2
    search_start = max(0, center - max_shift)
    search_end = min(len(correlation), center + max_shift)
    
    search_region = correlation[search_start:search_end]
    max_idx = np.argmax(search_region)
    
    # 轉換為偏移量（正數 = mono 延遲）- Convert to offset (positive = mono is delayed)
    offset = (search_start + max_idx) - center
    max_correlation = search_region[max_idx] / len(mono_segment)
    
    return int(offset), float(max_correlation)


def align_signals(mono, binaural, offset):
    """
    根據偵測到的偏移量對齊 mono 和 binaural 信號
    Align mono and binaural signals based on detected offset.
    
    參數 Args:
        mono: Mono 音訊信號，形狀 (1, T) 或 (C, T)
        binaural: Binaural 音訊信號，形狀 (2, T)
        offset: 需要移動的 samples 數（來自 find_alignment_offset）
    
    回傳 Returns:
        aligned_mono: 對齊後的 mono 信號
        aligned_binaural: 對齊後的 binaural 信號（裁剪以匹配長度）
    
    2026-01-24: 創建此函數以實作時間對齊
    """
    is_tensor = isinstance(mono, th.Tensor)
    
    if is_tensor:
        device = mono.device
        mono_np = mono.cpu().numpy()
        binaural_np = binaural.cpu().numpy()
    else:
        mono_np = mono
        binaural_np = binaural
    
    # 應用偏移量 - Apply offset (2026-01-24)
    if offset > 0:
        # Mono 延遲，裁剪 mono 的開頭 - Mono is delayed, trim beginning of mono
        aligned_mono = mono_np[:, offset:] if mono_np.ndim > 1 else mono_np[offset:]
        aligned_binaural = binaural_np
    elif offset < 0:
        # Binaural 延遲，裁剪 binaural 的開頭 - Binaural is delayed, trim beginning of binaural
        aligned_mono = mono_np
        aligned_binaural = binaural_np[:, -offset:] if binaural_np.ndim > 1 else binaural_np[-offset:]
    else:
        # 沒有偏移 - No offset
        aligned_mono = mono_np
        aligned_binaural = binaural_np
    
    # 裁剪到相同長度 - Trim to same length
    if aligned_mono.ndim > 1:
        min_len = min(aligned_mono.shape[-1], aligned_binaural.shape[-1])
        aligned_mono = aligned_mono[:, :min_len]
        aligned_binaural = aligned_binaural[:, :min_len]
    else:
        min_len = min(len(aligned_mono), len(aligned_binaural))
        aligned_mono = aligned_mono[:min_len]
        aligned_binaural = aligned_binaural[:min_len]
    
    # Convert back to tensor if needed
    if is_tensor:
        aligned_mono = th.from_numpy(aligned_mono).to(device)
        aligned_binaural = th.from_numpy(aligned_binaural).to(device)
    
    return aligned_mono, aligned_binaural


def diagnose_alignment(mono, binaural, offset, correlation, sample_rate=48000):
    """
    輸出對齊的診斷資訊
    Print diagnostic information about the alignment.
    
    參數 Args:
        mono: 原始 mono 信號
        binaural: 原始 binaural 信號
        offset: 偵測到的偏移量（samples）
        correlation: 互相關係數
        sample_rate: 採樣率 Hz
    
    2026-01-24: 創建此函數以輸出對齊診斷資訊
    """
    offset_ms = (offset / sample_rate) * 1000  # 轉換為毫秒 - Convert to milliseconds
    
    # Compute correlation before alignment
    if isinstance(mono, th.Tensor):
        mono_np = mono.cpu().numpy().flatten()
        binaural_np = binaural.cpu().numpy()[0, :] if binaural.ndim > 1 else binaural.cpu().numpy()
    else:
        mono_np = mono.flatten() if mono.ndim > 1 else mono
        binaural_np = binaural[0, :] if binaural.ndim > 1 else binaural
    
    min_len = min(len(mono_np), len(binaural_np))
    mono_segment = mono_np[:min_len]
    binaural_segment = binaural_np[:min_len]
    
    # Normalize
    mono_segment = (mono_segment - np.mean(mono_segment)) / (np.std(mono_segment) + 1e-8)
    binaural_segment = (binaural_segment - np.mean(binaural_segment)) / (np.std(binaural_segment) + 1e-8)
    
    # Correlation at zero lag
    corr_before = np.corrcoef(mono_segment, binaural_segment)[0, 1]
    
    print(f"  [ALIGNMENT] Detected offset: {offset:+d} samples ({offset_ms:+.2f} ms)")
    print(f"  [ALIGNMENT] Correlation: before={corr_before:.3f}, after={correlation:.3f}", end="")
    
    if abs(correlation - corr_before) > 0.05:
        print(" ✓ IMPROVED")
    else:
        print("")
    
    return {
        'offset_samples': offset,
        'offset_ms': offset_ms,
        'correlation_before': corr_before,
        'correlation_after': correlation
    }


# ============================================================================
# 2026-01-28: 新增 VAD 對齊函數 (Voice Activity Detection Alignment)
# 用於解決 mono 和 binaural 音訊延遲不匹配導致 Phase Error 過高的問題
# ============================================================================

def detect_speech_onset(signal, threshold_ratio=0.02, min_duration_ms=10, sample_rate=48000):
    """
    使用能量閾值偵測語音的起始點 (2026-01-28)
    Detect speech onset using energy threshold
    
    原理 Principle：
    1. 計算信號的瞬時能量（滑動視窗 RMS）
    2. 找到能量超過閾值的第一個點
    3. 確保該點後有持續的語音活動（避免噪音觸發）
    
    參數 Args:
        signal: 音訊信號，形狀 (C, T) 或 (T,) / Audio signal, shape (C, T) or (T,)
        threshold_ratio: 能量閾值相對於最大能量的比例（預設 2%）
                        / Energy threshold as ratio of max energy (default 2%)
        min_duration_ms: 超過閾值需持續的最小時間（毫秒）
                        / Minimum duration above threshold (milliseconds)
        sample_rate: 採樣率 / Sample rate
    
    回傳 Returns:
        onset_sample: 語音起始點的 sample 索引 / Sample index of speech onset
    """
    # 轉換為 numpy
    if isinstance(signal, th.Tensor):
        signal = signal.cpu().numpy()
    
    # 如果是多聲道，取平均（用於能量計算）
    # If multi-channel, average for energy computation
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)
    
    # 計算滑動視窗 RMS 能量（視窗大小：5ms）
    # Compute sliding window RMS energy (window size: 5ms)
    window_size = int(sample_rate * 0.005)  # 5ms 視窗
    if window_size < 1:
        window_size = 1
    
    # 計算平方信號的滑動平均
    squared = signal ** 2
    
    # 使用卷積計算滑動視窗能量
    # Use convolution for sliding window energy
    kernel = np.ones(window_size) / window_size
    energy = np.convolve(squared, kernel, mode='same')
    
    # 計算能量閾值（基於整體最大能量）
    # Compute energy threshold based on max energy
    max_energy = np.max(energy)
    threshold = max_energy * threshold_ratio
    
    # 找到超過閾值的樣本
    # Find samples above threshold
    above_threshold = energy > threshold
    
    # 確保持續時間足夠（避免噪音觸發）
    # Ensure minimum duration (avoid noise triggers)
    min_samples = int(sample_rate * min_duration_ms / 1000)
    
    # 找到第一個連續超過閾值 min_samples 的起點
    # Find first onset with continuous activity for min_samples
    onset_sample = 0
    consecutive_count = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above:
            if consecutive_count == 0:
                candidate_onset = i
            consecutive_count += 1
            if consecutive_count >= min_samples:
                onset_sample = candidate_onset
                break
        else:
            consecutive_count = 0
    
    return onset_sample


def align_by_speech_onset(pred_signal, ref_signal, sample_rate=48000, verbose=True):
    """
    根據語音起始點對齊兩個信號 (2026-01-28)
    Align signals based on speech onset detection
    
    解決的問題 Problem Solved：
    當 binauralized 和 reference 音訊的靜音開頭長度不同時，
    直接計算 Phase Error 會因為時間偏移而得到錯誤的高值。
    此函數找到兩個信號的語音起點，然後裁剪對齊。
    
    參數 Args:
        pred_signal: 預測的雙耳信號 (2, T) / Predicted binaural signal
        ref_signal: 參考的雙耳信號 (2, T) / Reference binaural signal
        sample_rate: 採樣率 / Sample rate
        verbose: 是否輸出診斷資訊 / Whether to print diagnostic info
    
    回傳 Returns:
        aligned_pred: 對齊後的預測信號 / Aligned predicted signal
        aligned_ref: 對齊後的參考信號 / Aligned reference signal
        alignment_info: 對齊資訊字典 / Alignment info dictionary
    """
    # 偵測語音起始點
    # Detect speech onset
    pred_onset = detect_speech_onset(pred_signal, sample_rate=sample_rate)
    ref_onset = detect_speech_onset(ref_signal, sample_rate=sample_rate)
    
    # 計算偏移量（正數 = pred 起始較晚）
    # Compute offset (positive = pred starts later)
    offset = pred_onset - ref_onset
    
    is_tensor = isinstance(pred_signal, th.Tensor)
    
    if is_tensor:
        device = pred_signal.device
        pred_np = pred_signal.cpu().numpy()
        ref_np = ref_signal.cpu().numpy()
    else:
        pred_np = pred_signal
        ref_np = ref_signal
    
    # Trim each signal from its own speech onset
    if offset != 0:
        aligned_pred = pred_np[:, pred_onset:]
        aligned_ref = ref_np[:, ref_onset:]
    else:
        aligned_pred = pred_np
        aligned_ref = ref_np
    
    # 裁剪到相同長度
    # Trim to same length
    min_len = min(aligned_pred.shape[-1], aligned_ref.shape[-1])
    aligned_pred = aligned_pred[:, :min_len]
    aligned_ref = aligned_ref[:, :min_len]
    
    # 轉回張量（如果需要）
    # Convert back to tensor if needed
    if is_tensor:
        aligned_pred = th.from_numpy(aligned_pred).to(device)
        aligned_ref = th.from_numpy(aligned_ref).to(device)
    
    # 輸出診斷資訊
    # Print diagnostic info
    if verbose:
        pred_onset_ms = pred_onset / sample_rate * 1000
        ref_onset_ms = ref_onset / sample_rate * 1000
        offset_ms = offset / sample_rate * 1000
        print(f"  [VAD] 預測語音起點: {pred_onset} samples ({pred_onset_ms:.1f}ms)")
        print(f"  [VAD] 參考語音起點: {ref_onset} samples ({ref_onset_ms:.1f}ms)")
        print(f"  [VAD] 應用偏移: {offset:+d} samples ({offset_ms:+.1f}ms)")
    
    alignment_info = {
        'pred_onset': pred_onset,
        'ref_onset': ref_onset,
        'offset_samples': offset,
        'offset_ms': offset / sample_rate * 1000,
        'aligned_length': min_len
    }
    
    return aligned_pred, aligned_ref, alignment_info
