"""
診斷所有角度的 ITD 值
Diagnose ITD values for all angles
"""

import numpy as np
import torch as th
import argparse
import os
from scipy.io import wavfile
from scipy.signal import correlate
from src.synthesis_utils import load_binaural_net, angle_to_tx_positions, chunked_forwarding

def load_model(model_path, blocks=3):
    """載入模型"""
    net = load_binaural_net(model_path, blocks)
    net.eval()
    return net

def compute_itd_samples(binaural_audio, sample_rate=48000, max_lag_ms=1.0):
    """
    計算 ITD（以樣本數為單位）
    使用互相關方法
    """
    left = binaural_audio[0, :]
    right = binaural_audio[1, :]
    
    max_lag_samples = int(max_lag_ms * sample_rate / 1000)
    
    # 計算互相關
    correlation = correlate(left, right, mode='same')
    center = len(correlation) // 2
    
    # 只看 ±max_lag_samples 範圍
    start = center - max_lag_samples
    end = center + max_lag_samples + 1
    correlation_window = correlation[start:end]
    
    # 找到最大相關的位置
    max_idx = np.argmax(correlation_window)
    itd_samples = max_idx - max_lag_samples
    
    return itd_samples

def compute_theoretical_itd(angle_deg, head_radius_cm=8.75, sound_speed_cm_s=34300, sample_rate=48000):
    """
    計算理論 ITD（以樣本數為單位）
    使用 Woodworth 公式
    ITD = (head_radius / sound_speed) * (sin(angle) + angle) * sample_rate
    """
    angle_rad = np.deg2rad(angle_deg)
    itd_seconds = (head_radius_cm / sound_speed_cm_s) * (np.sin(angle_rad) + angle_rad)
    itd_samples = itd_seconds * sample_rate
    return itd_samples

def synthesize_binaural(net, mono_audio, angle_deg, distance=1.0, sample_rate=48000):
    """
    合成雙耳音訊
    使用 synthesis_utils 的函數確保格式正確
    """
    # 準備輸入 - 確保形狀正確
    # mono_audio 是 (T,)，需要變成 (1, T)
    mono_tensor = th.from_numpy(mono_audio).float().unsqueeze(0)  # (1, T)
    
    # 對齊到 400 的倍數（與 synthesize.py 相同）
    target_samples = (mono_tensor.shape[-1] // 400) * 400
    mono_tensor = mono_tensor[:, :target_samples]
    num_frames = target_samples // 400
    
    # 使用 angle_to_tx_positions 產生正確的 view
    view_np = angle_to_tx_positions(angle_deg, distance, num_frames)
    view = th.from_numpy(view_np)  # (7, num_frames)，不需要 unsqueeze
    
    # 使用 chunked_forwarding 進行合成
    binaural = chunked_forwarding(net, mono_tensor, view)
    
    return binaural.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='診斷所有角度的 ITD 值')
    parser.add_argument('--model', type=str, required=True,
                       help='模型檔案路徑 (e.g., outputs_with_ipd/binaural_network.epoch-60.net)')
    parser.add_argument('--mono', type=str, default='dataset/mono.wav',
                       help='單聲道音訊檔案')
    parser.add_argument('--distance', type=float, default=1.0,
                       help='聲源距離（公尺）')
    parser.add_argument('--blocks', type=int, default=3,
                       help='WaveNet blocks')
    parser.add_argument('--output', type=str, default='itd_diagnosis.txt',
                       help='輸出診斷結果的檔案')
    args = parser.parse_args()
    
    # 載入模型
    print(f"載入模型: {args.model}")
    net = load_model(args.model, args.blocks)
    
    # 載入單聲道音訊
    print(f"載入音訊: {args.mono}")
    sample_rate, mono_audio = wavfile.read(args.mono)
    if len(mono_audio.shape) > 1:
        mono_audio = mono_audio[:, 0]
    mono_audio = mono_audio.astype(np.float32) / 32768.0
    
    # 測試角度
    angles = [-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90]
    
    print(f"\n{'='*80}")
    print(f"ITD 診斷報告")
    print(f"{'='*80}")
    print(f"模型: {args.model}")
    print(f"音訊: {args.mono}")
    print(f"{'='*80}\n")
    
    results = []
    
    for angle in angles:
        print(f"處理角度: {angle:+4d}°", end=" ... ")
        
        # 合成雙耳音訊（使用正確的 view 格式）
        binaural = synthesize_binaural(net, mono_audio, angle, args.distance, sample_rate)
        
        # 計算實際 ITD
        actual_itd_samples = compute_itd_samples(binaural, sample_rate)
        actual_itd_us = actual_itd_samples / sample_rate * 1e6
        
        # 計算理論 ITD
        theoretical_itd_samples = compute_theoretical_itd(angle, sample_rate=sample_rate)
        theoretical_itd_us = theoretical_itd_samples / sample_rate * 1e6
        
        # 計算誤差
        error_samples = actual_itd_samples - theoretical_itd_samples
        error_us = actual_itd_us - theoretical_itd_us
        error_percent = (error_samples / theoretical_itd_samples * 100) if theoretical_itd_samples != 0 else 0
        
        results.append({
            'angle': angle,
            'actual_samples': actual_itd_samples,
            'actual_us': actual_itd_us,
            'theoretical_samples': theoretical_itd_samples,
            'theoretical_us': theoretical_itd_us,
            'error_samples': error_samples,
            'error_us': error_us,
            'error_percent': error_percent
        })
        
        print(f"完成")
    
    # 顯示結果
    print(f"\n{'='*80}")
    print(f"{'角度':>6s} | {'實際ITD':>12s} | {'理論ITD':>12s} | {'誤差':>12s} | {'誤差%':>8s}")
    print(f"{'':>6s} | {'(samples)':>12s} | {'(samples)':>12s} | {'(samples)':>12s} | {'':>8s}")
    print(f"{'-'*80}")
    
    for r in results:
        status = "✅" if abs(r['error_percent']) < 20 else "⚠️" if abs(r['error_percent']) < 50 else "❌"
        print(f"{r['angle']:+6d}° | {r['actual_samples']:12.2f} | {r['theoretical_samples']:12.2f} | "
              f"{r['error_samples']:+12.2f} | {r['error_percent']:+7.1f}% {status}")
    
    print(f"{'='*80}\n")
    
    # 分析左右不對稱
    print(f"左右對稱性分析:")
    print(f"{'-'*80}")
    
    for angle in [15, 30, 45, 60, 75, 90]:
        left_result = next((r for r in results if r['angle'] == -angle), None)
        right_result = next((r for r in results if r['angle'] == angle), None)
        
        if left_result and right_result:
            left_error = abs(left_result['error_samples'])
            right_error = abs(right_result['error_samples'])
            asymmetry = right_error - left_error
            
            status = "✅ 對稱" if abs(asymmetry) < 2 else "⚠️ 輕微不對稱" if abs(asymmetry) < 5 else "❌ 嚴重不對稱"
            
            print(f"±{angle:2d}°: 左側誤差 {left_error:5.2f} samples, "
                  f"右側誤差 {right_error:5.2f} samples, "
                  f"差異 {asymmetry:+6.2f} samples {status}")
    
    print(f"{'='*80}\n")
    
    # 系統性偏差分析
    print(f"系統性偏差分析:")
    print(f"{'-'*80}")
    
    left_errors = [r['error_samples'] for r in results if r['angle'] < 0 and r['angle'] != 0]
    right_errors = [r['error_samples'] for r in results if r['angle'] > 0]
    all_errors = [r['error_samples'] for r in results if r['angle'] != 0]
    
    print(f"左側平均誤差: {np.mean(left_errors):+7.2f} samples ({np.mean(left_errors)/48*1000:+6.2f} μs)")
    print(f"右側平均誤差: {np.mean(right_errors):+7.2f} samples ({np.mean(right_errors)/48*1000:+6.2f} μs)")
    print(f"整體平均誤差: {np.mean(all_errors):+7.2f} samples ({np.mean(all_errors)/48*1000:+6.2f} μs)")
    print(f"整體標準差:   {np.std(all_errors):7.2f} samples ({np.std(all_errors)/48*1000:6.2f} μs)")
    
    if np.mean(right_errors) < np.mean(left_errors) - 2:
        print(f"\n❌ 右側 ITD 系統性偏小 → 角度會被低估")
    elif np.mean(left_errors) < np.mean(right_errors) - 2:
        print(f"\n❌ 左側 ITD 系統性偏小 → 角度會被低估")
    else:
        print(f"\n✅ 左右側 ITD 誤差相近")
    
    print(f"{'='*80}\n")
    
    # 保存結果
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(f"ITD 診斷報告\n")
        f.write(f"{'='*80}\n")
        f.write(f"模型: {args.model}\n")
        f.write(f"音訊: {args.mono}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"{'角度':>6s} | {'實際ITD':>12s} | {'理論ITD':>12s} | {'誤差':>12s} | {'誤差%':>8s}\n")
        f.write(f"{'-'*80}\n")
        
        for r in results:
            f.write(f"{r['angle']:+6d}° | {r['actual_samples']:12.2f} | {r['theoretical_samples']:12.2f} | "
                   f"{r['error_samples']:+12.2f} | {r['error_percent']:+7.1f}%\n")
        
        f.write(f"\n系統性偏差:\n")
        f.write(f"左側平均誤差: {np.mean(left_errors):+7.2f} samples\n")
        f.write(f"右側平均誤差: {np.mean(right_errors):+7.2f} samples\n")
        f.write(f"整體平均誤差: {np.mean(all_errors):+7.2f} samples\n")
    
    print(f"✅ 診斷結果已保存到: {args.output}")

if __name__ == "__main__":
    main()
