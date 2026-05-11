#!/usr/bin/env python3
"""
診斷工具：測量模型在不同角度輸出的實際 ILD 值
用於建立 KEMAR 特定的 ILD-角度映射表
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
import torch as th

from src.synthesis_utils import (
    angle_to_tx_positions,
    chunked_forwarding,
    load_binaural_net,
)


def compute_ild_db(binaural_audio, freq_range=(1000, 8000), sample_rate=48000):
    """
    計算雙耳音訊的 ILD (dB)
    
    Returns:
        ild_db: ILD 值（正值表示左耳較響）
        left_energy_db: 左耳能量 (dB)
        right_energy_db: 右耳能量 (dB)
    """
    if isinstance(binaural_audio, th.Tensor):
        binaural_audio = binaural_audio.detach().cpu().numpy()
    
    if binaural_audio.shape[0] != 2:
        binaural_audio = binaural_audio.T
    
    left = binaural_audio[0].astype(np.float64)
    right = binaural_audio[1].astype(np.float64)
    
    # 使用頻譜能量（在指定頻率範圍內）
    n = len(left)
    nfft = 1 << int(np.ceil(np.log2(n)))
    
    left_fft = np.fft.rfft(left, n=nfft)
    right_fft = np.fft.rfft(right, n=nfft)
    
    freqs = np.fft.rfftfreq(nfft, 1.0 / sample_rate)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    
    left_energy = np.sqrt(np.mean(np.abs(left_fft[freq_mask]) ** 2))
    right_energy = np.sqrt(np.mean(np.abs(right_fft[freq_mask]) ** 2))
    
    # 避免 log(0)
    if left_energy < 1e-10 or right_energy < 1e-10:
        return 0.0, -np.inf, -np.inf
    
    left_energy_db = 20 * np.log10(left_energy)
    right_energy_db = 20 * np.log10(right_energy)
    ild_db = left_energy_db - right_energy_db
    
    return ild_db, left_energy_db, right_energy_db


def main():
    p = argparse.ArgumentParser(description="測量模型在不同角度的實際 ILD 值")
    p.add_argument("--input", type=str, required=True, help="mono wav")
    p.add_argument("--model_file", type=str, default="outputs/binaural_network.newbob.net")
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument(
        "--angles",
        type=str,
        default="-90,-75,-60,-45,-30,-15,0,15,30,45,60,75,90",
        help="要測試的角度（逗號分隔）",
    )
    p.add_argument("--freq_range", type=str, default="1000,8000", help="頻率範圍 (Hz)")
    p.add_argument("--max_seconds", type=float, default=10.0, help="只用前 N 秒")
    p.add_argument("--output_csv", type=str, default="", help="輸出 CSV 檔案")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"錯誤: 找不到輸入 {args.input}")
        raise SystemExit(1)
    if not os.path.isfile(args.model_file):
        print(f"錯誤: 找不到模型 {args.model_file}")
        raise SystemExit(1)

    # 載入音訊
    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = th.from_numpy(audio).unsqueeze(0)

    if sr != 48000:
        import torchaudio.functional as F_audio
        mono = F_audio.resample(mono, sr, 48000)
        sr = 48000

    # 截斷到指定長度
    max_samples = int(args.max_seconds * 48000)
    if mono.shape[-1] > max_samples:
        mono = mono[:, :max_samples]
    
    target_samples = (mono.shape[-1] // 400) * 400
    mono = mono[:, :target_samples]
    num_frames = mono.shape[-1] // 400

    # 解析角度和頻率範圍
    angles = [float(x.strip()) for x in args.angles.split(",")]
    freq_range = tuple(float(x.strip()) for x in args.freq_range.split(","))

    # 載入模型
    net = load_binaural_net(args.model_file, blocks=args.blocks)

    print(f"測試音訊: {args.input}")
    print(f"音訊長度: {mono.shape[-1] / 48000:.2f} 秒")
    print(f"頻率範圍: {freq_range[0]:.0f}-{freq_range[1]:.0f} Hz")
    print(f"測試角度: {len(angles)} 個")
    print("=" * 80)
    print(f"{'角度':>8} {'ILD (dB)':>12} {'左耳 (dB)':>12} {'右耳 (dB)':>12} {'|ILD|':>10}")
    print("-" * 80)

    results = []
    for angle in angles:
        # 合成雙耳音訊
        view = th.from_numpy(angle_to_tx_positions(angle, args.distance, num_frames))
        binaural = chunked_forwarding(net, mono, view)
        
        # 計算 ILD
        ild_db, left_db, right_db = compute_ild_db(
            binaural.numpy(), freq_range=freq_range, sample_rate=48000
        )
        
        print(f"{angle:+8.1f} {ild_db:+12.2f} {left_db:+12.2f} {right_db:+12.2f} {abs(ild_db):10.2f}")
        
        results.append({
            "angle": angle,
            "ild_db": ild_db,
            "left_db": left_db,
            "right_db": right_db,
        })

    print("=" * 80)
    
    # 分析結果
    ild_values = [r["ild_db"] for r in results]
    max_ild = max(abs(x) for x in ild_values)
    
    print(f"\n統計:")
    print(f"  最大 |ILD|: {max_ild:.2f} dB")
    print(f"  ILD 範圍: {min(ild_values):+.2f} ~ {max(ild_values):+.2f} dB")
    
    # 找出對稱性
    print(f"\n對稱性檢查:")
    for i, angle in enumerate(angles):
        if angle >= 0:
            neg_angle = -angle
            if neg_angle in angles:
                j = angles.index(neg_angle)
                ild_pos = results[i]["ild_db"]
                ild_neg = results[j]["ild_db"]
                print(f"  {neg_angle:+.0f}° vs {angle:+.0f}°: {ild_neg:+.2f} dB vs {ild_pos:+.2f} dB (對稱誤差: {abs(ild_pos + ild_neg):.2f} dB)")
    
    # 輸出 CSV
    if args.output_csv:
        import csv
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["angle", "ild_db", "left_db", "right_db"])
            writer.writeheader()
            writer.writerows(results)
        print(f"\n已輸出: {args.output_csv}")
    
    # 建議的 max_ILD
    print(f"\n建議:")
    print(f"  使用 max_ILD = {max_ild:.1f} dB（而非預設的 20 dB）")
    print(f"  但線性映射仍然不準確，建議建立查找表或使用 ITD 方法")


if __name__ == "__main__":
    main()
