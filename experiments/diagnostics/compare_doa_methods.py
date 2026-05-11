#!/usr/bin/env python3
"""
比較 ITD 和 ILD 方法在實際合成音訊上的表現
"""
from __future__ import annotations

import argparse
import os

import soundfile as sf
import torch as th

from src.doa import gcc_phat_estimate, ild_estimate, hybrid_estimate
from src.synthesis_utils import (
    angle_to_tx_positions,
    chunked_forwarding,
    load_binaural_net,
    trim_binaural_for_gcc,
)


def main():
    p = argparse.ArgumentParser(description="比較不同 DOA 估計方法")
    p.add_argument("--input", type=str, required=True, help="mono wav")
    p.add_argument("--model_file", type=str, default="outputs/binaural_network.newbob.net")
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument(
        "--test_angles",
        type=str,
        default="-60,-30,0,30,60",
        help="要測試的角度（逗號分隔）",
    )
    p.add_argument("--max_seconds", type=float, default=10.0)
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"錯誤: 找不到輸入 {args.input}")
        raise SystemExit(1)

    # 載入音訊
    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = th.from_numpy(audio).unsqueeze(0)

    if sr != 48000:
        import torchaudio.functional as F_audio
        mono = F_audio.resample(mono, sr, 48000)

    # 截斷
    max_samples = int(args.max_seconds * 48000)
    if mono.shape[-1] > max_samples:
        mono = mono[:, :max_samples]
    
    target_samples = (mono.shape[-1] // 400) * 400
    mono = mono[:, :target_samples]
    num_frames = mono.shape[-1] // 400

    # 載入模型
    net = load_binaural_net(args.model_file, blocks=args.blocks)

    # 解析角度
    test_angles = [float(x.strip()) for x in args.test_angles.split(",")]

    print(f"測試音訊: {args.input}")
    print(f"音訊長度: {mono.shape[-1] / 48000:.2f} 秒")
    print("=" * 90)
    print(f"{'目標角度':>10} {'ITD估計':>10} {'ITD誤差':>10} {'ILD估計':>10} {'ILD誤差':>10} {'Hybrid估計':>10} {'Hybrid誤差':>10}")
    print("-" * 90)

    for target_angle in test_angles:
        # 合成雙耳音訊
        view = th.from_numpy(angle_to_tx_positions(target_angle, args.distance, num_frames))
        binaural = chunked_forwarding(net, mono, view)
        bio = binaural.numpy()
        bio_trimmed = trim_binaural_for_gcc(bio)

        # 三種方法估計
        pred_itd = gcc_phat_estimate(bio_trimmed, sample_rate=48000)
        pred_ild = ild_estimate(bio_trimmed, sample_rate=48000, method='spectral')
        pred_hybrid = hybrid_estimate(bio_trimmed, sample_rate=48000)

        # 計算誤差
        err_itd = abs(pred_itd - target_angle)
        err_ild = abs(pred_ild - target_angle)
        err_hybrid = abs(pred_hybrid - target_angle)

        print(
            f"{target_angle:+10.1f} "
            f"{pred_itd:+10.1f} {err_itd:10.1f} "
            f"{pred_ild:+10.1f} {err_ild:10.1f} "
            f"{pred_hybrid:+10.1f} {err_hybrid:10.1f}"
        )

    print("=" * 90)
    print("\n結論:")
    print("  - ITD (GCC-PHAT): 基於時間差，適合語音，通常誤差 <5°")
    print("  - ILD (強度差): 基於能量差，目前使用線性映射，誤差可能很大")
    print("  - Hybrid: 混合方法，ITD 權重 0.7，ILD 權重 0.3")
    print("\n建議: 對於語音信號，優先使用 ITD 方法")


if __name__ == "__main__":
    main()
