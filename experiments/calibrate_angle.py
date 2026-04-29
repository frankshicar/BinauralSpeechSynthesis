#!/usr/bin/env python3
"""
單點角度校準：對一組候選 tx 角合成雙耳音，用 GCC-PHAT 估感知角，找 |估計角 - 目標角| 最小者。

用法示例（左前 +60°）:
  python calibrate_angle.py --input dataset/testset/subject4/mono.wav --desired 60 \\
      --model_file outputs/binaural_network.newbob.net --candidates auto

右前 -60°:
  python calibrate_angle.py --input ... --desired -60 --candidates auto

候選自訂:
  python calibrate_angle.py --desired 60 --candidates 55,58,60,62,65,70,75

若要在每次合成時自動寫入 JSON 快取，請用 synthesize.py：
  --angle_cache my_cache.json --cache_auto_calibrate
"""
from __future__ import annotations

import argparse
import os

import soundfile as sf
import torch as th

from src.synthesis_utils import (
    angle_to_tx_positions,
    chunked_forwarding,
    find_best_tx_gcc,
    load_binaural_net,
    parse_calibration_candidates,
    truncate_mono_for_calibration,
)


def main():
    p = argparse.ArgumentParser(description="掃描 tx 角，找最接近目標感知角的設定")
    p.add_argument("--input", type=str, required=True, help="mono wav（48kHz 最佳）")
    p.add_argument("--desired", type=float, required=True, help="目標感知方位角（度），+左 -右")
    p.add_argument("--model_file", type=str, default="outputs/binaural_network.newbob.net")
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument(
        "--candidates",
        type=str,
        default="auto",
        help='候選 tx 角：逗號分隔，或 "auto"',
    )
    p.add_argument("--calibrate_half_span", type=float, default=12.0, help="auto 時掃描半寬（度）")
    p.add_argument("--calibrate_step", type=float, default=1.0, help="auto 時步長（度）")
    p.add_argument("--max_seconds", type=float, default=30.0, help="只用前 N 秒 mono 加速")
    p.add_argument("--save_best_wav", type=str, default="", help="若指定，輸出最佳 tx 的雙耳 wav")
    p.add_argument(
        "--method",
        type=str,
        default="itd",
        choices=["itd", "ild", "hybrid"],
        help="角度估計方法: itd (GCC-PHAT), ild (強度差), hybrid (混合)",
    )
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"錯誤: 找不到輸入 {args.input}")
        raise SystemExit(1)
    if not os.path.isfile(args.model_file):
        print(f"錯誤: 找不到模型 {args.model_file}")
        raise SystemExit(1)

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = th.from_numpy(audio).unsqueeze(0)

    if sr != 48000:
        import torchaudio.functional as F_audio

        mono = F_audio.resample(mono, sr, 48000)
        sr = 48000

    target_samples = (mono.shape[-1] // 400) * 400
    mono = mono[:, :target_samples]
    mono = truncate_mono_for_calibration(mono, args.max_seconds, sample_rate=48000)
    num_frames = mono.shape[-1] // 400
    if num_frames < 1:
        print("錯誤: 音訊太短")
        raise SystemExit(1)

    candidates = parse_calibration_candidates(
        args.candidates,
        args.desired,
        half_span=args.calibrate_half_span,
        step_deg=args.calibrate_step,
    )
    net = load_binaural_net(args.model_file, blocks=args.blocks)

    method_name = {
        "itd": "GCC-PHAT (ITD)",
        "ild": "ILD (強度差)",
        "hybrid": "Hybrid (ITD+ILD)",
    }[args.method]
    
    print(f"目標感知角 desired = {args.desired:+.1f}° | 候選 tx 數 = {len(candidates)}")
    print(f"估計方法: {method_name}")
    print("-" * 72)
    best_tx, best_err, best_pred, best_binaural = find_best_tx_gcc(
        net,
        mono,
        args.desired,
        args.distance,
        candidates,
        sample_rate=48000,
        verbose=True,
        method=args.method,
    )
    print("-" * 72)
    print(
        f"最佳: tx = {best_tx:+.2f}°  -> 估計 {best_pred:+.2f}°  |  誤差 {best_err:.2f}°"
    )
    print(
        f"建議: 使用 synthesize.py --angle_cache angle_tx_cache.json --cache_auto_calibrate "
        f"可自動寫入快取；或手動把 {args.desired:+.1f}° -> tx {best_tx:+.2f}° 記入 JSON。"
    )

    if args.save_best_wav:
        sf.write(args.save_best_wav, best_binaural.t().numpy().T, 48000)
        print(f"已寫入: {args.save_best_wav}")


if __name__ == "__main__":
    main()
