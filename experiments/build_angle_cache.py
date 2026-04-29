#!/usr/bin/env python3
"""
批次建立「目標角 -> tx」快取（angle_tx_cache.json）。

你不需要一個一個跑 synthesize.py；先用本腳本把常用角度全部校準並記住：

  python build_angle_cache.py \
    --input dataset/mono.wav \
    --angles_from_testset ./dataset/testset \
    --model_file outputs/binaural_network.newbob.net \
    --angle_cache angle_tx_cache.json

或自行指定角度列表：
  python build_angle_cache.py --input dataset/mono.wav --angles -90,-60,-45,-30,-15,0,15,30,45,60,75,90 ...

校準方法：對每個目標角，用 GCC-PHAT 掃描候選 tx，取誤差最小者寫入 JSON 快取。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import soundfile as sf
import torch as th

from src.synthesis_utils import (
    angle_cache_key,
    correct_angle_segmented,
    find_best_tx_gcc,
    load_angle_tx_cache,
    load_binaural_net,
    parse_calibration_candidates,
    save_angle_tx_cache,
    truncate_mono_for_calibration,
)


TESTSET_SUBJECT_ANGLES = {
    "subject1": -90.0,
    "subject2": -60.0,
    "subject8": -45.0,
    "subject3": -30.0,
    "subject10": -15.0,
    "subject4": 0.0,
    "subject11": 15.0,
    "subject5": 30.0,
    "subject9": 45.0,
    "subject6": 60.0,
    "subject12": 75.0,
    "subject7": 90.0,
}


def _parse_angles_csv(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    p = argparse.ArgumentParser(description="批次建立 angle->tx 快取（GCC 校準）")
    p.add_argument("--input", type=str, required=True, help="mono wav（校準用；48kHz 最佳）")
    p.add_argument("--model_file", type=str, required=True)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--distance", type=float, default=1.0)
    p.add_argument("--angle_cache", type=str, required=True, help="輸出 JSON 快取檔路徑")
    p.add_argument("--cache_key_decimals", type=int, default=2)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--angles", type=str, help="逗號分隔目標角，例如 -60,-45,0,30,60")
    src.add_argument(
        "--angles_from_testset",
        type=str,
        help="給 dataset/testset 目錄：使用內建 subject→角度表依序校準",
    )

    p.add_argument("--max_seconds", type=float, default=30.0, help="每個角度校準只用前 N 秒 mono")
    p.add_argument("--candidates", type=str, default="auto", help="auto 或逗號分隔候選 tx")
    p.add_argument("--half_span", type=float, default=12.0, help="auto 候選半寬（度）")
    p.add_argument("--step", type=float, default=1.0, help="auto 候選步長（度）")
    p.add_argument("--refresh", action="store_true", help="覆寫已存在的快取鍵")
    p.add_argument("--verbose", action="store_true", help="列印每個候選 tx 的 GCC 結果")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"找不到 input: {args.input}")
    if not os.path.isfile(args.model_file):
        raise SystemExit(f"找不到 model_file: {args.model_file}")

    audio, sr = sf.read(args.input, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = th.from_numpy(audio).unsqueeze(0)
    if sr != 48000:
        import torchaudio.functional as F_audio

        mono = F_audio.resample(mono, sr, 48000)

    mono_cal = truncate_mono_for_calibration(mono, args.max_seconds, sample_rate=48000)
    if mono_cal.shape[-1] < 400:
        raise SystemExit("音訊太短，無法校準")

    if args.angles:
        desired_angles = _parse_angles_csv(args.angles)
    else:
        # 依 testset 角度順序（右 -90 -> 左 +90）
        testset_dir = args.angles_from_testset
        if not os.path.isdir(testset_dir):
            raise SystemExit(f"找不到 testset 目錄: {testset_dir}")
        # 僅保留存在的 subject
        desired_angles = []
        for subj, ang in TESTSET_SUBJECT_ANGLES.items():
            if os.path.isdir(os.path.join(testset_dir, subj)):
                desired_angles.append(float(ang))

    # 去重且排序（由右到左）
    desired_angles = sorted(set(desired_angles))

    net = load_binaural_net(args.model_file, blocks=args.blocks)
    cache = load_angle_tx_cache(args.angle_cache)

    print(f"將校準 {len(desired_angles)} 個角度，並寫入 {args.angle_cache}")

    for a in desired_angles:
        key = angle_cache_key(a, decimals=args.cache_key_decimals)
        if key in cache and not args.refresh:
            print(f"跳過 [{key}°]：已存在 tx={cache[key]:+.2f}°")
            continue

        center = correct_angle_segmented(a)
        cands = parse_calibration_candidates(
            args.candidates,
            a,
            half_span=args.half_span,
            step_deg=args.step,
            center_tx=center,
        )
        print(f"[{key}°] GCC 校準：候選 {len(cands)} 個（center≈{center:+.2f}°）")
        best_tx, best_err, best_pred, _ = find_best_tx_gcc(
            net,
            mono_cal,
            a,
            args.distance,
            cands,
            sample_rate=48000,
            verbose=args.verbose,
        )
        cache[key] = best_tx
        save_angle_tx_cache(args.angle_cache, cache)
        print(f"  -> tx {best_tx:+.2f}° (GCC 估 {best_pred:+.2f}°, 誤差 {best_err:.2f}°)")

    print("完成。")


if __name__ == "__main__":
    main()

