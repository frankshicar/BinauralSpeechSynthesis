"""
雙耳音訊合成腳本
用法: python synthesize.py --input mono.wav --angle 90 --output output.wav --model_file outputs/binaural_network.newbob.net
角度定義: 0=正前方, +90=左方, -90=右方

角度快取（依權重／語料自動學 tx）:
  --angle_cache angle_tx_cache.json --cache_auto_calibrate
  第一次某角度會用 GCC 掃描最佳 tx 並寫入 JSON，之後同角度直接讀取。
"""

import argparse
import os

import soundfile as sf
import torch as th

from src.synthesis_utils import (
    angle_cache_key,
    angle_to_tx_positions,
    chunked_forwarding,
    correct_angle_curve,
    correct_angle_segmented,
    correct_angle_simple,
    find_best_tx_gcc,
    load_angle_tx_cache,
    load_binaural_net,
    parse_calibration_candidates,
    save_angle_tx_cache,
    truncate_mono_for_calibration,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="輸入 mono wav 檔案")
    parser.add_argument(
        "--angle",
        type=float,
        required=True,
        help="聲源方位角（度）: 0=正前方, +90=左, -90=右",
    )
    parser.add_argument("--output", type=str, default="binaural_output.wav", help="輸出雙耳 wav 檔案")
    parser.add_argument("--model_file", type=str, default="outputs/binaural_network.newbob.net")
    parser.add_argument("--distance", type=float, default=1.0, help="聲源距離（公尺）")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--no_correction", action="store_true", help="停用角度校正（忽略快取）")
    parser.add_argument(
        "--correction_mode",
        type=str,
        choices=("segmented", "curve", "simple"),
        default="segmented",
        help="無快取或快取未命中且未開自動校準時使用",
    )
    parser.add_argument(
        "--use_curve",
        action="store_true",
        help="等同 --correction_mode curve（與舊版相容）",
    )
    parser.add_argument(
        "--angle_cache",
        type=str,
        default="",
        help="JSON 路徑：記錄「目標角 -> 最佳 tx」；命中則直接套用",
    )
    parser.add_argument(
        "--cache_auto_calibrate",
        action="store_true",
        help="快取中沒有該角度時，用 GCC 掃描並寫入快取（較慢，但最準）",
    )
    parser.add_argument(
        "--cache_refresh",
        action="store_true",
        help="忽略快取命中，強制重新掃描並覆寫該角度條目",
    )
    parser.add_argument(
        "--cache_key_decimals",
        type=int,
        default=2,
        help="快取鍵四捨五入到小數位數（同鍵視為同一目標角）",
    )
    parser.add_argument(
        "--calibrate_candidates",
        type=str,
        default="auto",
        help='GCC 掃描候選 tx：auto 或逗號清單，如 "55,58,60,62"',
    )
    parser.add_argument(
        "--calibrate_half_span",
        type=float,
        default=12.0,
        help='auto 候選時，以 segmented 預估為中心向兩側延伸（度）',
    )
    parser.add_argument(
        "--calibrate_step",
        type=float,
        default=1.0,
        help="auto 候選時步長（度）；可改 0.5 做細掃",
    )
    parser.add_argument(
        "--calibrate_max_seconds",
        type=float,
        default=30.0,
        help="校準僅用前 N 秒 mono（合成仍用完整長度）",
    )
    parser.add_argument(
        "--calibrate_verbose",
        action="store_true",
        help="列印每個候選 tx 的 GCC 估計角與誤差",
    )
    args = parser.parse_args()

    mode = args.correction_mode
    if args.use_curve:
        mode = "curve"

    if not os.path.isfile(args.model_file):
        print(f"錯誤: 找不到模型檔 {args.model_file}")
        raise SystemExit(1)

    if not os.path.isfile(args.input):
        print(f"錯誤: 找不到輸入音檔 {args.input}")
        print("提示：repo 內常見可用路徑例如：")
        print("  - dataset/mono.wav")
        print("  - dataset/testset/subject4/mono.wav")
        raise SystemExit(1)

    try:
        audio, sr = sf.read(args.input, dtype="float32")
    except Exception as e:
        print(f"錯誤: 無法讀取音檔 {args.input}")
        print(f"  例外：{type(e).__name__}: {e}")
        print("提示：請確認檔案存在、格式為 wav、且有讀取權限；也可先改用：")
        print("  - dataset/mono.wav")
        print("  - dataset/testset/subject4/mono.wav")
        raise SystemExit(1)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    mono = th.from_numpy(audio).unsqueeze(0)

    if sr != 48000:
        import torchaudio.functional as F_audio

        mono = F_audio.resample(mono, sr, 48000)
        sr = 48000

    target_samples = (mono.shape[-1] // 400) * 400
    mono_full = mono[:, :target_samples]
    num_frames = target_samples // 400

    desired_angle = args.angle
    tx_angle: float
    cache_path = (args.angle_cache or "").strip()
    used_cache = False
    calibrated = False
    net: th.nn.Module | None = None

    if args.no_correction:
        tx_angle = float(desired_angle)
        print(f"未校正: 直接使用 tx = {tx_angle:+.1f}°")
    elif cache_path:
        key = angle_cache_key(desired_angle, decimals=args.cache_key_decimals)
        mapping = load_angle_tx_cache(cache_path)
        do_cal = args.cache_refresh or (key not in mapping and args.cache_auto_calibrate)

        if key in mapping and not args.cache_refresh:
            tx_angle = float(mapping[key])
            used_cache = True
            print(f"角度快取命中 [{key}°]: 使用 tx = {tx_angle:+.2f}°（來自 {cache_path}）")
        elif do_cal:
            net = load_binaural_net(args.model_file, blocks=args.blocks)
            mono_cal = truncate_mono_for_calibration(
                mono_full, args.calibrate_max_seconds, sample_rate=48000
            )
            if mono_cal.shape[-1] < 400:
                print("錯誤: 校準用音訊太短")
                raise SystemExit(1)
            cands = parse_calibration_candidates(
                args.calibrate_candidates,
                desired_angle,
                half_span=args.calibrate_half_span,
                step_deg=args.calibrate_step,
            )
            print(
                f"GCC 校準: 目標 {desired_angle:+.2f}° | 候選 {len(cands)} 個 | "
                f"校準長度 {mono_cal.shape[-1] / 48000:.2f}s"
            )
            if args.calibrate_verbose:
                print("-" * 40)
            best_tx, best_err, best_pred, _ = find_best_tx_gcc(
                net,
                mono_cal,
                desired_angle,
                args.distance,
                cands,
                sample_rate=48000,
                verbose=args.calibrate_verbose,
            )
            if args.calibrate_verbose:
                print("-" * 40)
            tx_angle = best_tx
            calibrated = True
            mapping[key] = best_tx
            save_angle_tx_cache(cache_path, mapping)
            print(
                f"已寫入快取 [{key}°] -> tx {best_tx:+.2f}° "
                f"(GCC 估 {best_pred:+.2f}°, 誤差 {best_err:.2f}°) -> {cache_path}"
            )
        else:
            if key not in mapping and not args.cache_auto_calibrate:
                print(
                    f"快取無條目 [{key}°] 且未加 --cache_auto_calibrate，改用 correction_mode={mode}"
                )
            if mode == "simple":
                tx_angle = correct_angle_simple(desired_angle)
                print(f"簡單補償: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")
            elif mode == "curve":
                tx_angle = correct_angle_curve(desired_angle)
                print(f"曲線插值: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")
            else:
                tx_angle = correct_angle_segmented(desired_angle)
                print(f"分段校正: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")
    else:
        if mode == "simple":
            tx_angle = correct_angle_simple(desired_angle)
            print(f"簡單補償: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")
        elif mode == "curve":
            tx_angle = correct_angle_curve(desired_angle)
            print(f"曲線插值: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")
        else:
            tx_angle = correct_angle_segmented(desired_angle)
            print(f"分段校正: 想要 {desired_angle:+.1f}° -> tx {tx_angle:+.1f}°")

    view = th.from_numpy(angle_to_tx_positions(tx_angle, args.distance, num_frames))

    if net is None:
        net = load_binaural_net(args.model_file, blocks=args.blocks)

    print(
        f"輸入: {args.input} ({audio.shape[0] / 48000:.2f}s, 48kHz) | "
        f"Frames: {num_frames} | correction_mode={mode} | "
        f"cache={'on' if cache_path else 'off'} used_hit={used_cache} calibrated={calibrated}"
    )

    binaural = chunked_forwarding(net, mono_full, view)
    sf.write(args.output, binaural.t().numpy(), 48000)
    print(f"輸出: {args.output}")


if __name__ == "__main__":
    main()
