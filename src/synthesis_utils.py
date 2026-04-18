"""
共用雙耳合成與角度校正工具（供 synthesize.py、calibrate_angle.py 使用）
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import torch as th
from scipy.interpolate import interp1d

from src.doa import gcc_phat_estimate, ild_estimate, hybrid_estimate
from src.models import BinauralNetwork

# ---------------------------------------------------------------------------
# 角度校正：直接補償（固定誤差值）
# 每個角度的固定補償量 = 目標角度 - 預測角度
# 使用方式：tx_angle = desired_angle + ANGLE_COMPENSATION[desired_angle]
# 2026-02-16: 從未校正模型的 evaluate.py 輸出中測得，並經過迭代優化
# ---------------------------------------------------------------------------
_ANGLE_COMPENSATION = {
     -90.0:   +0.0,   # 預測  -90.0°, 誤差   0.0°
     -75.0:   +3.8,   # 預測  -74.5°, 誤差   0.5° (二分搜尋校準)
     -60.0:   -3.8,   # 預測  -59.8°, 誤差   0.2°
     -45.0:   -5.6,   # 預測  -47.0°, 誤差   2.0°
     -30.0:   -4.0,   # 預測  -32.1°, 誤差   2.1°
     -15.0:   -3.1,   # 預測  -15.4°, 誤差   0.4°
      +0.0:   +1.9,   # 預測   -0.0°, 誤差   0.0°
     +15.0:   +7.0,   # 預測  +13.5°, 誤差   1.5°
     +30.0:  +12.8,   # 預測  +29.9°, 誤差   0.1°
     +45.0:  +16.0,   # 預測  +44.3°, 誤差   0.7°
     +60.0:  +16.1,   # 預測  +56.2°, 誤差   3.8°
     +75.0:   +7.5,   # 預測  +74.5°, 誤差   0.5° (二分搜尋校準)
     +90.0:   +0.0,   # 預測  +90.0°, 誤差   0.0°
}

# 曲線插值：實測 (perceived, tx)，依 perceived 排序後做 cubic（避免 x 非遞增）
# 2026-02-16: 暫時註解，改用原始模型輸出測量真實誤差
# _TX_ANGLES = np.array([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90], dtype=np.float64)
# _PERCEIVED = np.array(
#     [-90.0, -59.8, -47.0, -32.1, -11.5, 1.9, 7.6, 29.9, 44.3, 68.5, 63.8, 90.0],
#     dtype=np.float64,
# )
# _order = np.argsort(_PERCEIVED)
# _PERCEIVED_SORTED = _PERCEIVED[_order]
# _TX_SORTED = _TX_ANGLES[_order]
# _curve_correction = interp1d(_PERCEIVED_SORTED, _TX_SORTED, kind="cubic", fill_value="extrapolate")

# 分段線性：45°~90°（左側正角）與 -90°~-45°（右側負角）— 來自角度校正說明.md 映射表
# 2026-02-16: 暫時註解，改用原始模型輸出測量真實誤差
# _POS_DESIRED = np.array([45.0, 60.0, 75.0, 90.0], dtype=np.float64)
# _POS_TX = np.array([60.9, 77.5, 88.3, 90.0], dtype=np.float64)
# _NEG_DESIRED = np.array([-90.0, -75.0, -60.0, -45.0], dtype=np.float64)
# _NEG_TX = np.array([-90.0, -74.8, -62.8, -51.4], dtype=np.float64)


def correct_angle_simple(desired_angle: float) -> float:
    """
    直接補償法：查表取得該角度的固定補償量
    如果角度不在表中，返回原角度（不做補償）
    
    使用方式：
    1. 用 calibrate_angle.py 測量每個角度的誤差
    2. 將補償量填入 _ANGLE_COMPENSATION 字典
    3. 合成時會自動套用該角度的固定補償
    """
    compensation = _ANGLE_COMPENSATION.get(desired_angle, 0.0)
    corrected = desired_angle + compensation
    return float(np.clip(corrected, -90, 90))


def correct_angle_curve(desired_angle: float) -> float:
    """
    目標感知角 desired → 應設定的 tx 角（cubic，適用 |角| 較小與中間區）
    2026-02-16: 暫時停用，直接返回原始角度
    """
    # corrected = float(_curve_correction(desired_angle))
    # return float(np.clip(corrected, -90, 90))
    return float(np.clip(desired_angle, -90, 90))


def _linear_mid_positive(desired: float) -> float:
    """
    左側 +45°~+90°：分段線性（映射表）
    2026-02-16: 暫時停用，直接返回原始角度
    """
    # return float(np.interp(desired, _POS_DESIRED, _POS_TX))
    return float(desired)


def _linear_mid_negative(desired: float) -> float:
    """
    右側 -90°~-45°：分段線性（映射表）
    2026-02-16: 暫時停用，直接返回原始角度
    """
    # return float(np.interp(desired, _NEG_DESIRED, _NEG_TX))
    return float(desired)


def correct_angle_segmented(desired_angle: float) -> float:
    """
    分段校正（預設推薦）：
    - |角| ≤ 45°：cubic 曲線（與原 --use_curve 同資料，但 perceived 已排序）
    - |角| > 45°：45°~90° / -90°~-45° 用映射表線性內插，避免 +60~+90 非單調區拖垮 cubic
    
    2026-02-16: 暫時停用所有校正，直接返回原始角度
    """
    # a = float(desired_angle)
    # if abs(a) <= 45.0:
    #     return correct_angle_curve(a)
    # if a > 45.0:
    #     return float(np.clip(_linear_mid_positive(a), -90.0, 90.0))
    # return float(np.clip(_linear_mid_negative(a), -90.0, 90.0))
    return float(np.clip(desired_angle, -90, 90))


def angle_to_tx_positions(azimuth_deg: float, distance: float, num_frames: int) -> np.ndarray:
    rad = math.radians(azimuth_deg)
    x = distance * math.cos(rad)
    y = -distance * math.sin(rad)
    z = 0.0
    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    row = [x, y, z, qx, qy, qz, qw]
    return np.array([row] * num_frames, dtype=np.float32).T


def chunked_forwarding(net: BinauralNetwork, mono: th.Tensor, view: th.Tensor) -> th.Tensor:
    net.eval()
    if th.cuda.is_available():
        net.cuda()
        mono, view = mono.cuda(), view.cuda()

    chunk_size = 480000
    rec_field = net.receptive_field() + 1000
    rec_field -= rec_field % 400

    chunks = [
        {
            "mono": mono[:, max(0, i - rec_field) : i + chunk_size],
            "view": view[:, max(0, i - rec_field) // 400 : (i + chunk_size) // 400],
        }
        for i in range(0, mono.shape[-1], chunk_size)
    ]

    for i, chunk in enumerate(chunks):
        with th.no_grad():
            binaural = net(chunk["mono"].unsqueeze(0), chunk["view"].unsqueeze(0))["output"].squeeze(0)
            trim = chunk["mono"].shape[-1] - rec_field
            if i > 0 and trim > 0:
                binaural = binaural[:, -trim:]
            elif i > 0:
                # last chunk shorter than rec_field: keep all output
                pass
            chunk["binaural"] = binaural

    binaural = th.cat([c["binaural"] for c in chunks], dim=-1)
    return th.clamp(binaural, min=-1, max=1).cpu()


def load_binaural_net(model_file: str, blocks: int = 3) -> BinauralNetwork:
    net = BinauralNetwork(view_dim=7, wavenet_blocks=blocks)
    weights = th.load(model_file, map_location="cpu")
    if isinstance(weights, dict):
        net.load_state_dict(weights)
    else:
        net.load_state_dict(weights.state_dict())
    net.eval()
    return net


def angular_error_deg(pred: float, ground_truth: float) -> float:
    err = abs(pred - ground_truth)
    return min(err, 360.0 - err)


def trim_binaural_for_gcc(binaural_2t: np.ndarray) -> np.ndarray:
    """與 evaluate.py 一致：略去尾端近靜音，避免 GCC 被 padding 影響。"""
    left = binaural_2t[0, :]
    valid_indices = np.where(np.abs(left) > 1e-4)[0]
    if len(valid_indices) == 0:
        return binaural_2t
    valid_len = valid_indices[-1] + 1
    valid_len = int(valid_len * 0.95)
    return binaural_2t[:, :valid_len]


def angle_cache_key(desired_deg: float, decimals: int = 2) -> str:
    return f"{round(float(desired_deg), decimals):.{decimals}f}"


def load_angle_tx_cache(path: str) -> dict[str, float]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw: Any = json.load(f)
    if isinstance(raw, dict) and "map" in raw:
        raw = raw["map"]
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        out[str(k)] = float(v)
    return out


def save_angle_tx_cache(path: str, mapping: dict[str, float]) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    payload = {
        "format": "angle_tx_v1",
        "map": {k: float(mapping[k]) for k in sorted(mapping.keys(), key=lambda x: float(x))},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_calibration_candidates(
    spec: str,
    desired: float,
    *,
    half_span: float = 12.0,
    step_deg: float = 1.0,
    center_tx: float | None = None,
) -> np.ndarray:
    """
    spec == "auto": 以 center_tx（若 None 則用 segmented 預估）為中心，在 [-half_span, +half_span] 內掃描。
    否則為逗號分隔的 tx 角列表。
    """
    s = spec.strip().lower()
    if s == "auto":
        center = float(center_tx) if center_tx is not None else correct_angle_segmented(desired)
        lo = max(-90.0, min(center, desired) - half_span)
        hi = min(90.0, max(center, desired) + half_span)
        xs = np.arange(lo, hi + 0.5 * step_deg, step_deg)
        return np.unique(np.clip(np.round(xs, 4), -90.0, 90.0))
    parts = [float(x.strip()) for x in spec.split(",") if x.strip()]
    return np.array(sorted(set(parts)), dtype=np.float64)


def find_best_tx_gcc(
    net: BinauralNetwork,
    mono: th.Tensor,
    desired: float,
    distance: float,
    candidates: np.ndarray,
    *,
    sample_rate: int = 48000,
    verbose: bool = False,
    method: str = "itd",
) -> tuple[float, float, float, th.Tensor]:
    """
    對每個候選 tx 合成雙耳，使用指定方法估角，回傳誤差最小的 tx。

    Parameters:
        method: 角度估計方法
            'itd': GCC-PHAT (基於 ITD)
            'ild': ILD (基於強度差)
            'hybrid': ITD + ILD 混合

    Returns:
        best_tx, best_err_deg, best_pred_deg, binaural_best (2 x T)
    """
    target_samples = (mono.shape[-1] // 400) * 400
    mono_use = mono[:, :target_samples]
    num_frames = target_samples // 400
    if num_frames < 1:
        raise ValueError("mono 太短，無法校準")

    if len(candidates) == 0:
        raise ValueError("校準候選 tx 為空")

    best_tx = 0.0
    best_err = float("inf")
    best_pred = 0.0
    best_binaural: th.Tensor | None = None

    if verbose:
        print(f"{'tx_deg':>10} {'pred_deg':>10} {'err_deg':>10}")

    for tx in candidates:
        view = th.from_numpy(angle_to_tx_positions(float(tx), distance, num_frames))
        binaural = chunked_forwarding(net, mono_use, view)
        bio = binaural.numpy()
        bio = trim_binaural_for_gcc(bio)
        
        # 根據方法選擇角度估計函數
        if method == "itd":
            pred = gcc_phat_estimate(bio, sample_rate=sample_rate)
        elif method == "ild":
            pred = ild_estimate(bio, sample_rate=sample_rate, method='spectral')
        elif method == "hybrid":
            pred = hybrid_estimate(bio, sample_rate=sample_rate)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'itd', 'ild', or 'hybrid'")
        
        err = angular_error_deg(pred, desired)
        if verbose:
            print(f"{float(tx):10.2f} {pred:10.2f} {err:10.2f}")
        if err < best_err:
            best_err = err
            best_tx = float(tx)
            best_pred = float(pred)
            best_binaural = binaural

    assert best_binaural is not None
    return best_tx, best_err, best_pred, best_binaural


def truncate_mono_for_calibration(mono: th.Tensor, max_seconds: float, sample_rate: int = 48000) -> th.Tensor:
    """僅用於校準：取前 max_seconds，再對齊到 400 的倍數。"""
    nmax = int(max_seconds * sample_rate)
    if mono.shape[-1] <= nmax:
        t = mono.shape[-1] // 400 * 400
        return mono[:, :t]
    m = mono[:, :nmax]
    t = m.shape[-1] // 400 * 400
    return m[:, :t]
