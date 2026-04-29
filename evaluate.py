"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import numpy as np
import torch as th
# import torchaudio as ta # Replaced with soundfile
import soundfile as sf

from src.models import BinauralNetwork
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss, ITDLoss, ILDLoss
# 2026-01-24: 加入時間對齊模組以修正 Phase Error - Import alignment module to fix Phase Error
# 2026-01-28: 新增 align_by_speech_onset 以使用 VAD 對齊
from src.alignment import find_alignment_offset, align_signals, diagnose_alignment, align_by_speech_onset


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./dataset/test_13angles",
                    help="path to the test data")
parser.add_argument("--model_file",
                    type=str,
                    default="./outputs/binaural_network.net",
                    help="model file containing the trained binaural network weights")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./outputs",
                    help="directory to write binaural outputs to")
parser.add_argument("--blocks",
                    type=int,
                    default=3)
args = parser.parse_args()


def load_model(weights_path):
    # load network
    if not os.path.exists(weights_path):
        print("Error: Model weights not found in the outputs directory.")
        exit()

    network = th.load(weights_path)
    # Check if loaded object is a state dict or full model
    if isinstance(network, dict):
        print(f"Loaded state dict from: {weights_path}")
        return network # Return dict directly
    
    # If it's a model object
    network.eval()
    if th.cuda.is_available():
        network.cuda()
    print(f"Loaded model object: {weights_path}")
    return network.state_dict() # Always return state dict for consistency


def chunked_forwarding(net, mono, view):
    """
    分段處理雙耳化 (Binauralize mono input in chunks)
    
    使用 Overlap-Discard 策略處理長音訊，避免記憶體溢出
    Use Overlap-Discard strategy for long audio to avoid OOM
    
    參數 Parameters:
        net: 雙耳化網路 (Binauralization network)
        mono: 單聲道輸入訊號，形狀 1 x T (Mono input signal, shape 1 x T)
        view: Tx/Rx 位置/方向，形狀 7 x K，K = T/400 (View conditioning, shape 7 x K)
    
    返回 Returns:
        binaural: 雙耳輸出訊號，形狀 2 x T (Binaural output, shape 2 x T)
    
    修復註記 Fix Note:
        2026-01-26: 恢復原始實作以修復噪音問題
        Restored original implementation to fix noise issue
    """
    net.eval()
    if th.cuda.is_available():
        net.cuda()
        mono, view = mono.cuda(), view.cuda()

    # 分段參數 Chunking parameters
    chunk_size = 480000  # 10秒片段 (10s chunks at 48kHz)
    
    # 感受野 + 安全邊界 (Receptive field + safety margin)
    # 加 1000 samples 是因為 time warping 的感受野難以精確計算
    # Add 1000 as "safe bet" since warping has undefined receptive field
    rec_field = net.receptive_field() + 1000
    
    # 確保是 400 的倍數，以匹配音訊和 view 的頻率
    # Make sure it's a multiple of 400 to match audio and view frequencies
    rec_field -= rec_field % 400
    
    # 建立重疊片段 (Create overlapping chunks)
    # 每個片段包含前文 (context)，以確保邊界的連續性
    # Each chunk includes context to ensure continuity at boundaries
    chunks = [
        {
            "mono": mono[:, max(0, i-rec_field):i+chunk_size],
            "view": view[:, max(0, i-rec_field)//400:(i+chunk_size)//400]
        }
        for i in range(0, mono.shape[-1], chunk_size)
    ]

    # 處理每個片段 (Process each chunk)
    for i, chunk in enumerate(chunks):
        with th.no_grad():
            mono_chunk = chunk["mono"].unsqueeze(0)
            view_chunk = chunk["view"].unsqueeze(0)
            
            # 前向傳播 (Forward pass)
            binaural = net(mono_chunk, view_chunk)["output"].squeeze(0)
            
            # Overlap-Discard: 丟棄受前文污染的部分
            # Discard the part contaminated by context
            if i > 0:
                # 只保留後面的部分（不包含前文）
                # Keep only the latter part (excluding context)
                binaural = binaural[:, -(mono_chunk.shape[-1]-rec_field):]
            
            chunk["binaural"] = binaural

    # 連接所有片段 (Concatenate all chunks)
    binaural = th.cat([chunk["binaural"] for chunk in chunks], dim=-1)
    
    # 限制範圍到 [-1, 1] 以防止噪音和失真
    # Clamp to [-1, 1] to prevent noise and distortion
    # 2026-01-26: 這一步非常重要！缺少這個會導致噪音
    # This step is CRITICAL! Missing this causes noise
    binaural = th.clamp(binaural, min=-1, max=1).cpu()
    
    return binaural


def compute_metrics(binauralized, reference, ground_truth_angle=None):
    """
    計算評估指標 (Compute evaluation metrics)
    
    參數 Parameters:
        binauralized: 預測的雙耳訊號，形狀 2 x T (Predicted binaural signal, shape 2 x T)
        reference: 參考的雙耳訊號，形狀 2 x T (Reference binaural signal, shape 2 x T)
    
    返回 Returns:
        dict: 包含各種誤差指標 (Dictionary containing error metrics):
            - l2: L2 均方誤差 (Mean squared error)
            - amplitude: 振幅誤差 (Amplitude error in frequency domain)
            - phase: 相位誤差 (Phase error in frequency domain)
            - itd: ITD 誤差，單位微秒 (ITD error in microseconds)
            - ild: ILD 誤差，單位 dB (ILD error in dB)
            - samples: 樣本數 (Number of samples)
    """
    # 準備張量 Prepare tensors
    binauralized = binauralized.reshape(2, -1).cpu()
    if th.cuda.is_available():
        binauralized = binauralized.cuda()
        reference = reference.cuda()
    
    # 確保 binauralized 和 reference 長度一致（取較短者）
    # Ensure binauralized and reference have the same length (use shorter one)
    min_len = min(binauralized.shape[-1], reference.shape[-1])
    binauralized = binauralized[:, :min_len]
    reference = reference[:, :min_len]
    
    # 增加 batch 維度
    # Add batch dimension
    binauralized_batch = binauralized.unsqueeze(0)
    reference_batch = reference.unsqueeze(0)

    # 計算誤差指標 Compute error metrics
    # L2 和 Amplitude 使用長度對齊後的原始信號
    # L2 and Amplitude use length-aligned original signals
    l2_error = L2Loss()(binauralized_batch, reference_batch)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized_batch, reference_batch)
    
    # Phase 使用原始信號
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized_batch, reference_batch)
    
    # 2026-01-25: 新增 ITD 和 ILD 錯誤指標
    # Added ITD (Interaural Time Difference) and ILD (Interaural Level Difference) error metrics
    itd_error = ITDLoss(sample_rate=48000, max_shift_ms=1.0)(binauralized_batch, reference_batch)
    ild_error = ILDLoss(sample_rate=48000)(binauralized_batch, reference_batch)

    # 新增 Angular Error 指標，使用 GCC-PHAT 估計角度
    angular_error = None
    pred_angle = None
    if ground_truth_angle is not None:
        from src.doa import gcc_phat_estimate
        try:
            audio_np = binauralized.cpu().numpy()
            if audio_np.shape[0] != 2:
                audio_np = audio_np.T 
            
            left_channel = audio_np[0, :]
            right_channel = audio_np[1, :]
            
            valid_indices = np.where(np.abs(left_channel) > 1e-4)[0]
            if len(valid_indices) > 0:
                valid_len = valid_indices[-1] + 1
                valid_len = int(valid_len * 0.95) 
                left_channel = left_channel[:valid_len]
                right_channel = right_channel[:valid_len]

            trimmed_binaural = np.stack([left_channel, right_channel])
            pred_angle = gcc_phat_estimate(trimmed_binaural, sample_rate=48000)
            
            # 反轉角度符號以匹配訓練時的坐標系定義（Y+ 為右方 / Y- 為左方）
            # Invert the angle sign to match the coordinate system definition
            if pred_angle is not None:
                pred_angle = -pred_angle
            
            error = abs(pred_angle - ground_truth_angle)
            error = min(error, 360.0 - error)
            angular_error = error
        except Exception as e:
            print(f"    [Warning] Angular error computation failed: {e}")
            angular_error = None

    return {
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "itd": itd_error,
        "ild": ild_error,
        "angular_error": angular_error,
        "predicted_angle": pred_angle,
        "ground_truth_angle": ground_truth_angle,
        "samples": binauralized.shape[-1]
    }


# ============================================================================
# 主評估流程 (Main Evaluation Pipeline)
# ============================================================================

# 動態列出測試序列 (Dynamically list test sequences)
# 支援自訂資料集目錄結構 (Support custom dataset directory structure)
try:
    test_sequences = [d for d in os.listdir(args.dataset_directory) 
                     if os.path.isdir(os.path.join(args.dataset_directory, d))]
    test_sequences.sort()
except FileNotFoundError:
    print(f"Error: Dataset directory not found: {args.dataset_directory}")
    test_sequences = []

# 後備預設序列 (Fallback to default sequences)
if not test_sequences:
    test_sequences = [f"subject{i+1}" for i in range(6)] + ["validation_sequence"]

SUBJECT_ANGLES = {
    'subject1': -90.0,
    'subject2': -60.0,
    'subject3': -30.0,
    'subject4': 0.0,
    'subject5': 30.0,
    'subject6': 60.0,
    'subject7': 90.0,
}

# initialize network
net = BinauralNetwork(view_dim=7,
                      wavenet_blocks=args.blocks)
# load weights
net.load_state_dict(load_model(args.model_file), strict=False)
if th.cuda.is_available():
    net.cuda()
net.eval()

if not os.path.exists(args.artifacts_directory):
    os.makedirs(args.artifacts_directory)

# 初始化誤差列表 (Initialize error list)
errors = []

# 逐個處理每個測試序列 (Process each test sequence)
for test_sequence in test_sequences:
    print(f"處理 {test_sequence}... (Processing {test_sequence}...)")
    
    # ========================================================================
    # 1. 載入 Mono 音訊和 View 資訊 (Load mono audio and view)
    # ========================================================================
    try:
        # 2026-01-26: 使用 soundfile 取代 torchaudio (Use soundfile instead of torchaudio)
        audio_path = f"{args.dataset_directory}/{test_sequence}/mono.wav"
        data, sample_rate = sf.read(audio_path, dtype='float32')
        
        # 轉換為張量並調整形狀 (Convert to tensor and reshape)
        if data.ndim == 1:
            data = data.reshape(1, -1)  # (T,) -> (1, T)
        else:
            data = data.T  # (T, C) -> (C, T)
        mono = th.from_numpy(data)
    except Exception as e:
        print(f"  錯誤：無法載入 mono 音訊 (Error loading mono): {e}")
        continue
    
    # 載入 Transmitter 位置資訊 (Load transmitter positions)
    view = np.loadtxt(f"{args.dataset_directory}/{test_sequence}/tx_positions.txt").transpose().astype(np.float32)
    view = th.from_numpy(view)
    
    # ========================================================================
    # 2. 載入參考雙耳音訊 (Load reference binaural audio)
    # ========================================================================
    try:
        ref_path = f"{args.dataset_directory}/{test_sequence}/binaural.wav"
        ref_data, sr = sf.read(ref_path, dtype='float32')
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(1, -1)
        else:
            ref_data = ref_data.T
        reference = th.from_numpy(ref_data)
    except Exception as e:
        print(f"  警告：無法載入參考音訊，將跳過指標計算 (Warning: cannot load reference)")
        reference = None
    
    # ========================================================================
    # 3. 對齊長度 (Align lengths - Use Padding)
    # ========================================================================
    target_frames = view.shape[-1]
    target_samples = target_frames * 400
    
    original_mono_len = mono.shape[-1]
    
    if mono.shape[-1] < target_samples:
        padding_samples = target_samples - mono.shape[-1]
        print(f"  [Padding] Mono ({mono.shape[-1]}) < View ({target_samples}) -> Padding {padding_samples} samples")
        mono = th.nn.functional.pad(mono, (0, padding_samples))
    elif mono.shape[-1] > target_samples:
        print(f"  [Truncation] Mono ({mono.shape[-1]}) > View ({target_samples}) -> Truncating to {target_samples} samples")
        mono = mono[:, :target_samples]

    # ========================================================================
    # 4. 雙耳化處理 (Binauralization)
    # ========================================================================
    # 使用改進的 chunked_forwarding（已修復噪音問題）
    # Use improved chunked_forwarding (noise issue fixed)
    binaural = chunked_forwarding(net, mono, view)
    
    # 移除剛剛補上的 Padding，避免後半段神經網路生成的無聲噪音污染 GCC-PHAT 結果
    binaural = binaural[:, :original_mono_len]

    # ========================================================================
    # 5. 儲存輸出 (Save output)
    # ========================================================================
    output_path = f"{args.artifacts_directory}/{test_sequence}.wav"
    sf.write(output_path, binaural.t().detach().cpu().numpy(), sample_rate)

    # ========================================================================
    # 6. 計算評估指標 (Compute evaluation metrics)
    # ========================================================================
    if reference is not None:
        # 裁剪 reference 以匹配生成長度 (Trim reference to match generated length)
        reference = reference[:, :binaural.shape[-1]]
        
        gt_angle = SUBJECT_ANGLES.get(test_sequence, None)
        # 計算指標 (Compute metrics)
        metrics = compute_metrics(binaural, reference, ground_truth_angle=gt_angle)
        metrics['test_sequence'] = test_sequence
        errors.append(metrics)
        
        # 輸出指標 (Print metrics)
        if metrics['angular_error'] is not None:
            print(f"  -> L2: {metrics['l2']*1000:.3f}, "
                  f"Amp: {metrics['amplitude']:.3f}, "
                  f"Phase: {metrics['phase']:.3f}, "
                  f"ITD: {metrics['itd']:.1f}μs, "
                  f"ILD: {metrics['ild']:.2f}dB, "
                  f"Angle Error: {metrics['angular_error']:.1f}° (Pred: {metrics['predicted_angle']:.1f}°, GT: {metrics['ground_truth_angle']:.1f}°)")
        else:
            print(f"  -> L2: {metrics['l2']*1000:.3f}, "
                  f"Amp: {metrics['amplitude']:.3f}, "
                  f"Phase: {metrics['phase']:.3f}, "
                  f"ITD: {metrics['itd']:.1f}μs, "
                  f"ILD: {metrics['ild']:.2f}dB")

# ============================================================================
# 累積誤差計算與輸出 (Accumulate errors and print results)
# ============================================================================
if errors:
    # 計算每個序列的權重（按樣本數加權）
    # Calculate weights for each sequence (weighted by number of samples)
    sequence_weights = np.array([err["samples"] for err in errors])
    sequence_weights = sequence_weights / np.sum(sequence_weights)
    
    # 計算加權平均誤差 (Compute weighted average errors)
    l2_error = sum([err["l2"] * sequence_weights[i] for i, err in enumerate(errors)])
    amplitude_error = sum([err["amplitude"] * sequence_weights[i] for i, err in enumerate(errors)])
    phase_error = sum([err["phase"] * sequence_weights[i] for i, err in enumerate(errors)])
    
    # 2026-01-25: 累積 ITD 和 ILD 誤差 (Accumulate ITD and ILD errors)
    itd_error = sum([err["itd"] * sequence_weights[i] for i, err in enumerate(errors)])
    ild_error = sum([err["ild"] * sequence_weights[i] for i, err in enumerate(errors)])

    # 輸出測試集的累積誤差 (Print accumulated errors on testset)
    print("-" * 60)
    print(f"{'Subject':<10} {'GT 角度':<10} {'預測角度':<10} {'誤差':<10} {'ITD 誤差':<10}")
    for err in errors:
        if err['angular_error'] is not None:
            subj = err.get('test_sequence', 'unknown')
            gt = f"{err['ground_truth_angle']:+.0f}°".replace('+0°', '0°').replace('-0°', '0°')
            pred = f"{err['predicted_angle']:+.1f}°"
            ang_err = f"{err['angular_error']:.1f}°"
            itd = f"{err['itd']:.1f}μs"
            print(f"{subj:<10} {gt:<10} {pred:<10} {ang_err:<10} {itd:<10}")

    print("-" * 60)
    print("測試集累積誤差 (Accumulated Errors on Testset):")
    print("-" * 60)
    print(f"L2 誤差 (x10^3):     {l2_error * 1000:.3f}")
    print(f"振幅誤差 (Amplitude): {amplitude_error:.3f}")
    print(f"相位誤差 (Phase):     {phase_error:.3f}")
    print(f"ITD 誤差 (μs):       {itd_error:.1f}")
    print(f"ILD 誤差 (dB):       {ild_error:.2f}")
    print("-" * 60)

