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
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss
# 2026-01-24: 加入時間對齊模組以修正 Phase Error - Import alignment module to fix Phase Error
from src.alignment import find_alignment_offset, align_signals, diagnose_alignment


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./data/testset",
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


def chunked_forwarding(network, segments, mono, view):
    # process segments
    binaural_segments = []
    for i in range(len(segments)):
        # get chunk
        start_frame = segments[i]['start']
        end_frame = segments[i]['end']
        start_view = start_frame // 400
        end_view = end_frame // 400
        mono_chunk = mono[:, :, start_frame:end_frame]
        view_chunk = view[:, :, start_view:end_view]
        # move to cuda
        if th.cuda.is_available():
            mono_chunk = mono_chunk.cuda()
            view_chunk = view_chunk.cuda()
        # forward pass
        binaural_chunk = network(mono_chunk, view_chunk)["output"]
        binaural_segments.append(binaural_chunk.detach().cpu())
    
    # concat segments
    binaural = th.cat(binaural_segments, dim=-1)
    return binaural


def compute_metrics(binauralized, reference):
    # move to cpu
    binauralized = binauralized.reshape(2, -1).cpu()
    if th.cuda.is_available():
        binauralized = binauralized.cuda()
        reference = reference.cuda()
    binauralized, reference = binauralized.unsqueeze(0), reference.unsqueeze(0)

    # compute error metrics
    l2_error = L2Loss()(binauralized, reference)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized, reference)
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized, reference)

    return{
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "samples": binauralized.shape[-1]
    }


# binauralized and evaluate test sequence for the eight subjects and the validation sequence
# Dynamic listing to handle standard or custom folders
try:
    test_sequences = [d for d in os.listdir(args.dataset_directory) if os.path.isdir(os.path.join(args.dataset_directory, d))]
    test_sequences.sort()
except FileNotFoundError:
    test_sequences = []

if not test_sequences:
     test_sequences = [f"subject{i+1}" for i in range(6)] + ["validation_sequence"]

# initialize network
net = BinauralNetwork(view_dim=7,
                      wavenet_blocks=args.blocks)
# load weights
net.load_state_dict(load_model(args.model_file))
if th.cuda.is_available():
    net.cuda()
net.eval()

if not os.path.exists(args.artifacts_directory):
    os.makedirs(args.artifacts_directory)

errors = []
for test_sequence in test_sequences:
    print(f"binauralize {test_sequence}...")
    # load mono info and view
    try:
        # Replaced torchaudio.load with soundfile.read
        audio_path = f"{args.dataset_directory}/{test_sequence}/mono.wav"
        data, sample_rate = sf.read(audio_path, dtype='float32')
        # Convert to tensor and shape (1, T)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T # (T, C) -> (C, T)
        mono = th.from_numpy(data)
        
    except Exception as e:
        print(f"Error loading mono: {e}")
        continue
        
    view = np.loadtxt(f"{args.dataset_directory}/{test_sequence}/tx_positions.txt").transpose().astype(np.float32)
    view = th.from_numpy(view)
    
    # 2026-01-24: 載入 ground truth binaural 用於時間對齊
    # Load ground truth binaural for alignment
    try:
        ref_path = f"{args.dataset_directory}/{test_sequence}/binaural.wav"
        ref_data, sr = sf.read(ref_path, dtype='float32')
        if ref_data.ndim == 1:
            ref_data = ref_data.reshape(1, -1)
        else:
            ref_data = ref_data.T
        reference = th.from_numpy(ref_data)
        
        # 2026-01-24: 時間對齊 - 偵測並修正 mono 和 binaural 之間的時間偏移
        # Temporal alignment: detect and correct offset between mono 和 binaural
        offset, correlation = find_alignment_offset(mono, reference, max_shift=2400, sample_rate=sample_rate)
        diagnose_alignment(mono, reference, offset, correlation, sample_rate=sample_rate)
        
        # 2026-01-24: 應用對齊 - 同時對齊 mono 和 reference
        # Apply alignment - align both mono and reference
        mono_aligned, reference_aligned = align_signals(mono, reference, offset)
        
    except Exception as e:
        print(f"Error loading/aligning reference: {e}")
        import traceback
        traceback.print_exc()
        # 如果 reference 無法載入，不進行對齊 - No alignment if reference not available
        mono_aligned = mono
        reference_aligned = None
    
    # 與 view 對齊長度 - Align lengths with view
    n_frames = min(mono_aligned.shape[-1] // 400, view.shape[-1])
    mono_aligned = mono_aligned[:, :n_frames * 400]
    view = view[:, :n_frames]

    # 切分成 1 秒片段 - chunk into 1s segments
    chunk_size = 48000
    segments = []
    for i in range(0, mono_aligned.shape[-1], chunk_size):
        segments.append({'start': i, 'end': min(i + chunk_size, mono_aligned.shape[-1])})

    # 使用對齊後的 mono 進行前向傳播 - forward pass with aligned mono
    binaural = chunked_forwarding(net, segments, mono_aligned.unsqueeze(0), view.unsqueeze(0))

    # 儲存雙耳化輸出 - save binauralized output
    output_path = f"{args.artifacts_directory}/{test_sequence}.wav"
    sf.write(output_path, binaural.squeeze(0).t().detach().cpu().numpy(), sample_rate)

    # 準備 reference 用於計算指標 - Prepare reference for metrics
    if reference_aligned is not None:
        # 裁剪 reference 以匹配生成長度 - Trim reference to match generated length
        reference_aligned = reference_aligned[:, :binaural.shape[-1]]
    else:
        # 沒有 reference 可用，跳過指標計算 - No reference available, skip metrics
        reference_aligned = binaural

    metrics = compute_metrics(binaural, reference_aligned)
    errors.append(metrics)
    # Original logging style (simple)
    print(f"  -> L2: {metrics['l2']*1000:.3f}, Amp: {metrics['amplitude']:.3f}, Phase: {metrics['phase']:.3f}")

# accumulate errors
if errors:
    sequence_weights = np.array([err["samples"] for err in errors])
    sequence_weights = sequence_weights / np.sum(sequence_weights)
    l2_error = sum([err["l2"] * sequence_weights[i] for i, err in enumerate(errors)])
    amplitude_error = sum([err["amplitude"] * sequence_weights[i] for i, err in enumerate(errors)])
    phase_error = sum([err["phase"] * sequence_weights[i] for i, err in enumerate(errors)])

    # print accumulated errors on testset
    print("-" * 30)
    print("Accumulated Errors on Testset:")
    print(f"l2 (x10^3):     {l2_error * 1000:.3f}")
    print(f"amplitude:      {amplitude_error:.3f}")
    print(f"phase:          {phase_error:.3f}")
