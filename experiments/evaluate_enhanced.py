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
import soundfile as sf
from scipy.spatial.transform import Rotation as R

from src.models import BinauralNetwork
from src.losses import L2Loss, AmplitudeLoss, PhaseLoss, AngularError, ITDLoss, ILDLoss


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


def chunked_forwarding(net, mono, view):
    '''
    binauralized the mono input given the view
    :param net: binauralization network
    :param mono: 1 x T tensor containing the mono audio signal
    :param view: 7 x K tensor containing the view as 3D positions and quaternions for orientation (K = T / 400)
    :return: 2 x T tensor containing binauralized audio signal
    '''
    net.eval().cuda()
    mono, view = mono.cuda(), view.cuda()

    chunk_size = 480000  # forward in chunks of 10s
    rec_field = net.receptive_field() + 1000  # add 1000 samples as "safe bet" since warping has undefined rec. field
    rec_field -= rec_field % 400  # make sure rec_field is a multiple of 400 to match audio and view frequencies
    chunks = [
        {
            "mono": mono[:, max(0, i-rec_field):i+chunk_size],
            "view": view[:, max(0, i-rec_field)//400:(i+chunk_size)//400]
        }
        for i in range(0, mono.shape[-1], chunk_size)
    ]

    for i, chunk in enumerate(chunks):
        with th.no_grad():
            mono = chunk["mono"].unsqueeze(0)
            view = chunk["view"].unsqueeze(0)
            binaural = net(mono, view)["output"].squeeze(0)
            if i > 0:
                binaural = binaural[:, -(mono.shape[-1]-rec_field):]
            chunk["binaural"] = binaural

    binaural = th.cat([chunk["binaural"] for chunk in chunks], dim=-1)
    binaural = th.clamp(binaural, min=-1, max=1).cpu()
    return binaural


def compute_metrics(binauralized, reference):
    '''
    compute l2 error, amplitude error, and angular phase error for the given binaural and reference singal
    :param binauralized: 2 x T tensor containing predicted binaural signal
    :param reference: 2 x T tensor containing reference binaural signal
    :return: errors as a scalar value for each metric and the number of samples in the sequence
    '''
    binauralized, reference = binauralized.unsqueeze(0), reference.unsqueeze(0)

    # compute error metrics
    # compute error metrics
    l2_error = L2Loss()(binauralized, reference)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized, reference)
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized, reference)
    angular_error = AngularError(sample_rate=48000, ignore_below=0.2)(binauralized, reference)
    
    # 2026-01-25: 新增 ITD 和 ILD 錯誤指標 - Added ITD and ILD error metrics
    itd_error = ITDLoss(sample_rate=48000, max_shift_ms=1.0)(binauralized, reference)
    ild_error = ILDLoss(sample_rate=48000)(binauralized, reference)

    return{
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "angle": angular_error,
        "itd": itd_error,
        "ild": ild_error,
        "samples": binauralized.shape[-1]
    }


# binauralized and evaluate test sequence for the eight subjects and the validation sequence
# Dynamically list subdirectories in the dataset directory
try:
    test_sequences = [d for d in os.listdir(args.dataset_directory) if os.path.isdir(os.path.join(args.dataset_directory, d))]
    test_sequences.sort()
except FileNotFoundError:
    print(f"Dataset directory not found: {args.dataset_directory}")
    test_sequences = []

# If empty (maybe correct path but no folders?), fallback or just empty
if not test_sequences:
    # default fallback if dynamic list fails or is empty, though logical to just warn.
    # checking for specific known structure if strictly required
    test_sequences = [f"subject{i+1}" for i in range(6)] + ["validation_sequence"]

# initialize network
net = BinauralNetwork(view_dim=7,
                      warpnet_layers=4,
                      warpnet_channels=64,
                      wavenet_blocks=args.blocks,
                      layers_per_block=10,
                      wavenet_channels=64
                      )
net.load_from_file(args.model_file)

os.makedirs(f"{args.artifacts_directory}", exist_ok=True)

errors = []
for test_sequence in test_sequences:
    print(f"binauralize {test_sequence}...")

    # load mono input and view conditioning
    try:
        mono, sr = sf.read(f"{args.dataset_directory}/{test_sequence}/mono.wav", dtype='float32')
        mono = th.from_numpy(mono)
        if mono.ndim == 1:
            mono = mono.unsqueeze(0)
        else:
            mono = mono.t()
    except Exception as e:
        print(f"Error loading mono: {e}")
        continue
    view = np.loadtxt(f"{args.dataset_directory}/{test_sequence}/tx_positions.txt").transpose().astype(np.float32)
    
    # # 嘗試讀取 Rx positions (如果有的話)
    # # Try to load rx_positions.txt
    # rx_path = f"{args.dataset_directory}/{test_sequence}/rx_positions.txt"
    # if os.path.exists(rx_path):
    #     rx_view = np.loadtxt(rx_path).transpose().astype(np.float32)
        
    #     # 確保長度一致
    #     # Ensure lengths match
    #     min_len = min(view.shape[1], rx_view.shape[1])
    #     view = view[:, :min_len]
    #     rx_view = rx_view[:, :min_len]
        
    #     # 計算相對位置與姿態 (World Frame -> Rx Frame)
    #     # Calculate relative position and orientation
        
    #     # 1. 取出位置與四元數
    #     tx_pos = view[:3, :].T # (N, 3)
    #     tx_quat = view[3:, :].T # (N, 4)
    #     rx_pos = rx_view[:3, :].T # (N, 3)
    #     rx_quat = rx_view[3:, :].T # (N, 4)
        
    #     # 2. 轉換為 Scipy Rotation 物件
    #     r_tx = R.from_quat(tx_quat)
    #     r_rx = R.from_quat(rx_quat)
        
    #     # 3. 計算相對位置: (Tx - Rx) 在 Rx 座標系下的投影
    #     # Relative Position: (Tx - Rx) projected into Rx's coordinate frame
    #     # P_rel = R_rx_inv * (P_tx - P_rx)
    #     rel_pos = r_rx.inv().apply(tx_pos - rx_pos)
        
    #     # 4. 計算相對旋轉: Rx 看 Tx 的姿態
    #     # Relative Rotation: Tx orientation relative to Rx
    #     # R_rel = R_rx_inv * R_tx
    #     rel_rot = (r_rx.inv() * r_tx).as_quat()
        
    #     # 5. 組合回 view tensor (7, N)
    #     # Combine back to view tensor
    #     view = np.concatenate([rel_pos, rel_rot], axis=1).T
    #     view = view.astype(np.float32)

    view = th.from_numpy(view)
    
    # Align lengths
    n_frames = min(mono.shape[-1] // 400, view.shape[-1])
    mono = mono[..., :n_frames * 400]
    view = view[..., :n_frames]
    
    # sanity checks
    if not sr == 48000:
        raise Exception(f"sampling rate is expected to be 48000 but is {sr}.")


    # binauralize and save output
    binaural = chunked_forwarding(net, mono, view)
    sf.write(f"{args.artifacts_directory}/{test_sequence}.wav", binaural.t().detach().cpu().numpy(), sr)

    # compute error metrics
    reference, sr = sf.read(f"{args.dataset_directory}/{test_sequence}/binaural.wav", dtype='float32')
    reference = th.from_numpy(reference)
    if reference.ndim == 1:
        reference = reference.unsqueeze(0)
    else:
        reference = reference.t()
    
    # Trim reference to match generated binaural length
    if reference.shape[-1] > binaural.shape[-1]:
        reference = reference[..., :binaural.shape[-1]]
    elif reference.shape[-1] < binaural.shape[-1]:
        binaural = binaural[..., :reference.shape[-1]]

    metrics = compute_metrics(binaural, reference)
    errors.append(metrics)
    print(f"  -> L2誤差: {metrics['l2']*1000:.3f}, 振幅誤差(Amp): {metrics['amplitude']:.3f}, 相位誤差(Phase): {metrics['phase']:.3f}, 角度誤差(Angle): {metrics['angle']:.2f}度, ITD: {metrics['itd']:.1f}μs, ILD: {metrics['ild']:.2f}dB")

# accumulate errors
sequence_weights = np.array([err["samples"] for err in errors])
sequence_weights = sequence_weights / np.sum(sequence_weights)
l2_error = sum([err["l2"] * sequence_weights[i] for i, err in enumerate(errors)])
amplitude_error = sum([err["amplitude"] * sequence_weights[i] for i, err in enumerate(errors)])
phase_error = sum([err["phase"] * sequence_weights[i] for i, err in enumerate(errors)])
angular_error = sum([err["angle"] * sequence_weights[i] for i, err in enumerate(errors)])
# 2026-01-25: 累積 ITD 和 ILD 誤差 - Accumulate ITD and ILD errors
itd_error = sum([err["itd"] * sequence_weights[i] for i, err in enumerate(errors)])
ild_error = sum([err["ild"] * sequence_weights[i] for i, err in enumerate(errors)])

# print accumulated errors on testset
print("-" * 30)
print("測試集累積誤差 (Accumulated Errors on Testset):")
print(f"L2誤差 (x10^3):       {l2_error * 1000:.3f}")
print(f"振幅誤差 (Amplitude):   {amplitude_error:.3f}")
print(f"相位誤差 (Phase):       {phase_error:.3f}")
print(f"角度誤差 (Angle):       {angular_error:.2f} 度")
print(f"ITD誤差 (μs):         {itd_error:.1f}")
print(f"ILD誤差 (dB):         {ild_error:.2f}")

