#!/usr/bin/env python3
"""
診斷工具：檢查不同 mono 輸入是否產生不同的雙耳輸出
"""
import soundfile as sf
import torch as th
import numpy as np

from src.synthesis_utils import (
    angle_to_tx_positions,
    chunked_forwarding,
    load_binaural_net,
)
from src.doa import gcc_phat_estimate

# 載入兩個不同的 mono 檔案
print("載入 subject1...")
audio1, sr1 = sf.read("dataset/testset_org/subject1/mono.wav", dtype="float32")
if audio1.ndim > 1:
    audio1 = audio1.mean(axis=1)
mono1 = th.from_numpy(audio1).unsqueeze(0)

print("載入 subject2...")
audio2, sr2 = sf.read("dataset/testset_org/subject2/mono.wav", dtype="float32")
if audio2.ndim > 1:
    audio2 = audio2.mean(axis=1)
mono2 = th.from_numpy(audio2).unsqueeze(0)

# 檢查 mono 是否不同
print(f"\nmono1 shape: {mono1.shape}")
print(f"mono2 shape: {mono2.shape}")
print(f"mono1 mean: {mono1.mean():.6f}, std: {mono1.std():.6f}")
print(f"mono2 mean: {mono2.mean():.6f}, std: {mono2.std():.6f}")
print(f"mono1 vs mono2 相同？ {th.allclose(mono1[:, :min(mono1.shape[1], mono2.shape[1])], mono2[:, :min(mono1.shape[1], mono2.shape[1])])}")

# 截斷到相同長度
max_samples = int(10.0 * 48000)
mono1 = mono1[:, :max_samples]
mono2 = mono2[:, :max_samples]

target_samples = (mono1.shape[-1] // 400) * 400
mono1 = mono1[:, :target_samples]
mono2 = mono2[:, :target_samples]
num_frames = target_samples // 400

print(f"\n截斷後:")
print(f"mono1 shape: {mono1.shape}, frames: {num_frames}")
print(f"mono2 shape: {mono2.shape}, frames: {num_frames}")

# 載入模型
print("\n載入模型...")
net = load_binaural_net("outputs/binaural_network.newbob.net", blocks=3)

# 測試相同角度
test_angle = 60.0
print(f"\n測試角度: {test_angle}°")

view = th.from_numpy(angle_to_tx_positions(test_angle, 1.0, num_frames))
print(f"view shape: {view.shape}")

print("\n合成 subject1...")
binaural1 = chunked_forwarding(net, mono1, view)
print(f"binaural1 shape: {binaural1.shape}")
print(f"binaural1 mean: {binaural1.mean():.6f}, std: {binaural1.std():.6f}")

print("\n合成 subject2...")
binaural2 = chunked_forwarding(net, mono2, view)
print(f"binaural2 shape: {binaural2.shape}")
print(f"binaural2 mean: {binaural2.mean():.6f}, std: {binaural2.std():.6f}")

print(f"\nbinaural1 vs binaural2 相同？ {th.allclose(binaural1, binaural2)}")

# GCC-PHAT 估計
from src.synthesis_utils import trim_binaural_for_gcc

bio1 = trim_binaural_for_gcc(binaural1.numpy())
bio2 = trim_binaural_for_gcc(binaural2.numpy())

pred1 = gcc_phat_estimate(bio1, sample_rate=48000)
pred2 = gcc_phat_estimate(bio2, sample_rate=48000)

print(f"\nGCC-PHAT 估計:")
print(f"subject1: {pred1:+.2f}°")
print(f"subject2: {pred2:+.2f}°")
print(f"差異: {abs(pred1 - pred2):.2f}°")

if abs(pred1 - pred2) < 0.1:
    print("\n⚠️ 警告: 兩個不同的 mono 輸入產生了幾乎相同的角度估計！")
    print("可能的原因:")
    print("1. 模型主要依賴 view (tx_positions)，而非 mono 內容")
    print("2. GCC-PHAT 估計對 mono 內容不敏感")
    print("3. 兩個 mono 檔案的頻譜特性太相似")
else:
    print("\n✓ 正常: 不同的 mono 輸入產生了不同的角度估計")
