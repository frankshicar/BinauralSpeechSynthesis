"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse
import random
import numpy as np
import torch as th
from torch.utils.data import DataLoader, random_split

from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.finetuner import FineTuner

# ============================================================================
# 命令列參數解析 (Command-line argument parsing)
# ============================================================================
parser = argparse.ArgumentParser(description='微調雙耳語音合成模型 (Fine-tune Binaural Speech Synthesis Model)')

# Data arguments
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./dataset/trainset",
                    help="訓練資料路徑 (path to the training data)")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./output_finetune",
                    help="模型檔案輸出目錄 (directory to write model files to)")
parser.add_argument("--pretrained_model",
                    type=str,
                    default="./outputs/binaural_network.net",
                    help="預訓練模型路徑 (Path to pretrained model checkpoint)")

# Training mechanics arguments
parser.add_argument("--num_gpus",
                    type=int,
                    default=1,
                    help="訓練時使用的 GPU 數量 (number of GPUs used during training)")
parser.add_argument("--blocks",
                    type=int,
                    default=3,
                    help="WaveNet 的 block 數量 (number of wavenet blocks, 3=large model, 1=small model)")

# Fine-tuning specific arguments
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-5,
                    help="學習率，建議極低以保留原有能力 (Learning rate, keep extremely low e.g. 1e-5)")
parser.add_argument("--batch_size",
                    type=int,
                    default=4,
                    help="批次大小 (Batch size)")
parser.add_argument("--epochs",
                    type=int,
                    default=100,
                    help="總訓練輪數 (Total training epochs)")
parser.add_argument("--early_stopping_patience",
                    type=int,
                    default=5,
                    help="早停容忍度 (Early stopping patience in epochs)")
parser.add_argument("--dry_run",
                    action='store_true',
                    help="試跑模式：只跑 1 個 epoch 和少量資料 (Run 1 epoch with subset data for sanity check)")

args = parser.parse_args()

# Seeding for reproducibility
random.seed(42)
np.random.seed(42)
th.manual_seed(42)
if th.cuda.is_available():
    th.cuda.manual_seed_all(42)

# ============================================================================
# 訓練配置 (Training Configuration)
# ============================================================================
config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": args.learning_rate,
    "newbob_decay": 0.5,
    "newbob_max_decay": 0.01,
    "batch_size": args.batch_size,
    "mask_beginning": 1024,
    "loss_weights": {"l2": 1.0, "phase": 0.01},
    "save_frequency": 5,
    "epochs": args.epochs if not args.dry_run else 1,
    "num_gpus": args.num_gpus,
    "early_stopping_patience": args.early_stopping_patience,
}

print("="*40)
print(f"Fine-Tuning Configuration:")
print(f"  Dataset: {args.dataset_directory}")
print(f"  Pretrained Model: {args.pretrained_model}")
print(f"  Learning Rate: {config['learning_rate']}")
print(f"  Batch Size: {config['batch_size']}")
print(f"  Epochs: {config['epochs']}")
print(f"  Patience: {config['early_stopping_patience']}")
print("="*40)

# 建立輸出目錄
os.makedirs(config["artifacts_dir"], exist_ok=True)

# ============================================================================
# 載入資料集 (Load Dataset)
# ============================================================================
print("Loading dataset...", end=" ")
# Exclude bad subjects based on expert feedback (subject4, subject5 have negative correlation)
exclude_list = ['subject4', 'subject5']
print(f"Excluding subjects: {exclude_list}")

dataset = BinauralDataset(dataset_directory=args.dataset_directory, 
                         chunk_size_ms=200, 
                         overlap=0.5,
                         exclude_subjects=exclude_list)
print(f"Done. Total chunks: {len(dataset)}")

if args.dry_run:
    # Use valid subset logic to slice dataset if possible or just rely on random_split with small sizes
    total_len = len(dataset)
    train_size = 10
    val_size = 2
    dump_size = total_len - train_size - val_size
    train_set, val_set, _ = random_split(dataset, [train_size, val_size, dump_size])
else:
    # 90% Train, 10% Validation
    total_len = len(dataset)
    val_size = int(total_len * 0.1)
    if val_size < 1 and total_len > 1:
        val_size = 1
    train_size = total_len - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

print(f"Train set size: {len(train_set)}")
print(f"Val set size: {len(val_set)}")

# Create DataLoaders
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=4)

# ============================================================================
# 建立模型 (Create Model)
# ============================================================================
net = BinauralNetwork(
    view_dim=7,
    warpnet_layers=4,
    warpnet_channels=64,
    wavenet_blocks=args.blocks,
    layers_per_block=10,
    wavenet_channels=64
)

# Load Pretrained Weights
if args.pretrained_model:
    if os.path.isfile(args.pretrained_model):
        print(f"Loading pretrained model from {args.pretrained_model} ...")
        net.load_from_file(args.pretrained_model)
    else:
        print(f"WARNING: Pretrained model not found at {args.pretrained_model}")
        print("Starting from scratch? (Press Ctrl+C to abort if this is a mistake)")
        import time; time.sleep(5)
else:
    print("No pretrained model specified. Training from scratch.")

# ============================================================================
# Emergency Fix: Freeze Warping Network (2026-01-30)
# ============================================================================
print(f"Original Trainable parameters: {net.num_trainable_parameters():,}")
print("Freezing Warping Network to preserve spatial accuracy (ITD)...")

for param in net.warper.parameters():
    param.requires_grad = False

print(f"Active Trainable parameters (Refinement Only): {net.num_trainable_parameters():,}")

# ============================================================================
# 開始微調 (Start Fine-tuning)
# ============================================================================
tuner = FineTuner(config, net, train_loader, val_loader)
tuner.train()
