"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import argparse

from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer import Trainer

# ============================================================================
# 命令列參數解析 (Command-line argument parsing)
# ============================================================================
parser = argparse.ArgumentParser(description='訓練雙耳語音合成模型 (Train Binaural Speech Synthesis Model)')
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./data/trainset",
                    help="訓練資料路徑 (path to the training data)")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./outputs",
                    help="模型檔案輸出目錄 (directory to write model files to)")
parser.add_argument("--num_gpus",
                    type=int,
                    default=4,
                    help="訓練時使用的 GPU 數量 (number of GPUs used during training)")
parser.add_argument("--blocks",
                    type=int,
                    default=3,
                    help="WaveNet 的 block 數量 (number of wavenet blocks, 3=large model, 1=small model)")
parser.add_argument("--resume",
                    type=str,
                    default=None,
                    help="從 checkpoint 繼續訓練 (resume training from checkpoint, e.g., 'epoch-10')")
args = parser.parse_args()

# ============================================================================
# 訓練配置 (Training Configuration)
# ============================================================================
config = {
    "artifacts_dir": args.artifacts_directory,  # 輸出目錄 (Output directory)
    "learning_rate": 0.001,                     # 學習率 (Learning rate)
    "newbob_decay": 0.5,                        # NewBob 學習率衰減因子 (NewBob lr decay factor)
    "newbob_max_decay": 0.1,                    # 最大衰減值 (Maximum decay value, 改為 0.1 避免 lr 下降太快)
    "batch_size": 32,                           # 批次大小 (Batch size)
    "mask_beginning": 1024,                     # 遮罩開頭的樣本數 (Mask beginning samples in loss)
    "loss_weights": {"l2": 1.0, "phase": 0.01, "ipd": 0.1}, # 損失函數權重 (降低 IPD 權重到 0.1)
    "lambda_warp": 0.01,                        # Warp loss 權重 (懲罰 neural warp 偏離 geometric warp)
    "lambda_smooth": 0.001,                     # Warp smoothness loss 權重 (懲罰 warpfield 不平滑)
    "save_frequency": 10,                       # 每 N 個 epoch 保存一次 (Save every N epochs)
    "epochs": 100,                              # 總訓練輪數 (Total training epochs)
    "num_gpus": args.num_gpus,                  # GPU 數量 (Number of GPUs)
}

# 建立輸出目錄 (Create output directory)
os.makedirs(config["artifacts_dir"], exist_ok=True)

# ============================================================================
# 載入資料集 (Load Dataset)
# ============================================================================
# chunk_size_ms=200: 每個訓練片段 200 毫秒 (200ms per training chunk)
# overlap=0.5: 50% 重疊以增加資料多樣性 (50% overlap for data augmentation)
dataset = BinauralDataset(dataset_directory=args.dataset_directory, 
                         chunk_size_ms=200, 
                         overlap=0.5)

# ============================================================================
# 建立模型 (Create Model)
# ============================================================================
net = BinauralNetwork(
    view_dim=7,               # View 維度: 3D 位置 (x,y,z) + 四元數 (qx,qy,qz,qw)
    warpnet_layers=4,         # Warpnet 層數 (Number of warpnet layers)
    warpnet_channels=64,      # Warpnet 通道數 (Number of warpnet channels)
    wavenet_blocks=args.blocks,     # WaveNet block 數量 (Number of wavenet blocks)
    layers_per_block=10,      # 每個 block 的層數 (Layers per block)
    wavenet_channels=64       # WaveNet 通道數 (Number of wavenet channels)
)

# ============================================================================
# 輸出模型資訊 (Print Model Information)
# ============================================================================
print(f"感受野 (Receptive field): {net.receptive_field()} 樣本 (samples)")
print(f"訓練片段數 (Train on): {len(dataset.chunks)} 個片段 (chunks)")
print(f"可訓練參數數量 (Trainable parameters): {net.num_trainable_parameters():,}")

# ============================================================================
# 開始訓練 (Start Training)
# ============================================================================
# 將 resume 參數傳遞給 Trainer
trainer = Trainer(config, net, dataset, resume_from=args.resume)
trainer.train()

