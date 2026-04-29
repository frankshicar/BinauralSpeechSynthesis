"""
改進版訓練腳本 v4
- blocks=3 (大模型)
- 改進的學習率調度 (ReduceLROnPlateau)
- 分階段訓練策略
- 動態 loss 權重
"""

import os
import argparse

from src.dataset import BinauralDataset
from src.models import BinauralNetwork
from src.trainer_v4 import TrainerV4

parser = argparse.ArgumentParser(description='訓練雙耳語音合成模型 v4')
parser.add_argument("--dataset_directory",
                    type=str,
                    default="./dataset_original/trainset",
                    help="訓練資料路徑")
parser.add_argument("--artifacts_directory",
                    type=str,
                    default="./outputs_v4",
                    help="模型檔案輸出目錄")
parser.add_argument("--num_gpus",
                    type=int,
                    default=1,
                    help="訓練時使用的 GPU 數量")
parser.add_argument("--blocks",
                    type=int,
                    default=3,
                    help="WaveNet 的 block 數量 (3=大模型)")
parser.add_argument("--resume",
                    type=str,
                    default=None,
                    help="從 checkpoint 繼續訓練")
args = parser.parse_args()

# 訓練配置
config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 0.001,
    "batch_size": 16,  # 3 blocks 記憶體需求大，降低 batch size
    "mask_beginning": 1024,
    "save_frequency": 10,
    "epochs": 100,
    "num_gpus": args.num_gpus,
    
    # 改進的學習率調度
    "lr_scheduler": {
        "type": "ReduceLROnPlateau",
        "patience": 10,      # 10 個 epoch 不改善才衰減
        "factor": 0.5,       # 衰減因子
        "min_lr": 1e-5       # 最小學習率
    },
    
    # 分階段訓練策略
    "training_stages": {
        "stage1": {  # Epoch 0-29: 專注 L2 + Phase
            "epochs": [0, 30],
            "loss_weights": {"l2": 10.0, "phase": 0.01, "ipd": 0.0}
        },
        "stage2": {  # Epoch 30-69: 加入 IPD
            "epochs": [30, 70],
            "loss_weights": {"l2": 10.0, "phase": 0.01, "ipd": 0.1}
        },
        "stage3": {  # Epoch 70-100: 全部 loss + 微調
            "epochs": [70, 100],
            "loss_weights": {"l2": 15.0, "phase": 0.005, "ipd": 0.05}
        }
    },
    
    # Warp loss 權重
    "lambda_warp": 0.01,
    "lambda_smooth": 0.001,
}

os.makedirs(config["artifacts_dir"], exist_ok=True)

# 載入資料集
dataset = BinauralDataset(
    dataset_directory=args.dataset_directory,
    chunk_size_ms=200,
    overlap=0.5
)

# 建立模型 (3 blocks)
net = BinauralNetwork(
    view_dim=7,
    warpnet_layers=4,
    warpnet_channels=64,
    wavenet_blocks=args.blocks,
    layers_per_block=10,
    wavenet_channels=64
)

print(f"{'='*60}")
print(f"模型配置:")
print(f"  - WaveNet blocks: {args.blocks}")
print(f"  - 感受野: {net.receptive_field()} 樣本")
print(f"  - 可訓練參數: {net.num_trainable_parameters():,}")
print(f"  - 訓練片段數: {len(dataset.chunks)}")
print(f"{'='*60}\n")

# 開始訓練
trainer = TrainerV4(config, net, dataset, resume_from=args.resume)
trainer.train()
