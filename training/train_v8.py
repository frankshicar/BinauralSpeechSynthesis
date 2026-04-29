"""
BinauralTFNet v8.3 訓練腳本
修正 v8/v8.2 的根本問題：從 y_common 出發，而不是從 warped 出發

v8.3 修改內容：
1. SpecificBranch 從 y_common 出發（不是 warped）
2. SimpleDPAB 替代 DPAB（移除 Cross-attention）
3. Stage 1: 0-60（減少 20 epochs）
4. Stage 2: 60-160（增加 20 epochs）

三階段訓練：
  Stage 1 (0–60):    CommonBranch，L2（純 L2，ITD 只記錄）
  Stage 2 (60–160):  SpecificBranch（凍結 CommonBranch），Phase + IPD
  Stage 3 (160–180): 全部，L2×100 + Phase + IPD
"""

import os
import argparse

from src.dataset import BinauralDataset
from src.models_v2 import BinauralTFNet
from src.trainer_v8 import TrainerV8

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_directory", type=str, default="./dataset/train")
parser.add_argument("--artifacts_directory", type=str, default="./outputs_v8")
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--resume", type=str, default=None,
                    help="從 checkpoint 繼續，例如 'epoch-80'")
args = parser.parse_args()

config = {
    "artifacts_dir": args.artifacts_directory,
    "learning_rate": 3e-4,
    "batch_size": 16,  # v8.2: 32 → 16（因為模型變大）
    "mask_beginning": 1024,
    "epochs": 180,
    "num_gpus": args.num_gpus,
    "save_frequency": 10,
    "lr_scheduler": {
        "T_0": 80,       # 每個 Stage 一個 cosine 週期
        "eta_min": 5e-6,
    },
    "training_stages": {
        "stage1": {"epochs": [0,   60]},   # v8.3: 80 → 60
        "stage2": {"epochs": [60,  160]},  # v8.3: 增加 20 epochs
        "stage3": {"epochs": [160, 180]},
    },
    # loss weights
    "loss_weights": {
        "itd": 0.5,   # Stage 1: L2 + ITD×0.5
        "l2":  100.0, # Stage 3: L2×100 + Phase + IPD
    },
}

os.makedirs(config["artifacts_dir"], exist_ok=True)

dataset = BinauralDataset(
    dataset_directory=args.dataset_directory,
    chunk_size_ms=200,
    overlap=0.5,
)

net = BinauralTFNet(
    # 使用 models_v2.py 的默認參數（v8.2）
    # fft_size=1024, hop_size=256, tf_channels=256, tf_blocks=8
)

print(f"可訓練參數: {net.num_trainable_parameters():,}")
print(f"訓練片段數: {len(dataset.chunks)}")

trainer = TrainerV8(config, net, dataset, resume_from=args.resume)
trainer.train()
