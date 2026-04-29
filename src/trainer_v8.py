"""
Trainer v8 — BinauralTFNet 三階段訓練
Stage 1 (0–80):    CommonBranch，L2 + DifferentiableITD
Stage 2 (80–160):  SpecificBranch（凍結 CommonBranch），Phase + IPD
Stage 3 (160–180): 全部，L2×100 + Phase + IPD（fine-tune，監控 Phase 不退步）
"""

import tqdm
import time
import torch as th
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import os
import json
import csv
from datetime import datetime

from src.losses import L2Loss, PhaseLoss, IPDLoss, DifferentiableITDLoss


class TrainerV8:
    def __init__(self, config, net, dataset, resume_from=None):
        self.config = config
        self.net = net
        self.dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )

        gpus = list(range(config["num_gpus"]))
        self.net_parallel = th.nn.DataParallel(net, gpus)

        self.l2_loss   = L2Loss(mask_beginning=config["mask_beginning"])
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.ipd_loss   = IPDLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.itd_loss   = DifferentiableITDLoss(mask_beginning=config["mask_beginning"])

        self.log_dir = os.path.join(config["artifacts_dir"], "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.training_history = []
        self.start_epoch = 0
        self.total_iters = 0

        # 初始化 optimizer（Stage 1 全部參數）
        self._setup_optimizer(lr=config["learning_rate"])
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config["lr_scheduler"]["T_0"],
            eta_min=config["lr_scheduler"]["eta_min"]
        )

        if resume_from:
            self._resume(resume_from)
        else:
            self._save_config()

    # ── Stage 管理 ──────────────────────────────────────────

    def _get_stage(self, epoch):
        s = self.config["training_stages"]
        if epoch < s["stage2"]["epochs"][0]:
            return 1
        elif epoch < s["stage3"]["epochs"][0]:
            return 2
        else:
            return 3

    def _apply_stage(self, stage):
        """設定可訓練參數並重建 optimizer"""
        net = self.net
        if stage == 1:
            for p in net.parameters():
                p.requires_grad = True
            lr = self.config["learning_rate"]
        elif stage == 2:
            # 凍結 CommonBranch，只訓練 SpecificBranch
            for p in net.common.parameters():
                p.requires_grad = False
            for p in net.specific.parameters():
                p.requires_grad = True
            lr = self.config["learning_rate"]
        else:  # stage 3
            for p in net.parameters():
                p.requires_grad = True
            lr = min(self.optimizer.param_groups[0]['lr'], 3e-4)

        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"  [Stage {stage}] 可訓練參數: {trainable:,}")
        self._setup_optimizer(lr=lr)

    def _setup_optimizer(self, lr):
        weights = [p for p in self.net.parameters() if p.requires_grad]
        self.optimizer = Adam(weights, lr=lr)

    # ── Loss 計算 ────────────────────────────────────────────

    def _compute_loss(self, prediction, binaural, stage):
        output      = prediction["output"]    # B×2×T
        y_common    = prediction["y_common"]  # B×1×T
        y_common_gt = binaural.mean(dim=1, keepdim=True)
        warped      = prediction["warped"]    # B×2×T

        # 各 Stage 都記錄 phase/ipd，方便跨 Stage 比較
        with th.no_grad():
            phase_L = self.phase_loss(output[:, 0:1, :], binaural[:, 0:1, :])
            phase_R = self.phase_loss(output[:, 1:2, :], binaural[:, 1:2, :])
            ipd     = self.ipd_loss(output, binaural)

        if stage == 1:
            l2  = self.l2_loss(y_common, y_common_gt)
            itd = self.itd_loss(warped, binaural)
            loss = l2  # Stage 1 純 L2，ITD 只記錄不參與訓練
            return loss, {"loss": loss, "l2": l2, "itd": itd,
                          "phase_L": phase_L, "phase_R": phase_R, "ipd": ipd}

        elif stage == 2:
            phase_L = self.phase_loss(output[:, 0:1, :], binaural[:, 0:1, :])
            phase_R = self.phase_loss(output[:, 1:2, :], binaural[:, 1:2, :])
            ipd     = self.ipd_loss(output, binaural)
            loss = phase_L + phase_R + ipd
            return loss, {"loss": loss, "phase_L": phase_L, "phase_R": phase_R, "ipd": ipd}

        else:  # stage 3
            w    = self.config["loss_weights"]
            l2_L = self.l2_loss(output[:, 0:1, :], binaural[:, 0:1, :])
            l2_R = self.l2_loss(output[:, 1:2, :], binaural[:, 1:2, :])
            phase_L = self.phase_loss(output[:, 0:1, :], binaural[:, 0:1, :])
            phase_R = self.phase_loss(output[:, 1:2, :], binaural[:, 1:2, :])
            ipd     = self.ipd_loss(output, binaural)
            loss = (l2_L + l2_R) * w["l2"] + (phase_L + phase_R) + ipd
            return loss, {"loss": loss, "l2_L": l2_L, "l2_R": l2_R,
                          "phase_L": phase_L, "phase_R": phase_R, "ipd": ipd}

    # ── 訓練主迴圈 ───────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}\nBinauralTFNet v8 訓練開始\n{'='*60}\n")
        prev_stage = None

        for epoch in range(self.start_epoch, self.config["epochs"]):
            stage = self._get_stage(epoch)

            if stage != prev_stage:
                self._apply_stage(stage)
                # 重建 scheduler，每個 Stage 一個 cosine 週期
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer,
                    T_0=self.config["lr_scheduler"]["T_0"],
                    eta_min=self.config["lr_scheduler"]["eta_min"]
                )
                print(f"\n{'='*40}\nStage {stage} 開始 (epoch {epoch+1})\n{'='*40}")
                prev_stage = stage

            t0 = time.time()
            stats = {}
            pbar = tqdm.tqdm(self.dataloader)

            for mono, binaural, view in pbar:
                mono, binaural, view = mono.cuda(), binaural.cuda(), view.cuda()
                self.optimizer.zero_grad()

                pred = self.net_parallel(mono, view)
                loss, batch_stats = self._compute_loss(pred, binaural, stage)
                loss.backward()
                self.optimizer.step()
                self.total_iters += 1

                for k, v in batch_stats.items():
                    stats[k] = stats.get(k, 0) + v.item()
                pbar.set_description(f"[S{stage}] loss:{batch_stats['loss'].item():.4f}")

            for k in stats:
                stats[k] /= len(self.dataloader)

            self.scheduler.step(epoch + 1)
            lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0

            self._log(epoch, stage, stats, elapsed, lr)
            stat_str = "  ".join(f"{k}:{v:.4f}" for k, v in stats.items())
            print(f"epoch {epoch+1:3d} [S{stage}] {stat_str}  lr:{lr:.6f}  ({time.strftime('%H:%M:%S', time.gmtime(elapsed))})")

            if (epoch + 1) % self.config["save_frequency"] == 0:
                self._save(epoch)

        self._save(self.config["epochs"] - 1, suffix="final")
        print("\n訓練完成!")

    # ── 工具方法 ─────────────────────────────────────────────

    def _save(self, epoch, suffix=None):
        s = suffix or f"epoch-{epoch+1}"
        self.net.save(self.config["artifacts_dir"], suffix=s)
        th.save({
            "epoch": epoch + 1,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "total_iters": self.total_iters,
        }, f"{self.config['artifacts_dir']}/checkpoint.{s}.pth")

    def _resume(self, suffix):
        print(f"從 checkpoint 恢復: {suffix}")
        self.net.load(self.config["artifacts_dir"], suffix=suffix)
        ckpt = th.load(f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth")
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"]
        self.training_history = ckpt.get("training_history", [])
        self.total_iters = ckpt.get("total_iters", 0)
        print(f"從 epoch {self.start_epoch + 1} 繼續")

    def _log(self, epoch, stage, stats, elapsed, lr):
        entry = {
            "epoch": epoch + 1,
            "stage": stage,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": lr,
            "epoch_time_seconds": elapsed,
            **{k: float(v) for k, v in stats.items()},
        }
        self.training_history.append(entry)
        with open(os.path.join(self.log_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)

    def _save_config(self):
        with open(os.path.join(self.log_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                **{k: v for k, v in self.config.items() if k != "artifacts_dir"},
            }, f, indent=2, ensure_ascii=False)
