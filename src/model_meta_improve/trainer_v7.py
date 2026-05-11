"""
Trainer v7
- CosineAnnealingWarmRestarts 取代 ReduceLROnPlateau
- 每個 Stage 一個 cosine 週期，Stage 切換時自動 warm restart
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

from src.losses import L2Loss, PhaseLoss, IPDLoss


class TrainerV7:
    def __init__(self, config, net, dataset, resume_from=None):
        self.config = config
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=16,
            pin_memory=True
        )

        gpus = [i for i in range(config["num_gpus"])]
        self.net = th.nn.DataParallel(net, gpus)

        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = Adam(weights, lr=config["learning_rate"])

        # CosineAnnealingWarmRestarts
        # T_0 = Stage 長度（epoch 數），每個 Stage 結束時自動 warm restart
        lr_cfg = config["lr_scheduler"]
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=lr_cfg["T_0"],
            T_mult=lr_cfg.get("T_mult", 1),
            eta_min=lr_cfg["eta_min"]
        )

        self.l2_loss = L2Loss(mask_beginning=config["mask_beginning"])
        self.phase_loss = PhaseLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])
        self.ipd_loss = IPDLoss(sample_rate=48000, mask_beginning=config["mask_beginning"])

        self.total_iters = 0
        self.training_history = []
        self.log_dir = os.path.join(config["artifacts_dir"], "training_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.start_epoch = 0

        if resume_from:
            self._resume_from_checkpoint(resume_from)
            # 從 checkpoint resume 時，強制把 LR 設為安全低值，避免 warm restart 衝擊
            resume_lr = config.get("resume_lr_override", None)
            if resume_lr is not None:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = resume_lr
                print(f"  [Resume] LR 強制設為 {resume_lr}")
        else:
            self._save_config()

        self.net.train()

    def _set_trainable_params(self, epoch):
        """根據訓練階段設定可訓練的參數"""
        stages = self.config["training_stages"]
        stage2_start = stages["stage2"]["epochs"][0]
        stage3_start = stages["stage3"]["epochs"][0]

        if stage2_start <= epoch < stage3_start:
            # Stage 2：只訓練 Warpnet，凍結 WaveNet
            for name, param in self.net.module.named_parameters():
                param.requires_grad = 'warper' in name
            trainable = sum(p.numel() for p in self.net.module.parameters() if p.requires_grad)
            print(f"  [Stage 2] 凍結 WaveNet，只訓練 Warpnet ({trainable:,} 參數)")
        else:
            # Stage 1 / Stage 3：全部參數可訓練
            for param in self.net.module.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.net.module.parameters() if p.requires_grad)
            print(f"  [Stage 1/3] 全部參數可訓練 ({trainable:,} 參數)")

        # 重建 optimizer 以反映新的可訓練參數
        weights = filter(lambda x: x.requires_grad, self.net.module.parameters())
        self.optimizer = Adam(weights, lr=self.optimizer.param_groups[0]['lr'])

    def _get_loss_weights(self, epoch):
        for stage_name, stage_config in self.config["training_stages"].items():
            start, end = stage_config["epochs"]
            if start <= epoch < end:
                return stage_config["loss_weights"]
        return list(self.config["training_stages"].values())[-1]["loss_weights"]

    def _save_config(self):
        config_path = os.path.join(self.log_dir, "training_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "batch_size": self.config["batch_size"],
                "learning_rate": self.config["learning_rate"],
                "epochs": self.config["epochs"],
                "lr_scheduler": self.config["lr_scheduler"],
                "training_stages": self.config["training_stages"],
            }, f, indent=2, ensure_ascii=False)

    def _resume_from_checkpoint(self, suffix):
        print(f"\n{'='*60}")
        print(f"從 checkpoint 恢復: {suffix}")
        self.net.module.load(self.config["artifacts_dir"], suffix=suffix)

        checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        if os.path.exists(checkpoint_file):
            checkpoint_data = th.load(checkpoint_file)
            if 'optimizer_state_dict' in checkpoint_data:
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint_data:
                self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            if 'epoch' in checkpoint_data:
                self.start_epoch = checkpoint_data['epoch']
            if 'training_history' in checkpoint_data:
                self.training_history = checkpoint_data['training_history']
            if 'total_iters' in checkpoint_data:
                self.total_iters = checkpoint_data['total_iters']
        print(f"從 epoch {self.start_epoch + 1} 繼續")
        print(f"{'='*60}\n")

    def save_checkpoint(self, epoch, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)
        checkpoint_file = f"{self.config['artifacts_dir']}/checkpoint.{suffix}.pth"
        th.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'total_iters': self.total_iters,
        }, checkpoint_file)

    def _log_epoch(self, epoch, loss_stats, epoch_time, current_lr, loss_weights):
        entry = {
            "epoch": epoch + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "learning_rate": current_lr,
            "epoch_time_seconds": epoch_time,
            "epoch_time_formatted": time.strftime('%H:%M:%S', time.gmtime(epoch_time)),
            "loss_weights": loss_weights,
        }
        for k, v in loss_stats.items():
            entry[k] = float(v)
        self.training_history.append(entry)

        json_path = os.path.join(self.log_dir, "training_history.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)

        csv_path = os.path.join(self.log_dir, "training_history.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [k for k in self.training_history[0].keys() if k != 'loss_weights']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.training_history:
                writer.writerow({k: v for k, v in e.items() if k != 'loss_weights'})

    def train(self):
        print(f"\n{'='*60}")
        print(f"開始訓練 v7")
        print(f"訓練記錄: {self.log_dir}")
        if self.start_epoch > 0:
            print(f"從 epoch {self.start_epoch + 1} 繼續")
        print(f"{'='*60}\n")

        prev_weights = None

        for epoch in range(self.start_epoch, self.config["epochs"]):
            loss_weights = self._get_loss_weights(epoch)

            if prev_weights is not None and prev_weights != loss_weights:
                # Stage 切換：限制 LR 不超過 0.0003
                for param_group in self.optimizer.param_groups:
                    if param_group['lr'] > 0.0003:
                        param_group['lr'] = 0.0003
                current_lr_at_switch = self.optimizer.param_groups[0]['lr']
                print(f"\n{'='*60}")
                print(f"Stage 切換 (Epoch {epoch+1}) -> {loss_weights}")
                print(f"LR 限制為: {current_lr_at_switch:.6f}")
                print(f"{'='*60}\n")
            elif epoch == 0:
                print(f"Stage 1 開始: {loss_weights}\n")
                print(f"Stage 1 開始: {loss_weights}\n")

            prev_weights = loss_weights

            t_start = time.time()
            loss_stats = {}
            data_pbar = tqdm.tqdm(self.dataloader)

            for data in data_pbar:
                loss_new = self.train_iteration(data, loss_weights)
                for k, v in loss_new.items():
                    loss_stats[k] = loss_stats[k] + v if k in loss_stats else v
                data_pbar.set_description(f"loss: {loss_new['accumulated_loss'].item():.6f}")

            for k in loss_stats:
                loss_stats[k] /= len(self.dataloader)

            # CosineAnnealingWarmRestarts 每個 step 更新一次
            self.scheduler.step(epoch + 1)
            current_lr = self.optimizer.param_groups[0]['lr']

            t_end = time.time()
            epoch_time = t_end - t_start

            self._log_epoch(epoch, loss_stats, epoch_time, current_lr, loss_weights)

            loss_str = "  ".join([f"{k}:{v:.4f}" for k, v in loss_stats.items()])
            print(f"epoch {epoch+1:3d} {loss_str}  lr:{current_lr:.6f}  ({time.strftime('%H:%M:%S', time.gmtime(epoch_time))})")

            if self.config["save_frequency"] > 0 and (epoch + 1) % self.config["save_frequency"] == 0:
                self.save_checkpoint(epoch, suffix=f'epoch-{epoch+1}')

        self.save_checkpoint(self.config["epochs"] - 1, suffix='final')
        print("\n訓練完成!")

    def train_iteration(self, data, loss_weights):
        self.optimizer.zero_grad()

        mono, binaural, quats = data
        mono, binaural, quats = mono.cuda(), binaural.cuda(), quats.cuda()

        prediction = self.net.forward(mono, quats)

        l2 = self.l2_loss(prediction["output"], binaural)
        phase = self.phase_loss(prediction["output"], binaural)
        ipd = self.ipd_loss(prediction["output"], binaural)

        intermediate_binaural = th.cat([binaural] * len(prediction["intermediate"]), dim=1)
        intermediate_prediction = th.cat(prediction["intermediate"], dim=1)
        intermediate_l2 = self.l2_loss(intermediate_prediction, intermediate_binaural)
        intermediate_phase = self.phase_loss(intermediate_prediction, intermediate_binaural)

        intermediate_ipd = sum(
            self.ipd_loss(o, binaural) for o in prediction["intermediate"]
        ) / len(prediction["intermediate"])

        loss = (l2 + intermediate_l2) * loss_weights["l2"] + \
               (phase + intermediate_phase) * loss_weights["phase"] + \
               (ipd + intermediate_ipd) * loss_weights["ipd"]

        loss.backward()
        self.optimizer.step()
        self.total_iters += 1

        return {
            "l2": l2, "phase": phase, "ipd": ipd,
            "intermediate_l2": intermediate_l2,
            "intermediate_phase": intermediate_phase,
            "intermediate_ipd": intermediate_ipd,
            "accumulated_loss": loss,
        }
