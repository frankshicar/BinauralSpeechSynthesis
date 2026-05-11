"""
檢查 checkpoint 狀態的工具腳本
"""
import os
import json
import argparse
import torch as th
from datetime import datetime

def check_checkpoint(checkpoint_path):
    """
    檢查 checkpoint 文件的內容
    """
    print(f"\n{'='*60}")
    print(f"檢查 Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return
    
    # 載入 checkpoint
    checkpoint = th.load(checkpoint_path, map_location='cpu')
    
    # 基本資訊
    print("📦 Checkpoint 內容:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Epoch 資訊
    if 'epoch' in checkpoint:
        print(f"\n📅 Epoch: {checkpoint['epoch'] + 1}")
    
    # Optimizer 狀態
    if 'optimizer_state_dict' in checkpoint:
        print(f"\n⚙️  Optimizer 狀態:")
        opt_state = checkpoint['optimizer_state_dict']
        if 'param_groups' in opt_state:
            for i, pg in enumerate(opt_state['param_groups']):
                print(f"  - Param Group {i}:")
                print(f"    - Learning Rate: {pg['lr']}")
                if 'betas' in pg:
                    print(f"    - Betas: {pg['betas']}")
                if 'eps' in pg:
                    print(f"    - Eps: {pg['eps']}")
    
    # NewbobAdam 狀態
    if 'newbob_state' in checkpoint:
        print(f"\n🔄 NewbobAdam 狀態:")
        newbob = checkpoint['newbob_state']
        print(f"  - Last Epoch Loss: {newbob['last_epoch_loss']:.6f}")
        print(f"  - Total Decay: {newbob['total_decay']:.6f}")
        print(f"  - Effective LR: {newbob['total_decay'] * 0.001:.6f} (假設初始 LR=0.001)")
    
    # 訓練狀態
    if 'training_state' in checkpoint:
        print(f"\n📊 訓練狀態:")
        train_state = checkpoint['training_state']
        if 'total_iters' in train_state:
            print(f"  - Total Iterations: {train_state['total_iters']}")
        if 'training_history' in train_state:
            history = train_state['training_history']
            print(f"  - Training History Length: {len(history)} epochs")
            if history:
                last = history[-1]
                print(f"  - Last Recorded Epoch: {last['epoch']}")
                print(f"  - Last Loss: {last['accumulated_loss']:.6f}")
                print(f"  - Last LR: {last['learning_rate']:.6f}")
    
    # 文件大小
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"\n💾 文件大小: {file_size:.2f} MB")
    
    # 修改時間
    mod_time = os.path.getmtime(checkpoint_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
    print(f"📅 修改時間: {mod_time_str}")
    
    print(f"\n{'='*60}\n")

def list_checkpoints(artifacts_dir):
    """
    列出所有可用的 checkpoint
    """
    print(f"\n{'='*60}")
    print(f"可用的 Checkpoints: {artifacts_dir}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(artifacts_dir):
        print(f"❌ 目錄不存在: {artifacts_dir}")
        return
    
    # 查找所有 checkpoint 文件
    checkpoints = []
    for file in os.listdir(artifacts_dir):
        if file.endswith('.pth') and 'checkpoint' in file:
            checkpoints.append(file)
    
    if not checkpoints:
        print("❌ 沒有找到 checkpoint 文件")
        return
    
    # 排序
    checkpoints.sort()
    
    print(f"找到 {len(checkpoints)} 個 checkpoint:\n")
    
    for ckpt in checkpoints:
        ckpt_path = os.path.join(artifacts_dir, ckpt)
        file_size = os.path.getsize(ckpt_path) / (1024 * 1024)
        mod_time = os.path.getmtime(ckpt_path)
        mod_time_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        
        # 嘗試讀取 epoch 資訊
        try:
            checkpoint = th.load(ckpt_path, map_location='cpu')
            epoch = checkpoint.get('epoch', '?')
            lr = '?'
            if 'optimizer_state_dict' in checkpoint:
                opt_state = checkpoint['optimizer_state_dict']
                if 'param_groups' in opt_state and len(opt_state['param_groups']) > 0:
                    lr = f"{opt_state['param_groups'][0]['lr']:.6f}"
            
            print(f"  📦 {ckpt}")
            print(f"     - Epoch: {epoch + 1 if isinstance(epoch, int) else epoch}")
            print(f"     - LR: {lr}")
            print(f"     - Size: {file_size:.2f} MB")
            print(f"     - Modified: {mod_time_str}")
            print()
        except Exception as e:
            print(f"  ❌ {ckpt} (無法讀取: {e})")
            print()
    
    print(f"{'='*60}\n")

def check_training_history(artifacts_dir):
    """
    檢查訓練歷史
    """
    history_file = os.path.join(artifacts_dir, "training_logs", "training_history.json")
    
    print(f"\n{'='*60}")
    print(f"訓練歷史: {history_file}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(history_file):
        print(f"❌ 文件不存在: {history_file}")
        return
    
    with open(history_file, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    if not history:
        print("❌ 訓練歷史為空")
        return
    
    print(f"📊 總共 {len(history)} 個 epoch\n")
    
    # 顯示最近 5 個 epoch
    print("最近 5 個 epoch:")
    for entry in history[-5:]:
        print(f"  Epoch {entry['epoch']:3d} | "
              f"Loss: {entry['accumulated_loss']:.6f} | "
              f"LR: {entry['learning_rate']:.6f} | "
              f"Time: {entry['epoch_time_formatted']}")
    
    # 統計資訊
    print(f"\n📈 統計資訊:")
    losses = [e['accumulated_loss'] for e in history]
    lrs = [e['learning_rate'] for e in history]
    
    print(f"  - 最低 Loss: {min(losses):.6f} (Epoch {losses.index(min(losses)) + 1})")
    print(f"  - 最高 Loss: {max(losses):.6f} (Epoch {losses.index(max(losses)) + 1})")
    print(f"  - 當前 Loss: {losses[-1]:.6f}")
    print(f"  - 當前 LR: {lrs[-1]:.6f}")
    print(f"  - 初始 LR: {lrs[0]:.6f}")
    print(f"  - LR 衰減次數: {len(set(lrs)) - 1}")
    
    print(f"\n{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='檢查訓練 checkpoint 狀態')
    parser.add_argument('--artifacts_dir', type=str, default='./outputs',
                        help='模型輸出目錄')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='檢查特定的 checkpoint 文件（例如：epoch-40）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的 checkpoint')
    parser.add_argument('--history', action='store_true',
                        help='顯示訓練歷史')
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.artifacts_dir)
    
    if args.history:
        check_training_history(args.artifacts_dir)
    
    if args.checkpoint:
        # 構建完整路徑
        if args.checkpoint.endswith('.pth'):
            checkpoint_path = os.path.join(args.artifacts_dir, args.checkpoint)
        else:
            checkpoint_path = os.path.join(args.artifacts_dir, 
                                          f'binaural_network_checkpoint.{args.checkpoint}.pth')
        check_checkpoint(checkpoint_path)
    
    # 如果沒有指定任何選項，顯示幫助
    if not (args.list or args.history or args.checkpoint):
        print("\n使用範例:")
        print("  python check_checkpoint.py --list")
        print("  python check_checkpoint.py --history")
        print("  python check_checkpoint.py --checkpoint epoch-40")
        print("  python check_checkpoint.py --artifacts_dir ./outputs_with_ipd --list")
        print()

if __name__ == "__main__":
    main()
