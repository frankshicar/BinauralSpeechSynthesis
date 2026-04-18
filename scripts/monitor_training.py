"""
即時監控訓練進度
Real-time training monitor
"""

import json
import os
import time
import argparse
from datetime import datetime

def load_latest_history(log_dir):
    """載入最新的訓練歷史"""
    json_path = os.path.join(log_dir, "training_history.json")
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
        return history
    except:
        return None

def print_progress(history, last_epoch):
    """顯示訓練進度"""
    if not history:
        return 0
    
    current_epoch = len(history)
    
    # 只顯示新的 epoch
    if current_epoch > last_epoch:
        for entry in history[last_epoch:]:
            epoch = entry["epoch"]
            timestamp = entry["timestamp"]
            lr = entry["learning_rate"]
            
            # 顯示主要 loss
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} | {timestamp} | LR: {lr:.6f}")
            print(f"{'='*70}")
            
            # 顯示所有 loss
            loss_items = [(k, v) for k, v in entry.items() 
                         if k not in ["epoch", "timestamp", "learning_rate", 
                                     "epoch_time_seconds", "epoch_time_formatted"]]
            
            for k, v in loss_items:
                print(f"  {k:20s}: {v:.6f}")
            
            print(f"  {'epoch_time':20s}: {entry['epoch_time_formatted']}")
            
            # 如果有多個 epoch，顯示趨勢
            if len(history) > 1:
                prev_loss = history[-2]["accumulated_loss"]
                curr_loss = entry["accumulated_loss"]
                change = curr_loss - prev_loss
                change_pct = (change / prev_loss) * 100
                
                trend = "📉" if change < 0 else "📈"
                print(f"\n  Loss 變化: {change:+.6f} ({change_pct:+.2f}%) {trend}")
    
    return current_epoch

def monitor(log_dir, interval=10):
    """持續監控訓練進度"""
    print(f"{'='*70}")
    print(f"開始監控訓練進度")
    print(f"記錄目錄: {log_dir}")
    print(f"更新間隔: {interval} 秒")
    print(f"{'='*70}")
    print(f"按 Ctrl+C 停止監控\n")
    
    last_epoch = 0
    
    try:
        while True:
            history = load_latest_history(log_dir)
            
            if history:
                last_epoch = print_progress(history, last_epoch)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待訓練開始...", end='\r')
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print(f"監控已停止")
        print(f"{'='*70}\n")

def show_latest(log_dir):
    """顯示最新的訓練狀態"""
    history = load_latest_history(log_dir)
    
    if not history:
        print(f"❌ 找不到訓練記錄: {log_dir}")
        return
    
    latest = history[-1]
    
    print(f"\n{'='*70}")
    print(f"最新訓練狀態 (Latest Training Status)")
    print(f"{'='*70}")
    print(f"Epoch: {latest['epoch']}")
    print(f"時間: {latest['timestamp']}")
    print(f"學習率: {latest['learning_rate']:.6f}")
    print(f"\nLosses:")
    
    loss_items = [(k, v) for k, v in latest.items() 
                 if k not in ["epoch", "timestamp", "learning_rate", 
                             "epoch_time_seconds", "epoch_time_formatted"]]
    
    for k, v in loss_items:
        print(f"  {k:20s}: {v:.6f}")
    
    print(f"\n訓練時間: {latest['epoch_time_formatted']}")
    
    # 顯示進度
    if len(history) > 1:
        first_loss = history[0]["accumulated_loss"]
        current_loss = latest["accumulated_loss"]
        improvement = first_loss - current_loss
        improvement_pct = (improvement / first_loss) * 100
        
        print(f"\n總體改善:")
        print(f"  初始 loss: {first_loss:.6f}")
        print(f"  當前 loss: {current_loss:.6f}")
        print(f"  改善: {improvement:.6f} ({improvement_pct:.2f}%)")
    
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description='監控訓練進度')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='訓練記錄目錄路徑 (e.g., outputs/training_logs)')
    parser.add_argument('--mode', type=str, default='monitor', choices=['monitor', 'latest'],
                       help='模式: monitor (持續監控) 或 latest (顯示最新狀態)')
    parser.add_argument('--interval', type=int, default=10,
                       help='監控更新間隔（秒），預設 10 秒')
    args = parser.parse_args()
    
    if args.mode == 'monitor':
        monitor(args.log_dir, args.interval)
    else:
        show_latest(args.log_dir)

if __name__ == "__main__":
    main()
