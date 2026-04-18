"""
視覺化訓練記錄
Visualize training logs
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端

def load_training_history(log_dir):
    """載入訓練歷史記錄"""
    json_path = os.path.join(log_dir, "training_history.json")
    if not os.path.exists(json_path):
        print(f"❌ 找不到訓練記錄: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    return history

def plot_losses(history, output_dir):
    """繪製 loss 曲線"""
    epochs = [entry["epoch"] for entry in history]
    
    # 準備所有 loss 資料
    loss_keys = [k for k in history[0].keys() 
                 if k not in ["epoch", "timestamp", "learning_rate", "epoch_time_seconds", "epoch_time_formatted"]]
    
    # 創建子圖
    n_losses = len(loss_keys)
    n_cols = 3
    n_rows = (n_losses + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, loss_key in enumerate(loss_keys):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        loss_values = [entry[loss_key] for entry in history]
        ax.plot(epochs, loss_values, linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{loss_key}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 標註最小值
        min_idx = loss_values.index(min(loss_values))
        ax.plot(epochs[min_idx], loss_values[min_idx], 'r*', markersize=15, 
                label=f'Min: {loss_values[min_idx]:.6f} @ epoch {epochs[min_idx]}')
        ax.legend(fontsize=10)
    
    # 隱藏多餘的子圖
    for idx in range(n_losses, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Loss 曲線已保存到: {output_path}")
    plt.close()

def plot_learning_rate(history, output_dir):
    """繪製學習率變化"""
    epochs = [entry["epoch"] for entry in history]
    learning_rates = [entry["learning_rate"] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, linewidth=2, marker='o', markersize=4, color='green')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用對數刻度
    
    output_path = os.path.join(output_dir, "learning_rate.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 學習率曲線已保存到: {output_path}")
    plt.close()

def plot_epoch_time(history, output_dir):
    """繪製每個 epoch 的訓練時間"""
    epochs = [entry["epoch"] for entry in history]
    epoch_times = [entry["epoch_time_seconds"] for entry in history]
    
    plt.figure(figsize=(10, 6))
    plt.bar(epochs, epoch_times, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 標註平均時間
    avg_time = sum(epoch_times) / len(epoch_times)
    plt.axhline(y=avg_time, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_time:.2f}s')
    plt.legend(fontsize=10)
    
    output_path = os.path.join(output_dir, "epoch_time.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 訓練時間圖已保存到: {output_path}")
    plt.close()

def plot_main_losses_comparison(history, output_dir):
    """繪製主要 loss 的比較圖"""
    epochs = [entry["epoch"] for entry in history]
    
    plt.figure(figsize=(12, 8))
    
    # 繪製主要的 loss
    main_losses = ["l2", "phase", "ipd", "accumulated_loss"]
    colors = ['blue', 'green', 'orange', 'red']
    
    for loss_key, color in zip(main_losses, colors):
        if loss_key in history[0]:
            loss_values = [entry[loss_key] for entry in history]
            plt.plot(epochs, loss_values, linewidth=2, marker='o', markersize=3, 
                    color=color, label=loss_key, alpha=0.8)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Main Losses Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, "main_losses_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 主要 Loss 比較圖已保存到: {output_path}")
    plt.close()

def print_summary(history, log_dir):
    """顯示訓練總結"""
    summary_path = os.path.join(log_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"訓練總結 (Training Summary)")
        print(f"{'='*60}")
        print(f"總 epochs: {summary['total_epochs']}")
        print(f"總訓練時間: {summary['total_time_seconds']:.2f} 秒")
        print(f"開始時間: {summary['start_time']}")
        print(f"結束時間: {summary['end_time']}")
        print(f"\n最佳 epoch: {summary['best_epoch']['epoch']}")
        print(f"最佳 loss: {summary['best_epoch']['accumulated_loss']:.6f}")
        print(f"\nLoss 改善:")
        print(f"  初始: {summary['loss_improvement']['initial_loss']:.6f}")
        print(f"  最終: {summary['loss_improvement']['final_loss']:.6f}")
        print(f"  改善: {summary['loss_improvement']['improvement']:.6f} ({summary['loss_improvement']['improvement_percent']:.2f}%)")
        print(f"\n最終 losses:")
        for k, v in summary['final_losses'].items():
            print(f"  {k}: {v:.6f}")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='視覺化訓練記錄')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='訓練記錄目錄路徑 (e.g., outputs/training_logs)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='圖表輸出目錄 (預設為 log_dir)')
    args = parser.parse_args()
    
    # 設定輸出目錄
    output_dir = args.output_dir if args.output_dir else args.log_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入訓練歷史
    print(f"載入訓練記錄: {args.log_dir}")
    history = load_training_history(args.log_dir)
    
    if history is None:
        return
    
    print(f"找到 {len(history)} 個 epochs 的記錄\n")
    
    # 顯示總結
    print_summary(history, args.log_dir)
    
    # 繪製圖表
    print("生成視覺化圖表...")
    plot_losses(history, output_dir)
    plot_learning_rate(history, output_dir)
    plot_epoch_time(history, output_dir)
    plot_main_losses_comparison(history, output_dir)
    
    print(f"\n✅ 所有圖表已保存到: {output_dir}")

if __name__ == "__main__":
    main()
