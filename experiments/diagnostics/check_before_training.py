"""
訓練前檢查腳本 - 確保一切就緒
"""
import os
import sys
import torch as th

def check_dataset(dataset_dir):
    """檢查數據集"""
    print(f"\n{'='*60}")
    print("檢查數據集")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_dir):
        print(f"❌ 數據集目錄不存在: {dataset_dir}")
        return False
    
    print(f"✅ 數據集目錄存在: {dataset_dir}")
    
    # 檢查子目錄
    subjects = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not subjects:
        print(f"❌ 數據集目錄為空")
        return False
    
    print(f"✅ 找到 {len(subjects)} 個 subject 目錄")
    
    # 檢查第一個 subject 的文件
    first_subject = os.path.join(dataset_dir, subjects[0])
    required_files = ['binaural.wav', 'mono.wav', 'tx_positions.txt']
    
    for file in required_files:
        file_path = os.path.join(first_subject, file)
        if os.path.exists(file_path):
            print(f"  ✅ {file}")
        else:
            print(f"  ⚠️  {file} (可能不存在)")
    
    return True

def check_cuda():
    """檢查 CUDA"""
    print(f"\n{'='*60}")
    print("檢查 CUDA")
    print(f"{'='*60}")
    
    if th.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"  - CUDA 版本: {th.version.cuda}")
        print(f"  - GPU 數量: {th.cuda.device_count()}")
        for i in range(th.cuda.device_count()):
            print(f"  - GPU {i}: {th.cuda.get_device_name(i)}")
        return True
    else:
        print(f"⚠️  CUDA 不可用，將使用 CPU 訓練（會很慢）")
        return False

def check_imports():
    """檢查必要的模組"""
    print(f"\n{'='*60}")
    print("檢查 Python 模組")
    print(f"{'='*60}")
    
    modules = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} 未安裝")
            all_ok = False
    
    return all_ok

def check_code_modifications():
    """檢查代碼修改"""
    print(f"\n{'='*60}")
    print("檢查代碼修改")
    print(f"{'='*60}")
    
    # 檢查 src/models.py 是否有新的方法
    try:
        from src.models import GeometricWarper
        warper = GeometricWarper()
        
        # 檢查是否有 ear_offset 屬性
        if hasattr(warper, 'ear_offset'):
            print(f"✅ Geometric Warp ITD 修改已應用")
            print(f"  - ear_offset: {warper.ear_offset} m")
        else:
            print(f"⚠️  Geometric Warp ITD 修改可能未應用")
        
        # 檢查是否有新方法
        if hasattr(warper, '_listener_ear_positions'):
            print(f"✅ _listener_ear_positions 方法存在")
        else:
            print(f"⚠️  _listener_ear_positions 方法不存在")
            
    except Exception as e:
        print(f"❌ 無法檢查 GeometricWarper: {e}")
        return False
    
    # 檢查 src/utils.py 是否有新的方法
    try:
        from src.utils import Net
        net = Net()
        
        if hasattr(net, 'save_checkpoint'):
            print(f"✅ Checkpoint 系統已應用")
        else:
            print(f"⚠️  Checkpoint 系統可能未應用")
            
    except Exception as e:
        print(f"❌ 無法檢查 Net: {e}")
        return False
    
    return True

def check_output_dir(output_dir):
    """檢查輸出目錄"""
    print(f"\n{'='*60}")
    print("檢查輸出目錄")
    print(f"{'='*60}")
    
    if os.path.exists(output_dir):
        print(f"⚠️  輸出目錄已存在: {output_dir}")
        
        # 檢查是否有舊的訓練
        checkpoints = [f for f in os.listdir(output_dir) if f.endswith('.pth') or f.endswith('.net')]
        if checkpoints:
            print(f"  - 找到 {len(checkpoints)} 個 checkpoint 文件")
            print(f"  - 如果要繼續訓練，請使用 --resume 參數")
            print(f"  - 如果要重新開始，請使用新的輸出目錄")
    else:
        print(f"✅ 輸出目錄不存在，將自動創建: {output_dir}")
    
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='訓練前檢查')
    parser.add_argument('--dataset_directory', type=str, default='./dataset/trainset',
                        help='數據集目錄')
    parser.add_argument('--artifacts_directory', type=str, default='./outputs_with_new_geom_warp',
                        help='輸出目錄')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("訓練前檢查")
    print("="*60)
    
    checks = [
        ("數據集", lambda: check_dataset(args.dataset_directory)),
        ("CUDA", check_cuda),
        ("Python 模組", check_imports),
        ("代碼修改", check_code_modifications),
        ("輸出目錄", lambda: check_output_dir(args.artifacts_directory)),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 檢查 {name} 時發生錯誤: {e}")
            results.append((name, False))
    
    # 總結
    print(f"\n{'='*60}")
    print("檢查總結")
    print(f"{'='*60}")
    
    all_passed = True
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ 所有檢查通過，可以開始訓練！")
        print(f"\n建議的訓練命令:")
        print(f"python train.py \\")
        print(f"    --dataset_directory {args.dataset_directory} \\")
        print(f"    --artifacts_directory {args.artifacts_directory} \\")
        print(f"    --num_gpus 1 \\")
        print(f"    --blocks 3")
    else:
        print("⚠️  部分檢查未通過，請先解決問題")
        sys.exit(1)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
