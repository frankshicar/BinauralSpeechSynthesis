"""
迭代優化角度補償表
自動執行多輪優化，直到所有角度誤差都接近 0
"""
import subprocess
import re
import sys

MAX_ITERATIONS = 10
TARGET_ERROR = 0.5  # 目標誤差（度）


def run_evaluate():
    """執行 evaluate.py 並解析結果"""
    print("執行 evaluate.py...")
    result = subprocess.run(
        [
            "python", "evaluate.py",
            "--dataset_directory", "./dataset/testset",
            "--model_file", "outputs/binaural_network.newbob.net",
            "--artifacts_directory", "results_audio",
            "--blocks", "3"
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("錯誤：evaluate.py 執行失敗")
        print(result.stderr)
        return None
    
    # 解析輸出，提取每個 subject 的角度數據
    angle_data = {}
    lines = result.stdout.split('\n')
    
    for line in lines:
        # 匹配格式：Angle Error: X.X° (Pred: Y.Y°, GT: Z.Z°)
        match = re.search(r'Angle Error: ([\d.]+)° \(Pred: ([+-]?[\d.]+)°, GT: ([+-]?[\d.]+)°\)', line)
        if match:
            error = float(match.group(1))
            pred = float(match.group(2))
            gt = float(match.group(3))
            angle_data[gt] = {
                'error': error,
                'pred': pred,
                'gt': gt
            }
    
    # 提取平均誤差
    avg_error = None
    for line in lines:
        match = re.search(r'角度誤差 \(Angle Error\):\s+([\d.]+)°', line)
        if match:
            avg_error = float(match.group(1))
            break
    
    return angle_data, avg_error


def update_compensation_table(angle_data):
    """根據誤差更新補償表"""
    # 讀取當前的 synthesis_utils.py
    with open('src/synthesis_utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到 _ANGLE_COMPENSATION 字典的位置
    start_marker = '_ANGLE_COMPENSATION = {'
    end_marker = '}'
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("錯誤：找不到 _ANGLE_COMPENSATION")
        return False
    
    # 找到字典結束位置
    end_idx = content.find(end_marker, start_idx)
    if end_idx == -1:
        print("錯誤：找不到 _ANGLE_COMPENSATION 結束位置")
        return False
    
    # 解析當前的補償表
    current_comp = {}
    comp_section = content[start_idx:end_idx]
    for line in comp_section.split('\n'):
        match = re.search(r'([+-]?[\d.]+):\s*([+-]?[\d.]+),', line)
        if match:
            angle = float(match.group(1))
            comp = float(match.group(2))
            current_comp[angle] = comp
    
    # 計算新的補償量
    new_comp = {}
    for gt, data in sorted(angle_data.items()):
        old_comp = current_comp.get(gt, 0.0)
        error = data['error']
        pred = data['pred']
        
        # 新補償 = 舊補償 + (目標 - 預測)
        adjustment = gt - pred
        new_comp[gt] = old_comp + adjustment
        
        print(f"  {gt:+6.1f}°: 舊補償 {old_comp:+6.1f}° + 調整 {adjustment:+6.1f}° = 新補償 {new_comp[gt]:+6.1f}° (誤差 {error:.1f}°)")
    
    # 生成新的字典內容
    new_dict_lines = ['_ANGLE_COMPENSATION = {']
    for angle in sorted(new_comp.keys()):
        comp = new_comp[angle]
        pred = angle_data[angle]['pred']
        error = angle_data[angle]['error']
        new_dict_lines.append(f"    {angle:+6.1f}: {comp:+6.1f},   # 預測 {pred:+6.1f}°, 誤差 {error:5.1f}°")
    new_dict_lines.append('}')
    
    new_dict_str = '\n'.join(new_dict_lines)
    
    # 替換內容
    new_content = content[:start_idx] + new_dict_str + content[end_idx + 1:]
    
    # 寫回檔案
    with open('src/synthesis_utils.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True


def regenerate_tx_positions():
    """重新生成 tx_positions.txt"""
    print("重新生成 tx_positions.txt...")
    result = subprocess.run(
        ["python", "regenerate_tx_with_compensation.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("錯誤：regenerate_tx_with_compensation.py 執行失敗")
        print(result.stderr)
        return False
    
    return True


def main():
    print("=" * 60)
    print("迭代優化角度補償表")
    print("=" * 60)
    print()
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'=' * 60}")
        print(f"第 {iteration} 輪優化")
        print(f"{'=' * 60}\n")
        
        # 1. 重新生成 tx_positions.txt（使用當前的補償表）
        if not regenerate_tx_positions():
            print("重新生成失敗，中止")
            return
        
        print()
        
        # 2. 執行 evaluate.py
        result = run_evaluate()
        if result is None:
            print("評估失敗，中止")
            return
        
        angle_data, avg_error = result
        
        print()
        print(f"平均角度誤差: {avg_error:.2f}°")
        print()
        
        # 3. 檢查是否達到目標
        if avg_error <= TARGET_ERROR:
            print(f"✓ 已達到目標誤差 ({TARGET_ERROR}°)，優化完成！")
            break
        
        # 4. 更新補償表
        print("更新補償表：")
        if not update_compensation_table(angle_data):
            print("更新失敗，中止")
            return
        
        print()
    
    print()
    print("=" * 60)
    print("優化完成！")
    print("=" * 60)
    print()
    print("最終結果已儲存到 src/synthesis_utils.py")
    print("請執行以下指令驗證：")
    print("python evaluate.py --dataset_directory ./dataset/testset \\")
    print("  --model_file outputs/binaural_network.newbob.net \\")
    print("  --artifacts_directory results_audio --blocks 3")


if __name__ == "__main__":
    main()
