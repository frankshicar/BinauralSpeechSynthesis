# Geometric Warp ITD 升級完成

## 修改摘要
成功修改 Geometric Warp 以分別計算左右耳的距離，使 geometric warpfield 本身就包含理論 ITD。

## 修改的文件

### 1. `src/models.py` - GeometricWarper 類別
- ✓ 新增 `ear_offset = 0.0875` 參數（8.75cm）
- ✓ 新增 `_quaternion_to_right_vector()` 方法
- ✓ 新增 `_listener_ear_positions()` 方法
- ✓ 修改 `_3d_displacements()` 方法，返回 B x 2 x 3 x K 張量

### 2. `src/warping.py` - GeometricTimeWarper 類別
- ✓ 修改 `displacements2warpfield()` 方法，處理分離的左右耳位移

## 測試結果

### 測試 1: ITD 準確性測試 (`test_geometric_warp_itd.py`)
```
場景 1 - 聲源在右側 (0, 1, 0):
  左耳 warpfield: -155.25 samples
  右耳 warpfield: -131.33 samples
  ITD: 23.92 samples = 0.498 ms
  理論 ITD: 23.92 samples
  誤差: 0.00 samples ✓

場景 2 - 聲源在左側 (0, -1, 0):
  ITD: -23.92 samples = -0.498 ms
  符號正確 ✓
```

### 測試 2: 模型整合測試 (`test_model_integration.py`)
```
✓ 模型創建成功
✓ Forward pass 成功
✓ 輸出形狀正確 (B x 2 x T)
✓ 輸出無 NaN
✓ Warper 單獨運行成功
✓ Warpfield 計算成功
✓ 左右耳 warpfield 有差異（包含 ITD）
```

## 技術細節

### 原始實現的問題
```python
# 原始代碼：左右耳使用固定偏移
left_ear_offset = [0, -0.08, -0.22]
right_ear_offset = [0, 0.08, -0.22]
displacement = transmitter_pos + transmitter_mouth - ear_offset
```
- 問題：沒有考慮 listener 的頭部方向
- 結果：ITD 完全依賴 neural warp 學習

### 新實現
```python
# 新代碼：根據 listener 方向計算左右耳實際位置
right_vector = quaternion_to_right_vector(listener_orientation)
left_ear_pos = -ear_offset * right_vector
right_ear_pos = +ear_offset * right_vector

# 分別計算到左右耳的距離
displacement_left = transmitter_mouth_abs - left_ear_pos
displacement_right = transmitter_mouth_abs - right_ear_pos
```
- 優點：考慮頭部方向，物理上更準確
- 結果：geometric warp 提供準確的 ITD 基礎

## 預期訓練改善

### 1. L_warp Loss 應該降低
- Geometric warp 已經提供準確的 ITD
- Neural warp 只需學習細微修正
- 預期 L_warp 收斂更快、最終值更低

### 2. ITD 預測更準確
- 不再完全依賴 neural network 學習 ITD
- 物理模型提供強先驗知識

### 3. 訓練更穩定
- 減少需要學習的參數空間
- 更容易收斂到好的解

## 使用方法

### 重新訓練模型
```bash
# 使用新的 geometric warp 訓練
python train.py --config your_config.json
```

### 監控訓練
```bash
# 特別關注 L_warp 的變化
python monitor_training.py
python visualize_training.py
```

### 比較舊模型
建議保留舊模型的 checkpoint，以便比較：
- ITD 預測準確度
- L_warp loss 值
- 整體音質

## 向後兼容性

⚠️ **注意**：此修改改變了 geometric warp 的計算方式，因此：
- 舊的 checkpoint 無法直接使用新代碼
- 需要重新訓練模型
- 建議備份舊模型以便比較

## 下一步建議

1. **重新訓練模型**
   - 使用相同的超參數
   - 監控 L_warp loss 的變化

2. **評估改善**
   - 比較 ITD 預測準確度
   - 測量音質改善（如果有主觀測試）
   - 檢查 L_warp loss 是否降低

3. **可能的進一步優化**
   - 調整 `ear_offset` 參數（目前是 8.75cm）
   - 考慮頭部大小的個體差異
   - 添加更多的物理約束

## 相關文件
- `src/models.py` - 主要修改
- `src/warping.py` - Warpfield 計算
- `test_geometric_warp_itd.py` - ITD 測試
- `test_model_integration.py` - 整合測試
- `geometric_warp_itd_modification.md` - 詳細說明

## 作者
修改日期: 2026-04-14
