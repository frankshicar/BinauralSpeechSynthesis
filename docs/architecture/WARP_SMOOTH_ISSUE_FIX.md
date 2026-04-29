# Warp Smooth Loss 問題修正

## 問題發現

訓練時發現 `warp_smooth` loss 完全不變：

```
Epoch 1:  warp_smooth = 6.60e-08
Epoch 2:  warp_smooth = 6.60e-08
Epoch 5:  warp_smooth = 6.60e-08
Epoch 10: warp_smooth = 6.60e-08
```

## 根本原因

通過調試發現：

### 1. Geometric Warpfield 在時間上幾乎是常數

```
Geometric warpfield smoothness:
  - mean_abs_diff: 0.0000000000
  - mean_squared_diff: 0.0000000000

前 10 個時間步: [-155.32945, -155.32945, -155.32945, ...]
相鄰差異: [0, 0, 0, 0, ...]
```

### 2. 為什麼會這樣？

**數據採樣率差異**：
- View 數據（位置/方向）：120Hz
- 音訊數據：48kHz
- 每 400 個音訊樣本對應 1 個 view 樣本

**插值方式**：
```python
# src/warping.py
distance = th.sum(displacements**2, dim=2) ** 0.5  # B x 2 x K (120Hz)
distance = F.interpolate(distance, size=seq_length)  # B x 2 x T (48kHz)
```

`F.interpolate` 使用線性插值，在每 400 個樣本的區間內：
- Warpfield 變化是線性的
- 相鄰樣本的差異極小（斜率恆定）
- 導致 smoothness loss 接近 0

### 3. 實際數值

```
Raw smoothness (不帶權重): 0.0000000093
Raw smoothness * 0.001: 0.0000000000 (四捨五入後)
```

由於值太小，乘以 `lambda_smooth=0.001` 後變成 `6.60e-08`，在訓練過程中幾乎不變。

## 問題分析

### 原始設計
```python
# 對 total warpfield 計算 smoothness
warp_smooth_loss = self.warp_smoothness_loss(warpfields['total'])
```

### 為什麼不work？

1. **Geometric warp 主導**
   - Total warpfield = geometric + neural
   - Geometric warpfield 幅度大（~-143）
   - Neural warpfield 幅度小（~-0.03）
   - Geometric warp 已經很平滑（由插值保證）

2. **Neural warp 的貢獻被淹沒**
   - Neural warp 的不平滑性被 geometric warp 的平滑性掩蓋
   - Loss 值主要反映 geometric warp 的平滑度
   - 對 neural warp 沒有約束作用

3. **Loss 值太小**
   - 即使 neural warp 有變化，總體 smoothness 仍然很小
   - 乘以 `lambda_smooth=0.001` 後更小
   - 對訓練幾乎沒有影響

## 解決方案

### 修正：只對 Neural Warp 計算 Smoothness

```python
# 修正前
warp_smooth_loss = self.warp_smoothness_loss(warpfields['total'])

# 修正後
warp_smooth_loss = self.warp_smoothness_loss(warpfields['neural'])
```

### 理由

1. **Geometric warp 已經平滑**
   - 由線性插值保證
   - 不需要額外約束

2. **Neural warp 需要約束**
   - Neural network 可能學習到不平滑的修正
   - 需要 smoothness loss 來約束

3. **更有效的梯度**
   - 直接作用於 neural warp
   - 不被 geometric warp 的平滑性掩蓋

## 預期效果

修正後，`warp_smooth` loss 應該：

1. **有實際變化**
   - 反映 neural warp 的平滑度
   - 隨訓練而變化

2. **更大的數值**
   - Neural warp 的變化比 total warpfield 更明顯
   - Loss 值應該在 1e-6 到 1e-4 範圍

3. **實際約束 neural warp**
   - 防止 neural warp 產生時間上的突變
   - 提高音質

## 驗證方法

### 1. 檢查訓練記錄
```bash
python -c "
import json
with open('outputs_with_warp_loss/training_logs/training_history.json', 'r') as f:
    history = json.load(f)
    for i in range(min(10, len(history))):
        print(f'Epoch {history[i][\"epoch\"]}: warp_smooth = {history[i][\"warp_smooth\"]:.10f}')
"
```

應該看到 `warp_smooth` 有變化。

### 2. 使用調試腳本
```bash
python debug_warp_smooth.py
```

檢查：
- Neural warpfield smoothness 是否有合理的值
- 是否隨訓練而變化

## 其他可能的改進

### 選項 1：調整權重
如果 `warp_smooth` 仍然太小，可以增加權重：

```python
config = {
    "lambda_smooth": 0.01,  # 從 0.001 增加到 0.01
    # ...
}
```

### 選項 2：使用不同的 smoothness 度量
除了 L2 norm，還可以考慮：

```python
# 一階導數的絕對值
temporal_diff = warpfield[:, :, 1:] - warpfield[:, :, :-1]
smoothness_loss = th.mean(th.abs(temporal_diff))

# 二階導數（加速度）
second_diff = temporal_diff[:, :, 1:] - temporal_diff[:, :, :-1]
smoothness_loss = th.mean(second_diff ** 2)
```

### 選項 3：頻域 smoothness
在頻域中約束高頻成分：

```python
# FFT
warpfield_fft = th.fft.rfft(warpfield, dim=-1)
# 懲罰高頻成分
high_freq_penalty = th.mean(th.abs(warpfield_fft[:, :, freq_cutoff:]) ** 2)
```

## 總結

- ✅ **問題**：`warp_smooth` loss 不變是因為對 total warpfield 計算，被 geometric warp 的平滑性主導
- ✅ **修正**：改為只對 neural warp 計算 smoothness
- ✅ **預期**：Loss 值應該有變化，並實際約束 neural warp

修正已應用到 `src/trainer.py`。

## 修改的文件

- `src/trainer.py` - 修改 `train_iteration()` 方法
  ```python
  # 只對 neural warp 計算 smoothness
  warp_smooth_loss = self.warp_smoothness_loss(warpfields['neural'])
  ```

## 下一步

1. 重新開始訓練或從 checkpoint 繼續
2. 監控 `warp_smooth` loss 是否有變化
3. 如果仍然太小，考慮增加 `lambda_smooth` 權重
