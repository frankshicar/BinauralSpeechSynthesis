# Geometric Warp ITD 修改說明

## 修改日期
2026-04-14

## 問題描述
原始的 geometric warp 實現中，左右耳使用相同的 warpfield 計算方式，只是使用了固定的耳朵偏移量。這導致：
1. ITD（Interaural Time Difference）完全依賴 neural warp 學習
2. Neural warp 負擔過重，難以準確學習 ITD
3. 訓練收斂困難

## 解決方案
修改 geometric warp，使其分別計算聲源到左右耳的實際距離，從而在 geometric warpfield 中就包含理論 ITD。

## 修改內容

### 1. 新增 `ear_offset` 參數
```python
self.ear_offset = 0.0875  # 單耳到頭部中心距離，約 8.75cm
```

### 2. 新增 `_quaternion_to_right_vector` 方法
從 receiver 的 quaternion 方向計算右向量，用於確定左右耳的位置。

```python
def _quaternion_to_right_vector(self, quat):
    rot_mat = R.from_quat(quat)
    right_vector = rot_mat.apply([0, 1, 0])
    return right_vector
```

### 3. 新增 `_listener_ear_positions` 方法
根據 receiver 的方向，計算左右耳在世界坐標系中的實際位置。

```python
def _listener_ear_positions(self, view):
    # 計算右向量
    right_vector = self._quaternion_to_right_vector(listener_quat)
    
    # 左耳 = -ear_offset * right_vector
    # 右耳 = +ear_offset * right_vector
    left_ear_pos = -self.ear_offset * right_vector
    right_ear_pos = self.ear_offset * right_vector
    
    return left_ear_pos, right_ear_pos
```

### 4. 修改 `_3d_displacements` 方法
分別計算 transmitter mouth 到左右耳的位移向量。

```python
def _3d_displacements(self, view):
    transmitter_mouth_abs = transmitter_pos + transmitter_mouth
    left_ear_pos, right_ear_pos = self._listener_ear_positions(view)
    
    # 分別計算到左右耳的位移
    displacement_left = transmitter_mouth_abs - left_ear_pos
    displacement_right = transmitter_mouth_abs - right_ear_pos
    
    displacement = th.stack([displacement_left, displacement_right], dim=1)
    return displacement  # B x 2 x 3 x K
```

### 5. 更新 `displacements2warpfield` 方法
處理分離的左右耳位移，計算各自的 warpfield。

```python
def displacements2warpfield(self, displacements, seq_length):
    # displacements shape: B x 2 x 3 x K
    distance = th.sum(displacements**2, dim=2) ** 0.5  # B x 2 x K
    distance = F.interpolate(distance, size=seq_length)  # B x 2 x T
    warpfield = -distance / 343.0 * self.sampling_rate  # B x 2 x T
    return warpfield
```

## 測試結果

### 測試場景 1：聲源在右側
- Transmitter 位置: (0, 1, 0)
- 左耳 warpfield: -155.25 samples
- 右耳 warpfield: -131.33 samples
- **ITD: 23.92 samples = 0.498 ms** ✓
- 理論 ITD: 23.92 samples
- 誤差: 0.00 samples

### 測試場景 2：聲源在左側
- Transmitter 位置: (0, -1, 0)
- 左耳 warpfield: -131.33 samples
- 右耳 warpfield: -155.25 samples
- **ITD: -23.92 samples = -0.498 ms** ✓
- 符號正確：左側聲源，左耳先到達

## 預期效果

1. **減輕 Neural Warp 負擔**
   - Geometric warp 提供準確的 ITD 基礎
   - Neural warp 只需學習細微的 HRTF 效應和修正

2. **物理上更準確**
   - 聲音確實是分別到達左右耳的
   - 考慮了 listener 的頭部方向

3. **更容易收斂**
   - ITD 的主要部分由物理模型提供
   - 減少需要學習的參數空間

## 相關文件
- `src/models.py` - GeometricWarper 類別
- `src/warping.py` - GeometricTimeWarper 類別
- `test_geometric_warp_itd.py` - 測試腳本

## 下一步
1. 使用新的 geometric warp 重新訓練模型
2. 觀察 L_warp loss 的變化
3. 比較 ITD 預測準確度
4. 評估整體音質改善
