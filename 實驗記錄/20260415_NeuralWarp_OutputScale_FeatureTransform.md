# 20260415 — Neural Warp 修改：Output Scale + 特徵轉換

**日期**：2026-04-15  
**修改檔案**：`src/models.py`  
**修改類別**：`Warpnet`  
**目的**：改善單角度生成時的角度不准問題，特別是右側 +60° 誤差（15.7°）遠大於左側 -60° 誤差（3.8°）

---

## 問題背景

### 觀察到的症狀

| 角度 | 感知誤差 | 方向 |
|------|---------|------|
| -60°（左側） | 3.8° | 往右偏 |
| +60°（右側） | **15.7°** | 往右偏 |
| +30° | 8.6° | 往右偏 |
| -30° | 6.5° | 往右偏 |

右側誤差系統性地大於左側，且 ITD 幅度診斷顯示：

| 角度 | 理論 ITD (samples) | 實際 ITD (samples) | 達成率 |
|------|-------------------|-------------------|--------|
| +90° | 31.48 | 2 | **6.4%** |
| +60° | 23.43 | 3 | **12.8%** |
| -60° | -23.43 | -2 | **8.5%** |
| -90° | -31.48 | -1 | **3.2%** |

**根本原因**：neural warp 的輸出幅度嚴重不足，且原始 quaternion 輸入對左右對稱性不友好。

---

## 修改內容

### 修改 A：可學習的 Output Scale（左右耳獨立）

**位置**：`src/models.py`，`Warpnet.__init__`

#### 原始程式碼
```python
class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()

    def neural_warpfield(self, view, seq_length):
        warpfield = view
        for layer in self.layers:
            warpfield = F.pad(warpfield, pad=[1, 0])
            warpfield = F.relu(layer(warpfield))
        warpfield = self.linear(warpfield)
        warpfield = F.interpolate(warpfield, size=seq_length)
        return warpfield
```

#### 修改後程式碼
```python
class Warpnet(nn.Module):
    def __init__(self, layers=4, channels=64, view_dim=7):
        super().__init__()
        self.layers = [nn.Conv1d(view_dim if l == 0 else channels, channels, kernel_size=2) for l in range(layers)]
        self.layers = nn.ModuleList(self.layers)
        self.linear = nn.Conv1d(channels, 2, kernel_size=1)
        # 修改 A: 左右耳獨立的可學習 output scale，初始值 10.0
        self.output_scale = nn.Parameter(th.full((2, 1), 10.0))
        self.neural_warper = MonotoneTimeWarper()
        self.geometric_warper = GeometricWarper()
```

**設計理由**：
- `nn.Parameter(th.full((2, 1), 10.0))` — shape `(2, 1)` 可與 `B×2×T` broadcast
- 初始值 10.0 對應 `10 / 48000 * 343 ≈ 0.071m` 的 ITD 修正量，接近理論耳間距（0.175m）量級
- 左右耳**獨立**縮放，讓模型可以自行修正左右不對稱的偏差

---

### 修改 B：輸入特徵轉換（Quaternion → 幾何特徵）

**位置**：`src/models.py`，`Warpnet._view_to_features`（新增靜態方法）

#### 原始輸入
```
view: B × 7 × K
(x, y, z, qx, qy, qz, qw)  ← 原始位置 + quaternion
```

#### 修改後輸入
```python
@staticmethod
def _view_to_features(view):
    x, y, z = view[:, 0], view[:, 1], view[:, 2]
    dist = th.sqrt(x**2 + y**2 + z**2 + 1e-8)
    az = th.atan2(-y, x)          # Y+ = 右 → 負角度為右側
    el = th.asin((z / dist).clamp(-1, 1))
    feat = th.stack([az, el, dist, th.sin(az), th.cos(az), th.sin(el), th.cos(el)], dim=1)
    return feat
```

```
feat: B × 7 × K
(az, el, dist, sin_az, cos_az, sin_el, cos_el)  ← 幾何特徵
```

**設計理由**：

| 特徵 | 說明 | 對稱性 |
|------|------|--------|
| `az` | 方位角（弧度） | 奇函數：左右對稱角度符號相反 |
| `el` | 仰角（弧度） | 偶函數：上下對稱 |
| `dist` | 聲源距離 | 純量，無方向性 |
| `sin_az` | 方位角正弦 | 奇函數，強化左右區分 |
| `cos_az` | 方位角餘弦 | 偶函數，前後區分 |
| `sin_el` | 仰角正弦 | 上下區分 |
| `cos_el` | 仰角餘弦 | 仰角幅度 |

原始 quaternion `(qx, qy, qz, qw)` 對左右旋轉的表示是非線性的，網路難以學到對稱映射。轉換後 `sin_az` 對左右是奇函數，網路只需學到一個對稱的映射即可同時處理左右兩側。

**注意**：維度仍為 7，`Conv1d` 的 `in_channels` 不需調整。

---

### neural_warpfield 完整修改

```python
def neural_warpfield(self, view, seq_length):
    warpfield = self._view_to_features(view)   # 修改 B
    for layer in self.layers:
        warpfield = F.pad(warpfield, pad=[1, 0])
        warpfield = F.relu(layer(warpfield))
    warpfield = self.linear(warpfield)
    warpfield = warpfield * self.output_scale   # 修改 A
    warpfield = F.interpolate(warpfield, size=seq_length)
    return warpfield
```

---

## 架構對比

### 原始架構

```
view (B×7×K)
(x, y, z, qx, qy, qz, qw)
        │
        ▼
  Conv1d×4 + ReLU (causal, ch=64)
        │
  Conv1d → B×2×K
        │
  Interpolate → B×2×T
        │
  neural_warp (B×2×T)   ← 幅度受限於網路初始化，實測只有理論值 3-8%
```

### 修改後架構

```
view (B×7×K)
(x, y, z, qx, qy, qz, qw)
        │
        ▼ [修改 B]
  _view_to_features()
  (az, el, dist, sin_az, cos_az, sin_el, cos_el)
        │
  Conv1d×4 + ReLU (causal, ch=64)
        │
  Conv1d → B×2×K
        │ [修改 A]
  × output_scale (2×1, learnable, init=10.0)
  左耳 scale / 右耳 scale 獨立學習
        │
  Interpolate → B×2×T
        │
  neural_warp (B×2×T)   ← 初始幅度 ~10 samples，可學習調整
```

### 完整模型架構（修改後）

```
mono (B×1×T)          view (B×7×K)
                              │
                    ┌─────────┴──────────┐
                    │                    │
                    ▼                    ▼
           ┌─────────────────┐  ┌──────────────────────┐
           │  GeometricWarper│  │  Warpnet.neural_warp  │
           │                 │  │                       │
           │ mouth offset    │  │ _view_to_features()   │
           │ + ear offset    │  │ → (az,el,dist,        │
           │ → distance      │  │    sin/cos az/el)     │
           │ → -dist/343*sr  │  │       │               │
           │                 │  │  Conv1d×4 + ReLU      │
           │ geo_warp(B×2×T) │  │  Conv1d → B×2×K       │
           └────────┬────────┘  │  × output_scale(2×1)  │
                    │           │  Interpolate→ B×2×T   │
                    └─────┬─────┘                       │
                          ▼                             │
               geo_warp + neural_warp ◄─────────────────┘
                          │
               -ReLU(-·)  ← causality
                          │
               MonotoneTimeWarper
                          │
               warped (B×2×T)
                          │
               Conv1d (2→64)
                          │
               HyperConvWavenet
               (dilated conv, hyper-weight from view)
                          │
               WaveoutBlock × N (sin activation)
                          │
               output (B×2×T)
```

---

## 預期效果

| 問題 | 修改前 | 修改後（預期） |
|------|--------|--------------|
| ITD 幅度 | 理論值的 3-8% | 初始即達 ~30%，可學習至更高 |
| 右側誤差大於左側 | +60° 誤差 15.7° vs -60° 誤差 3.8° | 左右 scale 獨立，可自動修正不對稱 |
| 特徵對稱性 | quaternion 非線性 | sin_az 奇函數，對稱性明確 |

---

## 注意事項

1. **不相容舊 checkpoint**：`output_scale` 是新增的 `nn.Parameter`，舊的 `.net` 檔案載入時會有 key mismatch，需要從頭訓練或用 `strict=False` 載入。
2. **geometric warp 不受影響**：`GeometricWarper` 仍使用原始 `view`（需要 quaternion 計算旋轉矩陣），不受特徵轉換影響。
3. **HyperConvWavenet 不受影響**：`view` 直接傳入 wavenet 的 hyper-weight 生成，不經過特徵轉換。

---

**狀態**：✅ 程式碼已修改，待重新訓練驗證
