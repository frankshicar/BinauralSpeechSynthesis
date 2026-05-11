# GeoWarpFiLMNet v6：NeuralWarpCorrector + Per-frame FiLM Conditioning

**日期**：2026-05-01  
**基礎**：GeoWarpFiLMNet v5b  
**模型檔**：`src/models_geowarp_film_v6.py`  
**訓練腳本**：`training/train_geowarp_film_v6.py`  
**輸出目錄**：`geowarp_film_v6/`

---

## 1. 問題診斷（v5b 的根本缺陷）

### 1.1 v5b 的 val_ipd 無法改善

v5b Stage 2 訓練 43 個 epoch，val_ipd 從 2.26 只降到 2.24，幾乎沒有改善：

```
[S2] Ep  1  val_phase=0.899  val_ipd=2.3570
[S2] Ep  17 val_phase=0.853  val_ipd=2.2649  ← best
[S2] Ep  43 val_phase=0.850  val_ipd=2.2386  ← 43 epoch 後幾乎沒動
```

這是**架構瓶頸**，不是訓練不夠。

### 1.2 兩個根本問題

**問題 1：warpfield ITD 低估（缺少 neural 修正）**

Meta 的 Warpnet 設計：
```python
warpfield = geometric_warpfield + neural_warpfield  # 合併
```
`neural_warpfield` 是 4 層 Conv1d，學習修正 geometric warpfield 低估的 ITD。

v5b 直接用 `geometric_warper(mono, view)` 做時域 warping，沒有 neural 修正步驟。FiLM ResStack 在 STFT 域操作，無法修正時域 ITD 的低估。

**問題 2：FiLM conditioning 沒有時間解析度**

v5b 的 `FourierPositionEncoder`：
```python
velocity = (view[:, :, 1:] - view[:, :, :-1]).mean(dim=2)  # 全段平均速度
pos = view.mean(dim=2)                                       # 全段平均位置
return self.mlp(combined)  # (B, 256) ← 整段只有一個向量
```

整段音訊用同一個 `pos_feat` 做 FiLM conditioning。聲源在移動，但每個 STFT frame 的 γ/β 完全相同。這是動態場景 phase 不好、單角度也不好的根本原因。

---

## 2. v6 改進方案

### 2.1 NeuralWarpCorrector（仿 Meta Warpnet）

```python
class NeuralWarpCorrector(nn.Module):
    # 4 層 causal Conv1d，學習 delta warpfield
    # warpfield = geometric + delta
    # 確保 causality 後用 MonotoneTimeWarper 施加
```

前向流程改為：
```python
# v5b
y_init = self.geo_warper(mono, view)

# v6
geo_wf = self.geo_warper._warpfield(view, T)
y_init = self.neural_warp(mono, view, geo_wf)  # geometric + neural delta
```

### 2.2 TemporalPositionEncoder（per-frame conditioning）

```python
class TemporalPositionEncoder(nn.Module):
    # 輸入：view (B, 7, K)
    # 輸出：pos_feat (B, 256, T_stft)  ← 保留時間維度
    #
    # 特徵：pos Fourier + vel Fourier（差分）+ raw view
    # 用 Conv1d MLP 處理，最後 interpolate 到 T_stft
```

### 2.3 FiLMLayer per-frame

```python
# v5b：pos_feat (B, 256) → 全段同一個 gamma/beta
# v6：pos_feat (B, 256, T) → 每幀獨立的 gamma/beta
params = self.film_net(pos_feat)          # Conv1d: (B, num_bands*2, T)
gamma = params[:, self.band_ids, 0, :]   # (B, F, T)
beta  = params[:, self.band_ids, 1, :]   # (B, F, T)
return x * gamma.unsqueeze(1) + beta.unsqueeze(1)
```

---

## 3. 架構對比

| 面向 | v5b | v6 |
|------|-----|-----|
| Warp | GeometricWarper only | GeometricWarper + NeuralWarpCorrector |
| Conditioning 時間解析度 | 全段一個向量 (B, 256) | Per-frame (B, 256, T_stft) |
| 速度資訊 | 全段平均差分 | 逐幀差分 + Fourier encoding |
| FiLM γ/β | 每幀相同 | 每幀獨立 |
| 參數量 | ~2,000,000 | 2,203,718 (+~200K) |

### 與 DPATFNet 的比較

| 面向 | DPATFNet DPAB | v6 FiLM |
|------|--------------|---------|
| 動態建模 | Self-Attention mask (t±1) | 逐幀差分 Fourier encoding |
| 頻帶調製 | 全頻帶相同 conditioning | 64-band 獨立 γ/β（優勢） |
| TDW | 純 GeometricWarper | GeometricWarper + NeuralWarpCorrector（優勢） |
| Phase-L2 (reported) | 0.717 | TBD |
| IPD-L2 (reported) | 1.020 | TBD |

---

## 4. 訓練指令

```bash
# 背景執行
cd ~/frank/BinauralSpeechSynthesis
nohup bash scripts/start_train_geowarp_film_v6.sh > geowarp_film_v6/train_console.log 2>&1 &

# 監控
tail -f geowarp_film_v6/train.log

# 評估
python evaluate_geowarp_film.py --model_file geowarp_film_v6/best.net --dataset_directory dataset/testset
python evaluate_angular.py --model_file geowarp_film_v6/best.net --dataset_directory dataset/test_static
```

---

## 5. 預期效果

| 指標 | v5b | v6 預期 | DPATFNet |
|------|-----|---------|---------|
| val_phase | 0.853 | < 0.80 | 0.717 |
| val_ipd | 2.24 | < 1.5 | 1.020 |
| angular error (mean) | ~19° | < 10° | — |

**關鍵驗證點**：Stage 1 結束時 val_ipd 是否低於 2.0。若是，代表 NeuralWarpCorrector 有效修正 ITD。

---

## 6. 實驗結果

### 6.1 動態 testset 結果

| 指標 | v6 | v6.2 | v5b | Meta 3-block | DPATFNet |
|------|----|----|-----|-------------|---------|
| L2 (×10³) | **0.115** | 0.118 | 0.134 | 0.144 | 0.148 |
| Amplitude | 0.035 | 0.034 | 0.038 | 0.036 | 0.037 |
| Phase | **0.746** | 0.772 | 0.853 | 0.804 | 0.717 |
| val_ipd (train) | 2.17 | **0.27** | 2.24 | — | — |

> v6.2 的 val_ipd 從 2.17 降到 0.27，低頻 cosine IPD loss 確實有效。
> 但 phase 0.772 比 v6 的 0.746 差，原因是 cosine loss 數值範圍（0~2）與 MSE 不同，
> calibration 後 IPD weight 過高壓制了 phase loss 學習。

### 6.2 靜態角度誤差（test_static）

| Subject | GT | v6 pred | v6 err | v6.2 pred | v6.2 err | v5b err |
|---------|-----|---------|--------|-----------|---------|--------|
| subject1 | -90° | -90.0° | **0.0°** | -90.0° | **0.0°** | ~66° |
| subject2 | -60° | -49.9° | 10.1° | -52.9° | **7.1°** | ~18° |
| subject3 | -30° | -25.6° | **4.4°** | -23.5° | 6.5° | ~4° |
| subject4 | 0° | -1.9° | **1.9°** | -1.9° | **1.9°** | ~5° |
| subject5 | +30° | +21.4° | 8.6° | +19.4° | 10.6° | ~9° |
| subject6 | +60° | +44.3° | 15.7° | +39.2° | 20.8° | ~10° |
| subject7 | +90° | +85.6° | **4.4°** | +85.6° | **4.4°** | ~0° |
| **Mean** | | | **6.4°** | | **7.3°** | **~19°** |

> v6 在動態 testset（phase 0.746）和靜態角度（mean 6.4°）都優於 v6.2。
> v6.2 的 IPD loss 設計方向正確（val_ipd 大幅下降），但 loss weight 需要調整。

### 6.3 結論

- **v6 是目前最佳**：phase 0.746（接近 DPATFNet 0.717），angular error mean 6.4°（vs v5b 19°）
- **v6.2 的低頻 cosine IPD loss 方向正確**，但需要降低 IPD weight（避免壓制 phase）
- 下一步：v6.3 調整 IPD weight，或改用對數頻帶分配
