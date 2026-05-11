# GeoWarpFiLMNet v5：加入 ITD Consistency Loss

**日期**：2026-04-30  
**目標**：修正 v4 的靜態角度誤差問題，讓 GCC-PHAT 估角度更準確

---

## 1. 問題診斷（v4 的根本缺陷）

### 1.1 發現過程

用 `evaluate_angular.py` 評估 v4 在 `dataset/test_static` 的角度誤差：

```
subject1  gt=-90°  pred=-23.5°  err=66.5°   ← 嚴重失準
subject2  gt=-60°  pred=-29.9°  err=30.1°
subject3  gt=-30°  pred=-19.4°  err=10.6°
subject4  gt=  0°  pred=  3.8°  err= 3.8°
subject5  gt=+30°  pred=+27.7°  err= 2.3°   ← 幾乎完美
subject6  gt=+60°  pred=+44.3°  err=15.7°
subject7  gt=+90°  pred=+90.0°  err= 0.0°   ← 完美
Mean: 18.4°
```

**左右不對稱**：左側（+角度）幾乎完美，右側（-角度）系統性偏差。

### 1.2 根本原因分析

**步驟 1：確認 GCC-PHAT 的 ear_distance 設定**

發現 `src/doa.py` 的 `itd_to_azimuth` 使用 `ear_distance=0.215m`（標準人頭值），但 GeometricWarper 的耳間距是 `|[0,-0.08,...] - [0,0.08,...]| = 0.16m`。

→ **修正**：將 `ear_distance` 改為 `0.16m`，與訓練資料一致。

修正後角度誤差：18.4° → **14.6°**（改善，但 subject1 仍有 57.6°）

**步驟 2：分析 GeometricWarper 的 ITD**

GeometricWarper 在 ±90° 給出 ±23 samples（479μs），但 GCC-PHAT 需要 30.1 samples（627μs）才能估出 ±90°。

→ GeometricWarper 的 ITD 本身就低估了（因為 ear offset 只有 0.08m，不是 0.1075m）。

**步驟 3：分析 model 輸出的 ITD**

```
Subj    GT    GeoWarp_ITD   Model_ITD   GW_angle   M_angle
subject1  -90°    +23 smp       +0 smp    +90.0°     +0.0°
subject2  -60°    +18 smp       +0 smp    +53.5°     +0.0°
subject3  -30°    +10 smp      +11 smp    +26.5°    +29.4°
subject4    0°     +0 smp       -2 smp     +0.0°     -5.1°
subject5  +30°    -10 smp      -13 smp    -26.5°    -35.5°
subject6  +60°    -18 smp       +0 smp    -53.5°     +0.0°
subject7  +90°    -23 smp       +0 smp    -90.0°     +0.0°
```

**關鍵發現：Model 的 ITD ≈ 0**（subject1, 2, 6, 7 的 model ITD = 0 samples）

→ **Phase residual 把 GeometricWarper 建立的 ITD 完全抹掉了**

### 1.3 為什麼 Phase loss 好但 Angular error 差？

```
Phase loss = |angle(Y_pred_L) - angle(Y_gt_L)| + |angle(Y_pred_R) - angle(Y_gt_R)|
           = 每隻耳朵的絕對相位誤差（分開比）

Angular error ← GCC-PHAT 只看 IPD = angle(Y_L) - angle(Y_R)
```

模型可以讓兩耳的絕對相位都跟 GT 很接近（Phase loss 好），但同時把左右耳的相位差（ITD）抹掉（Angular error 差）。這兩個目標在 gradient 方向上不衝突，所以模型選擇了「讓每隻耳朵的絕對相位接近 GT，但不保留 ITD」的解。

---

## 2. v5 改進方案

### 2.1 ITD Consistency Loss

在 `src/losses.py` 加入 `ITDConsistencyLoss`：

```python
class ITDConsistencyLoss(nn.Module):
    """
    確保 model 輸出的 ITD 方向與 GeometricWarper 一致。
    使用 STFT cross-spectrum 的低頻相位（前 32 bins）估計 ITD。
    """
    def forward(self, Y_L, Y_R, Y_L_init, Y_R_init):
        pred_cross = Y_L[:, :K, :] * Y_R[:, :K, :].conj()
        geo_cross  = Y_L_init[:, :K, :] * Y_R_init[:, :K, :].conj()
        diff = atan2(sin(angle(pred_cross) - angle(geo_cross)),
                     cos(angle(pred_cross) - angle(geo_cross)))
        return (diff[geo_energy_mask] ** 2).mean()
```

**設計理由**：
- 只用低頻（前 32 bins，0~750Hz）：ITD 在低頻最有效，高頻 IPD 有 ambiguity
- 能量加權 mask：只在 GeoWarp 有能量的地方計算，避免靜音區域的 noise
- 懲罰 model 的 cross-spectrum phase 偏離 GeoWarp 的 cross-spectrum phase

### 2.2 訓練策略

| Stage | Loss | 目的 |
|-------|------|------|
| Stage 1 (30 ep) | `0.1×mag_anchor + 0.1×phase + 0.5×ITD` | 從一開始就保護 ITD |
| Stage 2 (80 ep) | `w_l2×L2 + w_phase×phase + w_ipd×IPD + 0.5×ITD` | 全局優化，持續保護 ITD |

**w_itd = 0.5**：小權重避免壓制 phase 學習，但足以防止 ITD 被抹掉。

### 2.3 同步修正

- `src/doa.py`：`ear_distance` 從 0.215m 改為 **0.16m**（與 GeometricWarper 一致）

---

## 3. 預期效果

| 指標 | v4 | v5 預期 |
|------|-----|---------|
| Phase (testset) | 0.831 | ≈ 0.83（略微退步或持平） |
| Angular error (test_static) | 18.4° | < 10°（顯著改善） |
| subject1 (-90°) | 66.5° | < 30° |

**風險**：ITD loss 可能與 Phase loss 有輕微衝突（ITD loss 只關心相位差，Phase loss 關心絕對相位），可能導致 Phase 略微退步。w_itd=0.5 是保守設定，如果 Phase 退步超過 0.02，考慮降低到 0.2。

---

## 4. 實驗結果

（待訓練完成後填入）

```
訓練指令：bash scripts/start_train_geowarp_film_v5.sh
監控：tail -f geowarp_film_v5/train.log
評估：python evaluate_geowarp_film.py --model_file geowarp_film_v5/best.net --dataset_directory dataset/testset
      python evaluate_angular.py --model_file geowarp_film_v5/best.net --dataset_directory dataset/test_static
```

### 4.1 動態 testset 結果

| 指標 | v5 | v4 | Meta small | Meta large | DPATFNet |
|------|----|----|-----------|-----------|---------|
| L2 (×10³) | TBD | 0.136 | 0.197 | 0.144 | 0.148 |
| Amplitude | TBD | 0.038 | 0.043 | 0.036 | 0.037 |
| Phase | TBD | 0.831 | 0.862 | 0.804 | 0.717 |

### 4.2 靜態角度誤差

| Subject | GT | v5 pred | v5 err | v4 err |
|---------|-----|---------|--------|--------|
| subject1 | -90° | TBD | TBD | 57.6° |
| subject2 | -60° | TBD | TBD | 17.9° |
| subject3 | -30° | TBD | TBD | 3.5° |
| subject4 | 0° | TBD | TBD | 5.1° |
| subject5 | +30° | TBD | TBD | 8.7° |
| subject6 | +60° | TBD | TBD | 9.7° |
| subject7 | +90° | TBD | TBD | 0.0° |
| **Mean** | | | **TBD** | **14.6°** |

---

## 5. 後續方向

如果 v5 的角度誤差仍然大（> 10°）：

1. **增加 w_itd**：從 0.5 提高到 1.0 或 2.0
2. **加入 GCC-PHAT 直接 loss**：直接最大化 cross-correlation peak（不可微，需要 soft approximation）
3. **修正 GeometricWarper 的 ear offset**：從 0.08m 改為 0.1075m，讓初始 ITD 直接對應 GCC-PHAT 的 ear_distance=0.16m

如果 v5 的 Phase 退步超過 0.02：

1. **降低 w_itd**：從 0.5 降到 0.2
2. **只在 Stage 2 加入 ITD loss**（Stage 1 不加）

---

## 6. 錯誤修正記錄（2026-05-01）

### 6.1 `ear_distance=0.16` 是錯的，已還原

**錯誤**：我曾把 `src/doa.py` 的 `ear_distance` 從 `0.215m` 改成 `0.16m`，理由是「與 GeometricWarper 的 ear offset 一致」。

**為什麼是錯的**：
- `ear_distance=0.215m` 是 KEMAR 的耳間距，訓練資料就是用 KEMAR 錄製的，GCC-PHAT 應該用這個值
- GeometricWarper 的 `left_ear_offset=[0,-0.08,-0.22]` 是 receiver 頭部**追蹤標記到耳朵的幾何偏移**，用來計算聲源到耳朵的 3D 位移，跟 GCC-PHAT 的耳間距是完全不同的東西
- 改成 0.16 後 Meta pretrained 的角度誤差從 4.0° 變成 10.4°，明顯變差

**已還原**：`src/doa.py` 的 `ear_distance` 已改回 `0.215m`

### 6.2 ITDConsistencyLoss 無效

**問題**：v5 的 Stage 1 log 顯示 `itd=0.0000`（從 Epoch 2 開始就是 0）

**原因**：ITDConsistencyLoss 計算 model 的低頻 cross-spectrum phase 與 GeoWarp 的差異。但 GeoWarp 的 phase 在 Stage 1 初期跟 model 的幾乎一樣（model 還沒學到任何修正），所以 loss = 0，沒有 gradient。這個 loss 設計在 Stage 1 完全無效。

**根本問題**：ITDConsistencyLoss 只能防止 model 偏離 GeoWarp，但 GeoWarp 的 ITD 本身就低估（23 samples vs 需要的 30 samples），所以即使 model 完全跟 GeoWarp 一致，角度誤差還是很大。

### 6.3 IPD weight 被 calibration 壓制

**問題**：Stage 2 的 calibrated weights 是 `L2=2334, Phase=1.0, IPD=0.36`，IPD 被完全壓制。

**原因**：calibration 的邏輯是讓各 loss 貢獻相等，但 IPD loss 的數值本來就比 Phase loss 小，所以 calibration 給它更小的 weight，反而讓它更沒影響力。

---

## 7. v5b 修改計畫

### 7.1 核心修改

**強制 IPD weight = Phase weight**，不讓 calibration 壓制 IPD：

```python
w = calibrate_weights(...)  # L2, Phase calibration 正常
w['ipd'] = w['phase']       # 強制 IPD = Phase = 1.0（不用 calibration）
```

**理由**：
- Phase loss 測絕對相位，IPD loss 測相位差，兩者應該有同等影響力
- Meta 的 BinauralNetwork 不需要 IPD loss 是因為 WaveNet 架構的 ITD 是硬編碼的，無法被抹掉
- 我們的 phase residual 架構可以自由選擇 ITD=0，所以必須用 IPD loss 強制保留 ITD

### 7.2 移除無效的 ITDConsistencyLoss

ITDConsistencyLoss 在實驗中完全無效（loss=0），且增加計算複雜度。v5b 移除它，專注在 IPD weight boosting。

### 7.3 預期效果

| 指標 | Meta 3-block | v4 | v5 | v5b 預期 |
|------|-------------|-----|-----|---------|
| Phase | 0.804 | 0.831 | 0.826 | ~0.83（略退步） |
| Angular error (mean) | **4.0°** | 19.1° | 19.1° | < 10° |
| subject1 (-90°) | 0.0° | 66.5° | 66.5° | < 30° |

### 7.4 風險

IPD weight 提高後，model 可能犧牲部分 Phase 來換取更好的 ITD。如果 Phase 退步超過 0.02（低於 0.85），考慮降低 IPD weight 到 0.5。

### 7.5 訓練指令

```bash
bash scripts/start_train_geowarp_film_v5b.sh
# 監控：tail -f geowarp_film_v5b/train.log
# 評估：
python evaluate_geowarp_film.py --model_file geowarp_film_v5b/best.net --dataset_directory dataset/testset
python evaluate_angular.py --model_file geowarp_film_v5b/best.net --dataset_directory dataset/test_static
```
