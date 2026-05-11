# GeoWarpFiLMNet v6.4 實驗計畫與診斷紀錄

**日期**: 2026-05-09  
**狀態**: 準備執行  
**建議下一步**: 直接執行 v6.4，不再先跑 v6.3

---

## 1. 為什麼不先跑 v6.3

上一輪 testset 評估結果：

| 模型 | L2 x10^3 | Amplitude | Phase | Angular |
|---|---:|---:|---:|---:|
| v6 baseline | 0.115 | 0.035 | 0.746 | 50.4 deg |
| v6.2 | 0.118 | 0.034 | 0.772 | 50.1 deg |
| v6.3 | 0.115 | 0.034 | 0.759 | 64.9 deg |

v6.3 不建議繼續作為下一輪主實驗，原因：

1. v6.3 宣稱 `NeuralWarpCorrector 4 -> 6 layers`，但 `GeoWarpFiLMNet(... warp_layers=4)` 仍覆蓋了 class default，實際 checkpoint 也是 4 層。
2. v6.3 log-frequency FiLM band allocation 有實作 bug：對 `F=513` 的 STFT bins，band edges 最後只到 176，bin 177-512 被錯誤留在 band 0。
3. v6.3 angular error 從 v6 的 50.4 deg 退化到 64.9 deg，符合上述 band bug 可能破壞高頻/定位 cue 的現象。

因此下一輪應直接跑 v6.4。v6.3 保留為「錯誤設計對照組」與失敗分析紀錄。

---

## 2. v6.4 假設

### 核心假設

GeoWarpFiLMNet 應該拆成兩個可觀察的學習角色：

1. **Neural warp**：主要修正低頻 ITD/IPD，也就是左右耳時間/相位差。
2. **FiLM stack**：主要修正頻譜包絡、ILD、HRTF 造成的頻帶能量差與音色。

如果這個假設正確，診斷圖應該看到：

- neural warp 後的 low-frequency IPD error 比 geometric warp 低。
- FiLM block 的 gamma/beta 在低頻對定位 cue 有可見修正，並在中高頻對 HRTF/timbre 做平滑調整。
- final output 的 spectral L1 / ILD error 比 neural warp 更接近 ground truth。

---

## 3. v6.4 改動

### 模型

新增：`src/models_geowarp_film_v6_4.py`

改動：

1. 修正 log-frequency FiLM band allocation，確保所有 513 個 STFT bins 都被覆蓋，且沒有空 band。
2. 預設 `warp_layers=6`，真正啟用 6-layer `NeuralWarpCorrector`。
3. FiLM 從 `x * gamma + beta` 改成 `x * (1 + gamma_delta) + beta`。
4. FiLM 最後一層 zero-init，讓每個 residual block 初始接近 identity，避免剛開始就破壞 v6 warm start。
5. `detach_geo_phase=False`，讓 phase residual 的梯度可以回到 neural warp path。
6. `forward(..., return_debug=True)` 可以輸出 geometric warp、neural warp、每個 FiLM block activation、gamma/beta 與 final audio。

### 訓練

新增：`training/train_geowarp_film_v6_4.py`

策略：

1. 從 `geowarp_film_v6/best.net` partial warm start。
2. 只載入 shape 相容的權重；新增的 neural warp 第 5/6 層從初始化開始。
3. Stage 1：`mag_anchor + phase + ITD`。
4. Stage 2：`L2 + Phase + IPD + ITD`。
5. checkpoint metric 用 `phase + 0.25 * itd`，避免只顧 phase 而犧牲定位。

### 頻譜診斷

新增：`scripts/diagnose_geowarp_film_v6_4_spectra.py`

輸出：

- `00_audio_stage_mean_spectrum.png`
- `01_delta_warpfield.png`
- `film_block_01_mean_spectrum.png` ~ `film_block_08_mean_spectrum.png`
- `film_block_XX_gamma_delta.png`
- `film_block_XX_beta.png`
- `summary.json`

`summary.json` 會包含：

- geometric warp / neural warp / final output 的 spectral L1
- low-frequency IPD MAE
- low-frequency ILD MAE
- FiLM band counts

---

## 4. 執行方式

### 訓練 v6.4

```bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
./scripts/start_train_geowarp_film_v6_4.sh
```

### 訓練後評估 testset

目前 `evaluate_geowarp_film_v6.py` import 固定指向 v6 class。v6.4 評估建議用 runtime alias：

```bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
/home/sbplab/anaconda3/bin/python - <<'PY'
import runpy, sys
import src.models_geowarp_film_v6_4 as v64
sys.modules['src.models_geowarp_film_v6'] = v64
sys.argv = [
    'evaluate_geowarp_film_v6.py',
    '--model_file', 'geowarp_film_v6_4/best.net',
    '--dataset_directory', 'dataset/testset',
    '--artifacts_directory', 'geowarp_film/eval_output_v6.4',
]
runpy.run_path('evaluate_geowarp_film_v6.py', run_name='__main__')
PY
```

### 產生頻譜診斷

```bash
cd /home/sbplab/frank/BinauralSpeechSynthesis
/home/sbplab/anaconda3/bin/python scripts/diagnose_geowarp_film_v6_4_spectra.py \
  --model_file geowarp_film_v6_4/best.net \
  --dataset_directory dataset/testset \
  --subject subject4 \
  --seconds 2.0 \
  --output_dir geowarp_film_v6_4/spectra_diagnostics/subject4
```

建議至少看三個角度：

- `subject1`: -90 deg，極端左側
- `subject4`: 0 deg，正前方
- `subject7`: +90 deg，極端右側

### 訓練中頻譜監控

2026-05-09 修正：原本 v6.4 只有離線診斷腳本，training loop 沒有定期輸出每層頻譜；這不符合本實驗的監控需求。已在 `training/train_geowarp_film_v6_4.py` 內接入 training-time spectra monitor。

輸出位置：

```text
geowarp_film_v6_4/training_spectra/
```

每個監控點會建立：

```text
stage0_epoch000/
stage1_epoch001/
stage1_epoch002/
...
stage2_epoch001/
...
```

每個目錄包含：

- `00_audio_stage_mean_spectrum.png`: GT / geometric warp / neural warp / final output 左右耳頻譜。
- `01_delta_warpfield.png`: neural warp 學到的左右耳 residual warpfield。
- `neural_warp_layer_01..06_temporal_spectrum.png`: 6 層 neural warp activation 的 temporal spectrum。
- `film_block_01..08_mean_spectrum.png`: 每個 FiLM block 的 input / conv_out / film_out / output 頻譜。
- `film_block_01..08_gamma_delta.png`: 每個 FiLM block 的 gamma_delta heatmap。
- `film_block_01..08_beta.png`: 每個 FiLM block 的 beta heatmap。
- `summary.json`: 機器可讀的 per-epoch 指標。
- `analysis.md`: 文字判讀，直接標示 neural warp IPD、final spectrum、final ILD 是否改善。

累積摘要：

```text
geowarp_film_v6_4/training_spectra/summary.jsonl
```

當前背景訓練已重啟為監控版，初始監控 `stage0_epoch000` 已產生。初始狀態觀察：

- Neural warp IPD: `0.6430 -> 0.6759`，尚未改善。
- Final spectrum L1: `3.80 -> 14.63 dB`，尚未改善。
- Final low-frequency ILD: `4.17 -> 6.02 dB`，尚未改善。
- FiLM 8 blocks 皆為 identity-like，符合 zero-init 設計；後續 epoch 應逐漸變 active。

---

## 5. 觀察判準

### Neural warp 有沒有學到

看 `summary.json`：

- `neural_warp.low_freq_ipd_mae_rad < geometric_warp.low_freq_ipd_mae_rad`
- `01_delta_warpfield.png` 不是全零，也不是劇烈雜訊；應該是平滑的左右耳延遲修正。

看 `00_audio_stage_mean_spectrum.png`：

- neural warp 不應大幅改變 magnitude envelope。
- 它主要應改變左右耳時間/相位關係。

### FiLM 有沒有學到

看每個 `film_block_XX_mean_spectrum.png`：

- activation spectrum 不應在某一層突然崩成全平或爆炸。
- final output 的頻譜包絡應比 neural warp 更靠近 GT。

看 `film_block_XX_gamma_delta.png` / `beta.png`：

- 不應全部接近 0；如果全部 0，FiLM 沒有學到。
- 不應只在單一錯誤 band 有巨大值；若出現，可能 band allocation 或 loss 權重有問題。
- 低頻變化應和 ITD/IPD 目標相關，中高頻變化應更像 HRTF/ILD/timbre 修正。

---

## 6. 預期結果

短期成功標準：

- v6.4 testset phase <= v6.3 的 0.759。
- angular error 不得比 v6 baseline 50.4 deg 更差。
- `summary.json` 中 neural warp 的 low-frequency IPD MAE 要比 geometric warp 低。

強成功標準：

- testset phase 接近或低於 v6 baseline 0.746。
- angular error 明顯低於 50 deg。
- final output spectral L1 / ILD MAE 低於 neural warp。

---

## 7. 風險與下一步

風險：

1. 6-layer neural warp 需要更多 training time，partial warm start 可能不夠穩。
2. ITD loss 權重過高可能犧牲 waveform L2 或 phase。
3. FiLM zero-init 讓 early epochs 變化較慢，但這是刻意換取穩定性。

若 v6.4 失敗：

1. 先看 spectra diagnostics，不只看 aggregate metrics。
2. 若 neural warp IPD 沒改善：調高 `w_itd` 或改 neural warp delta scale。
3. 若 FiLM 沒動：檢查 gamma/beta 是否仍接近 0，必要時提高 lr 或解除部分 zero-init。
4. 若 FiLM 爆掉：降低 lr、增加 beta/gamma clamp 或加 regularization。
