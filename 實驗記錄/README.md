# 實驗記錄總覽

本資料夾是正式實驗紀錄區。每次重要訓練、架構變更、比較結果、失敗分析
都應該放在這裡，而不是散在 repo 根目錄或 `docs/`。

## 快速入口

- `實驗總表.md` - 所有重要實驗的總覽與結果表。
- `05_DPATFNet路線實驗/20260509_GeoWarpFiLMNet_v6.4_實驗計畫.md` -
  目前 v6.4 的實驗計畫與監控設計。
- `05_DPATFNet路線實驗/20260501_GeoWarpFiLMNet_v6.md` - v6 相關紀錄。
- `04_HybridTFNet實驗/` - HybridTFNet 與 IPD-only 失敗分析。

## 分類

### `01_早期實驗/`

時間對齊、距離、GCC-PHAT、早期清理與噪音修復紀錄。

### `02_角度與定位/`

角度估計、IPD Loss、訓練紀錄功能、測試集設計與 Neural Warp 早期分析。

### `03_模型訓練_v4-v8/`

v5、v6、v7、v8 系列模型訓練紀錄與 v8.2/v8.3 計畫。

包含一份從錯誤 literal-unicode 目錄救回的備份：
`20260425_v8模型訓練記錄_literal_unicode_backup.md`。

### `04_HybridTFNet實驗/`

HybridTFNet、IPD-only、WaveformSpatializer、混合物理學習方法，以及相關失敗分析。

### `05_DPATFNet路線實驗/`

DPATFNet、GeoWarpFiLMNet v5/v6/v6.4、FiLM/Neural Warp 架構分析與目前主要研究線。

## 尚未歸類的根層紀錄

以下檔案保留在根層，因為它們是跨主題總結或舊紀錄引用目標：

- `20260427_E9_ImprovedResidual.md`
- `20260427_Phase學習失敗總結.md`
- `20260427_多Agent討論_分階段訓練.md`
- `20260427_最終實驗結果.md`
- `20260427_過夜實驗計劃.md`
- `project_structure.md`

## 新增紀錄規則

- 檔名使用 `YYYYMMDD_主題.md`。
- 新 run 放到對應主題資料夾，並更新 `實驗總表.md`。
- 技術設計文件放 `docs/architecture/`；實驗過程與結果放這裡。
- 若紀錄包含輸出位置，請寫清楚 dataset split、checkpoint、evaluate command、
  metric 定義，避免之後比較時混淆。
