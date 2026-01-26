# 實驗性程式碼目錄 (Experimental Scripts)

此目錄包含在研究過程中建立的實驗性腳本。這些腳本用於測試不同的方法和想法，但不是核心功能的一部分。

## 腳本說明

### evaluate_enhanced.py
評估腳本的增強版本，包含：
- 嘗試計算相對座標的程式碼（已註解掉）
- 角度誤差指標
- ITD/ILD 指標
- 中文輸出訊息

### find_best_angles.py
掃描不同角度以找到最佳的 Transmitter 位置。用於分析模型對不同空間位置的表現。

### generate_distance_variations.py
生成不同距離的 Transmitter 位置，用於測試模型的距離感知能力。

### generate_tx_positions.py
生成不同的 Transmitter 位置用於測試。

### normalize_positions.py
對位置資料進行正規化處理。

### test_itd_ild_metrics.py
測試 ITD (Interaural Time Difference) 和 ILD (Interaural Level Difference) 指標的實作正確性。

### regenerate_tx.py
重新生成 Transmitter 位置檔案。

---

**注意**：這些腳本可能與當前的主程式碼不同步，使用前請檢查相容性。
