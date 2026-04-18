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

### align_dataset.py
資料集時間對齊工具，用於同步音訊和位置資料。

### build_angle_cache.py
建立角度到最佳 Transmitter 位置的快取映射。

### calibrate_angle.py
校準特定角度的 Transmitter 位置，使用 GCC-PHAT 方法。

### calibrate_75_degrees.py
針對 75 度角度的專門校準腳本。

### compare_doa_methods.py
比較不同的 DOA (Direction of Arrival) 估計方法。

### count_trainset_angles.py
統計訓練集中各個角度的分布情況。

### debug_mono_input.py
除錯單聲道輸入處理的工具。

### diagnose_ild.py
診斷 ILD (Interaural Level Difference) 相關問題。

### optimize_angle_compensation.py
優化角度補償參數的實驗腳本。

### regenerate_all_tx_raw.py
批次重新生成所有 Transmitter 位置的原始資料。

### regenerate_tx_with_compensation.py
使用角度補償重新生成 Transmitter 位置。

### check_90deg_ild.py
檢查 ±90° 樣本的 ILD（Interaural Level Difference）值，診斷工具。

### check_90deg_itd.py
檢查 ±90° 樣本的實際 ITD 值，看是否超過物理極限。

### diagnose_itd_all_angles.py
診斷所有角度的 ITD 值，完整的 ITD 分析工具。

### test_training_logs.py
測試訓練記錄功能是否正常工作。

### test_ipd_gradient.py
測試 IPD Loss 的梯度傳播是否正確。

---

**注意**：這些腳本可能與當前的主程式碼不同步，使用前請檢查相容性。
