# 專案檔案結構說明 (Project File Structure)

## 核心腳本 (Core Scripts)

*   **`evaluate.py`** (恢復版)
    *   **用途**: 標準評估腳本。
    *   **狀態**: 已修復 `soundfile` 讀取與 Tensor 維度問題，但在功能上保持與原始專案一致 (無額外指標)。
    *   **適用**: 想要跑出最原始風格的數據時。

*   **`evaluate_enhanced.py`** (增強版)
    *   **用途**: 進階評估與除錯用。
    *   **狀態**: 包含 **Rx Position 整合**、**Angular Error 指標**、**中文輸出**。
    *   **適用**: 日常開發與除錯，能看到更詳細的方位資訊。

*   **`normalize_positions.py`**
    *   **用途**: 資料前處理工具。
    *   **功能**: 將指定資料夾內的 Tx 位置正規化到 1.0m，並強制設定面朝正前方 (Face Forward)，同時生成預設的 `rx_positions.txt`。

*   **`generate_distance_variations.py`**
    *   **用途**: 實驗數據生成工具。
    *   **功能**: 基於 Subject 1 自動生成多組不同距離 (20cm - 300cm) 的測試資料夾。

## 資料集結構 (Dataset Structure)

### 1. `dataset_original/` 
*   **用途**: **原始備份** (不可修改)。
*   **內容**: 包含最原始的 `testset` 和 `trainset`。若程式跑壞了，請從這裡還原。

### 2. `dataset/testset/`
*   **用途**: **標準測試集** (已修正)。
*   **狀態**: 
    *   所有 Tx 距離皆正規化為 **100cm**。
    *   所有 Tx 方向皆設定為 **Face Forward**。
    *   包含初始化 Rx 位置 (原點)。
*   **目的**: 用於評估模型在標準化場景下的基礎效能。

### 3. `dataset/testset_variations/`
*   **用途**: **距離感知實驗集**。
*   **內容**: 
    *   `subject1_dist_20cm`
    *   `subject1_dist_50cm`
    *   `subject1_dist_100cm` ... 等
*   **目的**: 專門用於測試模型對距離變化的反應能力。

## 輸出結果 (Outputs)

*   **`results_audio/`**: 對應 `dataset/testset` 的生成的雙耳音訊。
*   **`results_audio_variations/`**: 對應 `dataset/testset_variations` 的生成結果 (可用於聽感比較)。
