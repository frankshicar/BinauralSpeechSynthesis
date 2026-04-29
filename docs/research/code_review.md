# Code Review: HybridTFNet Implementation

## Reviewer A (Architecture Expert)

### 審查範圍
- `src/models_hybrid.py` - HybridTFNet 架構
- `train_hybrid_corrected.py` - 訓練腳本

---

### 1. 架構設計評價

#### ✅ 優點

**Time-Frequency 分離的概念是對的**
```python
# Time Branch: 學習 Phase
# Freq Branch: 學習 Magnitude
```
- 符合人耳感知特性（低頻 phase 重要，高頻 magnitude 重要）
- 理論上可以針對性優化

**FiLM 調制的使用**
```python
gamma = self.gamma_net(pos_feat)
beta = self.beta_net(pos_feat)
phase_feat = gamma * phase_feat + beta
```
- 比 Cross-Attention 更直接
- 梯度流動更好（實驗證實 position_encoder 梯度從 1e-5 提升到 0.3）

**STFT domain 處理**
```python
stft_feat = torch.stack([Y_mono.real, Y_mono.imag], dim=1)
phase_feat = self.stft_encoder(stft_feat)
```
- 每個頻率 bin 有獨立特徵（避免了 expand 的問題）
- 正確的做法

#### ❌ 問題

**Phase Difference 學習目標有根本缺陷**
```python
Phase_diff_L_gt = torch.angle(Y_L_gt / (Y_mono + 1e-8))
```
- GT 的 std = 1.82（接近 uniform）
- Wrapped MSE 對 uniform distribution 的梯度很弱
- 導致模型無法學到有意義的映射

**Fusion 方式有問題**
```python
Phase_L = Phase_mono + Phase_diff_L
Y_L = Mag_L * exp(1j * Phase_L)
```
- 如果 Phase_diff_L 是隨機的，Phase_L 也是隨機的
- 最終 L2 = GT magnitude + mono phase 的上限（0.0007）

**模型容量可能不足**
```python
tf_channels=128, tf_blocks=4
```
- 相比 DPATFNet 的 256 channels, 8 blocks
- 但這不是主要問題（主要問題在學習目標）

---

### 2. 訓練腳本評價

#### ✅ 優點

**Loss 設計合理**
```python
loss = loss_phase + 10 * loss_l2
```
- L2 權重較大，確保波形質量
- Phase loss 提供額外的監督信號

**詳細的指標記錄**
```python
history.append({
    'l2': l2_loss,
    'phase_L': phase_L_err,
    'phase_R': phase_R_err,
    'mag_L': mag_L_err,
    'mag_R': mag_R_err
})
```
- 方便診斷問題

**正確的 sample_rate**
```python
sample_rate=48000, n_fft=1024, hop_size=64
```
- 與資料匹配

#### ❌ 問題

**沒有 early stopping**
- Phase loss 已經證明學不動
- 應該在 10-20 epochs 就停止

**沒有 gradient checkpointing**
- 導致 batch_size 只能用 16
- 訓練速度慢

---

### 3. 總體評分

| 項目 | 評分 | 說明 |
|------|------|------|
| 架構創新性 | 8/10 | Time-Freq 分離 + FiLM 是好的想法 |
| 實現質量 | 7/10 | 代碼清晰，但有些細節可優化 |
| 學習目標 | 3/10 | Phase difference 有根本缺陷 |
| 可訓練性 | 4/10 | 模型在學習，但學不到正確的東西 |
| **總分** | **5.5/10** | 架構好，但學習目標錯誤 |

---

## Reviewer B (ML Engineer)

### 審查重點
- 訓練穩定性
- 記憶體效率
- 可重現性

---

### 1. 訓練穩定性

#### ✅ 優點

**Gradient clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
```
- 防止梯度爆炸

**Learning rate scheduler**
```python
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```
- 自動調整學習率

#### ❌ 問題

**Loss 沒有真正下降**
```
Epoch 1-9:
Train Loss: 6.47 → 6.35 (下降)
L2: 0.000719 → 0.000719 (不動)
```
- Train Loss 下降但 L2 不動 → 模型在優化錯誤的目標
- 這是學習目標的問題，不是訓練穩定性的問題

**沒有 warmup**
- 直接用 3e-4 的 LR
- 對於大模型可能需要 warmup

---

### 2. 記憶體效率

#### ❌ 主要問題

**沒有 gradient checkpointing**
```python
# TFResStack 的 forward
for block in self.blocks:
    x = block(x)  # 每個 block 都保存 activation
```
- 導致記憶體需求大
- batch_size 只能用 16

**建議改進**
```python
from torch.utils.checkpoint import checkpoint

for block in self.blocks:
    x = checkpoint(block, x, use_reentrant=False)
```
- 可以節省 40-60% 記憶體
- batch_size 可以提升到 32-64

**STFT 重複計算**
```python
# 在 loss 計算時重複做 STFT
Y_mono = torch.stft(mono, ...)
Y_L_gt = torch.stft(target[:, 0, :], ...)
```
- 可以預先計算並 cache

---

### 3. 可重現性

#### ✅ 優點

**Config 完整記錄**
```python
config = {
    'batch_size': 16,
    'learning_rate': 3e-4,
    'sample_rate': 48000,
    ...
}
```

**Checkpoint 保存完整**
```python
torch.save({
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'config': config
}, checkpoint_path)
```

#### ⚠️ 可改進

**沒有設置 random seed**
```python
# 應該加入
torch.manual_seed(42)
np.random.seed(42)
```

**沒有記錄 git commit hash**
- 無法追溯代碼版本

---

### 4. 總體評分

| 項目 | 評分 | 說明 |
|------|------|------|
| 訓練穩定性 | 6/10 | 基本穩定，但 loss 不下降 |
| 記憶體效率 | 5/10 | 沒有 gradient checkpointing |
| 可重現性 | 7/10 | Config 完整，但缺 random seed |
| 代碼質量 | 8/10 | 清晰易讀 |
| **總分** | **6.5/10** | 工程實現可以，但有優化空間 |

---

## 共識與建議

### 兩位 Reviewer 的共識

1. **架構設計方向是對的**（Time-Freq 分離 + FiLM）
2. **學習目標有根本問題**（Phase difference 學不到）
3. **實現質量可以**，但有優化空間

### 優先級建議

#### 🔴 高優先級（必須改）
1. **改變學習目標**：Phase difference → IPD only
2. **加入 gradient checkpointing**：提升 batch size

#### 🟡 中優先級（應該改）
1. **加入 random seed**：確保可重現性
2. **加入 early stopping**：避免浪費計算資源
3. **Cache STFT 結果**：減少重複計算

#### 🟢 低優先級（可以改）
1. **加入 warmup**：更穩定的訓練
2. **記錄 git hash**：更好的版本追溯
3. **增加模型容量**：channels 128 → 256（如果記憶體允許）

---

## 下一步行動

### 立即執行
1. ✅ 停止當前訓練（已證明無效）
2. ✅ 整理代碼和文件（已完成）
3. 🔄 實現 IPD-only 版本
4. 🔄 加入 gradient checkpointing

### 驗證目標
- IPD error 能否從 2.5 降到 <1.5
- L2 能否從 0.0007 降到 <0.0003
- 訓練 20 epochs 內看到明顯改善

---

## 評分總結

| Reviewer | 總分 | 主要問題 |
|----------|------|----------|
| A (Architecture) | 5.5/10 | 學習目標錯誤 |
| B (ML Engineer) | 6.5/10 | 記憶體效率低 |
| **平均** | **6.0/10** | 需要改進學習目標和記憶體優化 |

**結論**：代碼質量可以，但需要改變學習目標才能真正解決問題。
