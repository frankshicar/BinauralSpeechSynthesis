# Meta-Audio (DPATFNet) 複現計劃

**目標**: 找出為什麼 DPATFNet 比我們的方法好 16%

**時間**: 1-2 週

---

## 背景

### 當前結果
```
DPATFNet (E0):        L2 = 0.000719
Magnitude-only (E8b): L2 = 0.000857
差距: 16% (0.000138)
```

### 關鍵問題
1. DPATFNet 的 Magnitude 網路更好？
2. DPATFNet 的訓練策略更好？
3. DPATFNet 的 Phase 處理更好？
4. 我們的實現有 bug？

---

## Phase 1: 複現 Meta-Audio

### 1.1 獲取程式碼

**GitHub**: https://github.com/facebookresearch/meta-audio

```bash
cd /home/sbplab/frank
git clone https://github.com/facebookresearch/meta-audio.git
cd meta-audio
```

### 1.2 環境設置

**檢查依賴**:
```bash
cat requirements.txt
# 可能需要:
# - torch
# - torchaudio
# - numpy
# - scipy
# - librosa
```

**安裝**:
```bash
pip install -r requirements.txt
# 或使用我們現有的環境
```

### 1.3 理解程式碼結構

**關鍵檔案**:
```
meta-audio/
├── models/
│   └── dpatfnet.py          # DPATFNet 架構
├── data/
│   └── dataset.py           # 資料載入
├── train.py                 # 訓練腳本
├── config/
│   └── default.yaml         # 配置
└── utils/
    └── losses.py            # Loss functions
```

**需要理解**:
1. 模型架構細節
2. 訓練流程
3. Loss function 設計
4. 資料預處理

### 1.4 適配我們的資料集

**我們的資料格式**:
```python
dataset/
├── trainset/
│   ├── subject1/
│   │   ├── 0.wav    # Binaural
│   │   └── mono.wav # Mono
│   └── ...
└── valset/
    └── ...
```

**需要修改**:
1. Dataset class
2. Data loader
3. 預處理流程

### 1.5 訓練

**配置**:
```yaml
model:
  name: DPATFNet
  params:
    # 使用他們的預設參數

training:
  batch_size: 16
  learning_rate: 3e-4
  epochs: 100
  early_stopping: 15

data:
  train_dir: dataset/trainset
  val_dir: dataset/valset
  sample_rate: 48000
  chunk_size: 200ms
```

**執行**:
```bash
python train.py --config config/our_dataset.yaml
```

**預期時間**: 1-2 天

---

## Phase 2: 對比分析

### 2.1 結果對比

**Table: Meta-Audio vs Our Implementation**

| Metric | Meta-Audio (Official) | Our DPATFNet (E0) | Difference |
|--------|----------------------|-------------------|------------|
| Waveform L2 | ? | 0.000719 | ? |
| IPD | ? | 3.18 | ? |
| Phase L | ? | 0.9991 | ? |
| Phase R | ? | 0.9992 | ? |
| Magnitude | ? | - | ? |

**關鍵問題**:
1. 我們的實現正確嗎？
2. 差異來自哪裡？

### 2.2 架構對比

**檢查項目**:
1. ✅ Encoder 架構
2. ✅ Decoder 架構
3. ✅ Attention 機制
4. ✅ Skip connections
5. ✅ 參數數量

**對比方法**:
```python
# 載入兩個模型
official_model = load_official_model()
our_model = load_our_model()

# 對比架構
print(official_model)
print(our_model)

# 對比參數數量
print(f"Official: {count_params(official_model)}")
print(f"Ours: {count_params(our_model)}")
```

### 2.3 訓練對比

**檢查項目**:
1. Loss function
2. Optimizer
3. Learning rate schedule
4. Data augmentation
5. Regularization

**對比方法**:
```python
# 檢查 loss
official_loss = compute_official_loss(pred, target)
our_loss = compute_our_loss(pred, target)
print(f"Loss difference: {abs(official_loss - our_loss)}")

# 檢查 gradient
official_grad = compute_official_grad()
our_grad = compute_our_grad()
print(f"Gradient difference: {torch.norm(official_grad - our_grad)}")
```

### 2.4 推理對比

**檢查項目**:
1. STFT 參數
2. 預處理
3. 後處理
4. 輸出格式

**對比方法**:
```python
# 同一個輸入
mono, view = load_test_sample()

# 兩個模型推理
official_output = official_model(mono, view)
our_output = our_model(mono, view)

# 對比輸出
print(f"Output difference: {torch.norm(official_output - our_output)}")
```

---

## Phase 3: 找出差異

### 3.1 可能的差異來源

#### 3.1.1 架構差異

**檢查**:
- Layer 數量
- Hidden dimensions
- Activation functions
- Normalization

**如果發現差異**:
→ 修改我們的實現，重新訓練

#### 3.1.2 Loss 差異

**檢查**:
- Loss function 定義
- Loss weights
- 是否有額外的 regularization

**如果發現差異**:
→ 修改 loss，重新訓練

#### 3.1.3 訓練差異

**檢查**:
- Learning rate schedule
- Batch size
- Gradient clipping
- Weight initialization

**如果發現差異**:
→ 修改訓練配置，重新訓練

#### 3.1.4 資料差異

**檢查**:
- 預處理方式
- STFT 參數
- Normalization
- Data augmentation

**如果發現差異**:
→ 修改資料處理，重新訓練

### 3.2 系統性測試

**測試矩陣**:
```
Test 1: Official architecture + Our training
Test 2: Our architecture + Official training
Test 3: Official architecture + Official training (baseline)
Test 4: Our architecture + Our training (current)
```

**分析**:
- Test 1 vs Test 3: 訓練的影響
- Test 2 vs Test 3: 架構的影響
- Test 1 vs Test 2: 交互作用

---

## Phase 4: 改進我們的方法

### 4.1 借鑑 Meta-Audio 的優點

**如果 Meta-Audio 的架構更好**:
```python
# 採用他們的架構
class ImprovedMagnitudeNet(nn.Module):
    def __init__(self):
        # 使用 Meta-Audio 的設計
        pass
```

**如果 Meta-Audio 的訓練更好**:
```python
# 採用他們的訓練策略
optimizer = Adam(lr=their_lr)
scheduler = their_scheduler
loss = their_loss_function
```

### 4.2 結合我們的創新

**Magnitude-only + Meta-Audio 架構**:
```python
class ImprovedMagnitudeOnly(nn.Module):
    def __init__(self):
        # Meta-Audio 的 magnitude 網路
        self.magnitude_net = MetaAudioMagnitudeNet()
        
        # 我們的 physical ITD
        self.physical_itd = PhysicalITD()
    
    def forward(self, mono, view):
        # 更好的 magnitude
        mag_L, mag_R = self.magnitude_net(mono, view)
        
        # 簡單的 phase
        phase_L = mono_phase + physical_itd / 2
        phase_R = mono_phase - physical_itd / 2
        
        return mag_L * exp(1j * phase_L), mag_R * exp(1j * phase_R)
```

**預期**:
- 如果 Meta-Audio 的 magnitude 網路更好
- 我們的 magnitude-only 也會改善
- 可能接近或超越 DPATFNet

### 4.3 實驗驗證

**實驗 E9: Improved Magnitude-only**
```
Model: Meta-Audio magnitude net + Physical ITD
Target: L2 < 0.000719 (超越 DPATFNet)
```

**如果成功**:
- ✅ 證明 magnitude 網路是關鍵
- ✅ 證明 phase 學習不重要
- ✅ 更強的負面結果論文

**如果失敗**:
- ⚠️ 說明 DPATFNet 的優勢來自其他地方
- ⚠️ 需要更深入分析

---

## Phase 5: 整合到論文

### 5.1 如果改進成功

**論文結構調整**:
```
1. Introduction (不變)
2. Related Work (不變)
3. Methods
   - 8 種失敗的方法
   - Meta-Audio 分析
   - Improved Magnitude-only (成功)
4. Results
   - 8 種方法失敗
   - Improved Magnitude-only 成功
5. Analysis
   - 為什麼 phase 學習失敗
   - 為什麼 improved magnitude 成功
   - Magnitude 網路的重要性
6. Discussion (更新)
7. Conclusion (更新)
```

**新的貢獻**:
1. 系統性證明 phase 學習失敗
2. 找出 DPATFNet 的優勢來源
3. 提出更好的 magnitude-only 方法
4. 實用的解決方案

### 5.2 如果改進失敗

**論文結構不變**:
```
1-7. (原計劃)
8. Appendix: Meta-Audio Analysis
   - 複現結果
   - 差異分析
   - 為什麼我們的方法更差
```

**仍然有價值**:
1. 系統性的負面結果
2. 深入的對比分析
3. 為什麼某些方法更好

---

## 時間規劃

### Week 1: 複現 Meta-Audio
```
Day 1-2: 獲取程式碼，理解架構
Day 3-4: 適配資料集，開始訓練
Day 5-7: 訓練完成，初步對比
```

### Week 2: 分析和改進
```
Day 8-10: 詳細對比分析
Day 11-12: 實現改進方法
Day 13-14: 訓練和驗證
```

### Week 3: 整合到論文
```
Day 15-17: 更新論文內容
Day 18-19: 準備圖表和實驗
Day 20-21: 潤色和修改
```

---

## 檢查清單

### 複現階段
- [ ] Clone Meta-Audio repo
- [ ] 理解程式碼結構
- [ ] 適配我們的資料集
- [ ] 訓練 Meta-Audio 模型
- [ ] 驗證結果

### 對比階段
- [ ] 對比架構
- [ ] 對比訓練流程
- [ ] 對比推理流程
- [ ] 找出關鍵差異

### 改進階段
- [ ] 實現改進方法
- [ ] 訓練和驗證
- [ ] 對比所有方法
- [ ] 確定最佳方案

### 論文階段
- [ ] 更新論文內容
- [ ] 準備新的圖表
- [ ] 撰寫分析部分
- [ ] 完成初稿

---

## 預期結果

### 樂觀情況
```
Improved Magnitude-only: L2 < 0.000719
→ 超越 DPATFNet
→ 證明 magnitude 網路是關鍵
→ 更強的論文
```

### 中等情況
```
Improved Magnitude-only: L2 ≈ 0.000719
→ 持平 DPATFNet
→ 證明我們的分析正確
→ 實用的解決方案
```

### 悲觀情況
```
Improved Magnitude-only: L2 > 0.000719
→ 仍然比 DPATFNet 差
→ 需要更深入分析
→ 但仍有負面結果的價值
```

---

## 備用計劃

### 如果 Meta-Audio 程式碼不可用
- 從論文重新實現
- 更仔細地閱讀論文
- 聯繫作者

### 如果資料集不相容
- 轉換資料格式
- 或在他們的資料集上測試
- 對比兩個資料集的差異

### 如果時間不夠
- 優先完成負面結果論文
- Meta-Audio 分析作為 future work
- 或作為獨立的後續研究

---

## 開始行動

### 立即執行
```bash
# 1. Clone repo
cd /home/sbplab/frank
git clone https://github.com/facebookresearch/meta-audio.git

# 2. 檢查程式碼
cd meta-audio
ls -la
cat README.md

# 3. 理解結構
tree -L 2
```

### 今天完成
1. ✅ 獲取 Meta-Audio 程式碼
2. ✅ 理解基本結構
3. ✅ 規劃適配方案

### 明天開始
1. 適配資料集
2. 開始訓練
3. 並行撰寫論文

---

**Let's go!** 🚀
