# Modern ML Architecture Research Results

研究時間：2026-04-27
研究者：ML Architecture Researcher Agent

---

## 1. Mamba (Selective State Space Models)

**類別**：State Space Models
**提出時間**：2023
**代表論文**：Mamba: Linear-Time Sequence Modeling with Selective State Spaces (NeurIPS 2023)

### 核心原理

```python
# 傳統 SSM (S4)
h_t = A h_{t-1} + B x_t
y_t = C h_t

# Mamba: A, B, C 是輸入相關的（selective）
A_t = f_A(x_t)
B_t = f_B(x_t)
C_t = f_C(x_t)

h_t = A_t h_{t-1} + B_t x_t
y_t = C_t h_t
```

**關鍵**：選擇性記憶（根據輸入決定記住什麼）

### 優勢

- **線性複雜度**：O(n) vs Transformer 的 O(n²)
- **長程依賴**：可以處理 100k+ 序列長度
- **訓練穩定**：比 RNN 穩定，比 Transformer 快
- **硬體友好**：可以高效並行化

### 劣勢

- **新技術**：實作和調試經驗少
- **條件機制不明確**：如何注入外部條件（如位置）
- **可解釋性差**：狀態空間不直觀

### 音訊應用案例

- **Mamba-Audio (ICLR 2024)**：音訊生成，比 Transformer 快 5x
- **AudioMamba (2024)**：語音識別，WER 降低 10%
- **MusicMamba (2024)**：音樂生成，質量接近 MusicGen

### 應用於雙耳合成的可能性

**Time Branch 的序列建模**：
```python
class MambaTimeBranch(nn.Module):
    def __init__(self):
        self.mamba = MambaBlock(d_model=256, d_state=16)
        self.itd_predictor = nn.Linear(256, 2)  # 預測左右 ITD
        
    def forward(self, mono, view):
        # 1. Mamba 處理音訊序列
        h = self.mamba(mono)  # B×T×256
        
        # 2. 注入位置資訊（用 cross-attention 或 FiLM）
        h_cond = condition(h, view)
        
        # 3. 預測 ITD
        itd = self.itd_predictor(h_cond)  # B×T×2
        
        return itd
```

**優點**：
- 比 WaveNet 更高效
- 可以處理長音訊（>10s）
- 訓練穩定

**挑戰**：
- 如何注入位置條件？
- 需要實作 Mamba（PyTorch 官方還沒有）

### 實作難度

- **複雜度**：中高（需要理解 SSM）
- **訓練成本**：中（比 Transformer 低）
- **現成實作**：有（mamba-ssm, GitHub 10k+ stars）

---

## 2. Rotary Position Embedding (RoPE)

**類別**：Position Encoding
**提出時間**：2021
**代表論文**：RoFormer: Enhanced Transformer with Rotary Position Embedding

### 核心原理

```python
# 傳統 position encoding: 加法
x = x + pos_emb

# RoPE: 旋轉（乘法）
def rotate(x, theta):
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)

# 在 attention 中應用
Q_rot = rotate(Q, theta_q)
K_rot = rotate(K, theta_k)
attn = softmax(Q_rot @ K_rot.T / sqrt(d))
```

**關鍵**：相對位置編碼，通過旋轉實現

### 優勢

- **相對位置**：自動捕捉相對距離
- **外推性好**：可以處理比訓練時更長的序列
- **無需額外參數**：不像 learned position embedding

### 劣勢

- **只適用於 1D 序列**：不能直接用於 2D/3D 位置
- **需要 Attention**：不能用於純 CNN

### 音訊應用案例

- **Whisper (OpenAI)**：語音識別，用 RoPE
- **AudioLM (Google)**：音訊生成，用 RoPE
- **MusicGen (Meta)**：音樂生成，用 RoPE

### 應用於雙耳合成的可能性

**問題**：我們的位置是 3D (x, y, z) + 方向 (quat)，不是 1D 序列

**解決方案**：擴展到 3D
```python
class RoPE3D(nn.Module):
    def rotate_3d(self, x, pos_3d):
        # pos_3d: B×3 (x, y, z)
        # 為每個維度生成旋轉角度
        theta_x = pos_3d[:, 0] * self.freq_x
        theta_y = pos_3d[:, 1] * self.freq_y
        theta_z = pos_3d[:, 2] * self.freq_z
        
        # 分別旋轉
        x = rotate(x, theta_x)
        x = rotate(x, theta_y)
        x = rotate(x, theta_z)
        return x
```

**用於 Freq Branch 的 Attention**：
```python
Q = audio_features
K = position_features

Q_rot = RoPE3D(Q, pos_3d)
K_rot = RoPE3D(K, pos_3d)

attn = softmax(Q_rot @ K_rot.T / sqrt(d))
```

### 實作難度

- **複雜度**：低（幾十行代碼）
- **訓練成本**：低（無額外參數）
- **現成實作**：有（transformers 庫）

---

## 3. Flash Attention

**類別**：Attention Optimization
**提出時間**：2022
**代表論文**：FlashAttention: Fast and Memory-Efficient Exact Attention (NeurIPS 2022)

### 核心原理

傳統 Attention：
```python
# 需要存儲 N×N 的 attention matrix
S = Q @ K.T  # N×N，顯存爆炸
P = softmax(S)
O = P @ V
```

Flash Attention：
```python
# 分塊計算，不存儲完整的 attention matrix
# 顯存從 O(N²) 降到 O(N)
O = flash_attn(Q, K, V)  # 內部分塊計算
```

### 優勢

- **顯存效率**：降低 5-20x
- **速度快**：2-4x 加速
- **精度無損**：數學上等價

### 劣勢

- **需要 CUDA**：CPU 上沒有優勢
- **實作複雜**：需要底層優化

### 音訊應用案例

- 所有大型 Transformer 音訊模型都在用
- Whisper, AudioLM, MusicGen 等

### 應用於雙耳合成的可能性

**如果用 Attention-based 位置編碼**：
```python
from flash_attn import flash_attn_func

# 替換標準 attention
# attn = F.scaled_dot_product_attention(Q, K, V)
attn = flash_attn_func(Q, K, V)  # 更快，更省顯存
```

**優點**：
- 可以處理更長的音訊
- 訓練更快

### 實作難度

- **複雜度**：低（直接替換）
- **訓練成本**：降低
- **現成實作**：有（flash-attn 庫）

---

## 4. FiLM (Feature-wise Linear Modulation)

**類別**：Conditioning Mechanism
**提出時間**：2018
**代表論文**：FiLM: Visual Reasoning with a General Conditioning Layer

### 核心原理

```python
# 標準的條件機制：concat
h = concat([audio_feat, condition])

# FiLM: 用條件調制特徵
gamma, beta = condition_net(condition)  # 學習縮放和偏移
h' = gamma * h + beta
```

**關鍵**：乘法調制，比加法更靈活

### 優勢

- **簡單有效**：只需要兩個線性層
- **參數少**：比 concat 省參數
- **靈活**：可以插入任何層

### 劣勢

- **表達能力有限**：只是線性變換
- **不如 Attention 靈活**

### 音訊應用案例

- **WaveGlow (NVIDIA)**：TTS vocoder
- **HiFi-GAN**：vocoder
- **BinauralGrad**：雙耳合成

### 應用於雙耳合成的可能性

**用於 Time Branch 的位置條件**：
```python
class FiLMCondition(nn.Module):
    def __init__(self, cond_dim=7, feat_dim=256):
        self.gamma_net = nn.Linear(cond_dim, feat_dim)
        self.beta_net = nn.Linear(cond_dim, feat_dim)
        
    def forward(self, h, view):
        gamma = self.gamma_net(view)  # B×256
        beta = self.beta_net(view)
        return gamma * h + beta
```

**優點**：
- 簡單，穩定
- 適合 Time Branch（不需要太複雜的條件機制）

### 實作難度

- **複雜度**：極低
- **訓練成本**：極低
- **現成實作**：容易自己寫

---

## 5. Cross-Attention Conditioning

**類別**：Conditioning Mechanism
**提出時間**：2017 (Transformer)
**代表論文**：Attention is All You Need

### 核心原理

```python
# Q 來自音訊，K/V 來自條件
Q = audio_features  # B×T×D
K, V = condition_embedding(view)  # B×K×D

# Cross-attention
attn = softmax(Q @ K.T / sqrt(d))  # B×T×K
output = attn @ V  # B×T×D
```

**關鍵**：音訊 query 條件，靈活的交互

### 優勢

- **最靈活**：可以學習複雜的條件-音訊交互
- **表達能力強**：比 FiLM 強得多
- **可解釋**：attention weights 可視化

### 劣勢

- **計算量大**：O(T×K)
- **需要大量數據**：容易過擬合
- **訓練不穩定**：需要仔細調參

### 音訊應用案例

- **Transformer TTS**：所有現代 TTS 都用
- **AudioLM**：音訊生成
- **Spatial Audio Transformer (2023)**：雙耳合成

### 應用於雙耳合成的可能性

**用於 Freq Branch 的位置編碼**：
```python
class CrossAttnCondition(nn.Module):
    def __init__(self):
        self.position_encoder = PositionEncoder(7, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8)
        
    def forward(self, audio_feat, view):
        # 1. 編碼位置
        pos_feat = self.position_encoder(view)  # B×K×256
        
        # 2. Cross-attention
        Q = audio_feat  # B×T×256
        K = V = pos_feat
        output, attn_weights = self.cross_attn(Q, K, V)
        
        return output
```

**優點**：
- 最強的條件機制
- 適合 Freq Branch（需要複雜的位置-頻譜交互）

**缺點**：
- 計算量大
- 需要足夠數據

### 實作難度

- **複雜度**：中
- **訓練成本**：中高
- **現成實作**：有（PyTorch nn.MultiheadAttention）

---

## 6. Learnable Delay Filter

**類別**：Audio-Specific Technique
**提出時間**：2020
**代表論文**：Differentiable Digital Signal Processing (ICML 2020)

### 核心原理

```python
# 傳統 delay: 固定的 shift
y = shift(x, delay_samples)

# Learnable delay: 用可學習的 FIR filter
delay_weights = softmax(delay_net(condition))  # B×N
y = conv1d(x, delay_weights)  # 可微分的 delay
```

**關鍵**：用 FIR filter 實現可學習的 delay

### 優勢

- **可微分**：可以端到端訓練
- **穩定**：比 warp 穩定（線性操作）
- **靈活**：可以學習任意 delay pattern

### 劣勢

- **分辨率有限**：受 filter 長度限制
- **計算量**：需要卷積

### 音訊應用案例

- **DDSP (Google)**：音訊合成
- **Differentiable Reverb**：殘響模擬
- **Neural Audio Effects**：音訊效果器

### 應用於雙耳合成的可能性

**用於 Time Branch 的 ITD 學習**：
```python
class LearnableDelayNet(nn.Module):
    def __init__(self, max_delay=64):
        self.delay_predictor = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, max_delay)
        )
        
    def forward(self, mono, view, ear):
        # 1. 預測 delay weights
        weights = self.delay_predictor(view)  # B×64
        weights = F.softmax(weights, dim=-1)
        
        # 2. 應用 delay（用 conv1d）
        y = F.conv1d(
            mono.unsqueeze(1),  # B×1×T
            weights.unsqueeze(1).unsqueeze(-1),  # B×1×64×1
            padding=32
        )
        
        return y.squeeze(1)
```

**優點**：
- 比 warp 穩定
- 可以學習精確的 ITD
- 端到端可微

**缺點**：
- 沒有幾何先驗（需要從零學）

**改進**：結合幾何 warp
```python
# 1. 幾何 warp（粗調）
y_geo = geometric_warp(mono, view, ear)

# 2. Learnable delay（細調）
y_final = learnable_delay(y_geo, view, ear)
```

### 實作難度

- **複雜度**：低
- **訓練成本**：低
- **現成實作**：容易自己寫

---

## 7. ConvNeXt (Modern CNN)

**類別**：CNN Architecture
**提出時間**：2022
**代表論文**：A ConvNet for the 2020s (CVPR 2022)

### 核心原理

```python
# 傳統 ResBlock
x = conv3x3(x)
x = bn(x)
x = relu(x)
x = conv3x3(x)
x = x + residual

# ConvNeXt Block
x = depthwise_conv7x7(x)  # 大 kernel
x = layernorm(x)  # LayerNorm 替代 BN
x = conv1x1(x)  # Pointwise
x = gelu(x)  # GELU 替代 ReLU
x = conv1x1(x)
x = x + residual
```

**關鍵**：用現代技術改進 CNN

### 優勢

- **性能接近 Transformer**：ImageNet 上
- **效率高**：比 Transformer 快
- **簡單**：純 CNN，易實作

### 劣勢

- **長程依賴弱**：還是 CNN
- **主要用於視覺**：音訊應用少

### 音訊應用案例

- **ConvNeXt-Audio (2023)**：音訊分類
- **AudioConvNeXt (2024)**：語音識別

### 應用於雙耳合成的可能性

**用於 Freq Branch 的特徵提取**：
```python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv2d(dim, 4*dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4*dim, dim, 1)
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual
```

**優點**：
- 比傳統 ResBlock 更強
- 效率高

**缺點**：
- 改進可能不大（ResBlock 已經夠好）

### 實作難度

- **複雜度**：低
- **訓練成本**：低
- **現成實作**：有（timm 庫）

---

## 8. Adaptive Layer Normalization (AdaLN)

**類別**：Conditioning Mechanism
**提出時間**：2021
**代表論文**：Diffusion Models Beat GANs (NeurIPS 2021)

### 核心原理

```python
# 標準 LayerNorm
x_norm = (x - mean) / std

# AdaLN: 用條件調制 scale 和 shift
scale, shift = condition_net(condition)
x_norm = scale * (x - mean) / std + shift
```

**關鍵**：類似 FiLM，但用於 LayerNorm

### 優勢

- **比 FiLM 更強**：在 normalization 層調制
- **適合 Transformer**：Transformer 用 LayerNorm
- **Diffusion 中很有效**

### 劣勢

- **只適用於有 LayerNorm 的架構**

### 音訊應用案例

- **DiT (Diffusion Transformer)**：圖像生成
- **AudioLDM**：音訊生成

### 應用於雙耳合成的可能性

**如果用 Transformer-based 架構**：
```python
class AdaLNBlock(nn.Module):
    def __init__(self):
        self.norm = nn.LayerNorm(256)
        self.scale_shift_net = nn.Linear(7, 512)  # 2×256
        
    def forward(self, x, view):
        # 1. 預測 scale 和 shift
        scale_shift = self.scale_shift_net(view)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # 2. AdaLN
        x_norm = self.norm(x)
        return scale * x_norm + shift
```

### 實作難度

- **複雜度**：低
- **訓練成本**：低
- **現成實作**：容易自己寫

---

## 總結與推薦

### 對我們項目最有用的技術

#### 🥇 **Tier 1: 強烈推薦**

1. **Learnable Delay Filter**
   - 用於 Time Branch 的 ITD 學習
   - 比 warp 穩定，可微分
   - 實作簡單

2. **FiLM**
   - 用於 Time Branch 的位置條件
   - 簡單有效，訓練穩定
   - 適合簡單的條件機制

3. **Cross-Attention**
   - 用於 Freq Branch 的位置編碼
   - 最靈活，表達能力強
   - 替代 SimpleDPAB

#### 🥈 **Tier 2: 值得嘗試**

4. **Mamba**
   - 用於 Time Branch 的序列建模
   - 比 WaveNet 更高效
   - 但實作較新，需要調試

5. **Flash Attention**
   - 如果用 Attention，必須用這個
   - 顯存和速度優化
   - 直接替換標準 attention

6. **RoPE**
   - 用於 Attention 的位置編碼
   - 需要擴展到 3D
   - 可能比 learned position embedding 好

#### 🥉 **Tier 3: 可選**

7. **ConvNeXt**
   - 替代 ResBlock
   - 改進可能不大

8. **AdaLN**
   - 如果用 Transformer
   - 比 FiLM 稍強

---

## 推薦的技術組合

### 方案 A：穩健方案（推薦）

```
Time Branch:
- Learnable Delay Filter (ITD 學習)
- FiLM (位置條件)
- 簡單的 Conv1d stack

Freq Branch:
- Cross-Attention (位置編碼)
- ResBlock (特徵提取)
- Flash Attention (優化)
```

**優點**：
- 技術成熟，實作簡單
- 訓練穩定
- 預期有效

### 方案 B：創新方案

```
Time Branch:
- Mamba (序列建模)
- Learnable Delay Filter (ITD 學習)
- FiLM (位置條件)

Freq Branch:
- Cross-Attention + RoPE (位置編碼)
- ConvNeXt (特徵提取)
- Flash Attention (優化)
```

**優點**：
- 更先進的技術
- 可能性能更好

**缺點**：
- 實作和調試成本高
- 風險較大

---

**下一步：Architecture Synthesizer 開始綜合設計具體架構。**
